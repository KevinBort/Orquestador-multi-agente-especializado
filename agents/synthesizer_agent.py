"""Agente SINTETIZADOR: redacta la respuesta final con los resultados del estado.
Sin herramientas: una sola llamada al LLM con un contexto compacto."""

import re

from langchain_core.messages import AIMessage

import config

from agents.common import texto
from prompts import SYNTHESIZER_SYSTEM
from state import AgentState


def construir_contexto(state: AgentState) -> str:
    pregunta = state["messages"][0].content
    inv = state.get("research_output") or {}
    ana = state.get("analysis_output") or {}
    lineas = [f"PREGUNTA DEL USUARIO: {pregunta}", ""]

    fuentes = sorted({d["fuente"] for d in inv.get("documentos", [])})
    lineas.append("DATOS DEL INVESTIGADOR:")
    lineas.append(inv.get("resumen") or "(sin datos)")
    if fuentes:
        lineas.append(f"Fuentes documentales: {', '.join(fuentes)}")
    if inv.get("resenas"):
        lineas.append(f"Reseñas de socios recuperadas: {len(inv['resenas'])}")

    lineas.append("\nRESULTADOS DEL ANALISTA (calculados con herramientas):")
    s = ana.get("sentimiento")
    if s:
        c, p = s["conteo"], s["porcentajes"]
        lineas.append(
            f"- Sentimiento de {s['total']} reseñas: {c['positivo']} positivas ({p['positivo']}%), "
            f"{c['neutral']} neutrales ({p['neutral']}%), {c['negativo']} negativas ({p['negativo']}%)."
        )
    for k in ana.get("calculos", []):
        lineas.append(f"- Cálculo: {k['expresion']} = {k['resultado']}")
    if not s and not ana.get("calculos"):
        lineas.append("(el analista no aportó resultados)")

    return "\n".join(lineas)


# Si el LLM escribe igualmente una "Limitación...", se elimina: las reales las calcula el código.
_LIMITACION_DEL_LLM = re.compile(r"^[ \t*_#>-]*limitaci[oó]n(es)?\b.*", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def limpiar_limitaciones(respuesta: str) -> str:
    return _LIMITACION_DEL_LLM.sub("", respuesta).strip()


def _total_resenas_indexadas() -> int | None:
    try:
        from vector_db import abrir_coleccion
        return abrir_coleccion(config.COLECCION_RESENAS)._collection.count()
    except Exception:
        return None


def calcular_limitaciones(state: AgentState) -> list[str]:
    """Limitaciones REALES, deducidas del estado (no inventadas por el modelo)."""
    from supervisor import necesita_analisis
    ro = state.get("research_output") or {}
    ao = state.get("analysis_output") or {}
    out = []
    if "límite" in (state.get("instruction") or "").lower():
        out.append("El supervisor cerró por límite de pasos: el resultado puede no haber pasado toda su validación.")
    if ro.get("errores"):
        out.append(f"Hubo {len(ro['errores'])} error(es) de herramientas durante la investigación.")
    if necesita_analisis(state["messages"][0].content) and not (ao.get("sentimiento") or ao.get("calculos")):
        out.append("No se pudo completar el análisis solicitado.")
    s = ao.get("sentimiento")
    if s:
        total = _total_resenas_indexadas()
        if total and s["total"] < total:
            out.append(f"Los porcentajes se calcularon sobre {s['total']} de {total} reseñas "
                       "(las más relacionadas con la consulta), no sobre todas.")
    return out


def make_synthesizer_node(llm):
    def synthesizer_node(state: AgentState) -> dict:
        paso = state.get("step_count", 0)
        respuesta = limpiar_limitaciones(texto(llm.invoke([
            ("system", SYNTHESIZER_SYSTEM),
            ("user", construir_contexto(state)),
        ]).content))
        limites = calcular_limitaciones(state)
        if limites:
            respuesta += "\n\nLimitaciones:\n" + "\n".join(f"- {x}" for x in limites)
        return {
            "final_answer": respuesta,
            "task_completed": True,
            "messages": [AIMessage(content=respuesta, name="synthesizer")],
            "contributions": [{"agent": "synthesizer", "step": paso, "summary": "respuesta final redactada"}],
        }

    return synthesizer_node

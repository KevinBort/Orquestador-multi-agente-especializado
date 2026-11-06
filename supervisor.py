"""Nodo SUPERVISOR: router inteligente + controlador de flujo.

Combina dos capas, pensadas para un modelo pequeño:
  1. El LLM decide (salida estructurada con Literal) usando una rúbrica en el prompt.
  2. Guardas deterministas en Python corrigen decisiones que rompen las reglas
     (p. ej. sintetizar sin datos) y GARANTIZAN que el flujo termina:
       - MAX_STEPS: pasado el límite se fuerza al sintetizador.
       - MAX_REFINAMIENTOS: un especialista no puede repetirse sin fin.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

import config
from prompts import SUPERVISOR_SYSTEM
from state import AgentState

Agente = Literal["researcher", "analyst", "synthesizer"]

# Pistas de que la pregunta exige análisis (sentimiento o cálculo)
_NECESITA_ANALISIS = re.compile(
    r"porcentaje|%|sentimiento|opini|rese[ñn]|valoraci|calcul|promedio|"
    r"al a[ñn]o|por a[ñn]o|anual|en total|ahorr", re.IGNORECASE)


class Decision(BaseModel):
    # 'reason' va primero: el modelo razona antes de elegir.
    reason: str = Field(description="Una frase: por qué se elige ese agente")
    next: Agente = Field(description="Quién interviene ahora")
    instruction: str = Field(description="Tarea concreta y breve para ese agente; vacía si next es synthesizer")


# ------------------------------------------------------------------ utilidades
def _intervino(state: AgentState, agente: str) -> bool:
    return any(c["agent"] == agente for c in state.get("contributions", []))


def _hay_datos(state: AgentState) -> bool:
    ro = state.get("research_output") or {}
    return bool(ro.get("documentos") or ro.get("resenas"))


def necesita_analisis(pregunta: str) -> bool:
    return bool(_NECESITA_ANALISIS.search(pregunta))


def _siguiente_logico(state: AgentState, pregunta: str) -> Agente:
    if necesita_analisis(pregunta) and not _intervino(state, "analyst") and _hay_datos(state):
        return "analyst"
    return "synthesizer"


def instruccion_por_defecto(agente: str, pregunta: str) -> str:
    if agente == "researcher":
        return f"Busca los datos necesarios para responder: {pregunta}"
    if agente == "analyst":
        return f"Analiza los datos del investigador (sentimiento y/o cálculos) para responder: {pregunta}"
    return ""


def resumen_estado(state: AgentState) -> str:
    """Panel compacto para el supervisor (no se le pasa el historial completo)."""
    ro = state.get("research_output") or {}
    ao = state.get("analysis_output") or {}
    s = ao.get("sentimiento")
    lineas = [
        f"- Paso: {state.get('step_count', 0) + 1} de {config.MAX_STEPS}",
        f"- Investigador: {len(ro.get('documentos', []))} fragmentos, {len(ro.get('resenas', []))} reseñas."
        + (f" Resumen: {ro['resumen'][:300]}" if ro.get("resumen") else " Aún no intervino.")
        + (f" ERRORES DE HERRAMIENTAS: {ro['errores'][-2:]}" if ro.get("errores") else ""),
        "- Analista: "
        + (f"sentimiento de {s['total']} reseñas ({s['porcentajes']}); " if s else "sin sentimiento; ")
        + f"{len(ao.get('calculos', []))} cálculos"
        + (f" {ao['calculos']}" if ao.get("calculos") else "")
        + ("" if _intervino(state, "analyst") else " (aún no intervino)"),
        f"- Refinamientos usados: {state.get('refinements') or {}} (máximo {config.MAX_REFINAMIENTOS} por agente)",
    ]
    return "\n".join(lineas)


def reglas_de_respaldo(state: AgentState, pregunta: str) -> tuple[Agente, str, str]:
    """Si el LLM falla (JSON inválido, etc.) se enruta con reglas simples."""
    if not _hay_datos(state):
        return "researcher", "respaldo: faltan datos", instruccion_por_defecto("researcher", pregunta)
    sig = _siguiente_logico(state, pregunta)
    return sig, "respaldo por reglas", instruccion_por_defecto(sig, pregunta)


def aplicar_guardas(sig: Agente, state: AgentState, pregunta: str) -> tuple[Agente, str]:
    """Corrige decisiones que rompen las reglas. Devuelve (agente, motivo del cambio)."""
    ref = state.get("refinements") or {}
    motivo = ""

    # 1) Un especialista no puede repetirse más de MAX_REFINAMIENTOS veces
    if sig in ("researcher", "analyst") and _intervino(state, sig) \
            and ref.get(sig, 0) >= config.MAX_REFINAMIENTOS:
        sig, motivo = _siguiente_logico(state, pregunta), f"refinamientos agotados"

    # 2) No sintetizar sin haber analizado lo que la pregunta exige
    if sig == "synthesizer" and necesita_analisis(pregunta) \
            and not _intervino(state, "analyst") and _hay_datos(state):
        sig, motivo = "analyst", "la pregunta exige análisis y aún no se hizo"

    # 3) Nada de analizar ni sintetizar sin datos (salvo que el investigador ya agotó sus intentos)
    investigador_agotado = _intervino(state, "researcher") and ref.get("researcher", 0) >= config.MAX_REFINAMIENTOS
    if sig in ("analyst", "synthesizer") and not _hay_datos(state) and not investigador_agotado:
        sig, motivo = "researcher", "no hay datos investigados"

    return sig, motivo


# ------------------------------------------------------------------------ nodo
def make_supervisor(llm):
    decidir = llm.with_structured_output(Decision)

    def supervisor_node(state: AgentState) -> dict:
        paso = state.get("step_count", 0) + 1
        pregunta = state["messages"][0].content
        ref = dict(state.get("refinements") or {})

        if paso > config.MAX_STEPS:
            sig, razon = "synthesizer", "límite de pasos alcanzado"
            instr = "Se alcanzó el límite de pasos: responde con lo disponible e indica qué quedó incompleto."
        else:
            d = None
            try:
                d = decidir.invoke([
                    ("system", SUPERVISOR_SYSTEM),
                    ("user", f"PREGUNTA DEL USUARIO: {pregunta}\n\nESTADO ACTUAL:\n{resumen_estado(state)}\n\n¿Quién interviene ahora?"),
                ])
            except Exception:
                d = None
            if d is None:
                sig, razon, instr = reglas_de_respaldo(state, pregunta)
            else:
                sig, razon, instr = d.next, d.reason, d.instruction

            sig2, motivo = aplicar_guardas(sig, state, pregunta)
            if sig2 != sig:
                razon, sig = f"{razon} | GUARDA: {motivo}", sig2
                instr = instruccion_por_defecto(sig, pregunta)
            # Un modelo pequeño a veces pone el nombre del agente como instrucción ("researcher")
            if sig != "synthesizer" and (len(instr.strip()) < 15 or instr.strip().lower() in
                                         ("researcher", "analyst", "synthesizer")):
                instr = instruccion_por_defecto(sig, pregunta)
            if sig == "synthesizer":
                instr = ""   # el sintetizador no recibe tarea; evita que el LLM escriba "synthesizer"

        # ¿Es un refinamiento? (el agente ya había intervenido antes)
        feedback = ""
        if sig in ("researcher", "analyst") and _intervino(state, sig):
            ref[sig] = ref.get(sig, 0) + 1
            feedback = instr

        return {
            "next_agent": sig,
            "instruction": instr,
            "supervisor_feedback": feedback,
            "step_count": paso,
            "refinements": ref,
            "contributions": [{"agent": "supervisor", "step": paso, "summary": f"-> {sig}: {razon}"}],
        }

    return supervisor_node


def route(state: AgentState) -> Literal["researcher", "analyst", "synthesizer"]:
    """Arista condicional: el nodo siguiente es lo que decidió el supervisor."""
    return state["next_agent"]

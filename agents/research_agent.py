"""Agente INVESTIGADOR: ReAct con herramientas acotadas de búsqueda.

Recibe SOLO la instrucción del supervisor (no el historial). Los datos completos
los copia al estado el propio nodo desde los artifacts de las herramientas,
sin pasar por el LLM.
"""

from langgraph.prebuilt import create_react_agent

from agents.common import artifacts, ultimo_texto
from prompts import RESEARCHER_SYSTEM
from state import AgentState
from tools.search_tools import SEARCH_TOOLS


def make_research_node(llm):
    agent = create_react_agent(llm, SEARCH_TOOLS, prompt=RESEARCHER_SYSTEM)

    def research_node(state: AgentState) -> dict:
        paso = state.get("step_count", 0)
        tarea = state.get("instruction") or state["messages"][0].content
        fb = state.get("supervisor_feedback", "")
        pedido = tarea + (f"\nCORRECCIÓN PEDIDA POR EL SUPERVISOR: {fb}" if fb else "")
        previo = state.get("research_output") or {}

        def correr(texto: str):
            return agent.invoke({"messages": [("user", texto)]}, config={"recursion_limit": 10})

        try:
            res = correr(pedido)
            if not artifacts(res["messages"]):
                # El modelo respondió de memoria sin buscar. Se reintenta aquí mismo,
                # sin gastar un paso (ni un refinamiento) del supervisor.
                res = correr(pedido + "\nIMPORTANTE: debes usar una herramienta de búsqueda; "
                                      "no respondas de memoria.")
        except Exception as e:  # el supervisor verá que no hay datos y decidirá
            return {"contributions": [{"agent": "researcher", "step": paso,
                                       "summary": f"FALLÓ: {type(e).__name__}"}]}

        docs = list(previo.get("documentos", []))
        resenas = list(previo.get("resenas", []))
        nuevos_docs = nuevas_resenas = 0
        errores = list(previo.get("errores", []))
        for a in artifacts(res["messages"]):
            if a.get("tipo") == "error":
                errores.append(a.get("detalle", "error desconocido")[:200])
            elif a.get("tipo") == "documentos":
                for f in a["fragmentos"]:
                    if f not in docs:
                        docs.append(f); nuevos_docs += 1
            elif a.get("tipo") == "resenas":
                ids = {r["id"] for r in resenas}
                for r in a["resenas"]:
                    if r["id"] not in ids:
                        resenas.append(r); nuevas_resenas += 1

        resumen = ultimo_texto(res["messages"])
        if previo.get("resumen") and resumen:
            resumen = (previo["resumen"] + "\n" + resumen)[-1500:]
        resumen = resumen or previo.get("resumen", "")

        salida = {"documentos": docs, "resenas": resenas, "resumen": resumen, "errores": errores}
        extra = f", {len(errores)} ERRORES de herramientas" if errores else ""
        return {
            "research_output": salida,
            "messages": [("assistant", f"[researcher] {resumen}")],
            "contributions": [{"agent": "researcher", "step": paso,
                               "summary": f"+{nuevos_docs} fragmentos, +{nuevas_resenas} reseñas{extra}"}],
        }

    return research_node

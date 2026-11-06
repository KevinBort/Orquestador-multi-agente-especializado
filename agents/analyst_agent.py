"""Agente ANALISTA: ReAct con herramientas de cómputo.

Recibe la instrucción y el RESUMEN del investigador (no todo el historial).
Las reseñas completas las lee la herramienta directamente del estado (InjectedState).
"""

from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import create_react_agent

from agents.common import artifacts, ultimo_texto
from prompts import ANALYST_SYSTEM
from state import AgentState
from tools.analysis_tools import ANALYSIS_TOOLS


class AnalystState(MessagesState):
    """Estado del sub-agente: solo lo que sus herramientas necesitan leer."""
    research_output: dict
    remaining_steps: RemainingSteps


def make_analyst_node(llm):
    agent = create_react_agent(llm, ANALYSIS_TOOLS, prompt=ANALYST_SYSTEM,
                               state_schema=AnalystState)

    def analyst_node(state: AgentState) -> dict:
        paso = state.get("step_count", 0)
        inv = state.get("research_output") or {}
        previo = state.get("analysis_output") or {}
        tarea = state.get("instruction") or state["messages"][0].content
        fb = state.get("supervisor_feedback", "")
        pedido = (
            f"TAREA: {tarea}\n"
            + (f"CORRECCIÓN PEDIDA POR EL SUPERVISOR: {fb}\n" if fb else "")
            + f"DATOS DEL INVESTIGADOR:\n{inv.get('resumen', '(sin resumen)')}\n"
            f"Reseñas disponibles para analizar: {len(inv.get('resenas', []))}"
        )

        try:
            res = agent.invoke(
                {"messages": [("user", pedido)], "research_output": inv},
                config={"recursion_limit": 10},
            )
        except Exception as e:
            return {"contributions": [{"agent": "analyst", "step": paso,
                                       "summary": f"FALLÓ: {type(e).__name__}"}]}

        sentimiento = previo.get("sentimiento")
        calculos = list(previo.get("calculos", []))
        nuevos = 0
        for a in artifacts(res["messages"]):
            if a.get("tipo") == "sentimiento":
                sentimiento = a
            elif a.get("tipo") == "calculo":
                item = {"expresion": a["expresion"], "resultado": a["resultado"]}
                if item not in calculos:
                    calculos.append(item); nuevos += 1

        resumen = ultimo_texto(res["messages"]) or previo.get("resumen", "")
        partes = []
        if sentimiento:
            partes.append(f"sentimiento de {sentimiento['total']} reseñas")
        partes.append(f"+{nuevos} cálculos")
        return {
            "analysis_output": {"sentimiento": sentimiento, "calculos": calculos, "resumen": resumen},
            "messages": [("assistant", f"[analyst] {resumen}")],
            "contributions": [{"agent": "analyst", "step": paso, "summary": ", ".join(partes)}],
        }

    return analyst_node

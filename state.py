"""Estado compartido del grafo.

Es el "tablero" que todos los nodos leen y escriben. Cada campo tiene un dueño
claro, así se sabe qué agente aportó qué y no se pierde contexto.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


class Contribution(TypedDict):
    """Una entrada del registro de trazabilidad."""
    agent: str      # quién aportó
    step: int       # en qué paso del supervisor
    summary: str    # qué aportó (una línea)


class AgentState(MessagesState):
    # --- control de flujo (dueño: supervisor) ---
    next_agent: str                 # researcher | analyst | synthesizer | FINISH
    instruction: str                # tarea concreta para el siguiente especialista
    supervisor_feedback: str        # qué corregir si se pide refinamiento
    step_count: int                 # contador anti-bucle
    refinements: dict               # {"researcher": 0, "analyst": 1}: repeticiones usadas
    task_completed: bool

    # --- resultados (dueño: cada especialista) ---
    research_output: dict           # {"documentos": [...], "resenas": [...], "resumen": str}
    analysis_output: dict           # {"sentimiento": {...}, "calculos": [...], "resumen": str}
    final_answer: str               # (dueño: sintetizador)

    # --- trazabilidad: la lista se ACUMULA (reducer), nunca se pisa ---
    contributions: Annotated[list[Contribution], operator.add]


def initial_state(question: str) -> dict:
    """Estado inicial para lanzar el grafo con una pregunta del usuario."""
    return {
        "messages": [("user", question)],
        "next_agent": "",
        "instruction": "",
        "supervisor_feedback": "",
        "step_count": 0,
        "refinements": {},
        "task_completed": False,
        "research_output": {},
        "analysis_output": {},
        "final_answer": "",
        "contributions": [],
    }

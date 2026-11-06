"""Grafo principal: topología jerárquica hub-and-spoke.

    START -> supervisor -> {researcher | analyst | synthesizer}
    researcher -> supervisor      (todos los especialistas vuelven al hub)
    analyst    -> supervisor
    synthesizer -> END
"""

from langgraph.graph import END, START, StateGraph

from agents.analyst_agent import make_analyst_node
from agents.research_agent import make_research_node
from agents.synthesizer_agent import make_synthesizer_node
from llm import get_llm
from state import AgentState
from supervisor import make_supervisor, route


def build_graph(llm=None):
    llm = llm or get_llm(0)
    g = StateGraph(AgentState)
    g.add_node("supervisor", make_supervisor(llm))
    g.add_node("researcher", make_research_node(llm))
    g.add_node("analyst", make_analyst_node(llm))
    g.add_node("synthesizer", make_synthesizer_node(llm))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route)   # el Literal del retorno define los destinos
    g.add_edge("researcher", "supervisor")
    g.add_edge("analyst", "supervisor")
    g.add_edge("synthesizer", END)
    return g.compile()


def export_diagram(graph, carpeta="assets"):
    """Guarda el diagrama Mermaid (.mmd) y, si hay internet, también el PNG."""
    from pathlib import Path
    Path(carpeta).mkdir(exist_ok=True)
    mermaid = graph.get_graph().draw_mermaid()
    (Path(carpeta) / "graph.mmd").write_text(mermaid, encoding="utf-8")
    try:
        (Path(carpeta) / "graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    except Exception:
        pass  # draw_mermaid_png usa un servicio web; el .mmd siempre se guarda
    return mermaid

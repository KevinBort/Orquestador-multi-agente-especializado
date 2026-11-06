"""Muestra QUÉ busca realmente el investigador y QUÉ recibe, paso a paso.

  python diagnostico_investigador.py
  python diagnostico_investigador.py "tu pregunta"

Imprime cada llamada a herramienta (con la consulta que escribió el modelo), los
fragmentos que volvieron y el resumen final. Sirve para saber si un fallo es de
búsqueda (el fragmento correcto nunca llegó) o del modelo (llegó y no lo usó).
Requiere Ollama corriendo.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agents.common import texto
from llm import get_llm
from prompts import RESEARCHER_SYSTEM
from tools.search_tools import SEARCH_TOOLS

PREGUNTA = "¿Cuándo es la asamblea general y quién preside el consejo coordinador?"
PISTAS = ("primer domingo", "Galíndez")


def main(llm=None, pregunta: str = PREGUNTA) -> None:
    agente = create_react_agent(llm or get_llm(0), SEARCH_TOOLS, prompt=RESEARCHER_SYSTEM)
    pedido = f"Busca los datos necesarios para responder: {pregunta}"
    print(f"TAREA ENVIADA AL INVESTIGADOR:\n  {pedido}\n")
    res = agente.invoke({"messages": [("user", pedido)]}, config={"recursion_limit": 10})

    for m in res["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for c in m.tool_calls:
                print(f"LLAMADA  {c['name']}  consulta = {c['args'].get('consulta')!r}")
        elif isinstance(m, ToolMessage):
            art = m.artifact if isinstance(m.artifact, dict) else {}
            if art.get("tipo") == "documentos":
                for f in art["fragmentos"]:
                    hay = any(p in f["texto"] for p in PISTAS)
                    print(f"   -> {f['fuente']} :: {f['texto'][:60].replace(chr(10), ' ')!r}"
                          + ("   <-- CONTIENE LA RESPUESTA" if hay else ""))
            else:
                print(f"   -> {str(m.content)[:120]}")
    print(f"\nRESUMEN DEL INVESTIGADOR:\n  {texto(res['messages'][-1].content)}")


if __name__ == "__main__":
    main(pregunta=" ".join(sys.argv[1:]) or PREGUNTA)

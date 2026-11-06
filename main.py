"""Punto de entrada.

  python main.py                      # consulta de demostración
  python main.py "tu pregunta"        # consulta propia
  python main.py --diagram            # solo exporta el diagrama del grafo
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from graph import build_graph, export_diagram
from state import initial_state

DEMO = ("Investiga cuánto cuesta la cuota para un socio jubilado y qué opinan los socios "
        "sobre las cuotas. Dime el porcentaje de opiniones positivas y calcula cuánto "
        "pagaría un jubilado en un año.")


def mostrar(nodo: str, delta: dict) -> None:
    ultima = (delta.get("contributions") or [{}])[-1]
    if nodo == "supervisor":
        print(f"\n[paso {delta['step_count']}] SUPERVISOR {ultima['summary']}")
        if delta.get("instruction"):
            print(f"          instrucción: {delta['instruction']}")
    else:
        print(f"          {nodo.upper()}: {ultima.get('summary', '')}")


def ejecutar(pregunta: str) -> dict:
    graph = build_graph()
    estado = initial_state(pregunta)
    print(f"PREGUNTA: {pregunta}")
    for chunk in graph.stream(estado, stream_mode="updates", config={"recursion_limit": 40}):
        for nodo, delta in chunk.items():
            mostrar(nodo, delta)
            estado.update({k: v for k, v in delta.items() if k not in ("messages", "contributions")})
            estado["contributions"] = estado.get("contributions", []) + delta.get("contributions", [])
    print("\n" + "=" * 70 + "\nRESPUESTA FINAL\n" + "=" * 70)
    print(estado.get("final_answer", "(sin respuesta)"))
    return estado


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--diagram":
        print(export_diagram(build_graph()))
    else:
        ejecutar(" ".join(args) or DEMO)

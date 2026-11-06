"""Comprueba que TODOS los archivos del proyecto son la versión actual.

  python verificar_instalacion.py

Útil cuando se reemplazan archivos de a uno: un solo archivo viejo basta para que
el sistema se comporte como antes de las correcciones.
"""

from pathlib import Path

RAIZ = Path(__file__).parent

# archivo -> (texto que solo existe en la versión actual, qué corrige)
MARCAS = {
    "vector_db.py": ("reiniciar_cliente", "cliente de Chroma seguro entre hilos"),
    "retrieval.py": ("fusionar_rrf", "búsqueda híbrida"),
    "tools/search_tools.py": ("buscar_hibrido", "la herramienta usa la búsqueda híbrida"),
    "config.py": ('os.getenv("K_DOCS", "4")', "K_DOCS = 4 (editable desde el .env)"),
    "agents/research_agent.py": ("IMPORTANTE", "reintento si el modelo no busca"),
    "agents/synthesizer_agent.py": ("calcular_limitaciones", "limitaciones calculadas por código"),
    "supervisor.py": ("el sintetizador no recibe tarea", "instrucción vacía para el sintetizador"),
    "prompts.py": ("NO escribas limitaciones", "el LLM no inventa limitaciones"),
    "test_graph.py": ("test_reintento_si_no_busca", "pruebas del grafo completas"),
}

fallos = 0
for archivo, (marca, para_que) in MARCAS.items():
    ruta = RAIZ / archivo
    if not ruta.exists():
        estado, fallos = "NO EXISTE", fallos + 1
    elif marca in ruta.read_text(encoding="utf-8"):
        estado = "ok"
    else:
        estado, fallos = "VERSIÓN VIEJA", fallos + 1
    print(f"{estado:<14}{archivo:<32}{para_que}")

print("\nTodo está actualizado." if not fallos else f"\nFaltan {fallos} archivo(s) por reemplazar (los marcados arriba).")
raise SystemExit(1 if fallos else 0)

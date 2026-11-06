"""Muestra QUÉ fragmentos recupera la búsqueda para una pregunta y en qué posición.

  python diagnostico_busqueda.py
  python diagnostico_busqueda.py "tu pregunta"

Compara el orden SEMÁNTICO puro con el orden HÍBRIDO (semántico + palabras clave),
que es el que recibe el agente. Requiere Ollama corriendo con bge-m3.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import config
from retrieval import ordenar_hibrido
from vector_db import abrir_coleccion

PREGUNTA = "¿Cuándo es la asamblea general y quién preside el consejo coordinador?"
PISTAS = ("primer domingo", "Galíndez")   # texto que SÍ responde a la pregunta por defecto

pregunta = " ".join(sys.argv[1:]) or PREGUNTA
print(f"Pregunta: {pregunta}")
print(f"K_DOCS actual: {config.K_DOCS} (solo los {config.K_DOCS} primeros llegan al agente)\n")

docs = abrir_coleccion(config.COLECCION_DOCS).similarity_search(pregunta, k=12)


def mostrar(titulo, lista):
    print(titulo)
    for i, d in enumerate(lista, 1):
        marca = "  <-- contiene la respuesta" if any(p in d.page_content for p in PISTAS) else ""
        corte = "  [llega al agente]" if i <= config.K_DOCS else ""
        linea = d.page_content[:70].replace("\n", " ")
        print(f"{i:>2}. {d.metadata.get('source', '?')} :: {linea!r}{corte}{marca}")
    print()


mostrar("ORDEN SEMÁNTICO (solo embeddings)", docs)
mostrar("ORDEN HÍBRIDO (lo que usa el agente)", ordenar_hibrido(docs, pregunta))

"""Indexa data/resenas.json en una colección nueva de ChromaDB.

Es idempotente: ejecutarlo dos veces no duplica nada (usa ids estables).
Requiere Ollama corriendo con el modelo de embeddings (bge-m3).
"""

import json

from langchain_core.documents import Document

import config
from vector_db import abrir_coleccion


def ingestar(embeddings=None) -> int:
    datos = json.loads((config.DIR_DATOS / "resenas.json").read_text(encoding="utf-8"))
    docs = [
        Document(
            page_content=r["texto"],
            metadata={"id": r["id"], "autor": r["autor"], "fecha": r["fecha"]},
        )
        for r in datos
    ]
    store = abrir_coleccion(config.COLECCION_RESENAS, embeddings=embeddings, crear=True)
    store.add_documents(docs, ids=[r["id"] for r in datos])
    return len(docs)


if __name__ == "__main__":
    n = ingestar()
    print(f"OK: {n} reseñas indexadas en '{config.COLECCION_RESENAS}'")

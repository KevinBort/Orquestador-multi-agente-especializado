"""Herramientas del agente INVESTIGADOR (solo lectura sobre ChromaDB).

Usan response_format="content_and_artifact":
  - content  -> lo que ve el LLM (corto, para no saturar el contexto de un modelo de 4B)
  - artifact -> los datos completos y estructurados, que el nodo del investigador
                copia al estado SIN que el LLM tenga que reescribirlos (más fiable).
"""

from langchain_core.tools import tool

import config
from retrieval import buscar_hibrido
from vector_db import ColeccionNoEncontrada, abrir_coleccion


@tool(response_format="content_and_artifact")
def buscar_reglamento(consulta: str) -> tuple[str, dict]:
    """Busca en los documentos oficiales de la cooperativa Huerta Viva: reglamento
    de socios (cuotas, requisitos, parcelas, sanciones), guía de cultivo,
    compostaje, talleres y preguntas frecuentes. Úsala para datos y normas.
    Usa palabras clave concretas en la consulta (p. ej. "asamblea general consejo coordinador")."""
    try:
        docs = buscar_hibrido(abrir_coleccion(config.COLECCION_DOCS), consulta, k=config.K_DOCS)
    except ColeccionNoEncontrada as e:
        return f"ERROR: {e}", {"tipo": "error", "detalle": str(e)}
    except Exception as e:  # p. ej. Ollama apagado o fallo del modelo de embeddings
        detalle = f"{type(e).__name__}: {e}"
        return f"ERROR en la búsqueda: {detalle}", {"tipo": "error", "detalle": detalle}

    fragmentos = [
        {"fuente": d.metadata.get("source", "desconocida"), "texto": d.page_content}
        for d in docs
    ]
    contenido = "\n\n".join(
        f"[{f['fuente']}]\n{f['texto'][:1500]}" for f in fragmentos
    ) or "Sin resultados."
    return contenido, {"tipo": "documentos", "consulta": consulta, "fragmentos": fragmentos}


@tool(response_format="content_and_artifact")
def buscar_resenas(consulta: str) -> tuple[str, dict]:
    """Busca reseñas y opiniones de socios sobre la cooperativa Huerta Viva.
    Úsala cuando se pidan opiniones, valoraciones o sentimiento de los socios.
    Devuelve las reseñas más relacionadas con la consulta."""
    try:
        docs = abrir_coleccion(config.COLECCION_RESENAS).similarity_search(
            consulta, k=config.K_RESENAS
        )
    except ColeccionNoEncontrada as e:
        return f"ERROR: {e}", {"tipo": "error", "detalle": str(e)}
    except Exception as e:  # p. ej. Ollama apagado o fallo del modelo de embeddings
        detalle = f"{type(e).__name__}: {e}"
        return f"ERROR en la búsqueda: {detalle}", {"tipo": "error", "detalle": detalle}

    resenas = [
        {
            "id": d.metadata.get("id", ""),
            "autor": d.metadata.get("autor", ""),
            "fecha": d.metadata.get("fecha", ""),
            "texto": d.page_content,
        }
        for d in docs
    ]
    # El LLM solo ve un resumen corto; las reseñas completas viajan en el artifact.
    ejemplos = "\n".join(f"- {r['texto'][:90]}..." for r in resenas[:3])
    contenido = (
        f"Se recuperaron {len(resenas)} reseñas sobre '{consulta}' "
        f"(guardadas en el estado para el analista). Ejemplos:\n{ejemplos}"
    )
    return contenido, {"tipo": "resenas", "consulta": consulta, "resenas": resenas}


SEARCH_TOOLS = [buscar_reglamento, buscar_resenas]

"""Acceso a ChromaDB (carpeta vectorstore/) compartido por ingesta y herramientas."""

import threading

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config


class ColeccionNoEncontrada(Exception):
    pass


def crear_embeddings() -> OllamaEmbeddings:
    # IMPORTANTE: el mismo modelo que se usó para indexar (bge-m3 -> 1024 dimensiones)
    return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)


_lock = threading.Lock()
_client = None


def _cliente() -> chromadb.ClientAPI:
    """Cliente único y seguro entre hilos. Las dos herramientas del investigador se
    ejecutan EN PARALELO: sin el lock, ambas creaban el cliente a la vez y una fallaba."""
    global _client
    with _lock:
        if _client is None:
            _client = chromadb.PersistentClient(
                path=str(config.DIR_VECTORES), settings=Settings(anonymized_telemetry=False)
            )
        return _client


def reiniciar_cliente() -> None:
    """Olvida el cliente (útil en pruebas cuando cambia la carpeta)."""
    global _client
    with _lock:
        _client = None


def existe_coleccion(nombre: str) -> bool:
    """True/False solo para 'existe / no existe'. Cualquier OTRO error se propaga
    con su mensaje real en vez de disfrazarse de 'no existe'."""
    try:
        _cliente().get_collection(nombre)
        return True
    except NotFoundError:
        return False
    except ValueError as e:
        if "does not exist" in str(e).lower():
            return False
        raise


def abrir_coleccion(nombre: str, embeddings=None, crear: bool = False) -> Chroma:
    """Abre una colección. Si no existe y crear=False, falla con un mensaje claro
    (evita crear una colección vacía por error)."""
    if not crear and not existe_coleccion(nombre):
        raise ColeccionNoEncontrada(
            f"No existe la colección '{nombre}' en {config.DIR_VECTORES}. "
            "Revisa que copiaste la carpeta vectorstore/ completa "
            "(o ejecuta ingest_resenas.py para las reseñas)."
        )
    extra = {"collection_configuration": {"hnsw": {"space": "cosine"}}} if crear else {}
    return Chroma(
        client=_cliente(),
        collection_name=nombre,
        embedding_function=embeddings or crear_embeddings(),
        **extra,
    )

"""Configuración central. Todo lo que cambia entre máquinas vive en el .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

RAIZ = Path(__file__).parent
DIR_DATOS = RAIZ / "data"
DIR_VECTORES = RAIZ / "vectorstore"

# Colecciones de ChromaDB (las dos viven en la misma carpeta vectorstore/)
COLECCION_DOCS = "rag_docs"                 # reglamento y guías (pre-entrega anterior)
COLECCION_RESENAS = "resenas_huerta_viva"   # reseñas de socios (nueva, para sentimiento)

# Modelos locales (Ollama)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))  # 4096 cabe en una GPU de 4 GB

# Control de flujo (anti "Supervisor Infinito")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))   # pasos totales del supervisor
MAX_REFINAMIENTOS = 1                          # veces que un especialista puede repetir

# Recuperación
K_DOCS = int(os.getenv("K_DOCS", "4"))   # fragmentos de documentos por búsqueda
K_RESENAS = int(os.getenv("K_RESENAS", "12"))   # reseñas por búsqueda (hay 24 en total)

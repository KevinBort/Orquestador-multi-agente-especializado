"""Un único lugar para crear el modelo. Cambiar de proveedor = tocar este archivo."""

from langchain_ollama import ChatOllama

import config


def get_llm(temperature: float = 0) -> ChatOllama:
    return ChatOllama(
        model=config.LLM_MODEL,
        temperature=temperature,
        num_ctx=config.NUM_CTX,
        base_url=config.OLLAMA_BASE_URL,
    )

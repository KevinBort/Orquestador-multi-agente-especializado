"""Utilidades compartidas por los agentes."""

from langchain_core.messages import AIMessage, ToolMessage


def texto(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in contenido)
    return str(contenido)


def ultimo_texto(mensajes) -> str:
    """Último mensaje del modelo que no sea una llamada a herramienta."""
    for m in reversed(mensajes):
        if isinstance(m, AIMessage) and not m.tool_calls and texto(m.content).strip():
            return texto(m.content).strip()
    return ""


def artifacts(mensajes) -> list[dict]:
    """Datos estructurados que dejaron las herramientas (ToolMessage.artifact)."""
    return [m.artifact for m in mensajes
            if isinstance(m, ToolMessage) and isinstance(m.artifact, dict)]

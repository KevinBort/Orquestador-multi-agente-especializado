"""Quick check: does Ollama respond, call tools, and return structured output?"""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel

from llm import get_llm


@tool
def sumar(a: int, b: int) -> int:
    """Suma dos números enteros."""
    return a + b


class Ruta(BaseModel):
    siguiente: Literal["researcher", "analyst", "FINISH"]


llm = get_llm()

# 1) Plain response
print("1) Respuesta simple:")
print("  ", llm.invoke("Responde solo con la palabra: listo").content)

# 2) Tool calling (needed by create_react_agent)
print("2) Tool calling:")
resp = llm.bind_tools([sumar]).invoke("¿Cuánto es 17 + 25? Usa la herramienta.")
print("  ", resp.tool_calls)

# 3) Structured output (needed by the supervisor's routing)
print("3) Salida estructurada:")
ruta = llm.with_structured_output(Ruta).invoke(
    "Aún no se ha buscado ningún dato. ¿Quién debe actuar primero: "
    "researcher, analyst o FINISH?"
)
print("  ", ruta)
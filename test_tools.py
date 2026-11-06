"""Pruebas de las herramientas, SIN depender del grafo.

  python test_tools.py            -> pruebas offline (no necesitan Ollama)
  python test_tools.py --ollama   -> además prueba contra Ollama y tu vectorstore real
"""

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from state import AgentState

import config
import vector_db
from tools.analysis_tools import (analizar_lista, calcular, clasificar_lexico,
                                  analizar_sentimiento_resenas, evaluar_expresion)
from tools.search_tools import buscar_resenas, buscar_reglamento

# Etiquetas "correctas" para medir la calidad del clasificador (solo para pruebas)
ESPERADO = {
    "r01": "positivo", "r02": "negativo", "r03": "positivo", "r04": "neutral",
    "r05": "positivo", "r06": "negativo", "r07": "positivo", "r08": "negativo",
    "r09": "positivo", "r10": "neutral", "r11": "negativo", "r12": "positivo",
    "r13": "neutral", "r14": "positivo", "r15": "negativo", "r16": "positivo",
    "r17": "neutral", "r18": "negativo", "r19": "positivo", "r20": "neutral",
    "r21": "positivo", "r22": "negativo", "r23": "positivo", "r24": "neutral",
}


class EmbFalso(Embeddings):
    """Embeddings de juguete (bolsa de palabras) para probar sin Ollama."""
    def _v(self, t):
        v = [0.0] * 128
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1.0
        return v
    def embed_documents(self, ts): return [self._v(t) for t in ts]
    def embed_query(self, t): return self._v(t)


def ok(msg): print(f"  OK  {msg}")


def correr_tool_node(herramientas, llamada, estado_extra=None):
    """Ejecuta una herramienta como lo hará el agente: dentro de un grafo, vía ToolNode."""
    g = StateGraph(AgentState)
    g.add_node("tools", ToolNode(herramientas))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    entrada = {"messages": [AIMessage(content="", tool_calls=[llamada])], **(estado_extra or {})}
    return g.compile().invoke(entrada)["messages"][-1]


def test_calculo():
    print("[1] calcular")
    assert evaluar_expresion("4500*12") == 54000
    assert evaluar_expresion("(4500-2500)*12") == 24000
    for malo in ("__import__('os').system('ls')", "9**9**9", "open('x')"):
        try:
            evaluar_expresion(malo); raise SystemExit(f"FALLO: aceptó {malo}")
        except SystemExit: raise
        except Exception: pass
    ok("aritmética correcta y rechaza código peligroso")
    m = correr_tool_node([calcular], {"name": "calcular", "args": {"expresion": "2500*12+8000"}, "id": "1"})
    assert m.artifact["resultado"] == 38000, m
    ok(f"vía ToolNode: {m.content}")


def test_sentimiento_sin_llm():
    print("[2] sentimiento (plan B léxico, sin LLM)")
    datos = json.loads((config.DIR_DATOS / "resenas.json").read_text(encoding="utf-8"))
    res = analizar_lista(datos, clasificar_lexico)
    acc = sum(d["sentimiento"] == ESPERADO[d["id"]] for d in res["detalle"]) / res["total"]
    assert res["total"] == 24 and sum(res["conteo"].values()) == 24
    assert abs(sum(res["porcentajes"].values()) - 100) < 0.5
    ok(f"conteo {res['conteo']} | acierto del plan B: {acc:.0%}")

    # La herramienta lee las reseñas del ESTADO (InjectedState), no de sus argumentos
    estado = {"research_output": {"resenas": datos[:6]}}
    import tools.analysis_tools as at
    original = at._clasificador_llm
    at._clasificador_llm = lambda: clasificar_lexico      # evita necesitar Ollama
    try:
        m = correr_tool_node([analizar_sentimiento_resenas],
                             {"name": "analizar_sentimiento_resenas", "args": {}, "id": "2"}, estado)
    finally:
        at._clasificador_llm = original
    assert m.artifact["total"] == 6, m.artifact
    ok(f"InjectedState funciona: {m.content}")
    m = correr_tool_node([analizar_sentimiento_resenas],
                         {"name": "analizar_sentimiento_resenas", "args": {}, "id": "3"}, {"research_output": {}})
    assert m.artifact["tipo"] == "error"
    ok("sin reseñas en el estado devuelve un error claro (no revienta)")


def test_busqueda_y_ingesta_simuladas():
    print("[3] ingesta + búsqueda (embeddings simulados, base temporal)")
    from langchain_core.documents import Document
    import ingest_resenas
    with tempfile.TemporaryDirectory() as tmp:
        config.DIR_VECTORES = Path(tmp)
        vector_db.reiniciar_cliente()
        emb = EmbFalso()
        n = ingest_resenas.ingestar(embeddings=emb)
        n2 = ingest_resenas.ingestar(embeddings=emb)   # idempotencia
        assert vector_db._cliente().get_collection(config.COLECCION_RESENAS).count() == 24 == n == n2
        ok("ingesta idempotente: 24 reseñas, sin duplicar al repetir")

        docs = vector_db.abrir_coleccion(config.COLECCION_DOCS, embeddings=emb, crear=True)
        docs.add_documents([Document(page_content="La cuota social mensual es de 4.500 pesos",
                                     metadata={"source": "01_reglamento_socios.md"})], ids=["d1"])
        vector_db.crear_embeddings = lambda: emb        # las herramientas usarán el falso
        m = correr_tool_node([buscar_reglamento], {"name": "buscar_reglamento", "args": {"consulta": "cuota mensual"}, "id": "4"})
        assert m.artifact["tipo"] == "documentos" and "4.500" in m.content
        ok("buscar_reglamento devuelve texto y fuente")
        m = correr_tool_node([buscar_resenas], {"name": "buscar_resenas", "args": {"consulta": "cuota jubilada"}, "id": "5"})
        assert m.artifact["tipo"] == "resenas" and len(m.artifact["resenas"]) == config.K_RESENAS
        ok(f"buscar_resenas: el LLM ve {len(m.content)} caracteres; el artifact trae {len(m.artifact['resenas'])} reseñas completas")

        vector_db.reiniciar_cliente()
        config.DIR_VECTORES = Path(tmp) / "no_existe"
        m = correr_tool_node([buscar_reglamento], {"name": "buscar_reglamento", "args": {"consulta": "x"}, "id": "6"})
        assert m.artifact["tipo"] == "error"
        ok("colección inexistente -> error claro, sin crear una vacía")


def test_ollama():
    print("[4] contra Ollama y tu vectorstore REAL")
    import importlib
    importlib.reload(config); importlib.reload(vector_db)
    from langchain_ollama import OllamaEmbeddings
    import tools.search_tools as st, tools.analysis_tools as at
    importlib.reload(st); importlib.reload(at)

    r = vector_db.crear_embeddings().embed_query("hola")
    print(f"  embeddings: dimensión {len(r)} (tu colección rag_docs usa 1024)")
    assert len(r) == 1024, "El modelo de embeddings no coincide con el de la colección"

    m = correr_tool_node([st.buscar_reglamento], {"name": "buscar_reglamento",
                         "args": {"consulta": "cuota mensual jubilados"}, "id": "a"})
    print("  buscar_reglamento ->", m.content[:160].replace("\n", " "), "...")
    assert m.artifact["tipo"] == "documentos", m.content
    assert "2.500" in m.content or "4.500" in m.content, "no encontró el fragmento de cuotas"
    ok("el fragmento de cuotas aparece: bge-m3 es el modelo correcto")

    print("  indexando reseñas (primera vez tarda unos segundos)...")
    import ingest_resenas; ingest_resenas.ingestar()
    m = correr_tool_node([st.buscar_resenas], {"name": "buscar_resenas",
                         "args": {"consulta": "opinión de los socios sobre la cuota"}, "id": "b"})
    resenas = m.artifact["resenas"]; ok(f"{len(resenas)} reseñas recuperadas")

    print("  clasificando las 24 reseñas con el LLM (puede tardar 1-2 min)...")
    datos = json.loads((config.DIR_DATOS / "resenas.json").read_text(encoding="utf-8"))
    res = at.analizar_lista(datos, at._clasificador_llm())
    aciertos = sum(d["sentimiento"] == ESPERADO[d["id"]] for d in res["detalle"])
    print(f"  conteo: {res['conteo']} | porcentajes: {res['porcentajes']}")
    print(f"  acierto vs etiquetas esperadas: {aciertos}/24 | con plan B: {res['resenas_con_plan_b']}")
    for d in res["detalle"]:
        if d["sentimiento"] != ESPERADO[d["id"]]:
            print(f"    difiere {d['id']}: dijo {d['sentimiento']}, esperado {ESPERADO[d['id']]}")
    assert aciertos >= 17, "el clasificador acierta poco; revisa el modelo"
    ok("clasificador con calidad suficiente")


if __name__ == "__main__":
    test_calculo()
    test_sentimiento_sin_llm()
    test_busqueda_y_ingesta_simuladas()
    if "--ollama" in sys.argv:
        test_ollama()
    else:
        print("\n(Pruebas con Ollama omitidas. Ejecuta: python test_tools.py --ollama)")
    print("\nTODO OK")

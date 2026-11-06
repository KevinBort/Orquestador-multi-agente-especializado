"""Pruebas del GRAFO con un LLM simulado (no necesitan Ollama).

Verifican la lógica que NO depende del modelo: el enrutamiento, las guardas del
supervisor, los límites anti-bucle y la trazabilidad en el estado.
Para probar con el modelo real, usa main.py.
"""

import hashlib
import re
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import config
import ingest_resenas
import vector_db
import tools.analysis_tools as at
from graph import build_graph
from state import initial_state
from supervisor import Decision


class EmbFalso(Embeddings):
    def _v(self, t):
        v = [0.0] * 128
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1.0
        return v
    def embed_documents(self, ts): return [self._v(t) for t in ts]
    def embed_query(self, t): return self._v(t)


class LLMFalso(BaseChatModel):
    """Modelo de juguete: los especialistas siempre llaman sus herramientas y el
    supervisor sigue un guion (`guion`) o una función (`siempre`)."""
    guion: list = []
    siempre: str = ""
    llamadas_supervisor: int = 0

    @property
    def _llm_type(self): return "falso"

    def bind_tools(self, tools, **kw): return self

    def with_structured_output(self, schema, **kw):
        def decidir(_):
            self.llamadas_supervisor += 1
            sig = self.siempre or (self.guion.pop(0) if self.guion else "synthesizer")
            return Decision(reason="guion de prueba", next=sig, instruction="" if sig == "synthesizer" else f"haz tu parte ({sig})")
        return RunnableLambda(decidir)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        sistema = messages[0].content
        hubo_tools = any(isinstance(m, ToolMessage) for m in messages)
        if "Eres el INVESTIGADOR" in sistema:
            if not hubo_tools:
                m = AIMessage(content="", tool_calls=[
                    {"name": "buscar_resenas", "args": {"consulta": "cuota"}, "id": "c1"},
                    {"name": "buscar_reglamento", "args": {"consulta": "cuota jubilados"}, "id": "c2"}])
            else:
                m = AIMessage(content="Cuota reducida para jubilados: 2500 pesos (01_reglamento_socios.md).")
        elif "Eres el ANALISTA" in sistema:
            if not hubo_tools:
                m = AIMessage(content="", tool_calls=[
                    {"name": "analizar_sentimiento_resenas", "args": {}, "id": "a1"},
                    {"name": "calcular", "args": {"expresion": "2500*12"}, "id": "a2"}])
            else:
                m = AIMessage(content="Sentimiento calculado y cuota anual de 30000.")
        else:  # sintetizador
            m = AIMessage(content="RESPUESTA FINAL: un jubilado paga 30000 al año.")
        return ChatResult(generations=[ChatGeneration(message=m)])


def ok(msg): print(f"  OK  {msg}")


def correr(llm, pregunta):
    g = build_graph(llm)
    return g.invoke(initial_state(pregunta), config={"recursion_limit": 40})


def agentes_en_orden(est):
    return [c["summary"].split(":")[0].replace("-> ", "") for c in est["contributions"] if c["agent"] == "supervisor"]


PREG = "Dime el porcentaje de opiniones positivas sobre la cuota y cuánto paga al año un jubilado"
PREG_SIMPLE = "¿Cuándo es la asamblea general?"


def preparar_datos():
    tmp = tempfile.mkdtemp()
    config.DIR_VECTORES = Path(tmp)
    vector_db.reiniciar_cliente()
    emb = EmbFalso()
    ingest_resenas.ingestar(embeddings=emb)
    docs = vector_db.abrir_coleccion(config.COLECCION_DOCS, embeddings=emb, crear=True)
    docs.add_documents([Document(page_content="La cuota reducida para jubilados es de 2.500 pesos.",
                                 metadata={"source": "01_reglamento_socios.md"})], ids=["d1"])
    vector_db.crear_embeddings = lambda: emb
    at._clasificador_llm = lambda: at.clasificar_lexico   # sentimiento sin Ollama


def test_caso_feliz():
    print("[1] caso feliz: investigador -> analista -> sintetizador")
    est = correr(LLMFalso(guion=["researcher", "analyst", "synthesizer"]), PREG)
    assert agentes_en_orden(est) == ["researcher", "analyst", "synthesizer"], agentes_en_orden(est)
    assert est["task_completed"] and "30000" in est["final_answer"]
    ro, ao = est["research_output"], est["analysis_output"]
    assert ro["documentos"] and len(ro["resenas"]) == config.K_RESENAS
    assert ao["sentimiento"]["total"] == config.K_RESENAS and ao["calculos"][0]["resultado"] == 30000
    quien = [(c["agent"], c["step"]) for c in est["contributions"] if c["agent"] != "supervisor"]
    assert [a for a, _ in quien] == ["researcher", "analyst", "synthesizer"]
    ok(f"flujo correcto; InjectedState y artifacts funcionan; aportes rastreados: {quien}")


def test_guarda_sin_datos():
    print("[2] guarda: el LLM quiere sintetizar sin datos")
    est = correr(LLMFalso(guion=["synthesizer", "researcher", "analyst", "synthesizer"]), PREG)
    orden = agentes_en_orden(est)
    assert orden[0] == "researcher", orden
    assert "GUARDA" in est["contributions"][0]["summary"]
    ok(f"el supervisor corrigió la decisión -> {orden}")


def test_guarda_exige_analisis():
    print("[3] guarda: sintetizar sin analizar lo que la pregunta pide")
    est = correr(LLMFalso(guion=["researcher", "synthesizer", "synthesizer"]), PREG)
    assert agentes_en_orden(est) == ["researcher", "analyst", "synthesizer"], agentes_en_orden(est)
    ok("forzó al analista antes de cerrar")


def test_pregunta_simple_no_fuerza_analista():
    print("[4] pregunta de solo consulta: no se llama al analista")
    est = correr(LLMFalso(guion=["researcher", "synthesizer"]), PREG_SIMPLE)
    assert agentes_en_orden(est) == ["researcher", "synthesizer"], agentes_en_orden(est)
    assert not est["analysis_output"]
    ok("investigador -> sintetizador, sin pasos de más")


def test_refinamiento():
    print("[5] refinamiento: el supervisor pide repetir al analista una vez")
    est = correr(LLMFalso(guion=["researcher", "analyst", "analyst", "synthesizer"]), PREG)
    assert agentes_en_orden(est) == ["researcher", "analyst", "analyst", "synthesizer"]
    assert est["refinements"] == {"analyst": 1}
    ok(f"refinamiento aceptado y contado: {est['refinements']}")


def test_bucle_infinito():
    print("[6] anti-bucle: un LLM que siempre pide 'researcher'")
    llm = LLMFalso(siempre="researcher")
    est = correr(llm, PREG)
    orden = agentes_en_orden(est)
    assert orden[-1] == "synthesizer" and est["task_completed"], orden
    assert est["step_count"] <= config.MAX_STEPS + 1
    assert est["refinements"].get("researcher", 0) <= config.MAX_REFINAMIENTOS
    ok(f"terminó igual en {est['step_count']} pasos: {orden}")


def test_limite_de_pasos():
    print("[7] límite de pasos: MAX_STEPS=2 fuerza el cierre")
    original, config.MAX_STEPS = config.MAX_STEPS, 2
    try:
        est = correr(LLMFalso(siempre="researcher"), PREG)
    finally:
        config.MAX_STEPS = original
    assert est["task_completed"] and est["step_count"] == 3
    assert "límite" in est["contributions"][-2]["summary"]
    ok("al superar el límite, el supervisor envía al sintetizador")


def test_respaldo_si_llm_falla():
    print("[8] respaldo: si la salida estructurada del LLM falla")
    class Roto(LLMFalso):
        def with_structured_output(self, schema, **kw):
            def f(_): raise ValueError("JSON inválido")
            return RunnableLambda(f)
    est = correr(Roto(), PREG)
    assert agentes_en_orden(est) == ["researcher", "analyst", "synthesizer"], agentes_en_orden(est)
    ok("las reglas de respaldo completaron el flujo")


def test_diagrama():
    print("[9] diagrama Mermaid")
    mer = build_graph(LLMFalso()).get_graph().draw_mermaid()
    for n in ("supervisor", "researcher", "analyst", "synthesizer"):
        assert n in mer
    ok("el grafo se dibuja con los 4 nodos")


def test_concurrencia_herramientas():
    print("[10] regresión: las 2 herramientas del investigador en paralelo, 15 veces")
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode
    from state import AgentState
    from tools.search_tools import SEARCH_TOOLS
    g = StateGraph(AgentState); g.add_node("tools", ToolNode(SEARCH_TOOLS))
    g.add_edge(START, "tools"); g.add_edge("tools", END)
    app = g.compile()
    llamadas = AIMessage(content="", tool_calls=[
        {"name": "buscar_resenas", "args": {"consulta": "cuota"}, "id": "1"},
        {"name": "buscar_reglamento", "args": {"consulta": "cuota"}, "id": "2"}])
    for i in range(15):
        vector_db.reiniciar_cliente()          # cliente "en frío", como al arrancar main.py
        for m in app.invoke({"messages": [llamadas]})["messages"][1:]:
            assert not str(m.content).startswith("ERROR"), f"intento {i}: {m.content[:120]}"
    ok("15/15 ejecuciones sin errores (antes fallaba de forma intermitente)")


def test_errores_visibles():
    print("[11] si las herramientas fallan, el error llega al estado y el flujo termina")
    buena = config.DIR_VECTORES
    config.DIR_VECTORES = Path(tempfile.mkdtemp()) / "vacia"
    vector_db.reiniciar_cliente()
    try:
        est = correr(LLMFalso(guion=["researcher", "researcher", "synthesizer"]), PREG)
    finally:
        config.DIR_VECTORES = buena
        vector_db.reiniciar_cliente()
    errores = est["research_output"].get("errores", [])
    assert errores and "No existe la colección" in errores[0], errores
    assert est["task_completed"]
    assert "ERRORES" in est["contributions"][1]["summary"]
    ok(f"errores visibles ({len(errores)}) y el flujo terminó igualmente")


def test_limitaciones_calculadas():
    print("[12] el LLM no inventa limitaciones; las reales las añade el código")
    class Charlatan(LLMFalso):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if "Eres el SINTETIZADOR" in messages[0].content:
                m = AIMessage(content="Un jubilado paga 30000 al año.\n\nLimitación: INVENTADA por el modelo.")
                return ChatResult(generations=[ChatGeneration(message=m)])
            return super()._generate(messages, stop, run_manager, **kw)
    est = correr(Charlatan(guion=["researcher", "analyst", "synthesizer"]), PREG)
    r = est["final_answer"]
    assert "INVENTADA" not in r, r
    assert "Limitaciones:" in r and f"{config.K_RESENAS} de 24 reseñas" in r, r
    ok("se eliminó la limitación inventada y se añadió la real (12 de 24 reseñas)")

    original, config.K_RESENAS = config.K_RESENAS, 24
    try:
        est = correr(LLMFalso(guion=["researcher", "analyst", "synthesizer"]), PREG)
    finally:
        config.K_RESENAS = original
    assert "Limitaciones" not in est["final_answer"], est["final_answer"]
    ok("si se analizaron las 24 reseñas, no se añade ninguna limitación")


def test_reintento_si_no_busca():
    print("[13] si el investigador contesta sin usar herramientas, reintenta sin gastar pasos")
    class Perezoso(LLMFalso):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if "Eres el INVESTIGADOR" in messages[0].content and \
               not any("IMPORTANTE" in str(m.content) for m in messages):
                m = AIMessage(content="Creo que la cuota es de 3000 pesos (de memoria).")
                return ChatResult(generations=[ChatGeneration(message=m)])
            return super()._generate(messages, stop, run_manager, **kw)
    est = correr(Perezoso(guion=["researcher", "synthesizer"]), PREG_SIMPLE)
    assert est["research_output"]["documentos"], est["research_output"]
    assert agentes_en_orden(est) == ["researcher", "synthesizer"], agentes_en_orden(est)
    assert est["refinements"] == {}
    ok("buscó en el segundo intento; el supervisor no tuvo que gastar un refinamiento")


if __name__ == "__main__":
    preparar_datos()
    for t in (test_caso_feliz, test_guarda_sin_datos, test_guarda_exige_analisis,
              test_pregunta_simple_no_fuerza_analista, test_refinamiento, test_bucle_infinito,
              test_limite_de_pasos, test_respaldo_si_llm_falla, test_diagrama,
              test_concurrencia_herramientas, test_errores_visibles,
              test_limitaciones_calculadas, test_reintento_si_no_busca):
        t()
    print("\nTODO OK")

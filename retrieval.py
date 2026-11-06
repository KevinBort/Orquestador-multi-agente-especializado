"""Búsqueda HÍBRIDA: similitud semántica + coincidencia de palabras clave.

Los embeddings entienden el significado, pero con fragmentos largos que mezclan
varios temas distinguen poco (en una prueba real, el fragmento correcto quedó 4º
y al agente solo le llegaban 3). Las palabras clave ("asamblea", "consejo",
"preside") corrigen justo ese caso. Se combinan con Reciprocal Rank Fusion (RRF).
"""

import math
import re
import unicodedata

_STOP = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "que", "y", "o", "en", "es", "son",
    "del", "al", "se", "su", "sus", "por", "para", "con", "sin", "como", "cual", "cuales", "cuando",
    "donde", "quien", "quienes", "cuanto", "cuanta", "cuantos", "me", "mi", "lo", "le", "les",
    "hay", "ser", "esta", "este", "esto", "dime", "indica", "busca", "investiga", "sobre", "mas",
}
_PREFIJO = 5   # compara por raíz aproximada: "preside" y "presidido" -> "presi"


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.lower()


def raices(texto: str) -> set[str]:
    palabras = re.findall(r"[a-z0-9]+", _normalizar(texto))
    return {p[:_PREFIJO] for p in palabras if len(p) >= 3 and p not in _STOP}


def puntuacion_lexica(consulta: str, textos: list[str]) -> list[float]:
    """Suma, por cada raíz de la consulta presente en el texto, su peso IDF
    (las raíces raras pesan más que las que aparecen en todos los textos)."""
    q = raices(consulta)
    por_texto = [raices(t) for t in textos]
    n = len(textos)
    puntajes = []
    for rs in por_texto:
        p = 0.0
        for r in q & rs:
            df = sum(r in otros for otros in por_texto)
            p += math.log((n + 1) / (df + 0.5)) + 1
        puntajes.append(p)
    return puntajes


def fusionar_rrf(puntajes_lexicos: list[float], c: int = 60) -> list[int]:
    """Devuelve los índices ordenados. El orden semántico es el orden de entrada
    (índice 0 = el más cercano). RRF: score = 1/(c+rank_sem) + 1/(c+rank_lex)."""
    n = len(puntajes_lexicos)
    orden_lex = sorted((i for i in range(n) if puntajes_lexicos[i] > 0),
                       key=lambda i: (-puntajes_lexicos[i], i))
    rank_lex = {i: r for r, i in enumerate(orden_lex, 1)}
    score = {}
    for i in range(n):
        s = 1 / (c + i + 1)
        if i in rank_lex:
            s += 1 / (c + rank_lex[i])
        score[i] = s
    return sorted(range(n), key=lambda i: (-score[i], i))


def ordenar_hibrido(docs: list, consulta: str) -> list:
    """Reordena documentos (ya ordenados semánticamente) combinando palabras clave."""
    if not docs:
        return []
    orden = fusionar_rrf(puntuacion_lexica(consulta, [d.page_content for d in docs]))
    return [docs[i] for i in orden]


def buscar_hibrido(store, consulta: str, k: int) -> list:
    """Recupera un grupo amplio por similitud, lo reordena con palabras clave y devuelve k."""
    pool = max(4 * k, 12)
    return ordenar_hibrido(store.similarity_search(consulta, k=pool), consulta)[:k]

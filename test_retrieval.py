"""Pruebas de la búsqueda híbrida (offline)."""

from types import SimpleNamespace

from retrieval import buscar_hibrido, fusionar_rrf, ordenar_hibrido, puntuacion_lexica, raices


def doc(texto):
    return SimpleNamespace(page_content=texto)


def ok(msg): print(f"  OK  {msg}")


# Escenario real: el fragmento que responde estaba 4º en el ranking semántico
SEMANTICO = [
    doc("Cómo proponer un taller: presentá una propuesta al Consejo Coordinador."),
    doc("Jornadas comunitarias: ocho horas al mes. Sanciones decididas por la Asamblea General."),
    doc("Período de prueba: el aspirante no puede votar en la Asamblea."),
    doc("La Asamblea General se reúne el primer domingo de cada mes. El Consejo Coordinador "
        "está presidido por Marta Galíndez."),
    doc("Guía de cultivo: calendario de siembra."),
    doc("Banco de semillas: hasta cinco sobres por temporada."),
]
PREGUNTA = "¿Cuándo es la asamblea general y quién preside el consejo coordinador?"


def test_raices():
    print("[1] raíces: tildes, mayúsculas y variantes de palabra")
    assert raices("preside") & raices("presidido")
    assert raices("Galíndez") == raices("galindez")
    assert not (raices("¿Cuándo es la") & {"cuand", "es", "la"})
    ok("preside~presidido, Galíndez~galindez, y se ignoran las palabras vacías")


def test_sube_el_fragmento_correcto():
    print("[2] el fragmento correcto sube de la posición 4 a las primeras")
    assert SEMANTICO[3].page_content.startswith("La Asamblea General")
    orden = ordenar_hibrido(SEMANTICO, PREGUNTA)
    pos = [d.page_content for d in orden].index(SEMANTICO[3].page_content) + 1
    assert pos <= 2, pos
    ok(f"posición híbrida: {pos} (semántica: 4)")


def test_sin_coincidencias_no_cambia_el_orden():
    print("[3] sin palabras en común, se respeta el orden semántico")
    assert fusionar_rrf([0.0, 0.0, 0.0, 0.0]) == [0, 1, 2, 3]
    ok("orden original conservado")


def test_vacio_y_k():
    print("[4] casos límite")
    assert ordenar_hibrido([], "x") == []
    class Falso:
        def similarity_search(self, q, k): return SEMANTICO[:k]
    assert len(buscar_hibrido(Falso(), PREGUNTA, k=3)) == 3
    ok("lista vacía y recorte a k")


def test_idf_penaliza_palabras_comunes():
    print("[5] una palabra presente en todos los textos pesa menos que una rara")
    textos = ["cuota mensual jubilados", "cuota de ingreso", "cuota reducida"]
    p = puntuacion_lexica("cuota jubilados", textos)
    assert p[0] > p[1] and p[0] > p[2]
    ok("gana el texto con la palabra poco frecuente")


if __name__ == "__main__":
    for t in (test_raices, test_sube_el_fragmento_correcto, test_sin_coincidencias_no_cambia_el_orden,
              test_vacio_y_k, test_idf_penaliza_palabras_comunes):
        t()
    print("\nTODO OK")

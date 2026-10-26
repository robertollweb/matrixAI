"""138, reparación 2 — «solo toma un valor en las N filas»: N son FILAS, no valores.

La comprobación de la clase única sobre el CSV ENTERO pasaba `opciones=len(valores_crudos)`,
que son los valores DISTINTOS: un CSV de 100 filas con una sola clase decía «en las 1 filas
de entrenamiento» / «across the 1 training rows». Ahora cuenta las filas que traen objetivo
(con el mismo criterio de ausencia) y su plantilla no las llama «de entrenamiento», porque son
las del fichero. La del reparto (`no_entrenable_en_train`) sí mira las de entrenamiento: se fija
también su número, contado aquí por separado.
"""

from __future__ import annotations

from matrixai.training.objetivo import confirmar_desde_csv, corte_de_entrenamiento


def _csv(filas: list[tuple[int, str]]) -> str:
    return "x,y\n" + "".join(f"{x},{y}\n" for x, y in filas)


def _motivo_de_una_clase(confirmacion):
    hallados = [b for b in confirmacion.bloqueos if b.clave == "objetivo_con_una_sola_clase"]
    assert len(hallados) == 1, confirmacion.bloqueos
    return hallados[0].motivo


def test_cien_filas_de_una_clase_dicen_cien_filas_en_los_dos_idiomas():
    texto = _csv([(i, "no") for i in range(100)])
    for kw in ({"objetivo": "y"}, {"objetivo": "y", "tarea": "binary_classification", "clase_positiva": "no"}):
        m = _motivo_de_una_clase(confirmar_desde_csv(texto, **kw))
        assert "en las 100 filas del fichero que lo traen" in m["es"], m["es"]
        assert "across the 100 rows of the file that have it" in m["en"], m["en"]
        # ni el número de valores distintos, ni «de entrenamiento» (son todas las del fichero)
        assert " 1 filas" not in m["es"] and " 1 rows" not in m["en"]
        assert "entrenamiento" not in m["es"] and "training rows" not in m["en"]


def test_las_filas_sin_objetivo_no_se_cuentan():
    texto = _csv([(i, "no") for i in range(100)] + [(i, "") for i in range(100, 113)])
    m = _motivo_de_una_clase(confirmar_desde_csv(texto, objetivo="y"))
    assert "en las 100 filas del fichero" in m["es"], m["es"]
    assert "across the 100 rows of the file" in m["en"], m["en"]


def test_la_del_reparto_cuenta_las_filas_de_entrenamiento():
    # Ordenado por la clase: dos valores en el fichero, uno solo en lo que entrena.
    total = 100
    corte = corte_de_entrenamiento(total)
    assert 1 < corte < total
    texto = _csv([(i, "no") for i in range(corte)] + [(i, "si") for i in range(corte, total)])
    m = _motivo_de_una_clase(confirmar_desde_csv(texto, objetivo="y", tarea="binary_classification",
                                                 clase_positiva="si"))
    assert f"en las {corte} filas de entrenamiento" in m["es"], m["es"]
    assert f"across the {corte} training rows" in m["en"], m["en"]

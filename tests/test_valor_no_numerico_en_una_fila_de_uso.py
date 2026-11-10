"""29-10: un número mal escrito en una fila de USO es un dato de la persona, no un fallo del servidor.

`transformar_fila` hacía `float(valor)` sin más: «sesenta», «1.234,5» o «70,5» subían como `ValueError: could not convert
string to float`, y el Studio contestaba un 500 que la pantalla enseñaba como «la conexión con el servidor falló» (visto por
dos auditorías del 142). Ahora es `ValorNoNumerico` —sigue siendo un `ValueError`, quien lo recogía así no cambia— con la
columna, lo escrito y el motivo en los dos idiomas.
"""
from __future__ import annotations

import pytest

from matrixai.training.preparacion import ValorNoNumerico, ajustar_preparacion, transformar_fila


def _politica():
    filas = [{"edad": float(20 + i), "zona": "Norte" if i % 2 else "Sur", "y": str(i % 2)} for i in range(40)]
    return ajustar_preparacion(filas, objetivo="y", columnas=["edad", "zona"], admite_categoricas=True,
                               admite_faltantes=True)


@pytest.mark.parametrize("escrito", ["sesenta", "1.234,5", "70,5", "--3", "3 años"])
def test_un_numero_mal_escrito_dice_su_columna_y_su_motivo(escrito):
    with pytest.raises(ValorNoNumerico) as e:
        transformar_fila({"edad": escrito, "zona": "Norte"}, _politica())
    assert isinstance(e.value, ValueError)
    assert e.value.columna == "edad" and e.value.valor == escrito
    for idioma in ("es", "en"):
        assert "edad" in e.value.motivo[idioma] and escrito in e.value.motivo[idioma], e.value.motivo
        assert "{" not in e.value.motivo[idioma]


@pytest.mark.parametrize("escrito, numero", [("60", 60.0), ("-3.5", -3.5), (" 7 ", 7.0), ("1e3", 1000.0)])
def test_lo_que_se_lee_como_numero_sigue_leyendose(escrito, numero):
    assert transformar_fila({"edad": escrito, "zona": "Norte"}, _politica())["edad"] == numero

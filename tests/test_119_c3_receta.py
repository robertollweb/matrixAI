# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C3, reparación 3 (auditoría 3, hallazgo I2): la pasada tiene que medir
LA RECETA ENTERA del protocolo (v4 + enmiendas 1 y 2), no solo k, d_block y
n_blocks.

La auditoría cambió en el motor el lr, el weight_decay, el dropout, las
frecuencias de PLR y AdamW por Adam (sus sabotajes S4, S5, S6, S8, S10 y C14):
todo seguía verde, en las pruebas del motor y en las de la pasada. Aquí:

* el protocolo se LEE de sus tres ficheros sellados (qué valor y de dónde);
* cada CONSTANTE del motor se compara con él, una a una;
* un AJUSTE DE VERDAD, pequeño y con espías, dice qué se USA: la CLASE del
  optimizador y sus `lr`/`weight_decay`, los argumentos de la red, el
  recorte, el lote, la semilla y el plazo. Cambiar `AdamW` por `Adam` en la
  llamada no lo ve ninguna constante ni el `"optimizador": "adamw"` que el
  predictor escribe a mano: lo ve el espía;
* y la guardia de la pasada PARA con cualquiera de ellas cambiada.

Importa el motor directamente (`matrixai_engines.motores.densa_tabm`), para
que la pasada localizada elija este fichero cuando se toque el motor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

torch = pytest.importorskip("torch")

import pasada_119_c3 as p119  # noqa: E402

from matrixai_engines.motores import densa_tabm  # noqa: E402

#: Lo que dicen los tres ficheros SELLADOS, escrito a mano: si la lectura del
#: protocolo se rompe (un campo que ya no se encuentra), esto lo dice.
RECETA_SELLADA = {
    "k": 8, "d_block": 256, "n_blocks": 2, "dropout": 0.1, "d_embedding": 24,
    "n_frequencies": 48, "frequency_init_scale": 0.01, "optimizador": "adamw",
    "tasa_de_aprendizaje": 0.002, "weight_decay": 0.0003, "recorte_de_gradiente": 1.0,
    "lote": 256, "paciencia": 16, "fraccion_del_presupuesto_para_entrenar": 0.75,
    "semilla_del_ruido_de_cuantiles": 0, "para_tras_epocas_sin_mejora": 17,
    "metrica_de_parada": "validation_loss_del_ensamblado",
}


@pytest.fixture(scope="module")
def receta():
    return p119.receta_que_fija_el_protocolo()


@pytest.fixture(scope="module")
def medida():
    """UN ajuste de verdad con espías (~1-7 s): lo comparten las pruebas."""
    return p119.medir_la_receta_del_motor()


def test_la_receta_se_lee_de_los_tres_ficheros_del_protocolo(receta):
    assert {k: v["valor"] for k, v in receta.items()} == RECETA_SELLADA
    assert receta["tasa_de_aprendizaje"]["de"].startswith("protocolo_119_v4.json")
    assert receta["k"]["de"].startswith("protocolo_119_v4_enmienda_1.json")
    assert receta["para_tras_epocas_sin_mejora"]["de"].startswith(
        "protocolo_119_v4_enmienda_2.json")


def test_cada_constante_del_motor_es_la_del_protocolo(receta):
    """S4, S5, S6, S8 y C14 de la auditoría 3 (lr, weight_decay, dropout,
    n_frequencies): la constante, contra el protocolo, una a una."""
    for clave, nombre in p119.CONSTANTES_DE_LA_RECETA.items():
        assert p119._mismo_valor(getattr(densa_tabm, nombre), receta[clave]["valor"]), (
            f"{nombre} = {getattr(densa_tabm, nombre)!r} y el protocolo fija "
            f"{receta[clave]['valor']!r} ({receta[clave]['de']})")


def test_un_ajuste_de_verdad_usa_adamw_y_la_receta_del_protocolo(receta, medida):
    """S10: la CLASE del optimizador que se construye, no el nombre que se
    declara. Y el lr y el weight_decay de su grupo, los argumentos de la red,
    el recorte, el lote (min(256, 300 filas)), la semilla y el plazo."""
    assert medida["clase_del_optimizador"] == "torch.optim.adamw.AdamW"
    usado = medida["usado"]
    assert usado["optimizador"] == receta["optimizador"]["valor"] == "adamw"
    assert usado["tasa_de_aprendizaje"] == 0.002 and usado["weight_decay"] == 0.0003
    assert {k: usado[k] for k in ("k", "d_block", "n_blocks", "dropout", "d_embedding",
                                  "n_frequencies", "frequency_init_scale")} == {
        k: RECETA_SELLADA[k] for k in ("k", "d_block", "n_blocks", "dropout", "d_embedding",
                                       "n_frequencies", "frequency_init_scale")}
    assert usado["recorte_de_gradiente"] == 1.0
    assert usado["lote"] == 256 and 44 in medida["lotes_de_entrenamiento_vistos"]
    assert usado["fraccion_del_presupuesto_para_entrenar"] == 0.75
    assert usado["semilla_del_ruido_de_cuantiles"] == 0
    assert medida["d_embedding_de_la_preparacion"] == 24
    assert p119.distintas_de_la_receta_medida(medida, receta) == {}


def test_la_guardia_de_la_pasada_mide_la_receta_y_la_declara():
    configuracion = p119.exigir_la_configuracion_de_la_enmienda()
    assert configuracion["receta_medida_en_un_ajuste"]["usado"]["optimizador"] == "adamw"
    assert configuracion["constantes_de_la_receta_leidas_del_motor"]["tasa_de_aprendizaje"] == 0.002
    json.dumps(configuracion)  # va entera al resultado


@pytest.mark.parametrize("constante,valor", [
    ("TASA_DE_APRENDIZAJE", 0.001), ("WEIGHT_DECAY", 0.01), ("DROPOUT", 0.3),
    ("N_FREQUENCIES", 16), ("D_EMBEDDING", 16), ("FREQUENCY_INIT_SCALE", 0.1),
    ("NORMA_MAXIMA_DE_RECORTE", 2.0), ("LOTE_MAXIMO", 64), ("PACIENCIA", 15),
    ("FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR", 0.9), ("SEMILLA_DEL_RUIDO_DE_CUANTILES", 1)])
def test_la_guardia_para_con_cualquier_constante_de_la_receta_cambiada(monkeypatch, constante,
                                                                         valor):
    monkeypatch.setattr(densa_tabm, constante, valor)
    with pytest.raises(SystemExit, match=constante):
        p119.exigir_la_configuracion_de_la_enmienda()


def test_la_guardia_para_si_el_ajuste_usa_adam_aunque_las_constantes_esten_bien(monkeypatch):
    """El caso S10 sin tocar el fichero: `torch.optim.AdamW` construye un
    `Adam`. Ninguna constante cambia y el predictor sigue escribiendo
    «adamw»: solo el espía lo ve."""
    monkeypatch.setattr(p119, "_RECETA_MEDIDA", {})
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.Adam)
    with pytest.raises(SystemExit, match="USA otra receta.*optimizador"):
        p119.exigir_la_configuracion_de_la_enmienda()


def test_la_guardia_para_si_el_protocolo_no_dice_un_valor(tmp_path, monkeypatch):
    v4 = json.loads(p119.RUTA_DEL_PROTOCOLO_V4.read_text(encoding="utf-8"))
    del v4["receta_entrenamiento"]["learning_rate"]
    falso = tmp_path / "protocolo_119_v4.json"
    falso.write_text(json.dumps(v4, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(p119, "RUTA_DEL_PROTOCOLO_V4", falso)
    with pytest.raises(SystemExit, match="no dice.*tasa_de_aprendizaje"):
        p119.exigir_la_configuracion_de_la_enmienda(medir=False)


def _intento(**cambios) -> dict:
    registro = {"dataset": "d", "repeticion": 0, "pliegue": 0, "estado": "completed",
                "arquitectura": {"k": 8, "d_block": 256, "n_blocks": 2, "dropout": 0.1,
                                 "d_embedding": 24, "n_frequencies": 48,
                                 "frequency_init_scale": 0.01, "d_out": 1},
                "hiperparametros": {"optimizador": "adamw", "tasa_de_aprendizaje": 0.002,
                                    "weight_decay": 0.0003, "recorte_de_gradiente": 1.0,
                                    "lote": 256, "fraccion_del_presupuesto_para_entrenar": 0.75,
                                    "parada_temprana": {"paciencia": 16,
                                                        "para_tras_epocas_sin_mejora": 17,
                                                        "metrica": "validation_loss_del_ensamblado"}}}
    for ruta, valor in cambios.items():
        *padres, hoja = ruta.split("__")
        destino = registro
        for p in padres:
            destino = destino[p]
        if valor is None:
            del destino[hoja]
        else:
            destino[hoja] = valor
    return registro


def test_cada_intento_se_compara_con_la_receta_entera(receta):
    p119.exigir_la_arquitectura_del_intento(_intento(), receta=receta, n_train=300)
    p119.exigir_la_arquitectura_del_intento(_intento(hiperparametros__lote=100), receta=receta,
                                            n_train=100)
    for cambio in ({"hiperparametros__tasa_de_aprendizaje": 0.001},
                   {"hiperparametros__optimizador": "adam"},
                   {"arquitectura__dropout": 0.3}, {"arquitectura__n_frequencies": 16},
                   {"hiperparametros__parada_temprana__para_tras_epocas_sin_mejora": 16},
                   {"hiperparametros__lote": 128}):
        with pytest.raises(SystemExit, match="declara haber corrido"):
            p119.exigir_la_arquitectura_del_intento(_intento(**cambio), receta=receta,
                                                    n_train=300)
    with pytest.raises(SystemExit, match="SIN declarar.*weight_decay"):
        p119.exigir_la_arquitectura_del_intento(_intento(hiperparametros__weight_decay=None),
                                                receta=receta)

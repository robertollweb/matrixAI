"""JSON no tiene NaN, y el core lo estaba escribiendo.

Lo encontró Roberto entrenando: un run que diverge deja `train_loss:
NaN`, `json.dumps` lo escribe como el literal `NaN` —que no es JSON
válido— y a partir de ahí CONSULTAR el entrenamiento fallaba entero con
«Unexpected token 'N'». O sea que ni siquiera se podía ver que había
divergido: el fallo tapaba su propia causa.
"""
import json
import math

from matrixai.playground import json_seguro


def test_nan_se_vuelve_null_no_cero():
    # `null`, NUNCA `0`: un valor ausente no es un cero, y un cero en la
    # pérdida se leería como el mejor resultado posible justo cuando ha
    # pasado lo contrario.
    assert json_seguro(float("nan")) is None


def test_infinitos_tambien():
    assert json_seguro(float("inf")) is None
    assert json_seguro(float("-inf")) is None


def test_los_numeros_de_verdad_no_se_tocan():
    assert json_seguro(0.0) == 0.0
    assert json_seguro(-3.5) == -3.5
    assert json_seguro(0) == 0


def test_recursivo_por_donde_viven_las_perdidas():
    # Las pérdidas viven dentro de `epochs`, una lista de diccionarios:
    # sanear solo el nivel de arriba no habría arreglado nada.
    payload = {
        "ok": True,
        "epochs": [{"epoch": 1, "train_loss": float("nan"), "val_loss": 0.5}],
    }
    salida = json_seguro(payload)
    assert salida["epochs"][0]["train_loss"] is None
    assert salida["epochs"][0]["val_loss"] == 0.5
    assert salida["ok"] is True


def test_lo_que_sale_ES_json_valido():
    # La prueba que de verdad importa: que el navegador pueda leerlo.
    crudo = {"epochs": [{"train_loss": float("nan")}], "acc": float("inf")}
    texto = json.dumps(json_seguro(crudo))
    assert "NaN" not in texto
    assert json.loads(texto) == {"epochs": [{"train_loss": None}], "acc": None}


def test_sin_sanear_json_dumps_produce_algo_que_nadie_puede_leer():
    # Deja constancia de POR QUÉ hace falta: no es una manía, es que
    # `json.dumps` produce un literal que ningún parser acepta.
    texto = json.dumps({"train_loss": float("nan")})
    assert "NaN" in texto
    with __import__("pytest").raises(ValueError):
        json.dumps({"a": float("nan")}, allow_nan=False)

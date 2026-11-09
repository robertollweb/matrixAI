"""149-C7 — la sección `comportamiento` del recibo firmado, compuesta por el núcleo (`comportamiento_en_el_recibo`).

Se prueba con el banco REAL del núcleo (`monotonia_en_rejilla` sobre un puntuador de juguete con tres columnas de comportamiento conocido:
sube, oscila, plana) y con valores CENTINELA en las filas: ningún id de fila ni valor de fila real puede aparecer en lo compuesto.
"""
from __future__ import annotations

import json

import pytest

from matrixai.estudio import banco_en_el_paquete as paquete
from matrixai.estudio import comportamiento as nucleo
from matrixai.estudio import comportamiento_en_el_recibo as recibo
from matrixai.estudio.comportamiento_textos import IDIOMAS

CENTINELA_ID = "FILA-REAL-7741"
CENTINELA_VALOR = 31337.125


def _filas():
    return [{"edad": float(20 + i), "oscila": float(i % 7), "plana": 5.0, "ciudad": "Madrid"} for i in range(60)]


def _doc(columna, direccion, puntuar, tarea="regression"):
    filas = _filas()
    escala = nucleo.escala_de_la_salida(tarea, amplitud_p90_p10=10.0)
    ids = [f"{CENTINELA_ID}-{i}" for i in range(len(filas))]
    d = nucleo.monotonia_en_rejilla(columna, direccion, filas, puntuar, minimo=0.0, maximo=CENTINELA_VALOR, puntos=25, filas_m=40, semilla=149,
                                    escala=escala, medido_sobre="muestra_de_desarrollo", ids=ids, motor_de_arboles=True)
    return {"declaracion": {"id": 3, "columna": columna, "direccion": direccion, "momento": "antes", "declarada_en": "2026-11-08T10:00:00+00:00"},
            "documento": d.a_json(),
            "fila_tipica": {"medido_sobre": "fila_tipica", "de_donde": "valor_de_referencia_de_train", "columna": columna, "puntos": 3,
                            "curva": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]}}}


def resultado_de_juguete():
    sube = _doc("edad", "sube", lambda fs: [f["edad"] * 0.5 for f in fs])
    oscila = _doc("oscila", "sube", lambda fs: [(-1) ** int(f["oscila"]) * 3.0 for f in fs])
    plana = _doc("plana", "sube", lambda fs: [1.0 for _ in fs])
    return {"version": 2, "version_del_banco": nucleo.VERSION_DEL_COMPORTAMIENTO, "motor": "lightgbm", "tarea": "regression",
            "muestra": {"origen": "desarrollo", "semilla": 149, "filas": 60, "filas_de_desarrollo": 480},
            "parametros": {"semilla": 149, "filas_de_b": 200, "filas_de_c": 200, "puntos_de_la_rejilla": 25, "fracciones_de_b": [0.01, 0.05]},
            "escala": nucleo.escala_de_la_salida("regression", amplitud_p90_p10=10.0).a_json(),
            "tolerancia_de_partida": {"epsilon_relativo": 0.05, "q_relativo": 0.05},
            "monotonia": [sube, oscila, plana],
            "no_aplica": [{"columna": "ciudad", "prueba": "monotonia", "tipo": "categorica",
                           "motivo": {"es": "Es una columna de categorías.", "en": "It is a column of categories."}}]}


PROTOCOLO = {"semilla": 149, "filas_por_columna": 200, "puntos_de_la_rejilla": 25, "tope_de_tiempo_segundos": 120.0}


def declaraciones(*columnas):
    return [{"id": i + 1, "columna": c, "direccion": "sube", "momento": "antes", "declarada_en": "2026-11-08T10:00:00+00:00"}
            for i, c in enumerate(columnas)]


def seccion(columnas=("edad", "oscila", "plana", "ciudad", "tarde"), **extra):
    return recibo.seccion_del_recibo(
        declaraciones=declaraciones(*columnas), resultado=extra.pop("resultado", resultado_de_juguete()), protocolo=PROTOCOLO,
        sin_comprobar=extra.pop("sin_comprobar", {"tarde": recibo.motivo_del_tope(120.0)}), **extra)


def test_sin_declaraciones_no_hay_seccion():
    """I12: nadie declaró nada → `None` (la clave AUSENTE del cuerpo, nunca `null`)."""
    assert recibo.seccion_del_recibo(declaraciones=[], resultado=resultado_de_juguete(), protocolo=PROTOCOLO) is None
    assert recibo.seccion_del_recibo(declaraciones=[], resultado=None, protocolo=PROTOCOLO) is None


def test_cada_columna_declarada_figura_con_su_estado_y_su_marca():
    s = seccion()
    assert [c["declaracion"]["columna"] for c in s["columnas"]] == ["edad", "oscila", "plana", "ciudad", "tarde"]
    assert [c["estado"] for c in s["columnas"]] == ["comprobada", "comprobada", "no_se_pudo_comprobar", "no_aplica", "no_se_pudo_comprobar"]
    assert all(c["estado"] in recibo.ESTADOS for c in s["columnas"])
    # la marca la puso el servidor; el `id` interno del Studio no viaja
    assert all(c["declaracion"]["momento"] == "antes" and "id" not in c["declaracion"] for c in s["columnas"])
    veredictos = {c["declaracion"]["columna"]: c["veredicto"]["clave"] for c in s["columnas"] if "veredicto" in c}
    assert veredictos == {"edad": "sin_contraejemplos", "oscila": "con_contraejemplos", "plana": "no_se_pudo_comprobar"}
    # 149-C7 remate (a): estado y veredicto no se contradicen; lo medido se conserva y el motivo está
    plana = next(c for c in s["columnas"] if c["declaracion"]["columna"] == "plana")
    assert plana["estado"] == "no_se_pudo_comprobar" and plana["motivo"]["es"] and plana["motivo"]["en"]


def test_lo_que_no_se_pudo_comprobar_dice_por_que_en_los_dos_idiomas():
    s = seccion()
    ciudad, tarde = s["columnas"][3], s["columnas"][4]
    assert set(ciudad["motivo"]) == set(IDIOMAS) == set(tarde["motivo"])
    assert "120" in tarde["motivo"]["es"] and "120" in tarde["motivo"]["en"]
    assert "veredicto" not in tarde and "veredicto" not in ciudad, "no se inventa un «sin contraejemplos» para lo que no se comprobó"


def test_el_protocolo_dice_semilla_filas_rejilla_tolerancia_y_tope():
    p = seccion()["protocolo"]
    assert (p["semilla"], p["filas_por_columna"], p["puntos_de_la_rejilla"], p["tope_de_tiempo_segundos"]) == (149, 200, 25, 120.0)
    assert (p["filas_de_la_muestra"], p["filas_de_desarrollo"], p["medido_sobre"]) == (60, 480, "muestra_de_desarrollo")
    assert p["tolerancia"]["relativa"] == 0.05 and p["tolerancia"]["fraccion_minima"] == 0.05 and p["tolerancia"]["por_omision"] is True
    assert p["version_del_banco"] == nucleo.VERSION_DEL_COMPORTAMIENTO


def test_los_limites_van_en_los_dos_idiomas_y_sin_garantia():
    s = seccion()
    assert set(s["limites"]) == set(IDIOMAS)
    assert "garantía" in s["limites"]["es"] and "guarantee" in s["limites"]["en"]       # la frase fija DICE que no lo es
    assert all("limites" in c["veredicto"] for c in s["columnas"] if c["estado"] == "comprobada")


def test_NINGUNA_fila_real_en_la_seccion_ni_su_id_ni_sus_valores():
    """D5: solo agregados y contraejemplos sintéticos. Los ids centinela y el valor centinela (el máximo de la columna, que usa la
    rejilla) no aparecen como fila: el rango visto SÍ es un agregado, así que se busca el id y la clave de fila, no el máximo."""
    texto = json.dumps(seccion(), ensure_ascii=False)
    assert CENTINELA_ID not in texto
    paquete.exigir_sin_filas(seccion())
    claves = set()

    def recorre(x):
        if isinstance(x, dict):
            claves.update(x)
            [recorre(v) for v in x.values()]
        elif isinstance(x, list):
            [recorre(v) for v in x]
    recorre(seccion())
    assert not claves & paquete.CLAVES_DE_FILA, claves & paquete.CLAVES_DE_FILA


def test_una_composicion_que_dejase_pasar_una_fila_se_NIEGA_a_devolverla(monkeypatch):
    """La segunda cerradura: si la poda de `_monotonia` filtrase una clave de fila, `exigir_sin_filas` aborta y no sale sección."""
    real = recibo._monotonia
    monkeypatch.setattr(recibo, "_monotonia", lambda item, **k: {**real(item, **k), "ids": ["FILA-REAL-1"]})
    with pytest.raises(paquete.BancoConFilas):
        seccion()


def test_es_determinista_y_no_cambia_lo_que_recibe():
    entrada = resultado_de_juguete()
    antes = json.dumps(entrada, sort_keys=True)
    a, b = seccion(resultado=entrada), seccion(resultado=entrada)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(entrada, sort_keys=True) == antes
    a["columnas"][0]["estado"] = "otro"
    assert seccion(resultado=entrada)["columnas"][0]["estado"] == "comprobada", "devuelve una copia"


def test_sin_resultado_cada_columna_dice_por_que_no_se_pudo():
    motivo = {"es": "No hay muestra.", "en": "There is no sample."}
    s = recibo.seccion_del_recibo(declaraciones=declaraciones("edad"), resultado=None, protocolo=PROTOCOLO, motivo_general=motivo)
    assert s["columnas"][0]["estado"] == "no_se_pudo_comprobar" and s["columnas"][0]["motivo"] == motivo
    assert s["protocolo"]["tolerancia"] is None and s["protocolo"]["filas_de_la_muestra"] is None, "lo no medido es None, no un cero"
    assert set(s["limites"]) == set(IDIOMAS)


def test_los_motivos_de_no_se_pudo_comprobar_empiezan_en_mayuscula_y_acaban_en_punto():
    """Remate del 149-C8: la lectura de «no se pudo comprobar» empezaba en minúscula y acababa a medias. Cada motivo que sale de verdad."""
    from matrixai.estudio.comportamiento_textos import MOTIVOS
    claves = ["amplitud_no_distingue", "amplitud_no_distingue_con_filas_que_se_mueven", "sin_prediccion", "sin_filas", "rejilla_sin_rango",
              "escala_de_regresion_no_valida", "recalibracion_no_conserva_el_orden", "multiclase_sin_clase_declarada", "salida_no_numerica",
              "tope_de_tiempo_del_recibo", "fallo_en_el_recibo"]
    for clave in claves:
        for idioma in IDIOMAS:
            texto = MOTIVOS[clave][idioma]
            assert texto[0].isupper() and texto.rstrip().endswith("."), (clave, idioma, texto[:30], texto[-20:])

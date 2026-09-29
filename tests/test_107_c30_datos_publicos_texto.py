# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — los datos públicos de la medición de TEXTO salen de los registros sellados,
nunca de una mano.

`benchmarks/datos_publicos/texto_107_publico.json` es lo que leerá la página pública sobre
la medición de texto del contrato 107. Mismo patrón que `tests/test_117_c1_datos_publicos.py`
(117-C1, Fase 0): si alguien retoca una cifra, o un registro cambia y nadie regenera, esto
falla.

**Esta prueba NUNCA lee `benchmarks/texto_107c30/tareas/`** (el CSV, un enlace sin
versionar a datos locales que no se commitea): solo los JSON versionados de
`benchmarks/texto_107c30/` y `benchmarks/datos_publicos/`.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
TEXTO_107C30 = RAIZ / "benchmarks" / "texto_107c30"

_spec = importlib.util.spec_from_file_location(
    "datos_publicos_generar_texto_107", RAIZ / "benchmarks" / "datos_publicos" / "generar_texto_107.py")
generar_texto_107 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generar_texto_107)


def _publicado() -> dict:
    return json.loads(generar_texto_107.RUTA_PUBLICADO.read_text(encoding="utf-8"))


def test_lo_publicado_es_BYTE_A_BYTE_lo_que_sale_de_los_registros():
    assert generar_texto_107.RUTA_PUBLICADO.read_text(encoding="utf-8") == generar_texto_107.generar(), (
        "texto_107_publico.json no es lo que sale de los registros sellados: "
        "python3 benchmarks/datos_publicos/generar_texto_107.py --escribir, y mirar por qué cambió")


def test_cada_metrica_puntual_publicada_es_exactamente_la_de_su_fuente():
    """No solo que el JSON entero coincida byte a byte (eso ya lo prueba la anterior):
    aquí se releen las fuentes por separado y se compara CADA `metrica_puntual` publicada
    contra el número que la fuente correspondiente escribió, condición por condición."""
    publicado = _publicado()
    c30 = json.loads((TEXTO_107C30 / "resultado_c30.json").read_text(encoding="utf-8"))

    for tarea in ("A", "B", "C", "D"):
        condiciones = publicado["tareas"][tarea]["condiciones"]
        for clave in ("0_sin_texto", "1_embedding", "2_tfidf_lineal"):
            assert condiciones[clave]["metrica_puntual"] == (
                c30["tareas"][tarea]["condiciones"][clave]["metrica_puntual"]), (tarea, clave)

    respondedor_a = json.loads((TEXTO_107C30 / "resultado_c30_respondedor_a.json").read_text(encoding="utf-8"))
    respondedor_b = json.loads((TEXTO_107C30 / "resultado_c30_respondedor_b.json").read_text(encoding="utf-8"))
    assert publicado["tareas"]["A"]["condiciones"]["3_respondedor"]["metrica_puntual"] == (
        respondedor_a["tareas"]["A"]["condiciones"]["3_respondedor"]["metrica_puntual"])
    assert publicado["tareas"]["B"]["condiciones"]["3_respondedor"]["metrica_puntual"] == (
        respondedor_b["tareas"]["B"]["condiciones"]["3_respondedor"]["metrica_puntual"])

    jev_a = json.loads((TEXTO_107C30 / "resultado_c30_jev_A.json").read_text(encoding="utf-8"))
    assert publicado["tareas"]["A"]["condiciones"]["4_jev"]["metrica_puntual"] == (
        jev_a["condiciones"]["4_jev"]["metrica_puntual"])


def test_la_tarea_B_no_tiene_la_condicion_4_medida():
    """Decisión de Roberto: B son datos clínicos y NUNCA pasa por Jev. Tiene que salir
    `omitida` con esa clave de motivo exacta, no con la genérica de coste ni con un hueco."""
    publicado = _publicado()
    condicion_4_de_b = publicado["tareas"]["B"]["condiciones"]["4_jev"]
    assert condicion_4_de_b == {"omitida": True, "motivo": "nunca_con_jev_datos_clinicos"}
    assert "jev" not in publicado["tareas"]["B"]  # tampoco hay bloque de estadísticas de Jev para B


def test_la_tarea_D_condicion_3_y_4_omitidas_con_sus_claves():
    """D nunca se manda a Jev (fuera del registro) ni al respondedor local (coste),
    comprobado contra lo que los propios registros de (3) ya declaran para D."""
    publicado = _publicado()
    condiciones_d = publicado["tareas"]["D"]["condiciones"]
    assert condiciones_d["3_respondedor"] == {"omitida": True, "motivo": "no_medida_por_coste"}
    assert condiciones_d["4_jev"] == {"omitida": True, "motivo": "fuera_del_registro"}


def test_la_tarea_C_sin_su_resultado_de_jev_sale_pendiente():
    """`resultado_c30_jev_C.json` todavía no existe (se está midiendo). Se prueba con un
    directorio temporal que copia las entradas SIN ese fichero -- nunca se toca el árbol
    real -- para comprobar que el generador declara la (4) de C como `pendiente` en vez de
    fallar o inventar un hueco silencioso, y que sigue componiendo el resto con normalidad."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ficheros_a_copiar = [
            "resultado_c30.json", "resumen_tareas.json", "marcador_number_A.json",
            "resultado_c30_respondedor_a.json", "resultado_c30_respondedor_b.json",
            "resultado_c30_jev_A.json", "respuestas_c30_jev_A.json",
            "respuestas_c30_respondedor_a.json", "respuestas_c30_respondedor_b.json",
        ]
        for nombre in ficheros_a_copiar:
            origen = TEXTO_107C30 / nombre
            assert origen.exists(), f"falta {nombre} en el árbol real, no se puede montar la prueba"
            shutil.copy2(origen, tmp_path / nombre)
        # A PROPÓSITO no se copia resultado_c30_jev_C.json (que hoy no existe de todas
        # formas): esta prueba sigue siendo válida el día que aparezca en el árbol real.
        assert not (tmp_path / "resultado_c30_jev_C.json").exists()

        _spec2 = importlib.util.spec_from_file_location(
            "datos_publicos_generar_texto_107_tmp",
            RAIZ / "benchmarks" / "datos_publicos" / "generar_texto_107.py")
        modulo_tmp = importlib.util.module_from_spec(_spec2)
        _spec2.loader.exec_module(modulo_tmp)

        # Apunta el módulo recién cargado al directorio temporal, no al árbol real.
        modulo_tmp.TEXTO_107C30 = tmp_path
        modulo_tmp.RUTA_RESULTADO_C30 = tmp_path / "resultado_c30.json"
        modulo_tmp.RUTA_RESUMEN_TAREAS = tmp_path / "resumen_tareas.json"
        modulo_tmp.RUTA_MARCADOR_NUMBER_A = tmp_path / "marcador_number_A.json"
        modulo_tmp.RUTAS_RESPONDEDOR = {
            "A": tmp_path / "resultado_c30_respondedor_a.json",
            "B": tmp_path / "resultado_c30_respondedor_b.json",
        }
        modulo_tmp.RUTAS_RESPUESTAS_RESPONDEDOR = {
            "A": tmp_path / "respuestas_c30_respondedor_a.json",
            "B": tmp_path / "respuestas_c30_respondedor_b.json",
        }

        datos = json.loads(modulo_tmp.generar())

    assert datos["tareas"]["C"]["condiciones"]["4_jev"] == {"omitida": True, "motivo": "pendiente"}
    # El resto de C sigue compuesto con normalidad -- no es un fallo general disfrazado.
    assert datos["tareas"]["C"]["condiciones"]["0_sin_texto"]["metrica_puntual"] == pytest.approx(0.6534988182550998)
    assert datos["tareas"]["C"]["metrica"] == "rmse"
    assert datos["tareas"]["C"]["mayor_es_mejor"] is False
    # Y la tarea A, que sí tiene todo, no se ve afectada por copiar el directorio.
    assert datos["tareas"]["A"]["condiciones"]["4_jev"]["metrica_puntual"] == pytest.approx(0.7154689715878527)


def test_el_acuerdo_de_la_tarea_A_se_recalcula_de_forma_INDEPENDIENTE_y_coincide():
    """No basta con que el generador esté satisfecho consigo mismo: se vuelve a calcular el
    acuerdo pregunta a pregunta aquí, leyendo las dos cachés de respuestas directamente y
    quitando a mano el prefijo `"A\\x1f"` del respondedor local, sin pasar por
    `generar_texto_107._jev_de_las_respuestas`. Si el generador tuviera un error de
    desplazamiento en el prefijo o comparase la clave equivocada, esta prueba lo vería sin
    heredar el mismo fallo."""
    jev = json.loads((TEXTO_107C30 / "respuestas_c30_jev_A.json").read_text(encoding="utf-8"))
    local = json.loads((TEXTO_107C30 / "respuestas_c30_respondedor_a.json").read_text(encoding="utf-8"))

    local_sin_prefijo = {}
    for clave, valor in local.items():
        assert clave.startswith("A\x1f")
        local_sin_prefijo[clave[2:]] = valor  # "A" + "\x1f" = 2 caracteres

    comunes = set(jev) & set(local_sin_prefijo)
    assert len(comunes) == len(jev) == len(local_sin_prefijo), "se esperaban las mismas claves en las dos cachés"
    n_coincidencias = sum(1 for k in comunes if jev[k]["valor"] == local_sin_prefijo[k]["valor"])

    publicado = _publicado()
    acuerdo = publicado["tareas"]["A"]["jev"]["acuerdo_con_respondedor_local"]
    assert acuerdo["n_comparadas"] == len(comunes)
    assert acuerdo["n_coincidencias"] == n_coincidencias
    assert acuerdo["proporcion"] == pytest.approx(n_coincidencias / len(comunes))


def test_las_diferencias_descritas_de_A_coinciden_con_lo_que_los_propios_registros_ya_escribieron():
    """(4) vs (2) y (3) vs (2) ya están escritas como `..._diferencia_descrita` dentro de
    los registros sellados de (4) y (3): esta prueba las compara directamente contra lo
    publicado, sin pasar por el cálculo del generador -- si el generador recalculase con la
    métrica equivocada, esta prueba lo vería porque compara con el registro, no consigo
    misma."""
    jev_a = json.loads((TEXTO_107C30 / "resultado_c30_jev_A.json").read_text(encoding="utf-8"))
    respondedor_a = json.loads((TEXTO_107C30 / "resultado_c30_respondedor_a.json").read_text(encoding="utf-8"))
    publicado = _publicado()
    diferencias = publicado["tareas"]["A"]["diferencias_descritas"]

    assert diferencias["4_jev_menos_2_tfidf_lineal"]["diferencia"] == pytest.approx(
        jev_a["comparaciones"]["4_vs_2_diferencia_descrita"]["diferencia_4_menos_2"])
    assert diferencias["3_respondedor_menos_2_tfidf_lineal"]["diferencia"] == pytest.approx(
        respondedor_a["tareas"]["A"]["comparaciones"]["3_vs_2_diferencia_descrita"]["diferencia_3_menos_2"])
    # (4) vs (3) no está en NINGÚN registro sellado -- se calcula aquí y se comprueba contra
    # la resta directa de los dos `metrica_puntual` ya copiados y ya verificados arriba.
    m4 = publicado["tareas"]["A"]["condiciones"]["4_jev"]["metrica_puntual"]
    m3 = publicado["tareas"]["A"]["condiciones"]["3_respondedor"]["metrica_puntual"]
    assert diferencias["4_jev_menos_3_respondedor"]["diferencia"] == pytest.approx(m4 - m3)
    assert "nota" not in diferencias["4_jev_menos_3_respondedor"]  # nadie la redactó: no hay fuente que copiar


def test_una_cifra_retocada_en_un_registro_cambia_lo_publicado():
    """La otra mitad, igual que en 117-C1: probar el artefacto no es probar el código que
    lo produce. Se compone un JSON NUEVO con el mismo código de generación, a partir de un
    `resultado_c30.json` con una métrica cambiada -- sin tocar el fichero real."""
    original = json.loads((TEXTO_107C30 / "resultado_c30.json").read_text(encoding="utf-8"))
    tocado = json.loads(json.dumps(original))  # copia profunda vía JSON, sin depender de `copy`
    tocado["tareas"]["A"]["condiciones"]["2_tfidf_lineal"]["metrica_puntual"] += 0.111

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for nombre in ("resumen_tareas.json", "marcador_number_A.json",
                       "resultado_c30_respondedor_a.json", "resultado_c30_respondedor_b.json",
                       "resultado_c30_jev_A.json", "respuestas_c30_jev_A.json",
                       "respuestas_c30_respondedor_a.json", "respuestas_c30_respondedor_b.json"):
            shutil.copy2(TEXTO_107C30 / nombre, tmp_path / nombre)
        (tmp_path / "resultado_c30.json").write_text(json.dumps(tocado), encoding="utf-8")

        _spec3 = importlib.util.spec_from_file_location(
            "datos_publicos_generar_texto_107_sabotaje",
            RAIZ / "benchmarks" / "datos_publicos" / "generar_texto_107.py")
        modulo_tmp = importlib.util.module_from_spec(_spec3)
        _spec3.loader.exec_module(modulo_tmp)
        modulo_tmp.TEXTO_107C30 = tmp_path
        modulo_tmp.RUTA_RESULTADO_C30 = tmp_path / "resultado_c30.json"
        modulo_tmp.RUTA_RESUMEN_TAREAS = tmp_path / "resumen_tareas.json"
        modulo_tmp.RUTA_MARCADOR_NUMBER_A = tmp_path / "marcador_number_A.json"
        modulo_tmp.RUTAS_RESPONDEDOR = {
            "A": tmp_path / "resultado_c30_respondedor_a.json",
            "B": tmp_path / "resultado_c30_respondedor_b.json",
        }
        modulo_tmp.RUTAS_RESPUESTAS_RESPONDEDOR = {
            "A": tmp_path / "respuestas_c30_respondedor_a.json",
            "B": tmp_path / "respuestas_c30_respondedor_b.json",
        }

        # OJO: `modulo_tmp` se cargó con su propio `importlib.util.module_from_spec`, así que
        # su `DatosQueNoCuadran` es una clase DISTINTA (mismo nombre, otro objeto) de la del
        # módulo cargado arriba para el resto de la suite -- se captura la del propio módulo.
        with pytest.raises(modulo_tmp.DatosQueNoCuadran):
            # con 2_tfidf_lineal tocado, la diferencia descrita que YA escribió el
            # registro de (3) (3_vs_2_diferencia_descrita) deja de cuadrar: no se publica.
            modulo_tmp.generar()

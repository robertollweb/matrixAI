# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169 (rep1, I2 de la aud1) — el BUCLE de `medir_c169.main`, con un motor FALSO: ni descarga ni mide.

`ejecutar_intento_aislado` (el que lanza el motor en su subproceso) se sustituye por un falso que GRABA lo que recibe (motor, particiones,
presupuesto) y devuelve un intento completado con una declaración de brazo válida. `cj.asegurar_todos` (la descarga verificada) se sustituye por
un espía: no baja nada. La preparación (`c3.preparar_para_motor`) es la real, envuelta para grabar con qué filas se AJUSTA. Lo demás es el camino
real de `main`: protocolo v2 ampliado en memoria, lectura del ARFF (los ficheros de este equipo; sin ellos, se salta), partición, plan, caché y
escritura de la salida.

Lo que estas pruebas cazan (los sabotajes del medidor de la aud1): presupuesto fijo (M1), un hilo (M2), preparación ajustada con la prueba (M3),
otra semilla (M4), otra métrica de cierre (M5), reutilizar cualquier intento previo (M6), la rejilla primero (M7), no verificar los datos (M8) y
la validación usada como prueba (M9).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]
for _p in (_RAIZ / "benchmarks" / "c169", _RAIZ / "benchmarks" / "c163", _RAIZ / "benchmarks" / "fase0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analiza_c169 as an  # noqa: E402
import conjuntos_c169 as cj  # noqa: E402
import medir_c169 as m  # noqa: E402

CONJUNTO = "blood-transfusion-service-center"        # binaria (AUROC, no accuracy: M5 se ve), pequeña (120 s), 748 filas
DATA_ID = 1464


def _hiperparametros(brazo: str) -> dict:
    hip = {"politica_por_tamano": {"aplicada": True, "tamano": "pequeno"}}
    if brazo == an.REJILLA:
        hip["politica_por_tamano"]["validacion_mejor"] = 0.40
        hip["rejilla_por_validacion"] = {"aplicada": True, "mira_la_prueba": False, "criterio": "perdida_en_validation_del_pliegue",
                                         "configuraciones": [{"validacion_mejor": 0.40 + 0.01 * i} for i in range(8)], "elegida": 0}
    return hip


class MotorFalso:
    """El sustituto de `ejecutar_intento_aislado`: graba cada llamada y devuelve un intento `completed` sin entrenar nada."""

    def __init__(self):
        self.llamadas: list[dict] = []

    def __call__(self, motor, train, val, test, spec, presupuesto, *, candidate, split_plan_digest, dataset, pliegue, repeticion, **_):
        self.llamadas.append({"brazo": candidate, "motor": motor, "train": tuple(train.row_ids), "val": tuple(val.row_ids),
                              "test": tuple(test.row_ids), "presupuesto": presupuesto, "dataset": dataset, "pliegue": pliegue,
                              "repeticion": repeticion})
        informe = {"metrics": [{"metric_id": "auroc", "value": 0.70 + 0.01 * pliegue}, {"metric_id": "accuracy", "value": 0.75}]}
        return types.SimpleNamespace(estado="completed", informe=informe, recursos={"wall_seconds": 1.0, "cpu_seconds": 1.0, "peak_ram_mb": 50.0},
                                     config_efectiva={"hiperparametros": _hiperparametros(candidate)}, motivo_del_estado=None,
                                     engine_version="falso")


@pytest.fixture
def camino(monkeypatch, tmp_path):
    """El camino real de `main` con el motor falso, la descarga espiada, la preparación espiada y los sellos «puestos» (los de AHORA)."""
    import pasada_amplia_101_c5 as c5
    import pasada_exploratoria_101_c3 as c3
    if not (c3.ARFF_DIR / f"{DATA_ID}.arff").exists():
        pytest.skip("el ARFF de blood no está en este equipo (`medir_c169` lo baja y lo verifica)")
    # lo que `main` cambia de forma global, para que monkeypatch lo deje como estaba
    for modulo, atributo in ((c5, "cargar_arff"), (c3, "_CATALOGO_POR_DATA_ID"), (c3, "_RUTA_DEL_CATALOGO"), (c3, "RUTA_DEL_PROTOCOLO")):
        monkeypatch.setattr(modulo, atributo, getattr(modulo, atributo))
    monkeypatch.setattr(c3, "LECTURA_DECLARADA", dict(c3.LECTURA_DECLARADA))
    # los sellos: la regla sellada es la real; el del código, uno temporal con el código de AHORA
    sello = tmp_path / "SELLO_DEL_CODIGO.json"
    sello.write_text(json.dumps({"regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "sha256": an.codigo_de_ahora()}), encoding="utf-8")
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", sello)
    eventos: list = []
    falso = MotorFalso()
    monkeypatch.setattr(m, "ejecutar_intento_aislado", lambda *a, **k: (eventos.append("medir"), falso(*a, **k))[1])
    monkeypatch.setattr(cj, "asegurar_todos", lambda *a, **k: eventos.append("asegurar_todos") or {})
    preparaciones: list[dict] = []
    real_preparar = c3.preparar_para_motor

    def preparar(ajuste, todas, *a, **k):
        preparaciones.append({"ajuste": tuple(f["row_id"] for f in ajuste), "todas": tuple(f["row_id"] for f in todas)})
        return real_preparar(ajuste, todas, *a, **k)
    monkeypatch.setattr(c3, "preparar_para_motor", preparar)
    planes: dict = {}
    real_particiones = c5.particiones_base

    def particiones(ds, protocolo):
        eventos.append("particiones_base")
        out = real_particiones(ds, protocolo)
        planes[ds.nombre] = out
        return out
    monkeypatch.setattr(c5, "particiones_base", particiones)
    salida = tmp_path / "salida" / "resultado.json"

    def correr(*extra):
        m.main(["--solo", CONJUNTO, "--salida", str(salida), *extra])
        return json.loads(salida.read_text(encoding="utf-8"))
    return types.SimpleNamespace(correr=correr, falso=falso, eventos=eventos, preparaciones=preparaciones, planes=planes, salida=salida)


def _plan_de(camino):
    por_id, propuesta, *_ = camino.planes[CONJUNTO]
    ds = next(d for d in cj.conjuntos() if d.nombre == CONJUNTO)
    return por_id, propuesta, ds


def test_bucle_verifica_los_datos_ANTES_de_leer_o_medir_nada(camino):
    """M8: sin `cj.asegurar_todos()` se mediría un fichero sin verificar su sha256 contra la tabla apartada."""
    camino.correr()
    assert camino.eventos.count("asegurar_todos") == 1
    assert camino.eventos.index("asegurar_todos") < camino.eventos.index("particiones_base") < camino.eventos.index("medir")


def test_bucle_mide_los_5_pliegues_la_referencia_PRIMERO_con_el_presupuesto_del_cubo_4_hilos_y_la_semilla_0(camino):
    """M1 (presupuesto fijo), M2 (un hilo), M4 (otra semilla) y M7 (la rejilla primero): lo que RECIBE el motor, no lo que se graba."""
    camino.correr()
    ll = camino.falso.llamadas
    assert [(x["pliegue"], x["brazo"]) for x in ll] == [(f, b) for f in range(5) for b in an.BRAZOS]
    assert all(x["motor"].rejilla_por_validacion is (x["brazo"] == an.REJILLA) for x in ll)
    for x in ll:
        pre = x["presupuesto"]
        assert (pre.wall_seconds, pre.hilos, pre.seed) == (120.0, 4, 0), x["brazo"]
        assert x["repeticion"] == 0 and x["dataset"] == CONJUNTO


def test_bucle_la_prueba_es_la_prueba_fija_y_no_la_validacion_y_las_tres_particiones_son_las_del_plan(camino):
    """M9: con la validación en el lugar de la prueba, la métrica de cierre se calcularía sobre las filas con las que la rejilla ELIGE."""
    import pasada_119_c3 as c119
    camino.correr()
    por_id, propuesta, _ = _plan_de(camino)
    test_ids = tuple(propuesta.plan.observaciones_del_rol("test"))
    protocolo = cj.protocolo_ampliado(m.c3.protocolo_registrado())
    ds_pasada = next(d for d in m.c5.datasets_de_la_pasada(protocolo) if d.nombre == CONJUNTO)
    plan = {i.pliegue: i for i in c119.plan_de_intentos(ds_pasada, propuesta, protocolo, humo=False) if i.repeticion == 0}
    for x in camino.falso.llamadas:
        ip = plan[x["pliegue"]]
        assert x["test"] == test_ids and len(test_ids) == 150
        assert x["train"] == tuple(ip.entrena) and x["val"] == tuple(ip.valida)
        assert not (set(x["test"]) & (set(x["train"]) | set(x["val"]))) and not (set(x["train"]) & set(x["val"]))


def test_bucle_la_preparacion_se_AJUSTA_solo_con_entrena(camino):
    """M3: ajustar la preparación con entrena + prueba es una fuga (la imputación y la escala verían la prueba)."""
    camino.correr()
    assert len(camino.preparaciones) == len(camino.falso.llamadas) == 10
    for prep, x in zip(camino.preparaciones, camino.falso.llamadas):
        assert prep["ajuste"] == x["train"]
        assert prep["todas"] == x["train"] + x["val"] + x["test"]


def test_bucle_la_salida_graba_lo_que_paso_y_pasa_lo_declarado_del_analizador(camino):
    """M5 (otra métrica) y lo que el analizador exige de cada registro: métrica de su tarea, cubo, semilla, hilos GRABADOS, digests, n_test."""
    out = camino.correr()
    rs = out["resultados"]
    assert len(rs) == 10 and all(r["metrica_de_cierre"] == "auroc" and r["hilos"] == 4 and r["semilla"] == 0
                                 and r["presupuesto_wall_s"] == 120.0 and r["n_test"] == 150 for r in rs)
    c = out["conjuntos"][0]
    assert c["metrica_de_cierre"] == "auroc" and c["n_test"] == 150 and c["predictores"] == ["V1", "V2", "V3", "V4"]
    assert out["sello_del_codigo_al_medir"] == an.codigo_de_ahora()
    assert an.problemas_de_lo_declarado(out, es_sonda=False) == []
    assert not [p for p in an.problemas_de_completitud(out, es_sonda=True)]       # (sonda: aquí solo hay 1 de los 6)


def test_bucle_NO_reutiliza_un_intento_fallido_ni_de_otro_codigo_y_SI_el_suyo(camino):
    """M6: reutilizar cualquier previo cementaría un `failed` o un intento de otro código. Primero: todo se re-mide. Después: todo se reusa."""
    out = camino.correr()
    malos = []
    for i, r in enumerate(out["resultados"]):
        r = dict(r)
        if i % 2:
            r["estado"] = "failed"                                       # mismo código, pero no cuenta como medida
        else:
            r["entorno_digest"] = "otro-codigo"                          # completado, pero con otro código
        malos.append(r)
    camino.salida.write_text(json.dumps({**out, "resultados": malos}), encoding="utf-8")
    camino.falso.llamadas.clear()
    out2 = camino.correr()
    assert len(camino.falso.llamadas) == 10 and out2["n_reusados"] == 0
    assert all(r["estado"] == "completed" and r["entorno_digest"] == out2["digest_de_la_cache"]["entorno"] for r in out2["resultados"])
    camino.falso.llamadas.clear()
    out3 = camino.correr()
    assert camino.falso.llamadas == [] and out3["n_reusados"] == 10


def test_bucle_NO_reutiliza_un_previo_de_otra_metrica_otra_semilla_u_otros_hilos(camino):
    out = camino.correr()
    for campo, valor in (("metrica_de_cierre", "accuracy"), ("semilla", 7), ("hilos", 1)):
        camino.salida.write_text(json.dumps({**out, "resultados": [dict(r, **{campo: valor}) for r in out["resultados"]]}), encoding="utf-8")
        camino.falso.llamadas.clear()
        camino.correr()
        assert len(camino.falso.llamadas) == 10, campo


def test_bucle_una_medida_real_no_empieza_sin_el_sello_del_codigo_y_una_sonda_si(camino, monkeypatch, tmp_path):
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", tmp_path / "no_existe.json")
    with pytest.raises(SystemExit, match="sello del CÓDIGO"):
        camino.correr()
    assert camino.falso.llamadas == [] and "asegurar_todos" not in camino.eventos
    out = camino.correr("--sonda")
    assert out["es_sonda"] is True and len(camino.falso.llamadas) == 2 and out["sello_del_codigo_al_medir"] is None
    assert all(x["presupuesto"].wall_seconds == m.PRESUPUESTO_DE_LA_SONDA_S for x in camino.falso.llamadas)

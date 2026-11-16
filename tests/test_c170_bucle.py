# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""170-C1 — el BUCLE de `medir_c170.main`, con un motor FALSO: ni ajusta ni mide.

`ejecutar_intento_aislado` (el que lanza el motor en su subproceso) se sustituye por un falso que GRABA lo que recibe (motor, particiones, presupuesto) y devuelve un intento completado. El motor real
(`MotorEBM`) se usa para lo que `main` le pregunta (capacidades para preparar los datos, su nombre), con `motivo_si_no_puede_entrenar` forzado a «disponible» (en el host no hay `interpret-core`). La preparación
(`c3.preparar_para_motor`) es la real, envuelta para grabar con qué filas se AJUSTA. Lo demás es el camino real de `main`: protocolo v2, lectura del ARFF (los de este equipo; sin ellos, se salta), partición,
plan, caché y escritura de la salida.

Lo que estas pruebas cazan: presupuesto fijo, un hilo, otra semilla, preparación ajustada con la prueba, otra métrica, reutilizar cualquier intento previo (también uno fallido o de otro código), no comprobar
la partición ni los predictores contra la v2, medir sin sello, medir fuera del régimen o un sellado.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]
for _p in (_RAIZ / "benchmarks" / "c170", _RAIZ / "benchmarks" / "fase0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analiza_c170 as an  # noqa: E402
import medir_c170 as m  # noqa: E402

CONJUNTO = "diabetes"        # binaria (AUROC), pequeña (120 s), 768 filas, 8 predictores, 15 pliegues


class MotorFalso:
    """El sustituto de `ejecutar_intento_aislado`: graba cada llamada y devuelve un intento `completed` (o el estado que se le pida)."""

    def __init__(self):
        self.llamadas: list[dict] = []
        self.estado = "completed"

    def __call__(self, motor, train, val, test, spec, presupuesto, *, candidate, split_plan_digest, dataset, pliegue, repeticion, **_):
        self.llamadas.append({"motor": motor, "train": tuple(train.row_ids), "val": tuple(val.row_ids), "test": tuple(test.row_ids), "presupuesto": presupuesto,
                              "dataset": dataset, "pliegue": pliegue, "repeticion": repeticion, "spec": spec, "split_plan_digest": split_plan_digest, "candidate": candidate})
        informe = {"metrics": [{"metric_id": "auroc", "value": 0.70 + 0.01 * pliegue}, {"metric_id": "accuracy", "value": 0.75}]}
        ok = self.estado == "completed"
        # `config_efectiva` en el producto es el predictor ENTERO (con las tablas): el medidor NO puede guardarlo
        config = {"hiperparametros": {"interpret_core": "0.7.8", "parametros": {"n_jobs": presupuesto.hilos}}, "modelo": {"terminos": [{"scores": [0.0] * 1000}]}}
        return types.SimpleNamespace(estado=self.estado, informe=informe if ok else None, recursos={"wall_seconds": 1.5, "cpu_seconds": 3.0, "peak_ram_mb": 50.0} if ok else {},
                                     config_efectiva=config if ok else None, motivo_del_estado=None if ok else {"es": "tope de pared agotado", "en": "wall budget exhausted"},
                                     engine_version=an.VERSION_DEL_MOTOR)


@pytest.fixture
def camino(monkeypatch, tmp_path):
    """El camino real de `main` con el motor falso, la preparación espiada y los sellos «puestos» (los de AHORA)."""
    import pasada_amplia_101_c5 as c5
    import pasada_exploratoria_101_c3 as c3
    if not (c3.ARFF_DIR / "37.arff").exists():
        pytest.skip("el ARFF de diabetes no está en este equipo")
    for modulo, atributo in ((c5, "cargar_arff"), (c3, "_CATALOGO_POR_DATA_ID"), (c3, "_RUTA_DEL_CATALOGO"), (c3, "RUTA_DEL_PROTOCOLO")):
        monkeypatch.setattr(modulo, atributo, getattr(modulo, atributo))
    monkeypatch.setattr(c3, "LECTURA_DECLARADA", dict(c3.LECTURA_DECLARADA))
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", an.sellos_de_ahora()["regla_sha256"])
    sello = tmp_path / "SELLO_DEL_CODIGO.json"
    sello.write_text(json.dumps({"regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "sha256": an.codigo_de_ahora()}), encoding="utf-8")
    monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", sello)
    falso = MotorFalso()
    real_motor_para = m.motor_para

    def motor_para(nombre):
        motor = real_motor_para(nombre)
        monkeypatch.setattr(type(motor), "motivo_si_no_puede_entrenar", lambda self: None)       # en el host no hay `interpret-core`
        return motor
    monkeypatch.setattr(m, "motor_para", motor_para)
    monkeypatch.setattr(m, "ejecutar_intento_aislado", falso)
    preparaciones: list[dict] = []
    real_preparar = c3.preparar_para_motor

    def preparar(ajuste, todas, objetivo, predictores, motor):
        preparaciones.append({"ajuste": tuple(f["row_id"] for f in ajuste), "todas": tuple(f["row_id"] for f in todas), "motor": motor.nombre})
        return real_preparar(ajuste, todas, objetivo, predictores, motor)
    monkeypatch.setattr(c3, "preparar_para_motor", preparar)
    planes: dict = {}
    real_particiones = c5.particiones_base

    def particiones(ds, protocolo):
        out = real_particiones(ds, protocolo)
        planes[ds.nombre] = out
        return out
    monkeypatch.setattr(c5, "particiones_base", particiones)
    salida = tmp_path / "salida" / "resultado.json"

    def correr(*extra, solo=CONJUNTO):
        m.main(["--solo", solo, "--salida", str(salida), *extra])
        return json.loads(salida.read_text(encoding="utf-8"))
    return types.SimpleNamespace(correr=correr, falso=falso, preparaciones=preparaciones, planes=planes, salida=salida, monkeypatch=monkeypatch, ref=an.cargar_referencia())


def test_bucle_mide_los_15_pliegues_de_la_v2_con_el_presupuesto_del_cubo_4_hilos_y_la_semilla_de_cada_repeticion(camino):
    """Lo que RECIBE el motor, no lo que se graba: presupuesto de pared del cubo (120 s), 4 hilos y la semilla de la repetición; los pliegues son los de la v2 y en ese orden."""
    camino.correr()
    ll = camino.falso.llamadas
    m_v2 = camino.ref.meta[CONJUNTO]
    assert [(x["repeticion"], x["pliegue"]) for x in ll] == sorted(m_v2["claves"]) and len(ll) == 15
    for x in ll:
        pre = x["presupuesto"]
        semilla_v2 = m_v2["por_clave"][(x["repeticion"], x["pliegue"])]["semilla"]
        assert (pre.wall_seconds, pre.hilos, pre.seed) == (120.0, 4, semilla_v2), x
        assert x["motor"].nombre == "interpret.ebm" and x["dataset"] == CONJUNTO


def test_bucle_la_prueba_es_la_prueba_fija_y_la_particion_es_la_del_plan_de_la_v2(camino):
    camino.correr()
    por_id, propuesta, *_ = camino.planes[CONJUNTO]
    test_ids = tuple(propuesta.plan.observaciones_del_rol("test"))
    for x in camino.falso.llamadas:
        pliegue = propuesta.pliegues.pliegue_de(repeticion=x["repeticion"], pliegue=x["pliegue"])
        assert x["test"] == tuple(str(i) for i in test_ids)
        assert x["train"] == tuple(str(i) for i in pliegue.entrena) and x["val"] == tuple(str(i) for i in pliegue.valida)
        assert not set(x["train"]) & set(x["test"]) and not set(x["val"]) & set(x["test"])
        assert x["split_plan_digest"] == camino.ref.meta[CONJUNTO]["plan_digest"]


def test_bucle_la_preparacion_se_AJUSTA_solo_con_entrena_y_con_el_motor_del_ebm(camino):
    camino.correr()
    por_id, propuesta, *_ = camino.planes[CONJUNTO]
    test_ids = set(map(str, propuesta.plan.observaciones_del_rol("test")))
    assert len(camino.preparaciones) == 15
    for p in camino.preparaciones:
        assert not set(map(str, p["ajuste"])) & test_ids, "la preparación vio filas de la prueba"
        assert p["motor"] == "interpret.ebm"


def test_el_EBM_declara_las_mismas_capacidades_de_preparacion_que_lightgbm_que_es_la_preparacion_de_la_v2():
    """La regla dice «la preparación del motor LightGBM»: se prepara con el motor del EBM, así que sus capacidades de categóricas y faltantes TIENEN que ser las de LightGBM."""
    from matrixai_engines.registro_de_motores import motor_para
    a, b = motor_para("lightgbm").capabilities(), motor_para("interpret.ebm").capabilities()
    assert (a.admite_categoricas, a.admite_faltantes) == (b.admite_categoricas, b.admite_faltantes) == (True, True)


def test_bucle_la_salida_graba_lo_que_paso_y_pasa_lo_declarado_del_analizador(camino):
    out = camino.correr()
    assert out["es_sonda"] is False and out["parcial"] is False and len(out["resultados"]) == 15
    ref = camino.ref
    assert an.problemas_de_lo_declarado(out, ref) == [] and an.problemas_de_completitud(dict(out, es_sonda=True), ref, True) == []
    r = out["resultados"][0]
    assert r["hilos"] == 4 and r["estado"] == "completed" and r["metrica_de_cierre"] == "auroc" and r["tiempo_de_ajuste"] == 1.5
    assert r["hiperparametros"] == {"interpret_core": "0.7.8", "parametros": {"n_jobs": 4}}
    assert out["regla_sha256_sellada"] == an.REGLA_SHA256_SELLADA == out["regla_sha256_al_medir"]
    assert out["digest_de_la_cache"]["entorno"] and out["digest_de_la_cache"]["motor"] == m.digest_del_motor()
    assert out["conjuntos"][0]["pliegues_esperados"] == [list(k) for k in sorted(ref.meta[CONJUNTO]["claves"])]


def test_bucle_graba_los_commits_que_declara_el_lanzador_porque_el_contenedor_no_trae_git(camino, monkeypatch):
    monkeypatch.setenv("C170_COMMITS", '{"matrixAI":{"commit":"abc","ficheros_modificados":0}}')
    assert camino.correr()["commits_declarados_por_el_lanzador"] == {"matrixAI": {"commit": "abc", "ficheros_modificados": 0}}
    monkeypatch.delenv("C170_COMMITS")
    assert camino.correr("--forzar")["commits_declarados_por_el_lanzador"] is None


def test_bucle_NO_guarda_el_predictor_entero_solo_los_hiperparametros(camino):
    """`config_efectiva` es el predictor con TODAS las tablas del modelo: megas por pliegue. La salida no puede llevarlas."""
    camino.correr()
    texto = camino.salida.read_text(encoding="utf-8")
    assert '"modelo"' not in texto and "terminos" not in texto


def test_bucle_no_reutiliza_un_fallido_ni_uno_de_otro_codigo_y_si_el_suyo(camino):
    out = camino.correr()
    n = len(camino.falso.llamadas)
    # 1) lo suyo (completado, mismo código): se reutiliza, no se llama al motor
    camino.correr()
    assert len(camino.falso.llamadas) == n
    # 2) uno FALLIDO con el código bueno NO se reutiliza (una caída por carga ajena no se cementa)
    out["resultados"][0]["estado"] = "failed"
    camino.salida.write_text(json.dumps(out), encoding="utf-8")
    camino.correr()
    assert len(camino.falso.llamadas) == n + 1
    # 3) uno de OTRO código no se reutiliza
    out = json.loads(camino.salida.read_text(encoding="utf-8"))
    out["resultados"][1]["entorno_digest"] = "otro"
    camino.salida.write_text(json.dumps(out), encoding="utf-8")
    camino.correr()
    assert len(camino.falso.llamadas) == n + 2
    # 4) uno de otro presupuesto, otra semilla u otros hilos tampoco
    for campo, valor in (("presupuesto_wall_s", 5.0), ("semilla", 99), ("hilos", 1), ("metrica_de_cierre", "accuracy")):
        out = json.loads(camino.salida.read_text(encoding="utf-8"))
        out["resultados"][2][campo] = valor
        camino.salida.write_text(json.dumps(out), encoding="utf-8")
        antes = len(camino.falso.llamadas)
        camino.correr()
        assert len(camino.falso.llamadas) == antes + 1, campo


def test_bucle_un_intento_fallido_se_registra_con_su_motivo_y_la_medida_sigue(camino):
    camino.falso.estado = "failed"
    out = camino.correr()
    assert len(out["resultados"]) == 15 and all(r["estado"] == "failed" and r["motivo"] == "tope de pared agotado" for r in out["resultados"])
    assert out["parcial"] is False


def test_bucle_una_medida_real_no_empieza_sin_el_sello_de_la_regla(camino):
    camino.monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    with pytest.raises(SystemExit, match="NO está sellada"):
        camino.correr()
    assert camino.falso.llamadas == []


def test_bucle_una_regla_cambiada_despues_de_sellar_no_se_mide(camino):
    camino.monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    with pytest.raises(SystemExit, match="no es la sellada"):
        camino.correr()
    assert camino.falso.llamadas == []


def test_bucle_sin_el_sello_del_codigo_no_se_mide(camino, tmp_path):
    camino.monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", tmp_path / "no_existe.json")
    with pytest.raises(SystemExit, match="sello del CÓDIGO"):
        camino.correr()
    assert camino.falso.llamadas == []


def test_bucle_un_fichero_sellado_que_cambio_no_se_mide(camino, tmp_path):
    sha = dict(an.codigo_de_ahora())
    sha["benchmarks/c170/analiza_c170.py"] = "f" * 64
    ruta = tmp_path / "otro.json"
    ruta.write_text(json.dumps({"regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "sha256": sha}), encoding="utf-8")
    camino.monkeypatch.setattr(an, "RUTA_DEL_SELLO_DEL_CODIGO", ruta)
    with pytest.raises(SystemExit, match="analiza_c170.py"):
        camino.correr()


def test_bucle_una_sonda_se_salta_los_sellos_mide_un_pliegue_y_se_marca_sonda(camino):
    camino.monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    out = camino.correr("--sonda")
    assert out["es_sonda"] is True and out["tipo_de_ejecucion"] == "sonda" and len(out["resultados"]) == 1 and len(camino.falso.llamadas) == 1
    v = an.analizar(out, referencia=camino.ref, comprobar_sellos=False)
    assert v["veredicto"] == "SONDA"


def test_bucle_se_niega_fuera_del_regimen_y_con_un_sellado(camino):
    for nombre in ("micro-mass", "Internet-Advertisements", "Satellite", "letter", "no-existe"):
        with pytest.raises(SystemExit, match="régimen"):
            camino.correr(solo=nombre)
    assert camino.falso.llamadas == []


def test_bucle_se_niega_si_el_motor_no_esta_disponible(camino, monkeypatch):
    real = m.motor_para

    def sin_biblioteca(nombre):
        motor = real(nombre)
        monkeypatch.setattr(type(motor), "motivo_si_no_puede_entrenar", lambda self: {"es": "falta interpret-core", "en": "x"})
        return motor
    monkeypatch.setattr(m, "motor_para", sin_biblioteca)
    with pytest.raises(SystemExit, match="falta interpret-core"):
        camino.correr()
    assert camino.falso.llamadas == []


def test_bucle_se_niega_si_la_particion_de_hoy_no_es_la_de_la_v2(camino, monkeypatch):
    ref = camino.ref
    ref.meta[CONJUNTO]["plan_digest"] = "otra"
    ref.tabm["particion_comparada_con_la_v2"][CONJUNTO]["plan_digest"] = "otra"       # la referencia es coherente consigo misma; lo que no cuadra es la partición DE HOY
    monkeypatch.setattr(an, "cargar_referencia", lambda *a, **k: ref)
    with pytest.raises(SystemExit, match="no es la de la v2"):
        camino.correr()
    assert camino.falso.llamadas == []


def test_bucle_se_niega_si_el_numero_de_predictores_no_es_el_registrado(camino, monkeypatch):
    ref = camino.ref
    ref.meta[CONJUNTO]["n_predictores"] = 9
    monkeypatch.setattr(an, "cargar_referencia", lambda *a, **k: ref)
    with pytest.raises(SystemExit, match="predictores"):
        camino.correr()
    assert camino.falso.llamadas == []


def test_bucle_se_niega_si_el_sha256_de_los_datos_no_es_el_que_midio_la_v2(camino, monkeypatch):
    ref = camino.ref
    ref.meta[CONJUNTO]["datos_sha256"] = "0" * 64
    monkeypatch.setattr(an, "cargar_referencia", lambda *a, **k: ref)
    with pytest.raises(SystemExit, match="sha256"):
        camino.correr()
    assert camino.falso.llamadas == []


def test_bucle_el_techo_blando_de_memoria_para_y_guarda_lo_medido(camino):
    with pytest.raises(SystemExit, match="techo blando"):
        camino.correr("--memoria-max-mb", "10")
    guardado = json.loads(camino.salida.read_text(encoding="utf-8"))
    assert guardado["parcial"] is True and len(guardado["resultados"]) == 1


def test_el_plan_no_lee_datos_y_da_los_24_conjuntos_y_los_297_ajustes(capsys):
    m.main(["--plan"])
    texto = capsys.readouterr().out
    assert "24 conjuntos, 297 ajustes" in texto and "micro-mass" not in texto and "diabetes" in texto


def test_el_digest_del_entorno_cubre_el_analizador_y_el_medidor_y_la_pasada_116():
    nombres = {Path(f).name for f in m.ficheros_del_entorno()}
    assert {"analiza_c170.py", "medir_c170.py", "pasada_116c2_tabicl.py", "registro_de_motores.py", "pasada_amplia_101_c5.py"} <= nombres

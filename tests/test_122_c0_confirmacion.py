# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""122-C0 — el ARNÉS de la medida de confirmación (`benchmarks/contrato122/
c0_confirmacion.py`): TabM contra la red del núcleo en las condiciones del paquete CPU.

Perfil de prueba: 120 filas, segundos de presupuesto, sin red ni los datos grandes de
la Fase 0 (los ARFF no hacen falta: el conjunto es sintético y el ARFF de `main()` es
un fichero falso solo para que haya un sha256 que atar). Lo que se prueba, y por qué:

* que las TRES redes ven la MISMA partición y que el test no entra al ajuste;
* que la red del núcleo del modo experto va SIN plazo, con la receta del GENERADOR
  (no con la del estudio) y con `recortar_objetivo=False` — lo que manda el Studio;
* que la caché no reaprovecha un intento fallido ni uno con otro código/otra
  configuración/otros datos;
* que el JSON lleva procedencia y sello, y que componer uno NUEVO con el mismo código da
  el mismo sello (no solo que el escrito cuadre);
* las guardias: protocolo registrado, sellados fuera, núcleo del árbol;
* que la regla da el veredicto que dice (el mejor de las redes, un fallo = perdido).
"""
from __future__ import annotations

import copy
import json
import random
import re
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "contrato122"))
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import c0_confirmacion as c0  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

torch = pytest.importorskip("torch")

from matrixai.estudio.validacion import digest_canonico  # noqa: E402
from matrixai_engines.particiones import Presupuesto  # noqa: E402

_FASE0 = _RAIZ / "benchmarks" / "fase0"
_RUTA_PROTOCOLO = _RAIZ / "benchmarks" / "contrato122" / "protocolo_122_c0.json"
_NOMBRES_C0 = ["dresses-sales", "climate-model-simulation-crashes", "pc1", "mfeat-factors",
               "us_crime", "wilt", "pendigits", "house_16H"]


# ---------------------------------------------------------------------------
# Material: un conjunto sintético de 120 filas con las tres piezas que el guion espera
# ---------------------------------------------------------------------------

def _entrada(nombre="dresses-sales", data_id=23381, tarea="binary_classification",
             cubo="pequeno", n=120):
    return {"data_id": data_id, "nombre": nombre, "cubo_de_tamano": cubo, "tarea": tarea,
            "sellado": False, "n_filas": n}


def _ds(tarea="binary_classification"):
    ds = c5.DatasetDeLaPasada(_entrada(tarea=tarea))
    if tarea != "regression":
        ds.clases, ds.positiva = ("no", "si"), "si"
    return ds


def _conjunto(n=120, tarea="binary_classification", semilla=7):
    """`por_id`/`objetivo`/`predictores`/`spec`/`test_ids` sintéticos: el objetivo
    depende de `a` y `b` (con ruido), para que las redes aprendan algo."""
    azar = random.Random(semilla)
    por_id = {}
    for i in range(n):
        a, b = azar.random(), azar.random()
        c = azar.choice(["x", "y", "z"])
        if tarea == "regression":
            y = 3.0 * a - b + azar.gauss(0, 0.05)
        else:
            y = "si" if a + b + azar.gauss(0, 0.2) > 1.0 else "no"
        por_id[i] = {"row_id": i, "a": a, "b": b, "c": c, "y": y}
    ids = list(por_id)
    test_ids = tuple(ids[:24])
    entrena = tuple(ids[24:92])
    valida = tuple(ids[92:])
    ds = _ds(tarea)
    spec = c5.problem_spec_de(ds, "y", ("a", "b", "c"))
    return {"por_id": por_id, "objetivo": "y", "predictores": ("a", "b", "c"), "spec": spec,
            "test_ids": test_ids, "entrena": entrena, "valida": valida}


# ---------------------------------------------------------------------------
# 1. LA MISMA PARTICIÓN, y el test fuera del ajuste
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("red", c0.REDES)
def test_cada_red_recibe_exactamente_las_filas_de_la_particion(red):
    conj = _conjunto()
    train, val, test = c0.particiones_para(red, conj, conj["entrena"], conj["valida"],
                                           conj["test_ids"])
    assert set(train.row_ids) == {str(i) for i in conj["entrena"]}
    assert set(val.row_ids) == {str(i) for i in conj["valida"]}
    assert set(test.row_ids) == {str(i) for i in conj["test_ids"]}
    assert not set(train.row_ids) & set(test.row_ids)
    assert not set(val.row_ids) & set(test.row_ids)


def test_las_tres_redes_ven_la_misma_particion_y_el_digest_lo_dice():
    conj = _conjunto()
    vistos = {}
    for red in c0.REDES:
        train, val, test = c0.particiones_para(red, conj, conj["entrena"], conj["valida"],
                                               conj["test_ids"])
        vistos[red] = (c0.digest_de_ids(train.row_ids), c0.digest_de_ids(val.row_ids),
                       c0.digest_de_ids(test.row_ids))
    assert len(set(vistos.values())) == 1, vistos
    declarado = c0.ids_de_la_particion(conj["entrena"], conj["valida"], conj["test_ids"])
    assert (declarado["train"], declarado["validation"], declarado["test"]) == next(iter(vistos.values()))
    # y el digest SÍ distingue: mover una fila de validación a train lo cambia
    otra = c0.ids_de_la_particion(conj["entrena"] + conj["valida"][:1], conj["valida"][1:],
                                  conj["test_ids"])
    assert otra["train"] != declarado["train"]


class _Cola:
    def __init__(self):
        self.mensajes = []

    def put(self, m):
        self.mensajes.append(m)


def test_el_test_no_entra_a_fit_y_el_arnes_para_si_se_cuela(monkeypatch):
    """`_hijo` corre `ejecutar_intento` de verdad con un motor ESPÍA que apunta qué
    recibió `fit`: ni una fila del test. Y con el test contaminado (una fila de
    train dentro) el hijo declara `fuga` y NO ajusta."""
    from matrixai_engines.motores.baseline import MotorBaseline

    conj = _conjunto()
    train, val, test = c0.particiones_para(c0.NUCLEO_ESTUDIO, conj, conj["entrena"],
                                           conj["valida"], conj["test_ids"])
    llamadas = []

    class Espia(MotorBaseline):
        def fit(self, tr, va, spec, presupuesto, **kw):
            llamadas.append((set(tr.row_ids), set(va.row_ids) if va is not None else set()))
            return super().fit(tr, va, spec, presupuesto, **kw)

    monkeypatch.setattr(c0, "motor_de_la_red", lambda red: Espia())
    cola = _Cola()
    c0._hijo(cola, c0.NUCLEO_ESTUDIO, train, val, test, conj["spec"], None, 2, 0, "t", "plan",
             "ds", 0, 0)
    assert cola.mensajes[0][0] == "ok", cola.mensajes
    (vistas_train, vistas_val), = llamadas
    assert not (vistas_train | vistas_val) & set(test.row_ids)

    # contaminación: el test lleva una fila de train → el arnés (no el motor) para
    llamadas.clear()
    from matrixai_engines.particiones import Particion
    contaminado = Particion(row_ids=test.row_ids[:-1] + (train.row_ids[0],),
                            features=test.features, target=test.target)
    cola2 = _Cola()
    c0._hijo(cola2, c0.NUCLEO_ESTUDIO, train, val, contaminado, conj["spec"], None, 2, 0, "t",
             "plan", "ds", 0, 0)
    assert cola2.mensajes[0][0] == "fuga", cola2.mensajes
    assert llamadas == []


# ---------------------------------------------------------------------------
# 2. LA RED DEL NÚCLEO VA POR EL CAMINO DEL MODO EXPERTO, SIN PLAZO
# ---------------------------------------------------------------------------

def _espiar_el_entrenamiento(monkeypatch):
    """Sustituye `run_playground_training` del núcleo por un espía que apunta los
    argumentos y llama al verdadero: lo que el motor le pasa es lo que se afirma."""
    import matrixai.playground_api as api
    visto = {}
    real = api.run_playground_training

    def espia(mxai, training_text, csv_text, *args, **kwargs):
        visto.update(training_text=training_text, kwargs=dict(kwargs))
        return real(mxai, training_text, csv_text, *args, **kwargs)

    monkeypatch.setattr(api, "run_playground_training", espia)
    return visto


def _ajustar(motor, conj, wall):
    train, val, test = c0.particiones_para(c0.NUCLEO_ESTUDIO, conj, conj["entrena"],
                                           conj["valida"], conj["test_ids"])
    return motor.fit(train, val, conj["spec"], Presupuesto(wall_seconds=wall, hilos=2, seed=0),
                     candidate="t", split_plan_digest="plan")


def test_la_red_del_modo_experto_va_sin_plazo_con_la_receta_del_generador(monkeypatch):
    visto = _espiar_el_entrenamiento(monkeypatch)
    conj = _conjunto()
    resultado, fitted = _ajustar(c0.MotorDensaModoExperto(), conj, wall=None)
    assert resultado.state == "completed", resultado
    predictor = fitted.spec.predictor
    hp, efectivo = predictor["hiperparametros"], predictor["entrenamiento_efectivo"]
    # SIN PLAZO: nada se lo pidió al núcleo ni lo declara el resultado
    assert visto["kwargs"]["plazo"] is None
    assert efectivo["plazo_de_entrenamiento_segundos"] is None
    assert efectivo["parado_por_plazo"] is False
    # lo que manda `_studio_train_start` (A8) y el motor del estudio no manda
    assert visto["kwargs"]["recortar_objetivo"] is False
    # LA RECETA DEL GENERADOR, leída de lo que se ENTRENÓ (no de mis constantes)
    assert hp["optimizador"] == "sgd" and hp["tasa_de_aprendizaje"] == 0.01
    assert hp["parada_temprana"] == {"paciencia": None, "metrica": None}
    assert hp["lote"] == 8
    assert efectivo["epocas_ejecutadas"] == 50  # sin parada temprana: las 50, todas


def test_el_texto_que_entrena_es_el_del_generador_salvo_la_linea_split(monkeypatch):
    """Lo ata al GENERADOR del núcleo, no a una receta escrita aquí: el texto de
    entrenamiento del modo experto es el de `generate_project_from_dataset`; el
    único cambio es el SPLIT (la partición la decide el arnés)."""
    from matrixai.playground_api import generate_project_from_dataset
    from matrixai_engines.motores import densa as densa_mod
    visto = _espiar_el_entrenamiento(monkeypatch)
    conj = _conjunto()
    _ajustar(c0.MotorDensaModoExperto(), conj, wall=None)

    # el mismo CSV que escribe el motor → el mismo proyecto que el modo experto genera
    train, val, _ = c0.particiones_para(c0.NUCLEO_ESTUDIO, conj, conj["entrena"], conj["valida"],
                                        conj["test_ids"])
    columnas = densa_mod._columnas(("a", "b", "c"))
    csv_ = densa_mod._escribir_csv(columnas, "y", list(train.features) + list(val.features),
                                   list(train.target) + list(val.target))
    del columnas  # silencia linters
    generado = generate_project_from_dataset(
        csv_, "y", locale="es", column_type_overrides={"y": "categorical"})["training_text"]

    def sin_split(texto):
        return re.sub(r"^\s*SPLIT .*$", "", texto, flags=re.MULTILINE)

    assert sin_split(visto["training_text"]) == sin_split(generado)
    # y el del estudio SÍ es distinto: la prueba distingue las dos recetas
    visto.clear()
    _ajustar(c0.MotorDensaPropia(), conj, wall=None)
    assert sin_split(visto["training_text"]) != sin_split(generado)


def test_el_motor_del_modo_experto_se_niega_a_llevar_plazo():
    conj = _conjunto()
    with pytest.raises(ValueError, match="SIN plazo"):
        _ajustar(c0.MotorDensaModoExperto(), conj, wall=60.0)


def test_la_red_del_estudio_sin_plazo_tampoco_lo_pide(monkeypatch):
    """`nucleo_estudio` es el motor de la Fase 0 pero SIN plazo: con `wall=None` el
    núcleo no recibe ninguno, y con un presupuesto SÍ lo recibe (el contraste)."""
    visto = _espiar_el_entrenamiento(monkeypatch)
    conj = _conjunto()
    _ajustar(c0.MotorDensaPropia(), conj, wall=None)
    assert visto["kwargs"]["plazo"] is None
    _ajustar(c0.MotorDensaPropia(), conj, wall=100.0)
    assert visto["kwargs"]["plazo"] is not None


def test_los_presupuestos_por_red_las_del_nucleo_sin_plazo_y_tabm_con_el_de_su_cubo():
    proto = c0.cargar_protocolo(_RUTA_PROTOCOLO)
    for red in (c0.NUCLEO_MODO_EXPERTO, c0.NUCLEO_ESTUDIO):
        wall, tope = c0.presupuesto_de_la_red(red, cubo="mediano", proto=proto, semilla=0)
        assert wall is None and tope == proto["presupuesto"]["tope_duro_nucleo_s_por_cubo"]["mediano"]
    wall, tope = c0.presupuesto_de_la_red(c0.TABM, cubo="mediano", proto=proto, semilla=0)
    assert wall == 300.0 and tope == 300.0 + c0.MARGEN_DEL_TOPE_SOBRE_EL_PRESUPUESTO_S
    assert c0.HILOS == 2


# ---------------------------------------------------------------------------
# 3. EL PROCESO APARTE Y SU PICO DE MEMORIA (con un spawn de verdad)
# ---------------------------------------------------------------------------

def test_un_ajuste_aislado_declara_hilos_y_su_pico_de_memoria():
    conj = _conjunto()
    train, val, test = c0.particiones_para(c0.TABM, conj, conj["entrena"], conj["valida"],
                                           conj["test_ids"])
    salida = c0.ejecutar_aislado(
        c0.TABM, train, val, test, conj["spec"], wall_seconds=20.0, tope_duro_s=120.0, hilos=2,
        seed=0, candidate="t", split_plan_digest="plan", dataset="sintetico", pliegue=0,
        repeticion=0)
    assert salida["clase"] == "ok", salida
    assert salida["hilos_de_torch_al_entrar"] == 2 and salida["hilos_pedidos"] == 2
    # el pico es el de ESE proceso (un hijo con torch importado ya pesa ≫ 100 MB) y no menor
    # que lo que pesaba antes de ajustar
    assert salida["rss_pico_mb"] > 100
    assert salida["rss_pico_mb"] >= salida["rss_antes_del_ajuste_mb"] > 50
    efectivo = salida["config"]["entrenamiento_efectivo"]
    assert "parado_por_plazo" in efectivo and efectivo["epocas_ejecutadas"] >= 1
    # los pesos NO viajan en el mensaje
    assert "pesos" not in json.dumps(salida["config"])


def test_el_tope_duro_mata_al_proceso_y_queda_failed():
    conj = _conjunto()
    train, val, test = c0.particiones_para(c0.NUCLEO_ESTUDIO, conj, conj["entrena"],
                                           conj["valida"], conj["test_ids"])
    salida = c0.ejecutar_aislado(
        c0.NUCLEO_ESTUDIO, train, val, test, conj["spec"], wall_seconds=None, tope_duro_s=0.3,
        hilos=2, seed=0, candidate="t", split_plan_digest="plan", dataset="s", pliegue=0,
        repeticion=0)
    assert salida["clase"] == "tope"
    registro = c0.registro_de_un_intento(
        ds=_ds(), red=c0.NUCLEO_ESTUDIO, repeticion=0, pliegue=0, semilla=0, wall_s_pedido=None,
        tope_duro_s=0.3, particion={"n_train": 1}, metric_id="auroc", datos_sha256="x",
        entorno_digest="e", config_digest="c", preparacion_s=0.0, salida=salida)
    assert registro["estado"] == "failed" and "tope duro" in registro["motivo"]


# ---------------------------------------------------------------------------
# 4. LA CACHÉ
# ---------------------------------------------------------------------------

def _previo(**cambios):
    base = {"estado": "completed", "datos_sha256": "D", "entorno_digest": "E",
            "config_digest": "C"}
    base.update(cambios)
    return base


def _ok(previo, **cambios):
    kw = dict(entorno_digest="E", config_digest="C", datos_sha256="D")
    kw.update(cambios)
    return c0.reusable(previo, **kw)


def test_la_cache_reusa_solo_lo_medido_con_el_mismo_codigo_config_y_datos():
    assert _ok(_previo()) is True
    assert _ok(_previo(estado="completed_budget_limited")) is True
    # un intento FALLIDO con los digests buenos NO se reaprovecha (trampa nº 1)
    assert _ok(_previo(estado="failed")) is False
    assert _ok(_previo(estado="cancelled")) is False
    assert _ok(None) is False
    # otro código, otra configuración, otros datos
    assert _ok(_previo(), entorno_digest="OTRO") is False
    assert _ok(_previo(), config_digest="OTRA") is False
    assert _ok(_previo(), datos_sha256="OTROS") is False
    assert _ok(_previo(datos_sha256=None), datos_sha256=None) is False


def test_el_digest_de_la_configuracion_reacciona_a_cada_cosa_que_decide_el_intento():
    base = dict(red=c0.TABM, wall_s=120.0, tope_duro_s=150.0, hilos=2, semilla=0, repeticion=0,
                pliegue=0, particion={"train": "a", "validation": "b", "test": "c"})
    d0 = c0.digest_de_la_configuracion(**base)
    assert c0.digest_de_la_configuracion(**base) == d0
    for campo, otro in (("red", c0.NUCLEO_ESTUDIO), ("wall_s", 60.0), ("tope_duro_s", 99.0),
                        ("hilos", 4), ("semilla", 1), ("repeticion", 1), ("pliegue", 1),
                        ("particion", {"train": "a", "validation": "b", "test": "OTRO"})):
        assert c0.digest_de_la_configuracion(**{**base, campo: otro}) != d0, campo


def test_el_digest_del_entorno_cubre_el_codigo_de_la_red_los_datos_y_los_hilos():
    componentes = c0.componentes_del_digest_del_entorno()
    claves = set(componentes)
    assert any(k.startswith("c0:c0_confirmacion") for k in claves)
    assert any(k.endswith("motores/densa.py") for k in claves)
    assert any(k.endswith("motores/densa_tabm.py") for k in claves)
    assert any(k.endswith("playground.py") for k in claves)  # el núcleo que entrena
    assert "dato:protocolo_122_c0" in claves and "version:torch" in claves
    assert componentes["version:hilos"] == "2"
    d = c0.digest_del_entorno(componentes)
    assert c0.digest_del_entorno({**componentes, "version:hilos": "4"}) != d
    clave_de_codigo = next(k for k in claves if k.endswith("motores/densa.py"))
    assert c0.digest_del_entorno({**componentes, clave_de_codigo: "otro"}) != d
    # lo de terceros entra por versión, no por código: nada de numpy/torch en el cierre
    assert not any(k.startswith("fuera:") for k in claves)


def test_el_cierre_de_imports_no_entra_en_el_codigo_de_terceros(tmp_path):
    """En el paquete CPU los motores viven en `site-packages`, donde `numpy` o `torch`
    TAMBIÉN existen: sin la restricción el cierre se iba a miles de ficheros de
    terceros (83 s medidos en el contenedor, 3 s en el host). Aquí, un «site-packages»
    de mentira con un `numpy` y un `matrixai_engines`: solo el segundo entra."""
    (tmp_path / "numpy").mkdir()
    (tmp_path / "numpy" / "__init__.py").write_text("")
    (tmp_path / "matrixai_engines").mkdir()
    (tmp_path / "matrixai_engines" / "__init__.py").write_text("")
    raices = (tmp_path,)
    assert c0._ficheros_de_lo_propio("numpy.core", raices) == []
    assert c0._ficheros_de_lo_propio("torch", raices) == []
    assert c0._ficheros_de_lo_propio("matrixai_engines", raices) == [
        tmp_path / "matrixai_engines" / "__init__.py"]


# ---------------------------------------------------------------------------
# 5. EL PROTOCOLO Y LAS GUARDIAS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v2():
    c0.c6.preparar_protocolo_v2()
    return c3.protocolo_registrado()


def test_el_borrador_son_los_8_conjuntos_no_sellados_del_v2(v2):
    proto = c0.cargar_protocolo(_RUTA_PROTOCOLO)
    datasets = c0.exigir_el_protocolo(proto, v2, tipo="humo")
    assert [d.nombre for d in datasets] == _NOMBRES_C0
    assert not any(d.sellado for d in datasets)
    # cada pliegue pedido existe en la repetición 0 del protocolo (5 pliegues)
    assert all(set(e["pliegues"]) <= set(range(v2.particion.folds)) for e in proto["datasets"])
    regla = proto["regla_de_confirmacion"]
    assert regla["puntos"] == 2.0 and round(regla["fraccion_minima"] * 8) == 7


def test_una_pasada_real_exige_el_protocolo_registrado_y_su_digest(v2):
    proto = c0.cargar_protocolo(_RUTA_PROTOCOLO)
    # El fichero de disco ya está REGISTRADO (04-10) y su sello vale: la pasada real lo acepta.
    assert proto["estado"] == c0.ESTADO_REGISTRADO
    assert c0.exigir_el_protocolo(proto, v2, tipo="pasada")
    # Y en BORRADOR, no (el caso se fabrica en memoria: el fichero ya no lo está).
    borrador = dict(proto, estado="BORRADOR")
    with pytest.raises(SystemExit, match="exige"):
        c0.exigir_el_protocolo(borrador, v2, tipo="pasada")
    registrado = dict(proto, estado=c0.ESTADO_REGISTRADO, digest_sha256="0" * 64)
    with pytest.raises(SystemExit, match="digest"):
        c0.exigir_el_protocolo(registrado, v2, tipo="pasada")  # digest viejo
    registrado["digest_sha256"] = c0.autodigest(registrado)
    assert c0.exigir_el_protocolo(registrado, v2, tipo="pasada")
    # editar DESPUÉS de registrarlo se nota
    registrado["regla_de_confirmacion"] = dict(registrado["regla_de_confirmacion"], puntos=3.0)
    with pytest.raises(SystemExit, match="digest"):
        c0.exigir_el_protocolo(registrado, v2, tipo="pasada")


def test_un_conjunto_sellado_se_niega(v2):
    proto = c0.cargar_protocolo(_RUTA_PROTOCOLO)
    sellado = next(d for d in c5.datasets_de_la_pasada(v2) if d.sellado)
    proto["datasets"] = [{"nombre": sellado.nombre, "data_id": sellado.data_id,
                          "tarea": sellado.tarea, "cubo": sellado.cubo, "pliegues": [0]}]
    with pytest.raises(SystemExit, match="SELLADO"):
        c0.exigir_el_protocolo(proto, v2, tipo="humo")
    proto["datasets"] = [{"nombre": "no-existe", "data_id": 1, "tarea": "regression",
                          "cubo": "pequeno", "pliegues": [0]}]
    with pytest.raises(SystemExit, match="no está"):
        c0.exigir_el_protocolo(proto, v2, tipo="humo")


def test_el_nucleo_tiene_que_ser_el_del_arbol(monkeypatch):
    assert c0.exigir_el_nucleo_del_arbol().startswith(str(c0.RAIZ))
    monkeypatch.setattr(c0.matrixai, "__file__", "/usr/lib/python3/site-packages/matrixai/__init__.py")
    with pytest.raises(SystemExit, match="OTRO núcleo"):
        c0.exigir_el_nucleo_del_arbol()


# ---------------------------------------------------------------------------
# 6. EL VEREDICTO DE LA REGLA
# ---------------------------------------------------------------------------

def _registros(valores_por_dataset: dict, datasets, pliegues=(0,), estado_tabm="completed"):
    """`valores_por_dataset[nombre] = {red: valor}` → registros planos con la métrica de cierre."""
    metrica = c5.metrica_de_cierre_por_dataset(c3.protocolo_registrado(), datasets)
    filas = []
    for ds in datasets:
        for red, valor in valores_por_dataset[ds.nombre].items():
            for p in pliegues:
                filas.append({"dataset": ds.nombre, "motor": red, "repeticion": 0, "pliegue": p,
                              "estado": estado_tabm if red == c0.TABM else "completed",
                              metrica[ds.nombre]: valor, "cubo": ds.cubo,
                              "particion": {"n_train": 10}, "wall_s": 1.0, "rss_pico_mb": 100.0,
                              "parado_por_plazo": False, "epocas_ejecutadas": 5})
    return filas


@pytest.fixture(scope="module")
def datasets_c0(v2):
    proto = c0.cargar_protocolo(_RUTA_PROTOCOLO)
    return c0.exigir_el_protocolo(proto, v2, tipo="humo")


def _veredicto(filas, v2, datasets):
    proto = c0.cargar_protocolo(_RUTA_PROTOCOLO)
    pliegues = {d.nombre: [0] for d in datasets}
    return c0.veredicto_de_la_regla(filas, proto, v2, datasets, pliegues)


def _valores(distancias_de_tabm_a_la_mejor):
    """TabM a esa distancia (en puntos) de la mejor de las dos redes del núcleo."""
    out = {}
    for nombre, dist in zip(_NOMBRES_C0, distancias_de_tabm_a_la_mejor):
        out[nombre] = {c0.NUCLEO_MODO_EXPERTO: 0.80, c0.NUCLEO_ESTUDIO: 0.70,
                       c0.TABM: 0.80 - dist / 100.0}
    return out


def test_la_regla_cumple_con_7_de_8_y_no_con_6(v2, datasets_c0):
    cumple = _veredicto(_registros(_valores([0, 1, 1.99, 0, 1.9, 0, 0, 5.0]), datasets_c0), v2,
                        datasets_c0)
    assert cumple["cumplidos"] == 7 and cumple["de"] == 8
    assert cumple["tabm_completa_todos"] is True
    assert cumple["veredicto"] == "SE_OFRECE_RECOMENDADA_PARA_TABLAS"
    no = _veredicto(_registros(_valores([0, 1, 2.5, 0, 3.0, 0, 0, 5.0]), datasets_c0), v2,
                    datasets_c0)
    assert no["cumplidos"] == 5 and no["veredicto"] == "SE_OFRECE_SIN_RECOMENDAR"


def test_la_distancia_es_a_la_mejor_de_las_redes_no_a_la_del_estudio(v2, datasets_c0):
    """TabM pegada a la red del ESTUDIO (la débil) pero a 10 puntos de la del modo
    experto: no cumple. Si «la mejor» fuera la débil, este caso saldría 8/8."""
    valores = {n: {c0.NUCLEO_MODO_EXPERTO: 0.80, c0.NUCLEO_ESTUDIO: 0.70, c0.TABM: 0.70}
               for n in _NOMBRES_C0}
    v = _veredicto(_registros(valores, datasets_c0), v2, datasets_c0)
    assert v["cumplidos"] == 0 and v["veredicto"] == "SE_OFRECE_SIN_RECOMENDAR"
    assert all(d["mejor"] == c0.NUCLEO_MODO_EXPERTO for d in v["detalle"])


def test_un_fallo_de_tabm_en_un_conjunto_es_un_conjunto_perdido_y_no_completa(v2, datasets_c0):
    filas = _registros(_valores([0] * 8), datasets_c0)
    # TabM falla en un conjunto: aunque sus cifras fueran perfectas, no completa los 8
    for f in filas:
        if f["dataset"] == "pc1" and f["motor"] == c0.TABM:
            f["estado"] = "failed"
    v = _veredicto(filas, v2, datasets_c0)
    assert v["tabm_completa_todos"] is False
    assert v["conjuntos_incompletos_de_tabm"] == [{"dataset": "pc1", "faltan": [0]}]
    assert v["cumplidos"] == 7  # la regla de distancia sola daría 7/8...
    assert v["veredicto"] == "SE_OFRECE_SIN_RECOMENDAR"  # ...pero «completa los 8» no se cumple


def test_el_plazo_cortado_de_tabm_se_declara(v2, datasets_c0):
    filas = _registros(_valores([0] * 8), datasets_c0)
    filas[2]["parado_por_plazo"] = True  # la fila 2 es de TabM? se fuerza abajo
    for f in filas:
        f["parado_por_plazo"] = False
    tabm = [f for f in filas if f["motor"] == c0.TABM and f["dataset"] == "wilt"][0]
    tabm["parado_por_plazo"] = True
    v = _veredicto(filas, v2, datasets_c0)
    assert v["tabm_con_el_plazo_cortado"] == [{"dataset": "wilt", "pliegue": 0}]


# ---------------------------------------------------------------------------
# 7. EL JSON: procedencia, sello y composición de uno NUEVO
# ---------------------------------------------------------------------------

def _procedencia_fija():
    p = {"medido": "2026-10-04T00:00:00Z", "repositorios": {
        n: {"commit": "a" * 40, "arbol_sucio": False, "fuente": "git"}
        for n in ("matrixAI", "matrixai-engines", "matrixaistudio")},
        "versiones_de_bibliotecas": {"torch": "x"}, "maquina": {},
        "codigo_que_se_importa": {"matrixai": "n", "matrixai_engines": "m", "torch": "t"}, "carga_al_empezar": [0, 0, 0],
        "anclable": True, "avisos": []}
    p["procedencia_id"] = digest_canonico(p)
    return p


def _salida_falsa(valores):
    """Un `ejecutar_aislado` que devuelve, sin entrenar, lo que dicta la tabla
    `valores[(dataset, red)]`; cuenta las llamadas."""
    llamadas = []

    def falso(red, train, val, test, spec, *, wall_seconds, tope_duro_s, hilos, seed, candidate,
              split_plan_digest, dataset, pliegue, repeticion):
        llamadas.append((dataset, red, pliegue))
        v = valores[(dataset, red)]
        if v is None:
            return {"clase": "excepcion", "wall_s": 0.1, "error": "RuntimeError: a propósito",
                    "traza": "traza"}
        return {"clase": "ok", "wall_s": 1.5, "wall_del_intento_s": 1.0, "estado": "completed",
                "motivo_del_estado": None, "recursos": {"wall_seconds": 0.9, "cpu_seconds": 1.8},
                "informe": {"metrics": [{"metric_id": "auroc", "value": v},
                                        {"metric_id": "accuracy", "value": v}]},
                "pipeline_digest": "p" * 8, "engine_version": "x",
                "config": {"entrenamiento_efectivo": {"epocas_ejecutadas": 7, "mejor_epoca": 3,
                                                      "parado_por_plazo": False},
                           "hiperparametros": {"optimizador": "x"}},
                "rss_antes_del_ajuste_mb": 200.0, "rss_pico_mb": 250.0 + len(red),
                "hilos_de_torch_al_entrar": 2, "hilos_pedidos": hilos, "pid": 1}
    return falso, llamadas


class _PropuestaFalsa:
    class plan:  # noqa: N801
        @staticmethod
        def digest():
            return "plan-falso"

        @staticmethod
        def observaciones_del_rol(rol):
            return tuple(range(24))

    class pliegues:  # noqa: N801
        class _P:
            entrena = tuple(range(24, 92))
            valida = tuple(range(92, 120))

        @classmethod
        def pliegue_de(cls, repeticion, pliegue):
            return cls._P


@pytest.fixture()
def entorno_de_main(tmp_path, monkeypatch):
    """`main()` con conjunto sintético, procedencia fija y un ARFF falso (para tener su sha256)."""
    arff = tmp_path / "arff"
    arff.mkdir()
    (arff / "23381.arff").write_text("@relation falsa\n")
    proto = c0.cargar_protocolo(_RUTA_PROTOCOLO)
    proto["datasets"] = [e for e in proto["datasets"] if e["nombre"] == "dresses-sales"]
    proto["datasets"][0]["pliegues"] = [0]
    ruta_proto = tmp_path / "protocolo.json"
    ruta_proto.write_text(json.dumps(proto))
    conj = _conjunto()
    conj = dict(conj, propuesta=_PropuestaFalsa, particion_declarada={})
    monkeypatch.setattr(c0, "cargar_conjunto", lambda ds, v2_: conj)
    monkeypatch.setattr(c0, "componer_procedencia",
                        lambda **kw: dict(_procedencia_fija(), datos_de_entrada={
                            "dresses-sales": {"sha256": "SHA-ARFF"}}))
    valores = {("dresses-sales", r): v for r, v in
               ((c0.NUCLEO_ESTUDIO, 0.60), (c0.NUCLEO_MODO_EXPERTO, 0.70), (c0.TABM, 0.80))}
    falso, llamadas = _salida_falsa(valores)
    monkeypatch.setattr(c0, "ejecutar_aislado", falso)

    def correr(salida, *extra):
        c0.main(["--solo", "dresses-sales", "--protocolo", str(ruta_proto), "--arff", str(arff),
                 "--salida", str(salida), *extra])
        return json.loads(Path(salida).read_text())

    return correr, llamadas, valores, tmp_path


def test_main_escribe_procedencia_sello_y_el_veredicto(entorno_de_main):
    correr, llamadas, _, tmp = entorno_de_main
    doc = correr(tmp / "a.json")
    assert [l[1] for l in llamadas] == list(c0.REDES) and len(llamadas) == 3
    assert set(doc["procedencia"]["repositorios"]) == {"matrixAI", "matrixai-engines",
                                                       "matrixaistudio"}
    assert doc["tipo"] == "solo" and doc["parcial"] is False
    assert doc["veredicto"]["tabm_completa_todos"] is True
    assert c0.verificar_sello(doc)
    # los registros llevan lo que pide el encargo
    r = next(x for x in doc["resultados"] if x["motor"] == c0.TABM)
    for campo in ("auroc", "wall_s", "epocas_ejecutadas", "parado_por_plazo", "rss_pico_mb",
                  "hilos", "particion", "config_digest", "entorno_digest", "datos_sha256"):
        assert campo in r, campo
    assert r["hilos"] == 2
    # alterar una cifra rompe el sello
    doc["resultados"][0]["auroc"] = 0.99
    assert not c0.verificar_sello(doc)


def test_componer_uno_nuevo_con_el_mismo_codigo_da_el_mismo_sello(entorno_de_main):
    """No es leer el escrito: dos pasadas COMPLETAS desde cero (--forzar) con el
    mismo código y los mismos datos componen dos documentos con el MISMO sello; y
    cambiar una cifra o el digest del código lo cambia."""
    correr, llamadas, valores, tmp = entorno_de_main
    a = correr(tmp / "a.json", "--forzar")
    b = correr(tmp / "b.json", "--forzar")
    assert a["sello"] == b["sello"] and c0.verificar_sello(a) and c0.verificar_sello(b)
    assert len(llamadas) == 6  # las dos veces se ENTRENÓ de verdad (no se leyó el escrito)
    valores[("dresses-sales", c0.TABM)] = 0.81
    otra_cifra = correr(tmp / "c.json", "--forzar")
    assert otra_cifra["sello"] != a["sello"]


def test_el_sello_cambia_si_cambia_el_codigo(entorno_de_main, monkeypatch):
    correr, _, _, tmp = entorno_de_main
    a = correr(tmp / "a.json", "--forzar")
    monkeypatch.setattr(c0, "digest_del_entorno", lambda componentes: "OTRO-CODIGO")
    b = correr(tmp / "b.json", "--forzar")
    assert b["sello"] != a["sello"]
    assert b["resultados"][0]["entorno_digest"] == "OTRO-CODIGO"


def test_la_segunda_pasada_reusa_y_un_fallo_se_reintenta(entorno_de_main):
    correr, llamadas, valores, tmp = entorno_de_main
    salida = tmp / "r.json"
    valores[("dresses-sales", c0.TABM)] = None  # TabM falla
    primera = correr(salida)
    assert len(llamadas) == 3
    assert [r["estado"] for r in primera["resultados"] if r["motor"] == c0.TABM] == ["failed"]
    # el fallo NO se reaprovecha: TabM se vuelve a ejecutar y las otras dos se reusan
    llamadas.clear()
    valores[("dresses-sales", c0.TABM)] = 0.80
    segunda = correr(salida)
    assert llamadas == [("dresses-sales", c0.TABM, 0)]
    assert segunda["n_reusados"] == 2
    assert [r["estado"] for r in segunda["resultados"] if r["motor"] == c0.TABM] == ["completed"]
    # y una tercera lo reusa TODO
    llamadas.clear()
    tercera = correr(salida)
    assert llamadas == [] and tercera["n_reusados"] == 3


def test_con_otro_codigo_la_cache_no_reusa_nada(entorno_de_main, monkeypatch):
    correr, llamadas, _, tmp = entorno_de_main
    salida = tmp / "r.json"
    correr(salida)
    llamadas.clear()
    monkeypatch.setattr(c0, "digest_del_entorno", lambda componentes: "OTRO-CODIGO")
    correr(salida)
    assert len(llamadas) == 3


def test_otro_tipo_de_salida_no_se_mezcla(entorno_de_main):
    correr, _, _, tmp = entorno_de_main
    salida = tmp / "r.json"
    correr(salida)  # tipo «solo»
    with pytest.raises(SystemExit, match="no se mezclan"):
        c0.main(["--humo", "--salida", str(salida), "--arff", str(tmp / "arff")])


def test_un_sello_roto_ignora_la_cache(entorno_de_main):
    correr, llamadas, _, tmp = entorno_de_main
    salida = tmp / "r.json"
    correr(salida)
    doc = json.loads(salida.read_text())
    doc["resultados"][0]["auroc"] = 0.123  # sin recalcular el sello
    salida.write_text(json.dumps(doc))
    llamadas.clear()
    correr(salida)
    assert len(llamadas) == 3

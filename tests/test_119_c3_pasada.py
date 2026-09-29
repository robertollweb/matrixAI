# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C3 — el ARNÉS de la pasada del motor nuevo (`pasada_119_c3.py`), NO el
motor en sí (que tiene sus propias pruebas en `matrixai-engines`).

Nada aquí entrena nada: `main()` y `--estimar` corren con un MOTOR FALSO
(`ejecutar_intento_aislado` sustituido por uno que devuelve en milisegundos
lo que la prueba le dicta), sobre ARFF reales pequeños y la v2 real. Lo que
se prueba, por hallazgo de la auditoría 2 (29-09):

* I2 — cómo cuentan los fallos en las dos condiciones (casos fabricados);
* I4 — qué cubre el digest de la caché, y que reacciona;
* I5 — que `main()` hace lo que dice: reintenta un `failed`, una semilla
  por repetición, el presupuesto de cada cubo, todas las repeticiones, la
  referencia SIN el motor nuevo, el motor nuevo DENTRO del campo y la
  diferencia en el sentido nuevo − densa v2;
* I6 — las guardias (configuración del motor, cadena de protocolos,
  partición de la v2, arquitectura declarada);
* I7 — la fusión de la salida y dónde escribe cada tipo de ejecución;
* I1 — la cuenta de `--estimar`.

Y además, lo de siempre: los sellados se niegan y el caché no reusa un
`failed`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import pasada_119_c3 as p119  # noqa: E402
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

_FASE0 = _RAIZ / "benchmarks" / "fase0"
RUTA_PROTOCOLO_V2 = _FASE0 / "protocolo_exploratorio_v2.json"
PROTOCOLO_V2 = protocolo_mod.ProtocoloExploratorio.cargar(RUTA_PROTOCOLO_V2)
NUEVO, DENSA = p119.NOMBRE_MOTOR_NUEVO, p119.NOMBRE_DENSA_V2


def _registro(dataset: str, motor: str, valor: float | None, *, repeticion: int = 0,
              pliegue: int = 0, estado: str = "completed", metrica: str = "auroc") -> dict:
    return {"dataset": dataset, "motor": motor, "repeticion": repeticion,
            "pliegue": pliegue, "estado": estado, metrica: valor}


@pytest.fixture(scope="module")
def _v2():
    return json.loads((_FASE0 / "pasada_v2_113_resultado.json").read_text(encoding="utf-8"))


def _valores_v2(v2: dict, dataset: str, motor: str, metrica: str) -> dict:
    return {(r["repeticion"], r["pliegue"]): r[metrica] for r in v2["resultados"]
            if r["dataset"] == dataset and r["motor"] == motor and r["estado"] == "completed"}


# ---------------------------------------------------------------------------
# 0. EL MOTOR FALSO: `ejecutar_intento_aislado` en milisegundos
# ---------------------------------------------------------------------------

_METRICAS_POR_TAREA = {"binary_classification": ("auroc",),
                       "multiclass_classification": ("accuracy", "macro_f1"),
                       "regression": ("r2",)}


def _motor_falso(valor, *, llamadas: list, fallos=frozenset(), arquitectura=None,
                 revienta_en=None):
    """`valor(dataset, repeticion, pliegue) -> float`. `fallos`: claves
    (dataset, rep, pliegue) que salen `failed`. `revienta_en`: la clave en la
    que el «proceso» muere (simula el tope de la cola a media noche)."""
    arquitectura = arquitectura or {"k": 8, "d_block": 256, "n_blocks": 2, "dropout": 0.1}

    def ejecutar(motor, train, validation, test, spec, presupuesto, *, candidate,
                 split_plan_digest, dataset, pliegue, repeticion, **_):
        clave = (dataset, repeticion, pliegue)
        if clave == revienta_en:
            raise KeyboardInterrupt("la cola mata el proceso (fabricado)")
        llamadas.append({"dataset": dataset, "repeticion": repeticion, "pliegue": pliegue,
                         "semilla": presupuesto.seed, "wall_seconds": presupuesto.wall_seconds,
                         "hilos": presupuesto.hilos})
        if clave in fallos:
            return SimpleNamespace(
                estado="failed", informe=None, recursos=None, config_efectiva=None,
                motivo_del_estado={"es": "fallo fabricado por la prueba", "en": "fabricated"},
                traza="Traceback (fabricado)", engine_version=None, pipeline_digest=None)
        v = valor(dataset, repeticion, pliegue)
        return SimpleNamespace(
            estado="completed",
            informe={"metrics": [{"metric_id": m, "value": v}
                                 for m in _METRICAS_POR_TAREA[spec.task]]},
            recursos={"wall_seconds": 0.01, "cpu_seconds": 0.02, "peak_ram_mb": 12.0},
            config_efectiva={
                "entrenamiento_efectivo": {"epocas_ejecutadas": 20, "mejor_epoca": 4,
                                           "parado_por_plazo": False},
                "arquitectura": dict(arquitectura),
                "hiperparametros": {"lote": 256, "optimizador": "adamw"},
                "pesos": {"no": "se guardan"}},
            motivo_del_estado=None, traza=None, engine_version="1.0.0+falso",
            pipeline_digest=f"pd-{dataset}-{repeticion}-{pliegue}")
    return ejecutar


def _correr_main(monkeypatch, argv, ejecutor):
    monkeypatch.setattr(p119, "ejecutar_intento_aislado", ejecutor)
    p119.main(argv)


# El escenario principal: cuatro conjuntos, uno por cada cosa que tiene que
# pasar, y UNA pasada falsa de la que salen muchas comprobaciones.
#   balance-scale: el nuevo medio punto por DEBAJO de catboost, que es el mejor
#       de la v2 sin la densa (la densa v2 es la mejor, 4,5 puntos por encima):
#       cumple SOLO si el nuevo compite en el sitio de la densa (S6).
#   kc2: el nuevo 5 puntos POR ENCIMA de la densa v2 en cada pliegue: mejora
#       (S7), y la densa v2, que es la mejor de la v2, sigue cumpliendo en la
#       referencia SIN el nuevo (S5).
#   yeast: 12 pliegues de 15 (la partición da 4 por repetición); el nuevo muy
#       por encima pero con UN fallo -> inferioridad por la regla (b).
#   wilt (mediano, 300 s): el nuevo 3 puntos por debajo: inferioridad.
_ESCENARIO = "balance-scale,kc2,yeast,wilt"
_FALLO_DE_YEAST = ("yeast", 1, 2)


@pytest.fixture(scope="module")
def pasada_falsa(tmp_path_factory, _v2):
    salida = tmp_path_factory.mktemp("pasada") / "resultado.json"
    catboost_bs = _valores_v2(_v2, "balance-scale", "catboost", "accuracy")
    densa = {n: _valores_v2(_v2, n, DENSA, m) for n, m in
             (("kc2", "auroc"), ("yeast", "accuracy"), ("wilt", "auroc"))}

    def valor(dataset, rep, pl):
        if dataset == "balance-scale":
            return catboost_bs[(rep, pl)] - 0.005
        delta = {"kc2": 0.05, "yeast": 0.10, "wilt": -0.03}[dataset]
        return densa[dataset][(rep, pl)] + delta

    llamadas: list = []
    with pytest.MonkeyPatch.context() as mp:
        _correr_main(mp, ["--solo", _ESCENARIO, "--salida", str(salida)],
                     _motor_falso(valor, llamadas=llamadas, fallos={_FALLO_DE_YEAST}))
    return {"salida": salida, "payload": json.loads(salida.read_text(encoding="utf-8")),
            "llamadas": llamadas}


# ---------------------------------------------------------------------------
# 1. LOS SELLADOS SE NIEGAN (a diferencia de 118, que solo avisaba)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _datasets():
    c6.preparar_protocolo_v2()
    protocolo = PROTOCOLO_V2
    todos = c5.datasets_de_la_pasada(protocolo)
    no_sellados = c6.datasets_no_sellados(protocolo)
    return todos, no_sellados


def test_sin_solo_son_los_32_no_sellados(_datasets):
    todos, no_sellados = _datasets
    datasets, subconjunto = p119.datasets_de_c3(todos, no_sellados, solo=None)
    assert len(datasets) == 32
    assert all(not d.sellado for d in datasets)
    assert datasets == no_sellados
    assert subconjunto is None


def test_solo_un_no_sellado_se_acepta(_datasets):
    todos, no_sellados = _datasets
    datasets, subconjunto = p119.datasets_de_c3(todos, no_sellados, solo="diabetes")
    assert [d.nombre for d in datasets] == ["diabetes"]
    assert subconjunto == ["diabetes"]


def test_solo_un_sellado_se_niega(_datasets):
    todos, no_sellados = _datasets
    sellados = [d.nombre for d in todos if d.sellado]
    assert sellados
    with pytest.raises(SystemExit, match="SELLADO"):
        p119.datasets_de_c3(todos, no_sellados, solo=sellados[0])


def test_solo_mezcla_sellado_y_no_sellado_tambien_se_niega(_datasets):
    todos, no_sellados = _datasets
    sellados = [d.nombre for d in todos if d.sellado]
    with pytest.raises(SystemExit, match="SELLADO"):
        p119.datasets_de_c3(todos, no_sellados, solo=f"diabetes,{sellados[0]}")


def test_solo_conjunto_desconocido_para(_datasets):
    todos, no_sellados = _datasets
    with pytest.raises(SystemExit, match="no están en el protocolo"):
        p119.datasets_de_c3(todos, no_sellados, solo="esto-no-existe-en-ningun-protocolo")


def test_los_32_no_sellados_coinciden_con_el_protocolo_v4():
    v4 = json.loads((_FASE0 / "protocolo_119_v4.json").read_text(encoding="utf-8"))
    sellados_v4 = sorted(d["nombre"] for d in v4["datasets"] if d["sellado"])
    sellados_v2 = sorted(d.nombre for d in c5.datasets_de_la_pasada(PROTOCOLO_V2) if d.sellado)
    assert sellados_v4 == sellados_v2
    assert len(sellados_v4) == 8


def test_v2_y_v4_son_identicos_byte_a_byte_en_los_bloques_que_se_miden():
    """Lo que el docstring del guion afirma, comprobado de verdad: el v2 (el
    que se lee) y el v4 (el que se cita) traen los MISMOS bytes canónicos en
    particion, presupuesto, regla_de_cierre y datasets."""
    v2 = json.loads(RUTA_PROTOCOLO_V2.read_text(encoding="utf-8"))
    v4 = json.loads((_FASE0 / "protocolo_119_v4.json").read_text(encoding="utf-8"))
    assert p119.BLOQUES_QUE_LA_V2_Y_EL_V4_COMPARTEN == (
        "particion", "presupuesto", "regla_de_cierre", "datasets")
    for bloque in p119.BLOQUES_QUE_LA_V2_Y_EL_V4_COMPARTEN:
        a = json.dumps(v2[bloque], sort_keys=True, ensure_ascii=False).encode("utf-8")
        b = json.dumps(v4[bloque], sort_keys=True, ensure_ascii=False).encode("utf-8")
        assert a == b, bloque


# ---------------------------------------------------------------------------
# 2. EL CACHÉ NO REUTILIZA UN `failed` (ni un ARFF distinto)
# ---------------------------------------------------------------------------

def _previo(**cambios) -> dict:
    return {"entorno_digest": "e1", "motor_digest": "m1", "presupuesto_wall_s": 120.0,
            "estado": "completed", "datos_sha256": "d1", **cambios}


def test_reusable_c3_acepta_completed_con_digests_iguales():
    assert p119._reusable_c3(_previo(), "e1", "m1", 120.0, datos_sha256="d1") is True


def test_reusable_c3_acepta_completed_budget_limited():
    assert p119._reusable_c3(_previo(estado="completed_budget_limited"), "e1", "m1", 120.0,
                             datos_sha256="d1") is True


def test_reusable_c3_rechaza_failed_aunque_los_digests_coincidan():
    """LA TRAMPA Nº1 de «ANTES DE RELANZAR UNA PASADA» (CLAUDE.md)."""
    previo = _previo(estado="failed")
    assert c3._reusable(previo, "e1", "m1", 120.0) is True  # la trampa es real
    assert p119._reusable_c3(previo, "e1", "m1", 120.0, datos_sha256="d1") is False


def test_reusable_c3_rechaza_cancelled():
    assert p119._reusable_c3(_previo(estado="cancelled"), "e1", "m1", 120.0,
                             datos_sha256="d1") is False


def test_reusable_c3_rechaza_digest_de_entorno_distinto():
    assert p119._reusable_c3(_previo(), "e2", "m1", 120.0, datos_sha256="d1") is False


def test_reusable_c3_rechaza_un_arff_distinto_o_sin_sha():
    assert p119._reusable_c3(_previo(), "e1", "m1", 120.0, datos_sha256="d2") is False
    assert p119._reusable_c3(_previo(), "e1", "m1", 120.0, datos_sha256=None) is False
    sin_sha = _previo()
    del sin_sha["datos_sha256"]
    assert p119._reusable_c3(sin_sha, "e1", "m1", 120.0, datos_sha256="d1") is False


def test_reusable_c3_ninguno_previo():
    assert p119._reusable_c3(None, "e1", "m1", 120.0, datos_sha256="d1") is False


# ---------------------------------------------------------------------------
# 3. EL DIGEST DE LA CACHÉ (I4)
# ---------------------------------------------------------------------------

def test_digest_de_reacciona_al_contenido_de_los_ficheros(tmp_path):
    f1 = tmp_path / "a.py"
    f1.write_text("contenido version 1\n", encoding="utf-8")
    d1 = p119._digest_de((f1,))
    f1.write_text("contenido version 1\n", encoding="utf-8")
    assert p119._digest_de((f1,)) == d1
    f1.write_text("contenido version 2 -- una linea distinta\n", encoding="utf-8")
    assert p119._digest_de((f1,)) != d1


def test_digest_motor_nuevo_incluye_los_tres_ficheros_declarados():
    nombres = {p.name for p in p119._FICHEROS_DEL_MOTOR_NUEVO}
    assert nombres == {"densa_tabm.py", "tabm_plr.py", "preparacion_tabm.py"}
    for ruta in p119._FICHEROS_DEL_MOTOR_NUEVO:
        assert ruta.exists(), f"{ruta} no existe"


def test_digest_motor_nuevo_es_estable_y_no_vacio():
    assert p119._digest_motor_nuevo() == p119._digest_motor_nuevo()
    assert len(p119._digest_motor_nuevo()) == 16


@pytest.fixture(scope="module")
def _componentes():
    return p119.componentes_del_digest_del_entorno()


def test_el_digest_cubre_lo_que_la_auditoria_encontro_fuera(_componentes):
    """Los que la auditoría 2 midió FUERA del digest, por su nombre."""
    for etiqueta in ("core:matrixai/estudio/metricas.py", "core:matrixai/estudio/esquemas.py",
                     "engines:matrixai_engines/tipos_de_columna.py",
                     "core:benchmarks/fase0/protocolo.py", "core:benchmarks/fase0/pasada_v2_113.py",
                     "core:benchmarks/fase0/pasada_114c6_ensamblado.py",
                     "core:benchmarks/fase0/pasada_119_c3.py",
                     "engines:matrixai_engines/harness.py", "engines:matrixai_engines/subproceso.py",
                     "engines:matrixai_engines/motores/densa_tabm.py",
                     "engines:matrixai_engines/redes/tabm_plr.py",
                     "engines:matrixai_engines/redes/preparacion_tabm.py",
                     "dato:protocolo_exploratorio_v2", "dato:protocolo_119_v4",
                     "dato:protocolo_119_v4_enmienda_1",
                     "version:torch", "version:numpy", "version:scikit-learn", "version:python"):
        assert etiqueta in _componentes, etiqueta


def test_las_etiquetas_no_dependen_de_donde_esta_el_arbol(_componentes):
    """Un worktree de la cola y el árbol principal tienen que dar el MISMO
    digest: ninguna etiqueta lleva una ruta absoluta."""
    assert all(e.split(":", 1)[0] in ("core", "engines", "dato", "version") for e in _componentes)
    assert not any("/home/" in e or "/tmp/" in e for e in _componentes)


def test_lo_que_importa_el_guion_de_los_repos_esta_en_el_digest():
    """Cota inferior medida: todo módulo de los repos que el GUION tiene
    cargado (lo que arrastra al importarse y al preparar el protocolo) está
    en el digest. En un proceso APARTE: en el de pytest hay cargados otros
    ficheros de prueba que el guion no importa."""
    import subprocess
    orden = ("import json, pasada_119_c3 as p; p.c6.preparar_protocolo_v2(); "
             "p.exigir_la_configuracion_de_la_enmienda(); "
             "print(json.dumps(p.lo_que_no_cubre_el_digest()))")
    hecho = subprocess.run([sys.executable, "-c", orden], cwd=_FASE0, capture_output=True,
                           text=True, timeout=300)
    assert hecho.returncode == 0, hecho.stderr[-2000:]
    nota = json.loads(hecho.stdout.strip().splitlines()[-1])
    assert nota["modulos_de_los_repos_cargados_por_este_proceso_fuera_del_digest"] == []
    for clave in ("bibliotecas_de_terceros", "imports_dinamicos_en_el_cierre", "datos",
                  "enmienda_2", "como_se_calcula"):
        assert nota[clave]


def test_el_cierre_sigue_los_imports_de_dentro_de_las_funciones(tmp_path):
    """El padre no importa lo que el hijo importa DENTRO de una función: el
    cierre tiene que verlo igual."""
    paquete = tmp_path / "paq"
    paquete.mkdir()
    (paquete / "__init__.py").write_text("", encoding="utf-8")
    (paquete / "perezoso.py").write_text("X = 1\n", encoding="utf-8")
    (paquete / "relativo.py").write_text("Y = 2\n", encoding="utf-8")
    (paquete / "entrada.py").write_text(
        "from . import relativo\n"
        "def f():\n    import paq.perezoso\n    import numpy\n    return paq.perezoso.X\n",
        encoding="utf-8")
    cierre, _ = p119._cierre_de_imports((paquete / "entrada.py").resolve(), (tmp_path.resolve(),))
    nombres = {f.name for f in cierre}
    assert {"entrada.py", "perezoso.py", "relativo.py", "__init__.py"} <= nombres


def test_el_digest_reacciona_a_metricas_py(monkeypatch):
    antes = p119._digest_entorno()
    original = c3._digest_fichero

    def alterado(ruta):
        d = original(ruta)
        return "0" * 16 if Path(ruta).name == "metricas.py" and "estudio" in str(ruta) else d
    monkeypatch.setattr(c3, "_digest_fichero", alterado)
    assert p119._digest_entorno() != antes


def test_el_digest_reacciona_a_la_version_de_torch(monkeypatch):
    antes = p119._digest_entorno()
    real = p119.versiones_de_bibliotecas()
    monkeypatch.setattr(p119, "versiones_de_bibliotecas",
                        lambda: {**real, "torch": "9.9.9+otra"})
    assert p119._digest_entorno() != antes


@pytest.mark.parametrize("constante", ["RUTA_DEL_PROTOCOLO_V4", "RUTA_DE_LA_ENMIENDA_1",
                                       "RUTA_DEL_PROTOCOLO_V2"])
def test_el_digest_reacciona_a_cada_fichero_de_protocolo(tmp_path, monkeypatch, constante):
    """El v4, su enmienda (S8 de la auditoría: quitarla del digest salía
    verde) y el v2 que DE VERDAD se lee."""
    falso = tmp_path / "protocolo_falso.json"
    falso.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(p119, constante, falso)
    antes = p119._digest_entorno()
    falso.write_text('{"version": 2}', encoding="utf-8")
    assert p119._digest_entorno() != antes


# ---------------------------------------------------------------------------
# 4. LAS GUARDIAS (I6)
# ---------------------------------------------------------------------------

def test_la_enmienda_1_declara_k8_d256_n2():
    enmienda = json.loads((_FASE0 / "protocolo_119_v4_enmienda_1.json").read_text("utf-8"))
    assert p119.configuracion_que_declara_la_enmienda(enmienda["que_cambia"]) == {
        "k": 8, "d_block": 256, "n_blocks": 2}
    assert p119.configuracion_que_declara_la_enmienda("k=16 y d_block=512 (n_blocks=3)") == {
        "k": 16, "d_block": 512, "n_blocks": 3}
    assert p119.configuracion_que_declara_la_enmienda("sin números")["k"] is None


def test_el_motor_usa_la_configuracion_de_la_enmienda():
    leida = p119.exigir_la_configuracion_de_la_enmienda()
    assert leida["leida_del_motor"] == {"k": 8, "d_block": 256, "n_blocks": 2}


@pytest.mark.parametrize("constante,valor", [("K_CABEZAS", 16), ("D_BLOCK", 512),
                                             ("N_BLOCKS", 3)])
def test_para_si_el_motor_no_usa_la_configuracion_de_la_enmienda(monkeypatch, constante, valor):
    modulo = sys.modules[p119.MotorDensaTabM.__module__]
    monkeypatch.setattr(modulo, constante, valor)
    with pytest.raises(SystemExit, match="enmienda 1"):
        p119.exigir_la_configuracion_de_la_enmienda()


def test_para_si_el_motor_ya_no_tiene_la_constante(monkeypatch):
    monkeypatch.delattr(sys.modules[p119.MotorDensaTabM.__module__], "K_CABEZAS")
    with pytest.raises(SystemExit):
        p119.exigir_la_configuracion_de_la_enmienda()


def test_para_si_la_enmienda_y_el_guion_no_dicen_lo_mismo(tmp_path, monkeypatch):
    enmienda = json.loads((_FASE0 / "protocolo_119_v4_enmienda_1.json").read_text("utf-8"))
    enmienda["que_cambia"] = "k=16 y d_block=256 (n_blocks=2)"
    falsa = tmp_path / "enmienda.json"
    falsa.write_text(json.dumps(enmienda), encoding="utf-8")
    monkeypatch.setattr(p119, "RUTA_DE_LA_ENMIENDA_1", falsa)
    with pytest.raises(SystemExit, match="declara"):
        p119.exigir_la_configuracion_de_la_enmienda()


def test_la_cadena_real_de_protocolos_cuadra():
    cadena = p119.exigir_los_protocolos_encadenados()
    assert cadena["enmienda_1"]["de"]["digest_sha256"] == cadena["v4"]["digest_sha256_declarado"]
    assert all(cadena["bloques_identicos_v2_v4"].values())


def _con_digest(payload: dict) -> dict:
    return {**payload, "digest_sha256": p119.autodigest(payload)}


def test_para_si_el_v4_no_cuadra_con_su_digest(tmp_path, monkeypatch):
    v4 = json.loads((_FASE0 / "protocolo_119_v4.json").read_text("utf-8"))
    v4["fecha_registro"] = "2026-12-31"  # tocado DESPUÉS de registrarlo
    falso = tmp_path / "v4.json"
    falso.write_text(json.dumps(v4), encoding="utf-8")
    monkeypatch.setattr(p119, "RUTA_DEL_PROTOCOLO_V4", falso)
    with pytest.raises(SystemExit, match="NO cuadra"):
        p119.exigir_los_protocolos_encadenados()


def test_para_si_la_enmienda_no_es_de_este_v4(tmp_path, monkeypatch):
    enmienda = json.loads((_FASE0 / "protocolo_119_v4_enmienda_1.json").read_text("utf-8"))
    enmienda["de"] = {**enmienda["de"], "digest_sha256": "f" * 64}
    falsa = tmp_path / "enmienda.json"
    falsa.write_text(json.dumps(_con_digest(enmienda)), encoding="utf-8")
    monkeypatch.setattr(p119, "RUTA_DE_LA_ENMIENDA_1", falsa)
    with pytest.raises(SystemExit, match="dice ser"):
        p119.exigir_los_protocolos_encadenados()


def test_para_si_el_v2_y_el_v4_divergen(tmp_path, monkeypatch):
    v2 = json.loads(RUTA_PROTOCOLO_V2.read_text("utf-8"))
    v2["particion"] = {**v2["particion"], "folds": 4}
    falso = tmp_path / "v2.json"
    falso.write_text(json.dumps(v2), encoding="utf-8")
    monkeypatch.setattr(p119, "RUTA_DEL_PROTOCOLO_V2", falso)
    with pytest.raises(SystemExit, match="'particion'"):
        p119.exigir_los_protocolos_encadenados()


def test_la_arquitectura_de_cada_intento_se_comprueba():
    bueno = {"dataset": "d", "repeticion": 0, "pliegue": 0, "estado": "completed",
             "arquitectura": {"k": 8, "d_block": 256, "n_blocks": 2}}
    p119.exigir_la_arquitectura_del_intento(bueno)
    p119.exigir_la_arquitectura_del_intento({**bueno, "estado": "failed", "arquitectura": None})
    with pytest.raises(SystemExit, match="declara haber corrido"):
        p119.exigir_la_arquitectura_del_intento(
            {**bueno, "arquitectura": {"k": 16, "d_block": 256, "n_blocks": 2}})
    with pytest.raises(SystemExit, match="SIN declarar"):
        p119.exigir_la_arquitectura_del_intento({**bueno, "arquitectura": None})


def test_la_particion_se_compara_con_la_de_la_v2():
    ds = SimpleNamespace(nombre="dsA", data_id=1)
    particion = {"plan_digest": "abc", "n_pliegues_obtenidos": 15, "limites": []}
    lectura = {"objetivo": "y", "columnas_excluidas": []}
    payload = {"particion_por_dataset": {"dsA": dict(particion)},
               "lectura_de_los_datos": {"1": dict(lectura)}}
    assert p119.exigir_la_particion_de_la_v2(ds, particion, lectura, payload)["identica_a_la_v2"]
    with pytest.raises(SystemExit, match="plan_digest"):
        p119.exigir_la_particion_de_la_v2(ds, {**particion, "plan_digest": "otro"}, lectura,
                                          payload)
    with pytest.raises(SystemExit, match="LECTURA"):
        p119.exigir_la_particion_de_la_v2(ds, particion, {**lectura, "objetivo": "z"}, payload)
    with pytest.raises(SystemExit, match="no registra"):
        p119.exigir_la_particion_de_la_v2(SimpleNamespace(nombre="dsB", data_id=2), particion,
                                          lectura, payload)


def test_main_para_si_el_motor_declara_otra_arquitectura_y_guarda_lo_medido(tmp_path,
                                                                           monkeypatch):
    salida = tmp_path / "r.json"
    llamadas: list = []
    with pytest.raises(SystemExit, match="declara haber corrido"):
        _correr_main(monkeypatch, ["--solo", "kc2", "--salida", str(salida)],
                     _motor_falso(lambda *a: 0.9, llamadas=llamadas,
                                  arquitectura={"k": 16, "d_block": 256, "n_blocks": 2}))
    assert len(llamadas) == 1
    guardado = json.loads(salida.read_text(encoding="utf-8"))
    assert guardado["parcial"] is True and len(guardado["resultados"]) == 1


# ---------------------------------------------------------------------------
# 5. EL VEREDICTO CON FALLOS (I2), sobre casos fabricados
# ---------------------------------------------------------------------------

_ESPERADAS_3 = [(0, 0), (0, 1), (0, 2)]


def _serie(motor, valores, *, dataset="dsA", fallos=(), ausentes=(), sin_metrica=()):
    salida = []
    for (rep, pl), v in zip(_ESPERADAS_3, valores):
        if (rep, pl) in ausentes:
            continue
        estado = "failed" if (rep, pl) in fallos else "completed"
        valor = None if ((rep, pl) in fallos or (rep, pl) in sin_metrica) else v
        salida.append(_registro(dataset, motor, valor, repeticion=rep, pliegue=pl, estado=estado))
    return salida


def test_fallos_del_motor_distingue_ausente_fallido_y_sin_metrica():
    registros = _serie(NUEVO, [0.9, 0.9, 0.9], fallos={(0, 0)}, ausentes={(0, 1)},
                       sin_metrica={(0, 2)})
    fallos = p119.fallos_del_motor(registros, NUEVO, "auroc", _ESPERADAS_3)
    assert [(f["pliegue"], f["tipo"]) for f in fallos] == [
        (0, "no_completado"), (1, "ausente"), (2, "sin_metrica_de_cierre")]
    assert p119.fallos_del_motor(_serie(NUEVO, [0.9] * 3), NUEVO, "auroc", _ESPERADAS_3) == []


def _cond1(nuevo, v2_otros, v2_densa, esperadas=_ESPERADAS_3):
    regla = PROTOCOLO_V2.regla_de_cierre
    return p119.veredicto_final(
        resultados_v2=v2_otros + v2_densa, resultados_del_motor_nuevo=nuevo,
        veredictos_por_conjunto=[], esperadas_por_conjunto={"dsA": esperadas}, regla=regla,
        metrica_por_dataset={"dsA": "auroc"}, nombres_de_los_conjuntos=["dsA"])


@pytest.mark.parametrize("hueco", ["fallos", "ausentes", "sin_metrica"])
def test_condicion_1_cualquier_intento_sin_medida_del_nuevo_pierde_el_conjunto(hueco):
    """(a): el nuevo, a 0 puntos del mejor en lo que completó, PIERDE el
    conjunto si le falta un intento por cualquiera de las tres vías. Sin la
    reparación, un AUSENTE o un completado SIN métrica no contaban."""
    otros = _serie("lightgbm", [0.80, 0.80, 0.80])
    densa = _serie(DENSA, [0.79, 0.79, 0.79])
    completo = _cond1(_serie(NUEVO, [0.80, 0.80, 0.80]), otros, densa)
    assert completo["cumplidos_con_el_motor_nuevo"]["cumplidos"] == 1
    con_hueco = _cond1(_serie(NUEVO, [0.80, 0.80, 0.80], **{hueco: {(0, 1)}}), otros, densa)
    assert con_hueco["cumplidos_con_el_motor_nuevo"]["cumplidos"] == 0
    assert con_hueco["cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"]["cumplidos"] == 1


def test_condicion_1_la_densa_v2_con_un_intento_ausente_tambien_pierde():
    """El mismo criterio para los dos lados."""
    otros = _serie("lightgbm", [0.80, 0.80, 0.80])
    v = _cond1(_serie(NUEVO, [0.5] * 3), otros, _serie(DENSA, [0.80] * 3, ausentes={(0, 2)}))
    assert v["cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"]["cumplidos"] == 0


def test_condicion_1_la_referencia_se_mide_sin_el_motor_nuevo():
    """(S5) La densa v2 es la mejor de la v2; el nuevo la supera. Medida en
    el campo CON el nuevo (o sin la densa, que es lo que queda en el campo),
    la densa v2 dejaría de cumplir."""
    otros = _serie("lightgbm", [0.80, 0.80, 0.80])
    v = _cond1(_serie(NUEVO, [0.99] * 3), otros, _serie(DENSA, [0.90] * 3))
    assert v["cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"]["cumplidos"] == 1
    assert v["cumplidos_con_el_motor_nuevo"]["cumplidos"] == 1


def test_condicion_1_el_motor_nuevo_compite_en_el_sitio_de_la_densa_v2():
    """(S6) La densa v2 es la mejor por 10 puntos; el nuevo, a 1 punto de
    lightgbm. En el campo (sin la densa) cumple; con la densa dentro, no."""
    otros = _serie("lightgbm", [0.80, 0.80, 0.80])
    v = _cond1(_serie(NUEVO, [0.79] * 3), otros, _serie(DENSA, [0.90] * 3))
    assert v["cumplidos_con_el_motor_nuevo"]["cumplidos"] == 1


def _cond2(nuevo, densa, esperadas=None):
    esperadas = esperadas or [(r, p) for r in range(3) for p in range(5)]
    return p119.veredicto_del_conjunto_c3(dataset="dsA", metric_id="auroc",
                                          registros_del_motor_nuevo=nuevo,
                                          registros_de_la_densa_v2=densa, esperadas=esperadas)


def _quince(motor, valor_de, *, estado_de=lambda k: "completed"):
    return [_registro("dsA", motor, (None if estado_de((r, p)) != "completed" else valor_de((r, p))),
                      repeticion=r, pliegue=p, estado=estado_de((r, p)))
            for r in range(3) for p in range(5)]


def _base(k):
    return 0.70 + 0.01 * (k[0] * 5 + k[1])


def test_condicion_2_la_diferencia_es_nuevo_menos_densa_v2():
    """(S7) El nuevo por encima en todos los pliegues: MEJORA, no
    inferioridad."""
    v = _cond2(_quince(NUEVO, lambda k: _base(k) + 0.05), _quince(DENSA, _base))
    assert v["mejora"] is True and v["inferioridad"] is False
    assert v["intervalo"]["diferencia"] == "motor nuevo - densa v2"
    v = _cond2(_quince(NUEVO, lambda k: _base(k) - 0.05), _quince(DENSA, _base))
    assert v["mejora"] is False and v["inferioridad"] is True


def test_condicion_2_fallos_del_nuevo_con_la_densa_v2_completa_es_inferioridad():
    """(b) El nuevo muy por encima en los 14 que completó y UN fallo: la
    emparejada diría mejora; la regla dice INFERIORIDAD."""
    v = _cond2(_quince(NUEVO, lambda k: _base(k) + 0.10,
                       estado_de=lambda k: "failed" if k == (1, 2) else "completed"),
               _quince(DENSA, _base))
    assert v["la_emparejada_decide"] is False
    assert v["inferioridad"] is True and v["mejora"] is False
    assert v["regla_aplicada"] == "b_inferioridad_por_fallos"


def test_condicion_2_la_densa_v2_sin_ningun_completado_es_sin_comparacion():
    """(b) Allstate/KDDCup09: la densa v2 falló los 5. Ni mejora ni
    inferioridad, aunque el nuevo complete todo."""
    esperadas = [(0, p) for p in range(5)]
    nuevo = [_registro("dsA", NUEVO, 0.9, pliegue=p) for p in range(5)]
    densa = [_registro("dsA", DENSA, None, pliegue=p, estado="failed") for p in range(5)]
    v = _cond2(nuevo, densa, esperadas)
    assert v["sin_comparacion"] is True
    assert v["mejora"] is False and v["inferioridad"] is False


def test_condicion_2_emparejada_solo_sobre_los_completos_en_los_dos():
    """(c) La densa v2 falló (0,1) — con un valor viejo en el registro, para
    probar que es el ESTADO lo que lo saca, no la ausencia del número."""
    densa = _quince(DENSA, _base)
    for r in densa:
        if (r["repeticion"], r["pliegue"]) == (0, 1):
            r.update(estado="failed", auroc=0.0)
    nuevo = _quince(NUEVO, lambda k: 0.5 if k == (0, 1) else _base(k) - 0.002)
    v = _cond2(nuevo, densa)
    assert v["n_pliegues_comunes"] == 14
    assert v["inferioridad"] is True and v["mejora"] is False
    assert v["regla_aplicada"] == "c_emparejada"


def test_condicion_2_fallos_en_los_dos_es_un_caso_no_previsto_y_se_dice():
    v = _cond2(_quince(NUEVO, _base, estado_de=lambda k: "failed" if k == (2, 4) else "completed"),
               _quince(DENSA, _base, estado_de=lambda k: "failed" if k == (0, 0) else "completed"))
    assert v["caso_no_previsto"] is True


def test_en_la_v2_real_la_densa_no_tiene_fallos_parciales_en_los_32(_v2):
    """El caso no previsto es INERTE en C3: en los 32 no sellados la densa v2
    completa todos sus intentos o falla todos (Allstate, KDDCup09). Si la v2
    cambiara, esta prueba tiene que chillar antes de medir."""
    no_sellados = {d.nombre for d in c6.datasets_no_sellados(PROTOCOLO_V2)}
    por_ds: dict[str, set] = {}
    for r in _v2["resultados"]:
        if r["motor"] == DENSA and r["dataset"] in no_sellados:
            por_ds.setdefault(r["dataset"], set()).add(r["estado"] == "completed")
    assert len(por_ds) == 32
    assert all(len(s) == 1 for s in por_ds.values()), {k: s for k, s in por_ds.items()
                                                        if len(s) > 1}
    assert sorted(k for k, s in por_ds.items() if s == {False}) == [
        "Allstate_Claims_Severity", "KDDCup09_appetency"]


def test_veredicto_por_conjunto_registra_los_fallos_con_su_motivo():
    """(d) Cuántos y por qué."""
    v = _cond2(_quince(NUEVO, _base, estado_de=lambda k: "failed" if k in {(0, 0), (1, 1)}
                       else "completed"), _quince(DENSA, _base))
    assert v["fallos_del_motor_nuevo"]["n"] == 2
    assert {(f["repeticion"], f["pliegue"]) for f in v["fallos_del_motor_nuevo"]["detalle"]} == {
        (0, 0), (1, 1)}
    assert all(f["tipo"] == "no_completado" for f in v["fallos_del_motor_nuevo"]["detalle"])
    assert "palanca" not in json.dumps(v)


def _cumplidos_de(n: int, de: int = 5) -> dict:
    return {"cumplidos": n, "datasets": de, "fraccion": n / de, "cumple_la_regla": n / de >= 0.8}


def test_veredicto_de_c3_sube():
    veredictos = [{"dataset": f"d{i}", "mejora": i < 3, "inferioridad": i == 3}
                  for i in range(5)]
    v = p119.veredicto_de_c3(veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(5),
                             cumplidos_de_la_densa_v2=_cumplidos_de(3))
    assert v["sube_los_cumplidos"] is True
    assert v["mejoras_superan_inferioridades"] is True
    assert (v["n_mejora"], v["n_inferioridad"]) == (3, 1)
    assert v["cumple_la_regla_de_subida"] is True
    assert "matrixai.dense.tabm_cpu" in v["campo"]


def test_veredicto_de_c3_no_sube_porque_no_sube_los_cumplidos():
    veredictos = [{"dataset": f"d{i}", "mejora": True, "inferioridad": False} for i in range(5)]
    v = p119.veredicto_de_c3(veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(3),
                             cumplidos_de_la_densa_v2=_cumplidos_de(3))
    assert v["mejoras_superan_inferioridades"] is True
    assert v["sube_los_cumplidos"] is False
    assert v["cumple_la_regla_de_subida"] is False


def test_veredicto_de_c3_no_sube_porque_no_hay_mas_mejoras_que_inferioridades():
    veredictos = [{"dataset": f"d{i}", "mejora": i < 2, "inferioridad": i >= 2}
                  for i in range(5)]
    v = p119.veredicto_de_c3(veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(5),
                             cumplidos_de_la_densa_v2=_cumplidos_de(3))
    assert v["mejoras_superan_inferioridades"] is False
    assert v["cumple_la_regla_de_subida"] is False


def test_veredicto_de_c3_cuenta_los_sin_comparacion_aparte():
    veredictos = [{"dataset": "a", "mejora": True, "inferioridad": False},
                  {"dataset": "b", "mejora": False, "inferioridad": False,
                   "sin_comparacion": True}]
    v = p119.veredicto_de_c3(veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(2, 2),
                             cumplidos_de_la_densa_v2=_cumplidos_de(1, 2))
    assert (v["n_mejora"], v["n_inferioridad"], v["n_sin_comparacion"]) == (1, 0, 1)
    assert v["conjuntos_sin_comparacion"] == ["b"]


def test_veredicto_de_c3_reusa_veredicto_de_la_palanca_de_118():
    import pasada_118_palanca as p118
    veredictos = [{"dataset": "d0", "mejora": True, "inferioridad": False}]
    directo = p118.veredicto_de_la_palanca(
        veredictos, cumplidos_con_la_palanca=_cumplidos_de(1, de=1),
        cumplidos_de_la_densa_v2=_cumplidos_de(0, de=1))
    envuelto = p119.veredicto_de_c3(
        veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(1, de=1),
        cumplidos_de_la_densa_v2=_cumplidos_de(0, de=1))
    assert envuelto["cumple_la_regla_de_subida"] == directo["la_palanca_ayuda"]
    assert envuelto["n_mejora"] == directo["n_mejora"]


def test_campo_de_la_comparacion_sustituye_la_densa_por_el_motor_nuevo():
    import pasada_118_palanca as p118
    resultados_v2 = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", DENSA, 0.70),
        _registro("dsB", "lightgbm", 0.75), _registro("dsB", DENSA, 0.72),
    ]
    nuevo = [_registro("dsA", NUEVO, 0.90)]
    campo = p118.campo_de_la_comparacion(resultados_v2, nuevo, {"dsA"})
    assert [r for r in campo if r["dataset"] == "dsA" and r["motor"] == NUEVO] == nuevo
    assert [r for r in campo if r["dataset"] == "dsA" and r["motor"] == DENSA] == []
    assert [r for r in campo if r["dataset"] == "dsB" and r["motor"] == DENSA] == [
        _registro("dsB", DENSA, 0.72)]


def test_cumplidos_de_se_acota_a_los_conjuntos_pedidos():
    regla = PROTOCOLO_V2.regla_de_cierre
    resultados = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", NUEVO, 0.79),
        _registro("dsX", "lightgbm", 0.90), _registro("dsX", NUEVO, 0.10),
    ]
    salida = p119._cumplidos_de(resultados, regla, motor=NUEVO,
                                metrica_por_dataset={"dsA": "auroc"},
                                nombres_de_los_conjuntos=["dsA"])
    assert salida == {"cumplidos": 1, "datasets": 1, "fraccion": 1.0, "cumple_la_regla": True}


def test_nombre_motor_nuevo_coincide_con_la_clase_real():
    from matrixai_engines.motores.densa_tabm import MotorDensaTabM
    assert p119.NOMBRE_MOTOR_NUEVO == MotorDensaTabM().nombre


# ---------------------------------------------------------------------------
# 6. main() DE VERDAD, con el motor falso (I5 y el resto por el camino real)
# ---------------------------------------------------------------------------

def test_main_llama_una_vez_por_intento_de_la_particion(pasada_falsa):
    """(S12) TODAS las repeticiones, y los pliegues que la partición trae:
    15 + 15 + 12 (yeast) + 15."""
    por_ds: dict[str, set] = {}
    for c in pasada_falsa["llamadas"]:
        por_ds.setdefault(c["dataset"], set()).add((c["repeticion"], c["pliegue"]))
    assert {k: len(v) for k, v in por_ds.items()} == {
        "balance-scale": 15, "kc2": 15, "yeast": 12, "wilt": 15}
    assert {r for r, _ in por_ds["kc2"]} == {0, 1, 2}


def test_main_una_semilla_por_repeticion(pasada_falsa):
    """(S10) La semilla de SU repetición, no la de la primera para todas."""
    semillas = {(c["repeticion"], c["semilla"]) for c in pasada_falsa["llamadas"]}
    assert semillas == {(0, 0), (1, 1), (2, 2)}


def test_main_el_presupuesto_de_cada_cubo(pasada_falsa):
    """(S11) 120 s el pequeño, 300 s el mediano; y los 4 hilos."""
    tope = {c["dataset"]: c["wall_seconds"] for c in pasada_falsa["llamadas"]}
    assert tope == {"balance-scale": 120.0, "kc2": 120.0, "yeast": 120.0, "wilt": 300.0}
    assert {c["hilos"] for c in pasada_falsa["llamadas"]} == {4}


def test_main_la_referencia_es_la_v2_sin_el_motor_nuevo(pasada_falsa):
    """(S5) La densa v2 cumple en kc2 y balance-scale (es la mejor de la v2)
    aunque el nuevo la supere en kc2."""
    v = pasada_falsa["payload"]["veredicto"]
    ref = v["cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"]
    assert ref["cumplidos"] == 2
    assert {d["dataset"] for d in ref["detalle"] if d["cumple"]} == {"balance-scale", "kc2"}


def test_main_el_motor_nuevo_compite_en_el_sitio_de_la_densa(pasada_falsa):
    """(S6) balance-scale: el nuevo, medio punto por debajo de catboost y 5
    por debajo de la densa v2, cumple SOLO si la sustituye en el campo."""
    nuevo = pasada_falsa["payload"]["veredicto"]["cumplidos_con_el_motor_nuevo"]
    assert {d["dataset"]: d["cumple"] for d in nuevo["detalle"]} == {
        "balance-scale": True, "kc2": True, "yeast": False, "wilt": False}
    yeast = [d for d in nuevo["detalle"] if d["dataset"] == "yeast"][0]
    assert yeast["perdido_por_fallo"] is True


def test_main_la_diferencia_emparejada_es_nuevo_menos_densa(pasada_falsa):
    """(S7) kc2 mejora; wilt, inferioridad; yeast, inferioridad por su fallo."""
    por_ds = {v["dataset"]: v for v in pasada_falsa["payload"]["veredicto_por_conjunto"]}
    assert por_ds["kc2"]["mejora"] is True and por_ds["kc2"]["inferioridad"] is False
    assert por_ds["wilt"]["inferioridad"] is True
    assert por_ds["yeast"]["inferioridad"] is True
    assert por_ds["yeast"]["regla_aplicada"] == "b_inferioridad_por_fallos"
    assert por_ds["yeast"]["fallos_del_motor_nuevo"]["n"] == 1
    v = pasada_falsa["payload"]["veredicto"]
    assert v["conjuntos_con_mejora"] == ["kc2"]
    assert v["conjuntos_con_fallos_del_motor_nuevo"] == ["yeast"]


def test_main_guarda_lo_que_el_motor_declara(pasada_falsa):
    """arquitectura, hiperparámetros, traza, engine_version, pipeline_digest,
    y la preparación del padre aparte de wall_s. Los pesos, NUNCA."""
    resultados = pasada_falsa["payload"]["resultados"]
    completado = [r for r in resultados if r["estado"] == "completed"][0]
    for campo in ("arquitectura", "hiperparametros", "engine_version", "pipeline_digest",
                  "split_plan_digest", "preparacion_s", "wall_s", "datos_sha256", "traza"):
        assert campo in completado, campo
    assert completado["arquitectura"]["k"] == 8
    assert completado["pipeline_digest"].startswith("pd-")
    fallido = [r for r in resultados if r["estado"] == "failed"]
    assert len(fallido) == 1 and fallido[0]["traza"] == "Traceback (fabricado)"
    assert "pesos" not in json.dumps(resultados)
    tiempos = pasada_falsa["payload"]["tiempos_del_padre_por_dataset"]
    assert tiempos["kc2"]["n_preparaciones"] == 15 and tiempos["kc2"]["carga_s"] > 0


def test_main_concilia_los_intentos_con_el_plan(pasada_falsa):
    """yeast: 15 pedidos, 12 que la partición admite; y el hueco del plan
    (60 pedidos, 57 medidos) queda explicado entero."""
    payload = pasada_falsa["payload"]
    assert payload["intentos_por_conjunto"]["yeast"] == {
        "pedidos_por_el_protocolo": 15, "admitidos_por_la_particion": 12,
        "planeados_en_esta_ejecucion": 12, "medidos": 12, "completos": 11,
        "limites": payload["particion_por_dataset"]["yeast"]["limites"]}
    reconciliado = payload["por_que_n_intentos_no_es_el_del_plan"]
    assert (reconciliado["plan"], reconciliado["medidos"], reconciliado["cuadra"]) == (60, 57, True)


def test_main_registra_la_particion_y_es_la_de_la_v2(pasada_falsa, _v2):
    payload = pasada_falsa["payload"]
    for nombre in ("balance-scale", "kc2", "yeast", "wilt"):
        assert payload["particion_por_dataset"][nombre] == _v2["particion_por_dataset"][nombre]
        assert payload["particion_comparada_con_la_v2"][nombre]["identica_a_la_v2"] is True
        assert payload["pliegues_digest_por_dataset"][nombre]


def test_main_declara_allstate_y_la_enmienda_2(pasada_falsa):
    payload = pasada_falsa["payload"]
    assert "≈10" in payload["allstate_declarado"]
    assert payload["allstate_medido"]["medido"] is False
    assert "presente" in payload["enmienda_2"]
    assert payload["tipo_de_ejecucion"] == "solo"
    assert payload["lo_que_no_cubre_el_digest"]["imports_dinamicos_en_el_cierre"]
    assert payload["digest_de_la_cache"]["entorno"] == payload["resultados"][0]["entorno_digest"]


def test_allstate_medido_dice_las_epocas_de_verdad():
    registros = [{"dataset": "Allstate_Claims_Severity", "repeticion": 0, "pliegue": p,
                  "estado": "completed",
                  "entrenamiento_efectivo": {"epocas_ejecutadas": 9 + p, "mejor_epoca": 8,
                                             "parado_por_plazo": True}} for p in range(5)]
    medido = p119.allstate_medido(registros)
    assert (medido["epocas_min"], medido["epocas_max"], medido["minimo_de_la_regla"]) == (9, 13, 17)
    assert medido["todos_por_debajo_del_minimo"] is True


def test_plan_de_intentos_semilla_presupuesto_y_repeticiones(_datasets):
    """Lo mismo que S10-S12, sobre la función que decide el plan."""
    todos, _ = _datasets
    por_nombre = {d.nombre: d for d in todos}
    pliegues = SimpleNamespace(pliegue_de=lambda repeticion, pliegue: SimpleNamespace(
        entrena=("a",), valida=("b",)))
    propuesta = SimpleNamespace(pliegues=pliegues)
    for nombre, reps, wall in (("kc2", 3, 120.0), ("wilt", 3, 300.0),
                               ("Allstate_Claims_Severity", 1, 600.0)):
        plan = p119.plan_de_intentos(por_nombre[nombre], propuesta, PROTOCOLO_V2, humo=False)
        assert len(plan) == 5 * reps
        assert {(i.repeticion, i.semilla) for i in plan} == {(r, r) for r in range(reps)}
        assert {i.presupuesto_wall_s for i in plan} == {wall}
    humo = p119.plan_de_intentos(por_nombre["kc2"], propuesta, PROTOCOLO_V2, humo=True)
    assert [(i.repeticion, i.pliegue) for i in humo] == [(0, 0)]


# ---------------------------------------------------------------------------
# 7. EL CACHÉ, POR main() (S4) Y LA FUSIÓN (I7)
# ---------------------------------------------------------------------------

def _claves(llamadas, dataset=None):
    return sorted((c["repeticion"], c["pliegue"]) for c in llamadas
                  if dataset is None or c["dataset"] == dataset)


def test_main_reintenta_un_failed_un_ausente_y_un_arff_distinto(tmp_path, monkeypatch):
    """(S4) `main()` reusa con `_reusable_c3`, no con `c3._reusable`: un
    `failed` con los digests buenos se vuelve a medir. Y también lo que falta
    del fichero y lo medido sobre otro ARFF."""
    salida = tmp_path / "r.json"
    primera: list = []
    _correr_main(monkeypatch, ["--solo", "kc2", "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.9, llamadas=primera))
    assert len(primera) == 15
    payload = json.loads(salida.read_text(encoding="utf-8"))
    for r in payload["resultados"]:
        if (r["repeticion"], r["pliegue"]) == (1, 3):
            r["estado"] = "failed"
        if (r["repeticion"], r["pliegue"]) == (0, 0):
            r["datos_sha256"] = "otro-arff"
    payload["resultados"] = [r for r in payload["resultados"]
                             if (r["repeticion"], r["pliegue"]) != (2, 4)]
    salida.write_text(json.dumps(payload), encoding="utf-8")
    segunda: list = []
    _correr_main(monkeypatch, ["--solo", "kc2", "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.9, llamadas=segunda))
    assert _claves(segunda) == [(0, 0), (1, 3), (2, 4)]
    final = json.loads(salida.read_text(encoding="utf-8"))
    assert final["n_reusados"] == 12 and len(final["resultados"]) == 15
    assert all(r["estado"] == "completed" for r in final["resultados"])


def test_la_salida_se_fusiona_con_la_que_ya_hay(tmp_path, monkeypatch):
    salida = tmp_path / "r.json"
    _correr_main(monkeypatch, ["--solo", "kc2", "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.9, llamadas=[]))
    _correr_main(monkeypatch, ["--solo", "dresses-sales", "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.6, llamadas=[]))
    payload = json.loads(salida.read_text(encoding="utf-8"))
    por_ds = {}
    for r in payload["resultados"]:
        por_ds[r["dataset"]] = por_ds.get(r["dataset"], 0) + 1
    assert por_ds == {"kc2": 15, "dresses-sales": 15}
    assert payload["n_registros_conservados_de_ejecuciones_anteriores"] == 15
    assert payload["registros_conservados_por_dataset"] == {"kc2": 15}
    assert [d["nombre"] for d in payload["datasets_declarados"]] == ["dresses-sales"]


def test_un_corte_a_media_noche_no_borra_lo_de_la_noche_anterior(tmp_path, monkeypatch):
    """LA NOCHE QUE SE PERDÍA: el fichero ya trae un conjunto medido; la
    ejecución siguiente guarda su primer punto de control y muere. El
    fichero tiene que seguir trayendo lo de antes."""
    salida = tmp_path / "r.json"
    _correr_main(monkeypatch, ["--solo", "yeast", "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.6, llamadas=[]))
    with pytest.raises(KeyboardInterrupt):
        _correr_main(monkeypatch, ["--solo", "kc2,dresses-sales", "--salida", str(salida)],
                     _motor_falso(lambda *a: 0.9, llamadas=[],
                                  revienta_en=("dresses-sales", 0, 1)))
    payload = json.loads(salida.read_text(encoding="utf-8"))
    por_ds = {}
    for r in payload["resultados"]:
        por_ds[r["dataset"]] = por_ds.get(r["dataset"], 0) + 1
    assert por_ds["yeast"] == 12
    assert por_ds["kc2"] == 15
    assert payload["parcial"] is True


def test_humo_solo_y_estimar_no_escriben_por_omision_en_el_resultado_real():
    real = p119.RUTA_DEL_RESULTADO.resolve()
    assert p119.ruta_de_salida("pasada", None) == real
    for tipo in ("solo", "humo", "estimar"):
        assert p119.ruta_de_salida(tipo, None) != real
        with pytest.raises(SystemExit, match="resultado real"):
            p119.ruta_de_salida(tipo, str(p119.RUTA_DEL_RESULTADO))


def test_salida_admite_una_ruta_absoluta_fuera_del_arbol(tmp_path, monkeypatch):
    fuera = tmp_path / "cola-nocturna" / "resultados" / "119-c3" / "r.json"
    assert p119.ruta_de_salida("pasada", str(fuera)) == fuera.resolve()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert p119.ruta_de_salida("solo", "~/x.json") == (tmp_path / "x.json").resolve()
    _correr_main(monkeypatch, ["--humo", "--salida", str(fuera)],
                 _motor_falso(lambda *a: 0.9, llamadas=[]))
    payload = json.loads(fuera.read_text(encoding="utf-8"))
    assert payload["tipo_de_ejecucion"] == "humo" and len(payload["resultados"]) == 2


def test_un_humo_no_pisa_un_resultado_real(tmp_path, monkeypatch):
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"corte": "119-C3", "tipo_de_ejecucion": "pasada",
                                "resultados": [_registro("kc2", NUEVO, 0.9)]}), encoding="utf-8")
    antes = hashlib.sha256(real.read_bytes()).hexdigest()
    for argv in (["--humo"], ["--solo", "kc2"]):
        llamadas: list = []
        with pytest.raises(SystemExit, match="no se mezclan"):
            _correr_main(monkeypatch, argv + ["--salida", str(real)],
                         _motor_falso(lambda *a: 0.9, llamadas=llamadas))
        assert llamadas == []
    assert hashlib.sha256(real.read_bytes()).hexdigest() == antes


def test_tipo_del_fichero_reconoce_los_de_antes_del_campo():
    assert p119.tipo_del_fichero({"es_humo": True}) == "humo"
    assert p119.tipo_del_fichero({"es_humo": False, "es_subconjunto_de_prueba": True}) == "solo"
    assert p119.tipo_del_fichero({"es_humo": False, "es_subconjunto_de_prueba": False}) == "pasada"
    assert p119.tipo_del_fichero({"sub_corte": "estimacion (no mide el corte)"}) == "estimar"


def test_fusionar_resultados_conserva_lo_no_tocado():
    previos = [_registro("a", NUEVO, 0.1, pliegue=p) for p in range(3)]
    nuevos = [_registro("a", NUEVO, 0.9, pliegue=1), _registro("b", NUEVO, 0.5)]
    fusion, conservados = p119.fusionar_resultados(previos, nuevos)
    assert [(r["dataset"], r["pliegue"], r["auroc"]) for r in fusion] == [
        ("a", 0, 0.1), ("a", 2, 0.1), ("a", 1, 0.9), ("b", 0, 0.5)]
    assert len(conservados) == 2


# ---------------------------------------------------------------------------
# 8. --estimar (I1)
# ---------------------------------------------------------------------------

def _medida(cubo, celdas, wall, *, carga=1.0, prep=0.1, estado="completed"):
    return {"cubo": cubo, "celdas": celdas, "wall_s": wall, "carga_s": carga,
            "preparacion_s": prep, "estado": estado}


def _estimar(medidas, conjuntos, **kw):
    return p119.estimar_desde_medidas(medidas, conjuntos, margen_s=30.0,
                                      ventana_nocturna_s=22500, ventana_de_dia_s=54000, **kw)


def test_estimar_interpola_por_celdas_y_usa_la_cota_por_encima():
    medidas = {"a": _medida("pequeno", 1000, 10.0), "b": _medida("pequeno", 3000, 30.0)}
    conjuntos = [{"nombre": n, "cubo": "pequeno", "celdas": c, "n_intentos": 15,
                  "presupuesto_wall_s": 120.0}
                 for n, c in (("a", 1000), ("b", 3000), ("medio", 2000), ("grande", 9000),
                              ("chico", 10))]
    por = {d["nombre"]: d for d in _estimar(medidas, conjuntos)["por_conjunto"]}
    assert por["a"]["fuente"] == "medido" and por["a"]["s_por_intento"] == 10.0
    assert por["medio"]["s_por_intento"] == 20.0
    assert por["grande"]["s_por_intento"] == 150.0 and por["grande"]["fuente"].startswith("cota")
    assert por["chico"]["s_por_intento"] == 10.0


def test_estimar_suma_la_preparacion_del_padre_y_nunca_pasa_del_techo():
    medidas = {"a": _medida("mediano", 1000, 400.0, carga=50.0, prep=5.0, estado="failed")}
    conjuntos = [{"nombre": "a", "cubo": "mediano", "celdas": 1000, "n_intentos": 10,
                  "presupuesto_wall_s": 300.0}]
    e = _estimar(medidas, conjuntos)
    # 50 de carga + 10 x (5 de preparación + 330 de techo, no 400)
    assert e["total_horas"] == round((50 + 10 * (5 + 330)) / 3600, 2)
    assert e["intentos_que_fallaron_al_estimar"] == ["a"]
    assert "aviso_de_fallos" in e


def test_estimar_dice_si_cabe_de_noche_y_cuantas_noches_con_un_proceso():
    medidas = {"a": _medida("grande", 10, 600.0, carga=0.0, prep=0.0)}
    conjuntos = [{"nombre": f"g{i}", "cubo": "grande", "celdas": 10, "n_intentos": 5,
                  "presupuesto_wall_s": 600.0} for i in range(9)]
    e = _estimar(medidas, conjuntos, ventana_de_dia_desde_ahora_s=3600)
    assert e["procesos"] == 1
    assert e["ventana_nocturna"]["cabe"] is False  # 45 x 600 s = 7,5 h > 6,25 h
    assert e["ventana_nocturna"]["noches_necesarias"] == 2
    assert e["de_dia"]["cabe"] is True and e["de_dia"]["cabe_desde_ahora"] is False
    assert not any("2_procesos" in k or "paralelo" in k for k in json.dumps(e).split('"'))


def test_cmd_estimar_mide_escribe_aparte_y_declara_los_cubos_sin_medida(tmp_path, monkeypatch):
    salida = tmp_path / "estimacion.json"
    llamadas: list = []
    _correr_main(monkeypatch, ["--estimar", "--estimar-conjuntos", "kc2,wilt",
                               "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.9, llamadas=llamadas))
    assert _claves(llamadas) == [(0, 0), (0, 0)]
    e = json.loads(salida.read_text(encoding="utf-8"))
    assert e["tipo_de_ejecucion"] == "estimar"
    assert set(e["medidas"]) == {"kc2", "wilt"}
    for m in e["medidas"].values():
        assert m["carga_s"] > 0 and m["preparacion_s"] > 0 and m["hilos"] == 4
    assert e["cubos_sin_medida"] == ["grande"]
    grandes = [d for d in e["estimacion"]["por_conjunto"] if d["cubo"] == "grande"]
    assert len(grandes) == 9 and all(d["fuente"].startswith("cota") for d in grandes)
    assert e["estimacion"]["procesos"] == 1
    assert "2_procesos" not in json.dumps(e)
    assert e["para_encolar"]["tope_s"] == 22500


def test_estimar_no_pisa_un_resultado_de_pasada(tmp_path, monkeypatch):
    real = tmp_path / "r.json"
    real.write_text(json.dumps({"tipo_de_ejecucion": "pasada", "resultados": []}), "utf-8")
    with pytest.raises(SystemExit, match="no es una estimación"):
        _correr_main(monkeypatch, ["--estimar", "--estimar-conjuntos", "kc2",
                                   "--salida", str(real)],
                     _motor_falso(lambda *a: 0.9, llamadas=[]))


def test_la_lista_de_la_estimacion_mide_mas_de_uno_por_cubo_y_los_anchos(_datasets):
    todos, _ = _datasets
    por_nombre = {d.nombre: d for d in todos}
    por_cubo: dict[str, list] = {}
    for n in p119.CONJUNTOS_DE_LA_ESTIMACION:
        assert not por_nombre[n].sellado
        por_cubo.setdefault(por_nombre[n].cubo, []).append(n)
    assert all(len(v) >= 2 for v in por_cubo.values()) and set(por_cubo) == {
        "pequeno", "mediano", "grande"}
    assert {"micro-mass", "mfeat-factors", "Internet-Advertisements"} <= set(
        p119.CONJUNTOS_DE_LA_ESTIMACION)


# ---------------------------------------------------------------------------
# 9. LA ENMIENDA 2: la pasada REAL la exige, con su cadena; las pruebas no
# ---------------------------------------------------------------------------

_RUTA_ENMIENDA_2 = _FASE0 / "protocolo_119_v4_enmienda_2.json"


class _Arranco(Exception):
    """La pasada real pasó de sus guardias: aquí se corta, antes de medir."""


def _arrancar_la_pasada_real(monkeypatch, salida: Path) -> list:
    """`main()` SIN --solo/--humo/--estimar, cortado en cuanto pasa de las
    guardias (la llamada siguiente, `c3.protocolo_registrado`, lanza
    `_Arranco`): nunca se mide nada."""
    def parar():
        raise _Arranco()
    monkeypatch.setattr(c3, "protocolo_registrado", parar)
    llamadas: list = []
    _correr_main(monkeypatch, ["--salida", str(salida)],
                 _motor_falso(lambda *a: 0.9, llamadas=llamadas))
    return llamadas


def _enmienda_2_rota(tmp_path: Path, caso: str) -> Path:
    e2 = json.loads(_RUTA_ENMIENDA_2.read_text(encoding="utf-8"))
    de = dict(e2["de"])
    if caso == "no_cita_el_v4":
        de["digest_sha256"] = "f" * 64
    elif caso == "no_cita_la_enmienda_1":
        de["enmienda_anterior"] = {**de["enmienda_anterior"], "digest_sha256": "e" * 64}
    e2 = _con_digest({**e2, "de": de})
    if caso == "digest_propio":
        e2["veredicto"] = "tocado DESPUÉS de registrarlo"
    ruta = tmp_path / "enmienda_2.json"
    ruta.write_text(json.dumps(e2, ensure_ascii=False), encoding="utf-8")
    return ruta


def test_la_cadena_de_la_enmienda_2_registrada_cuadra():
    cadena = p119.cadena_de_la_enmienda_2()
    assert cadena["presente"] is True and cadena["cuadra"] is True, cadena["problemas"]
    assert cadena["sha256_del_fichero"] == hashlib.sha256(_RUTA_ENMIENDA_2.read_bytes()).hexdigest()


@pytest.mark.parametrize("caso", ["ausente", "digest_propio", "no_cita_el_v4",
                                  "no_cita_la_enmienda_1"])
def test_la_pasada_real_se_niega_sin_la_enmienda_2_o_con_su_cadena_rota(tmp_path, monkeypatch,
                                                                        caso):
    ruta = tmp_path / "no_existe.json" if caso == "ausente" else _enmienda_2_rota(tmp_path, caso)
    monkeypatch.setattr(p119, "RUTA_DE_LA_ENMIENDA_2", ruta)
    salida = tmp_path / "real.json"
    with pytest.raises(SystemExit, match="enmienda 2"):
        _arrancar_la_pasada_real(monkeypatch, salida)
    assert not salida.exists()


def test_la_pasada_real_arranca_con_la_enmienda_2_registrada(tmp_path, monkeypatch):
    """El control: con la enmienda 2 de verdad, la pasada real pasa de la
    guardia (y se corta justo después)."""
    with pytest.raises(_Arranco):
        _arrancar_la_pasada_real(monkeypatch, tmp_path / "real.json")


def test_solo_humo_y_estimar_corren_sin_la_enmienda_2_y_lo_dicen(tmp_path, monkeypatch):
    monkeypatch.setattr(p119, "RUTA_DE_LA_ENMIENDA_2", tmp_path / "no_existe.json")
    salida = tmp_path / "solo.json"
    _correr_main(monkeypatch, ["--solo", "dresses-sales", "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.6, llamadas=[]))
    payload = json.loads(salida.read_text(encoding="utf-8"))
    assert payload["enmienda_2"]["presente"] is False and payload["enmienda_2"]["cuadra"] is False
    assert payload["procedencia"]["datos_de_entrada"]["protocolo_119_v4_enmienda_2"]["sha256"] is None
    estimacion = tmp_path / "estimacion.json"
    _correr_main(monkeypatch, ["--estimar", "--estimar-conjuntos", "kc2", "--salida",
                               str(estimacion)], _motor_falso(lambda *a: 0.9, llamadas=[]))
    assert json.loads(estimacion.read_text(encoding="utf-8"))["enmienda_2"]["presente"] is False


def test_la_enmienda_2_queda_en_la_procedencia_y_no_en_el_digest(tmp_path, monkeypatch):
    salida = tmp_path / "solo.json"
    _correr_main(monkeypatch, ["--solo", "dresses-sales", "--salida", str(salida)],
                 _motor_falso(lambda *a: 0.6, llamadas=[]))
    payload = json.loads(salida.read_text(encoding="utf-8"))
    sha = hashlib.sha256(_RUTA_ENMIENDA_2.read_bytes()).hexdigest()
    assert payload["procedencia"]["datos_de_entrada"]["protocolo_119_v4_enmienda_2"]["sha256"] == sha
    assert payload["enmienda_2"]["presente"] is True and payload["enmienda_2"]["cuadra"] is True
    assert not any("enmienda_2" in e for e in payload["digest_de_la_cache"]["componentes"])
    copia = tmp_path / "enmienda_2.json"
    copia.write_bytes(_RUTA_ENMIENDA_2.read_bytes())
    monkeypatch.setattr(p119, "RUTA_DE_LA_ENMIENDA_2", copia)
    antes = p119._digest_entorno()
    copia.write_text("{}", encoding="utf-8")
    assert p119._digest_entorno() == antes  # solo cambia cómo se CUENTA: no invalida la caché

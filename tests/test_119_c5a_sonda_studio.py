# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C5a — el ARNÉS de la sonda en condiciones de Studio (`sonda_studio_119_c5a.py`).

Nada aquí entrena nada de verdad: `main()` corre con un MOTOR FALSO
(`ejecutar_intento_aislado` sustituido) sobre `dresses-sales` (500 filas), y la
regla se prueba con registros fabricados donde el resultado se sabe a mano.

Lo que se prueba, por pieza (cada una tiene su sabotaje):

* los segundos por intento SALEN DEL STUDIO (reserva, pesos, reparto), no se
  escriben: cambiar la reserva del Studio cambia lo que la sonda da;
* los 8 sellados no entran (AL ARRANCAR, antes de cargar nada) y la guarda no
  vigila en vacío;
* la regla compone cada línea (completa / aprende / compite / cabe) con casos
  que cumplen y que no, en el borde, y un intento fallido/agotado/ausente
  cuenta como tal; su digest cambia con la regla y la pasada real lo exige;
* el pico de memoria es el del HIJO (un hijo real, no el del padre);
* el anclaje: la pasada real se niega con un árbol sucio, y el Studio es el
  tercer repo anclado;
* el caché: un `failed` no se reusa, la condición entra en la clave, hilos y
  presupuesto también, y `--forzar` ignora el caché.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
_FASE0 = _RAIZ / "benchmarks" / "fase0"
sys.path.insert(0, str(_FASE0))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

# El guion del WORKTREE, por su ruta (el `.pth` editable resuelve `matrixai` desde el árbol
# principal; aquí se carga el fichero que hay junto a este test, y se comprueba).
_ESPECIFICACION = importlib.util.spec_from_file_location(
    "sonda_studio_119_c5a_bajo_prueba", _FASE0 / "sonda_studio_119_c5a.py")
sonda = importlib.util.module_from_spec(_ESPECIFICACION)
sys.modules[_ESPECIFICACION.name] = sonda
_ESPECIFICACION.loader.exec_module(sonda)

import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as c3p  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402
from matrixai_engines.subproceso import motivo as motivo_real  # noqa: E402

TABM, DENSA = sonda.NOMBRE_TABM, sonda.NOMBRE_DENSA_DE_HOY
LGBM, HGB = "lightgbm", "sklearn.hgb"
METRICA = "auroc"


@pytest.fixture(scope="module")
def estudio_job():
    return sonda._estudio_job()


@pytest.fixture(scope="module")
def estudio(estudio_job):
    return sonda.condiciones_del_studio(estudio_job)


@pytest.fixture(scope="module")
def protocolo_y_todos():
    c6.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    return protocolo, c5.datasets_de_la_pasada(protocolo)


REGLA = protocolo_mod.ProtocoloExploratorio.cargar(c6.RUTA_DEL_PROTOCOLO_V2).regla_de_cierre


# ---------------------------------------------------------------------------
# 0. Se carga el guion del worktree
# ---------------------------------------------------------------------------

def test_el_guion_y_el_nucleo_que_se_prueban_son_los_del_worktree():
    assert Path(sonda.__file__).resolve() == (_FASE0 / "sonda_studio_119_c5a.py").resolve()
    import matrixai  # noqa: PLC0415

    assert Path(matrixai.__file__).resolve().is_relative_to(_RAIZ.resolve())


def test_el_ejecutor_real_guardado_es_el_de_subproceso():
    import matrixai_engines.subproceso as subproceso  # noqa: PLC0415

    assert sonda._EJECUTAR_INTENTO_AISLADO_REAL is subproceso.ejecutar_intento_aislado


# ---------------------------------------------------------------------------
# 1. LAS CONDICIONES SALEN DEL STUDIO
# ---------------------------------------------------------------------------

def test_los_segundos_por_intento_cuadran_a_mano_con_el_studio(estudio, estudio_job):
    r = estudio_job._RESERVA_POR_DEFECTO
    pesos = estudio_job._PESO_DE_BUSQUEDA
    total = sum(pesos[m] for m in estudio_job._MOTORES_PERMITIDOS)
    intentos = estudio["leido_del_codigo_del_studio"]["folds"] * \
        estudio["leido_del_codigo_del_studio"]["repeats"]
    a_mano = r.busqueda * pesos[DENSA] / total / intentos
    assert estudio["segundos_por_intento"][TABM] == pytest.approx(a_mano)
    assert estudio["segundos_por_intento"][DENSA] == pytest.approx(a_mano)
    assert estudio["segundos_por_intento"][TABM] == pytest.approx(22.4)  # el de C5.1, hoy
    assert estudio["segundos_por_intento"][LGBM] == pytest.approx(r.busqueda / total / intentos)
    assert estudio["segundos_del_ajuste_final"][TABM] == pytest.approx(
        r.ajuste_final * pesos[DENSA] / total)
    assert estudio["segundos_del_ajuste_final"][TABM] == pytest.approx(16.0)
    assert estudio["plazo_de_entrenamiento_s"][TABM] == pytest.approx(16.8)
    assert estudio["plazo_del_ajuste_final_s"][TABM] == pytest.approx(12.0)
    assert estudio["hilos"] == 1


def test_cambiar_la_reserva_del_studio_cambia_lo_que_da_la_sonda(estudio_job, monkeypatch):
    """Prueba de que los segundos SALEN del Studio y no están escritos: con
    otra reserva, otros segundos. Un 22,4 escrito a mano la dejaría en 22,4."""
    reserva = estudio_job._RESERVA_POR_DEFECTO
    otra = type(reserva)(busqueda=280.0, ajuste_final=40.0, calibracion=10.0, evaluacion=10.0)
    monkeypatch.setattr(estudio_job, "_RESERVA_POR_DEFECTO", otra)
    nuevo = sonda.condiciones_del_studio(estudio_job)
    assert nuevo["segundos_por_intento"][TABM] == pytest.approx(44.8)
    assert nuevo["segundos_del_ajuste_final"][TABM] == pytest.approx(32.0)


def test_cambiar_el_peso_de_la_densa_cambia_el_reparto_de_tabm(estudio_job, monkeypatch):
    pesos = dict(estudio_job._PESO_DE_BUSQUEDA, **{DENSA: 3.0})
    monkeypatch.setattr(estudio_job, "_PESO_DE_BUSQUEDA", pesos)
    nuevo = sonda.condiciones_del_studio(estudio_job)
    total = sum(pesos[m] for m in estudio_job._MOTORES_PERMITIDOS)
    assert nuevo["segundos_por_intento"][TABM] == pytest.approx(140.0 * 3.0 / total / 5)


def test_pliegues_hilos_y_reajuste_se_leen_del_codigo_del_studio(estudio):
    leido = estudio["leido_del_codigo_del_studio"]
    assert (leido["folds"], leido["repeats"], leido["hilos"]) == (5, 1, 1)
    assert leido["reajuste_en_un_proceso_aparte"] is True
    assert estudio["n_intentos_por_motor"] == 5
    assert estudio["fuente"]["sha256"] and estudio["fuente"]["fichero"].endswith("estudio_job.py")


_CODIGO_FABRICADO = '''
folds = int(payload.get("folds", {f}))
repeats = int(payload.get("repeats", {r}))
a = Presupuesto(wall_seconds=x[m],
                hilos={h}, seed=repeticion)
b = Presupuesto(wall_seconds=y, hilos={h}, seed=0)
_REAJUSTE_EN_UN_PROCESO_APARTE = True
'''


def test_lo_leido_del_codigo_sigue_al_codigo():
    texto = _CODIGO_FABRICADO.format(f=3, r=2, h=4)
    leido = sonda.lo_que_el_studio_da_por_omision(texto)
    assert (leido["folds"], leido["repeats"], leido["hilos"]) == (3, 2, 4)


def test_si_el_codigo_del_studio_discrepa_o_no_lo_dice_la_sonda_para():
    discrepa = _CODIGO_FABRICADO.format(f=3, r=2, h=4).replace("hilos=4, seed=0", "hilos=2, seed=0")
    with pytest.raises(SystemExit, match="DISTINTOS"):
        sonda.lo_que_el_studio_da_por_omision(discrepa)
    with pytest.raises(SystemExit, match="no encuentro"):
        sonda.lo_que_el_studio_da_por_omision("nada de esto")
    semilla_rara = _CODIGO_FABRICADO.format(f=5, r=1, h=1).replace("seed=0", "seed=7")
    with pytest.raises(SystemExit, match="siembra"):
        sonda.lo_que_el_studio_da_por_omision(semilla_rara)


def test_el_margen_y_la_fraccion_son_los_del_studio_y_del_motor(estudio, estudio_job):
    from matrixai_engines.motores import densa_tabm  # noqa: PLC0415

    assert sonda.MARGEN_S == estudio_job.MARGEN_POR_DEFECTO_SEGUNDOS
    assert estudio["margen_del_subproceso_s"] == estudio_job.MARGEN_POR_DEFECTO_SEGUNDOS
    assert estudio["fraccion_del_presupuesto_para_entrenar"][TABM] == \
        densa_tabm.FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR
    assert estudio["plazo_de_entrenamiento_s"][TABM] == pytest.approx(
        estudio["segundos_por_intento"][TABM] * densa_tabm.FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR)


def test_si_el_studio_ya_trae_tabm_la_sonda_para(estudio_job, monkeypatch):
    pesos = dict(estudio_job._PESO_DE_BUSQUEDA, **{TABM: 12.0})
    monkeypatch.setattr(estudio_job, "_PESO_DE_BUSQUEDA", pesos)
    with pytest.raises(SystemExit, match="ya trae"):
        sonda.condiciones_del_studio(estudio_job)


# ---------------------------------------------------------------------------
# 2. LOS SELLADOS NO SE TOCAN
# ---------------------------------------------------------------------------

def test_los_13_son_los_de_c54_y_ninguno_esta_sellado(protocolo_y_todos):
    _, todos = protocolo_y_todos
    datasets, subconjunto = sonda.datasets_de_c5a(todos)
    assert subconjunto is None and len(datasets) == 13
    assert [d.nombre for d in datasets] == [n for n, _ in sonda.CONJUNTOS_DE_C5A]
    assert not any(d.sellado for d in datasets)
    assert sum(1 for d in todos if d.sellado) == sonda.NUMERO_DE_SELLADOS_DEL_PROTOCOLO == 8


@pytest.mark.parametrize("sellado", ["pc3", "banknote-authentication", "diamonds", "letter",
                                     "kr-vs-kp", "splice", "Satellite", "pol"])
def test_cada_uno_de_los_8_sellados_se_niega_al_arrancar(protocolo_y_todos, sellado):
    _, todos = protocolo_y_todos
    with pytest.raises(SystemExit, match="AL ARRANCAR"):
        sonda.datasets_de_c5a(todos, solo=sellado)
    with pytest.raises(SystemExit, match="AL ARRANCAR"):
        sonda.datasets_de_c5a(todos, solo=f"dresses-sales,{sellado}")


def test_main_con_un_sellado_para_antes_de_cargar_nada(tmp_path, monkeypatch):
    llamadas: list = []

    def centinela(*a, **kw):
        llamadas.append(kw.get("dataset"))
        raise AssertionError("se llegó al motor con un sellado")

    def no_cargar(*a, **kw):
        raise AssertionError("se cargó un conjunto con un sellado en la lista")

    monkeypatch.setattr(sonda, "ejecutar_intento_aislado", centinela)
    monkeypatch.setattr(c5, "particiones_base", no_cargar)
    salida = tmp_path / "solo.json"
    with pytest.raises(SystemExit, match="AL ARRANCAR"):
        sonda.main(["--solo", "pc3", "--salida", str(salida)])
    assert llamadas == [] and not salida.exists()


def test_la_guarda_no_vigila_en_vacio_si_el_protocolo_no_declara_sellados():
    """Un `sellado` que dejara de leerse (todos False) dejaría pasar a cualquiera."""
    sin_sellados = [SimpleNamespace(nombre=n, sellado=False) for n, _ in sonda.CONJUNTOS_DE_C5A]
    with pytest.raises(SystemExit, match="sin saber cuáles son"):
        sonda.exigir_que_ningun_sellado_entre(["dresses-sales"], sin_sellados)


def test_la_segunda_guarda_para_antes_de_cargar_el_conjunto(protocolo_y_todos, estudio, monkeypatch):
    protocolo, todos = protocolo_y_todos
    sellado = next(d for d in todos if d.sellado)

    def no_cargar(*a, **kw):
        raise AssertionError("se cargó un sellado")

    monkeypatch.setattr(c5, "particiones_base", no_cargar)
    with pytest.raises(SystemExit, match="SELLADO"):
        sonda.medir_conjunto(sellado, protocolo, estudio, metric_id="auroc", pliegues=(0,),
                             con_curva=False, con_reajuste=False, cache_previo={}, forzar=False,
                             entorno_digest="x", procedencia={"datos_de_entrada": {}}, motores={})


def test_solo_fuera_de_los_13_se_niega(protocolo_y_todos):
    _, todos = protocolo_y_todos
    with pytest.raises(SystemExit, match="no son de los 13"):
        sonda.datasets_de_c5a(todos, solo="kc2")


# ---------------------------------------------------------------------------
# 3. LA REGLA
# ---------------------------------------------------------------------------

def _reg(dataset, motor, condicion="S", pliegue=0, valor=0.9, *, estado="completed", **extra):
    r = {"dataset": dataset, "motor": motor, "condicion": condicion, "repeticion": 0,
         "pliegue": pliegue, "estado": estado, METRICA: valor,
         "plazo_antes_de_la_primera_epoca_completa": False, "epocas_completas": 5, "mejor_epoca": 3,
         "pico_hijo_mb": 500.0, "checkpoint_json_bytes": 1_000_000}
    r.update(extra)
    return r


def _campo(nombres, *, tabm=0.90, densa=0.88, lgbm=0.89, hgb=0.89, pliegues=(0, 1), **cambios):
    """Los cuatro motores de S en cada conjunto y pliegue. `cambios`: por conjunto,
    un dict de overrides por motor."""
    registros = []
    for d in nombres:
        valores = {TABM: tabm, DENSA: densa, LGBM: lgbm, HGB: hgb}
        valores.update(cambios.get(d, {}))
        for motor, v in valores.items():
            for p in pliegues:
                registros.append(_reg(d, motor, pliegue=p, valor=v))
    return registros


def _veredicto(registros, nombres):
    return sonda.componer_veredicto(registros, nombres=nombres,
                                    metrica_por_dataset={n: METRICA for n in nombres},
                                    regla_de_cierre=REGLA)


_DIEZ = [f"ds{i}" for i in range(10)]


def test_un_campo_donde_tabm_gana_en_todo_cumple_las_cuatro_lineas():
    v = _veredicto(_campo(_DIEZ), _DIEZ)
    assert v["completa"]["numero"] == "20/20" and v["completa"]["cumple"]
    assert v["aprende"]["numero"] == "10/10" and v["aprende"]["cumple"]
    assert v["compite"]["cumplidos_de_la_cartera"] == "10/10" and v["compite"]["cumple"]
    assert v["compite"]["conjuntos_perdidos_contra_la_densa_de_hoy"] == 0
    assert v["cabe"]["cumple"] and v["cumple_las_cuatro"]


@pytest.mark.parametrize("fallan, cumple", [(2, True), (3, False)])
def test_completa_en_el_borde_del_90_por_ciento(fallan, cumple):
    registros = _campo(_DIEZ)
    quitados = 0
    for r in registros:
        if r["motor"] == TABM and quitados < fallan:
            r["estado"] = "failed"
            quitados += 1
    v = _veredicto(registros, _DIEZ)
    assert v["completa"]["numero"] == f"{20 - fallan}/20"
    assert v["completa"]["cumple"] is cumple


@pytest.mark.parametrize("como", ["failed", "agotado", "ausente", "sin_metrica", "cancelled"])
def test_un_intento_que_no_completa_cuenta_como_tal_aunque_los_demas_vayan_bien(como):
    nombres = _DIEZ[:5]
    registros = _campo(nombres)
    objetivo = next(r for r in registros if r["motor"] == TABM and r["dataset"] == "ds0"
                    and r["pliegue"] == 0)
    if como == "ausente":
        registros.remove(objetivo)
    elif como == "sin_metrica":
        objetivo[METRICA] = None
    elif como == "agotado":
        objetivo.update(estado="failed", clasificacion_del_fallo="agotado_por_el_techo",
                        **{METRICA: None})
    else:
        objetivo["estado"] = como
    v = _veredicto(registros, nombres)
    assert v["completa"]["numero"] == "9/10"
    caido = v["completa"]["intentos_que_no_completan"]
    assert [(c["dataset"], c["pliegue"]) for c in caido] == [("ds0", 0)]
    # y TabM no cumple la cartera en ese conjunto (un fallo es un conjunto perdido)
    assert v["compite"]["cumplidos_de_la_cartera"] == "4/5"
    assert v["aprende"]["por_conjunto"]["ds0"]["aprende"] is False


def test_medido_cuenta_completed_budget_limited_y_no_un_failed():
    ok = _reg("a", TABM, estado="completed_budget_limited")
    assert sonda._medido(ok, METRICA)
    assert not sonda._medido(_reg("a", TABM, estado="failed"), METRICA)
    assert not sonda._medido(None, METRICA)


@pytest.mark.parametrize("sin_epoca, cumple", [(1, True), (2, False)])
def test_aprende_en_el_borde_del_80_por_ciento(sin_epoca, cumple):
    nombres = _DIEZ[:5]
    registros = _campo(nombres)
    marcados = 0
    for r in registros:
        if r["motor"] == TABM and r["pliegue"] == 1 and marcados < sin_epoca:
            r["plazo_antes_de_la_primera_epoca_completa"] = True
            r["epocas_completas"] = 0
            marcados += 1
    v = _veredicto(registros, nombres)
    assert v["aprende"]["numero"] == f"{5 - sin_epoca}/5"
    assert v["aprende"]["cumple"] is cumple


def test_compite_no_cumple_si_tabm_esta_a_mas_de_dos_puntos_del_mejor():
    nombres = _DIEZ[:5]
    # lightgbm 3 puntos por encima de TabM en TODOS los conjuntos
    v = _veredicto(_campo(nombres, tabm=0.87, densa=0.80, lgbm=0.90, hgb=0.86), nombres)
    assert v["compite"]["cumplidos_de_la_cartera"] == "0/5"
    assert not v["compite"]["cumple"] and not v["cumple_las_cuatro"]


def test_compite_en_el_borde_de_dos_puntos_de_la_cartera():
    """(Sin el 2,000 exacto: 0,90 - 0,88 en coma flotante da 2,0000000000000018, y eso es de la
    función del protocolo, no de esta sonda.)"""
    nombres = _DIEZ[:5]
    justo = _veredicto(_campo(nombres, tabm=0.8805, densa=0.80, lgbm=0.90, hgb=0.86), nombres)
    fuera = _veredicto(_campo(nombres, tabm=0.8795, densa=0.80, lgbm=0.90, hgb=0.86), nombres)
    assert justo["compite"]["cumplidos_de_la_cartera"] == "5/5"
    assert fuera["compite"]["cumplidos_de_la_cartera"] == "0/5"


@pytest.mark.parametrize("pierde_en, cumple", [(2, True), (3, False)])
def test_compite_exige_no_perder_contra_la_densa_de_hoy_en_mas_de_dos(pierde_en, cumple):
    nombres = _DIEZ[:10]
    # TabM cumple la cartera en todos (a <= 2 puntos del mejor), pero en `pierde_en` conjuntos la
    # densa le gana por 0,5 puntos.
    cambios = {n: {DENSA: 0.905, TABM: 0.90, LGBM: 0.90, HGB: 0.90} for n in nombres[:pierde_en]}
    v = _veredicto(_campo(nombres, **cambios), nombres)
    assert v["compite"]["cumple_la_cartera"] is True
    assert v["compite"]["conjuntos_perdidos_contra_la_densa_de_hoy"] == pierde_en
    assert v["compite"]["cumple_contra_la_densa"] is cumple and v["compite"]["cumple"] is cumple


def test_si_la_densa_tiene_un_intento_sin_medida_no_hay_perdida_comparable():
    nombres = _DIEZ[:4]
    registros = _campo(nombres)
    for r in registros:
        if r["motor"] == DENSA and r["dataset"] == "ds0" and r["pliegue"] == 1:
            r["estado"] = "failed"
    v = _veredicto(registros, nombres)
    d0 = v["compite"]["contra_la_densa_de_hoy"]["ds0"]
    assert d0["pierde"] is False and "no comparable" in d0["motivo"]


def test_si_tabm_falla_donde_la_densa_completa_pierde_contra_ella():
    nombres = _DIEZ[:4]
    registros = _campo(nombres)
    for r in registros:
        if r["motor"] == TABM and r["dataset"] == "ds0" and r["pliegue"] == 0:
            r["estado"] = "failed"
    v = _veredicto(registros, nombres)
    assert v["compite"]["contra_la_densa_de_hoy"]["ds0"]["pierde"] is True
    assert v["compite"]["conjuntos_perdidos_contra_la_densa_de_hoy"] == 1


@pytest.mark.parametrize("dataset, pico, cumple", [
    ("dresses-sales", 2048.0, True), ("dresses-sales", 2049.0, False),
    ("wilt", 3584.0, True), ("wilt", 3585.0, False), ("wilt", 2500.0, True)])
def test_cabe_con_los_topes_distintos_de_los_ejemplos_y_del_resto(dataset, pico, cumple):
    registros = _campo([dataset])
    for r in registros:
        if r["motor"] == TABM:
            r["pico_hijo_mb"] = pico
    assert _veredicto(registros, [dataset])["cabe"]["cumple"] is cumple


def test_cabe_un_pico_sin_medir_no_cabe_y_el_reajuste_cuenta():
    registros = _campo(["wilt"])
    registros.append(_reg("wilt", TABM, "reajuste", 0, pico_hijo_mb=None))
    cabe = _veredicto(registros, ["wilt"])["cabe"]
    assert cabe["cumple"] is False and cabe["sin_pico_medido"][0]["condicion"] == "reajuste"
    registros[-1]["pico_hijo_mb"] = 9000.0
    cabe = _veredicto(registros, ["wilt"])["cabe"]
    assert cabe["cumple"] is False and cabe["pasan_del_tope_de_memoria"][0]["condicion"] == "reajuste"


def test_cabe_ignora_la_curva_que_no_es_la_del_studio():
    registros = _campo(["wilt"])
    registros.append(_reg("wilt", TABM, "S+120", 0, pico_hijo_mb=99999.0))
    assert _veredicto(registros, ["wilt"])["cabe"]["cumple"] is True


def test_cabe_el_checkpoint_solo_cuenta_en_los_ejemplos():
    grande = 21 * 1024 * 1024
    ejemplo = _campo(["pc1"])
    resto = _campo(["wilt"])
    for lista in (ejemplo, resto):
        for r in lista:
            if r["motor"] == TABM:
                r["checkpoint_json_bytes"] = grande
    assert _veredicto(ejemplo, ["pc1"])["cabe"]["cumple"] is False
    assert _veredicto(resto, ["wilt"])["cabe"]["cumple"] is True


def test_la_consecuencia_y_la_regla_viajan_en_el_veredicto():
    v = _veredicto(_campo(_DIEZ[:2]), _DIEZ[:2])
    assert set(v["regla"]) == {"completa", "aprende", "compite", "cabe"}
    assert "C5 se detiene" in v["consecuencia"]


def test_el_digest_de_la_regla_es_estable_y_cambia_con_la_regla(monkeypatch):
    a = sonda.digest_de_la_regla()
    assert a == sonda.digest_de_la_regla() and len(a) == 64
    cambiada = json.loads(json.dumps(sonda.REGLA_C5A))
    cambiada["completa"]["fraccion_minima"] = 0.5
    monkeypatch.setattr(sonda, "REGLA_C5A", cambiada)
    assert sonda.digest_de_la_regla() != a


def test_la_pasada_real_exige_el_digest_registrado():
    actual = sonda.digest_de_la_regla()
    with pytest.raises(SystemExit, match="exige --regla-registrada"):
        sonda.exigir_la_regla_registrada(None, "pasada")
    with pytest.raises(SystemExit, match="no coincide"):
        sonda.exigir_la_regla_registrada("0" * 64, "pasada")
    assert sonda.exigir_la_regla_registrada(actual, "pasada")["coincide"] is True
    assert sonda.exigir_la_regla_registrada(None, "solo")["coincide"] is False  # no es la real


# ---------------------------------------------------------------------------
# 4. EL REGISTRO DE UN INTENTO
# ---------------------------------------------------------------------------

def _intento(estado="completed", *, motivo_es=None, config=None, valor=0.9, traza=None):
    return SimpleNamespace(
        estado=estado, informe={"metrics": [{"metric_id": "auroc", "value": valor}]}
        if estado.startswith("completed") else None,
        recursos={"wall_seconds": 3.5, "cpu_seconds": 3.4}, traza=traza,
        motivo_del_estado={"es": motivo_es, "en": "x"} if motivo_es else None,
        config_efectiva=config, engine_version="1.1.0+falso", pipeline_digest="pd")


_CONFIG_TABM = {
    "entrenamiento_efectivo": {
        "epocas_ejecutadas": 7, "epocas_completas": 6, "mejor_epoca": 4, "parado_por_plazo": True,
        "parado_por_paciencia": False, "ultima_epoca_parcial": True,
        "plazo_antes_de_la_primera_epoca_completa": False,
        "plazo_de_entrenamiento_segundos": 16.8, "mejor_perdida_de_validacion": 0.3},
    "arquitectura": {"k": 8, "d_block": 256},
    "pesos": {"a": {"forma": [10, 20], "base64": "AAAA"}, "b": {"forma": [5], "base64": "BB"}}}


class _PicoFijo:
    pico_mb, pico_suma_rss_mb = 812.5, 900.0


def _registro(intento, **kw):
    ds = SimpleNamespace(nombre="dresses-sales", data_id=1, cubo="pequeno", tarea="binary_classification",
                         sellado=False)
    base = dict(ds=ds, condicion="S", motor_nombre=TABM, pliegue=0, wall_s=22.4, semilla=0,
                metric_id="auroc", intento=intento, transcurrido_s=21.0, preparacion_s=0.4,
                pico=_PicoFijo(), hilos=1, entorno_digest="e", datos_sha256="d",
                procedencia_id="p", split_plan_digest="s")
    base.update(kw)
    return sonda.registro_de_un_intento(**base)


def test_el_registro_lleva_todo_lo_que_pide_c54():
    r = _registro(_intento(config=_CONFIG_TABM))
    assert r["estado"] == "completed" and r["wall_s"] == 21.0 and r["tiempo_de_ajuste_s"] == 3.5
    assert (r["epocas_ejecutadas"], r["epocas_completas"], r["mejor_epoca"]) == (7, 6, 4)
    assert r["parado_por_plazo"] is True
    assert r["plazo_antes_de_la_primera_epoca_completa"] is False
    assert r["n_parametros"] == 10 * 20 + 5
    assert r["checkpoint_json_bytes"] == len(json.dumps(_CONFIG_TABM, ensure_ascii=False,
                                                        default=str).encode())
    assert r["pico_hijo_mb"] == 812.5 and r["metrica_de_cierre"] == "auroc" and r["auroc"] == 0.9
    assert (r["hilos"], r["presupuesto_wall_s"], r["techo_externo_s"]) == (1, 22.4, 22.4 + sonda.MARGEN_S)
    assert r["clasificacion_del_fallo"] is None and r["condicion"] == "S"
    assert "pesos" not in json.dumps({k: v for k, v in r.items() if k != "checkpoint_json_bytes"})


def test_los_parametros_no_se_inventan_si_el_predictor_no_trae_formas():
    assert sonda.n_parametros_de({"mxai": "programa"}) is None
    assert sonda.n_parametros_de(None) is None
    assert sonda.n_parametros_de({"pesos": {"a": "texto"}}) is None


def test_un_intento_fallido_se_registra_sin_metrica_y_con_su_clasificacion():
    texto = motivo_real("intento_agoto_el_tiempo", valor=52.4)["es"]
    r = _registro(_intento("failed", motivo_es=texto, traza="Traceback"))
    assert r["estado"] == "failed" and r["clasificacion_del_fallo"] == "agotado_por_el_techo"
    assert r.get("auroc") is None and r["checkpoint_json_bytes"] is None and r["traza"] == "Traceback"


@pytest.mark.parametrize("clave, valor, esperada", [
    ("intento_agoto_el_tiempo", 52.4, "agotado_por_el_techo"),
    ("intento_sin_resultado", -9, "murio_sin_resultado"),
    ("intento_fallo_no_declarado_en_subproceso", "ValueError: x", "excepcion_no_declarada")])
def test_los_prefijos_de_los_fallos_son_los_de_los_motores(clave, valor, esperada):
    assert sonda.clasificar_el_fallo("failed", motivo_real(clave, valor=valor)["es"]) == esperada


def test_clasificar_el_fallo_otros_casos():
    assert sonda.clasificar_el_fallo("completed", None) is None
    assert sonda.clasificar_el_fallo("completed_budget_limited", "x") is None
    assert sonda.clasificar_el_fallo("cancelled", None) == "cancelado"
    assert sonda.clasificar_el_fallo("failed", "el motor dijo no") == "fallo_declarado_por_el_motor"


# ---------------------------------------------------------------------------
# 5. EL PICO DE MEMORIA ES EL DEL HIJO
# ---------------------------------------------------------------------------

def test_el_pico_es_el_del_hijo_y_sin_hijo_no_hay_dato():
    with sonda._PicoDelHijo(cada_s=0.05) as sin_hijo:
        _ocupa = b"x" * (200 * 1024 * 1024)  # el PADRE ocupa 200 MB: no cuenta
        import time  # noqa: PLC0415

        time.sleep(0.3)
    assert sin_hijo.pico_mb is None and sin_hijo.pico_suma_rss_mb is None
    del _ocupa
    with sonda._PicoDelHijo(cada_s=0.05) as con_hijo:
        proceso = subprocess.Popen([sys.executable, "-c",
                                    "import time; x=b'x'*(120*1024*1024); time.sleep(1.5)"])
        proceso.wait()
    assert con_hijo.pico_mb is not None and 100.0 <= con_hijo.pico_mb < 400.0
    assert con_hijo.pico_suma_rss_mb >= con_hijo.pico_mb * 0.9


# ---------------------------------------------------------------------------
# 6. EL PLAN DE UN PLIEGUE
# ---------------------------------------------------------------------------

def test_el_plan_del_pliegue_0_lleva_s_reajuste_y_curva(estudio):
    plan = sonda.plan_de_un_pliegue(estudio, 0)
    s = [p for p in plan if p["condicion"] == "S"]
    assert [p["motor"] for p in s] == [TABM, DENSA, LGBM, HGB]
    assert [p["wall_s"] for p in s] == pytest.approx([22.4, 22.4, 140 / 15 / 5, 140 / 15 / 5])
    reajuste = [p for p in plan if p["condicion"] == "reajuste"]
    assert [p["wall_s"] for p in reajuste] == pytest.approx([16.0, 16.0, 20 / 15, 20 / 15])
    curva = [p for p in plan if p["condicion"].startswith("S+")]
    assert [(p["condicion"], p["motor"], p["wall_s"]) for p in curva] == [
        ("S+60", TABM, 60.0), ("S+120", TABM, 120.0)]
    assert len(plan) == 10


def test_el_pliegue_1_no_reajusta_y_sin_curva_solo_queda_s(estudio):
    assert [p["condicion"] for p in sonda.plan_de_un_pliegue(estudio, 1)].count("reajuste") == 0
    solo_s = sonda.plan_de_un_pliegue(estudio, 1, con_curva=False, con_reajuste=False)
    assert {p["condicion"] for p in solo_s} == {"S"} and len(solo_s) == 4


# ---------------------------------------------------------------------------
# 7. EL CACHÉ
# ---------------------------------------------------------------------------

def _previo(**cambios):
    r = _registro(_intento(config=_CONFIG_TABM), entorno_digest="E")
    r["motor_digest"] = sonda.digest_del_motor(TABM)
    r.update(cambios)
    return r


def _usa(previo, **kw):
    base = dict(entorno_digest="E", motor_nombre=TABM, wall_s=22.4, datos_sha256="d", hilos=1,
                margen_s=sonda.MARGEN_S)
    base.update(kw)
    return sonda.reusable(previo, base.pop("entorno_digest"), base.pop("motor_nombre"),
                          base.pop("wall_s"), **base)


def test_un_registro_bueno_del_mismo_codigo_se_reusa():
    assert _usa(_previo()) is True
    assert _usa(_previo(estado="completed_budget_limited")) is True


@pytest.mark.parametrize("estado", ["failed", "cancelled"])
def test_un_registro_que_no_es_medida_no_se_reusa_aunque_los_digests_cuadren(estado):
    assert _usa(_previo(estado=estado)) is False


@pytest.mark.parametrize("cambio", [
    {"entorno_digest": "otro"}, {"motor_digest": "otro"}, {"presupuesto_wall_s": 60.0},
    {"datos_sha256": "otros"}, {"hilos": 4}, {"margen_s": 99.0}])
def test_cualquier_cambio_de_condicion_invalida_el_registro(cambio):
    assert _usa(_previo(**cambio)) is False


def test_el_caso_none_no_se_reusa():
    assert _usa(None) is False
    assert _usa(_previo(), datos_sha256=None) is False


def test_la_condicion_entra_en_la_clave_y_los_reintentos_se_apuntan():
    s = _reg("a", TABM, "S", 0)
    s60 = _reg("a", TABM, "S+60", 0)
    assert sonda._clave(s) != sonda._clave(s60)
    previo_fallido = dict(s, estado="failed", motivo="agotado")
    nuevo_bueno = dict(s)
    en_fichero, reintentados = sonda.fusionar([previo_fallido, s60], [nuevo_bueno])
    assert len(en_fichero) == 2 and sonda._clave(s60) in {sonda._clave(r) for r in en_fichero}
    assert [r["estado"] for r in reintentados] == ["failed"]
    # un previo BUENO que se vuelve a medir (--forzar) no es un reintento
    assert sonda.fusionar([s], [dict(s)])[1] == []


def test_cargar_la_salida_previa_no_mezcla_tipos(tmp_path):
    ruta = tmp_path / "x.json"
    ruta.write_text(json.dumps({"tipo_de_ejecucion": "humo", "resultados": []}))
    with pytest.raises(SystemExit, match="no se mezclan"):
        sonda.cargar_salida_previa(ruta, "pasada")
    ruta.write_text(json.dumps({"tipo_de_ejecucion": "solo", "resultados": [_reg("a", TABM)]}))
    cache, payload = sonda.cargar_salida_previa(ruta, "solo")
    assert list(cache) == [("a", TABM, "S", 0, 0)] and payload["tipo_de_ejecucion"] == "solo"


# ---------------------------------------------------------------------------
# 8. EL ANCLAJE
# ---------------------------------------------------------------------------

def test_la_pasada_real_se_niega_con_un_arbol_sucio():
    sucia = {"anclable": False, "avisos": ["matrixaistudio: ARBOL SUCIO al medir"]}
    with pytest.raises(SystemExit, match="NO ANCLABLE"):
        sonda.exigir_el_anclaje(sucia, "pasada")
    sonda.exigir_el_anclaje(sucia, "solo")  # avisa, no para
    sonda.exigir_el_anclaje({"anclable": True, "avisos": []}, "pasada")


def test_el_studio_es_el_tercer_repositorio_anclado_y_se_restaura_la_lista(protocolo_y_todos):
    _, todos = protocolo_y_todos
    antes = dict(c3._RUTAS_DE_REPOSITORIO)
    dresses = [d for d in todos if d.nombre == "dresses-sales"]
    p = sonda.procedencia_de_la_medicion(digests={"entorno": "x"}, datasets=dresses)
    assert set(p["repositorios"]) == {"matrixAI", "matrixai-engines", "matrixaistudio"}
    assert p["repositorios"]["matrixaistudio"]["commit"]
    assert c3._RUTAS_DE_REPOSITORIO == antes
    assert "dresses-sales" in p["datos_de_entrada"] and p["datos_de_entrada"]["dresses-sales"]["sha256"]


def test_un_estudio_sucio_vuelve_la_procedencia_no_anclable(protocolo_y_todos, monkeypatch):
    _, todos = protocolo_y_todos
    original = c3._estado_del_repositorio
    raiz_studio = sonda._raiz_del_studio()

    def estado(raiz):
        e = original(raiz)
        if Path(raiz).resolve() == raiz_studio:
            e = dict(e, arbol_sucio=True, ficheros_modificados=["frontend/x.tsx"])
        return e

    monkeypatch.setattr(c3, "_estado_del_repositorio", estado)
    p = sonda.procedencia_de_la_medicion(digests={"entorno": "x"},
                                         datasets=[d for d in todos if d.nombre == "dresses-sales"])
    assert p["anclable"] is False and any("matrixaistudio" in a for a in p["avisos"])
    with pytest.raises(SystemExit, match="NO ANCLABLE"):
        sonda.exigir_el_anclaje(p, "pasada")


def test_main_de_la_pasada_real_con_arbol_sucio_para_antes_de_medir(tmp_path, monkeypatch):
    llamadas: list = []
    monkeypatch.setattr(sonda, "ejecutar_intento_aislado",
                        lambda *a, **kw: llamadas.append(1) or (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(sonda, "procedencia_de_la_medicion",
                        lambda **kw: {"anclable": False, "avisos": ["matrixAI: ARBOL SUCIO"],
                                      "procedencia_id": "p", "datos_de_entrada": {},
                                      "digests_de_codigo": {}})
    monkeypatch.setattr(sonda, "componentes_del_digest_del_entorno", lambda: {"x": "y"})
    # la pasada real, con el digest registrado, pero sin medir: se niega por el árbol
    with pytest.raises(SystemExit, match="NO ANCLABLE"):
        sonda.main(["--regla-registrada", sonda.digest_de_la_regla(), "--salida",
                    str(tmp_path / "pasada.json")])
    assert llamadas == []


# ---------------------------------------------------------------------------
# 9. main DE PUNTA A PUNTA CON MOTOR FALSO
# ---------------------------------------------------------------------------

_VALOR = {TABM: 0.95, DENSA: 0.90, LGBM: 0.93, HGB: 0.93}


def _ejecutor_falso(llamadas, *, falla=frozenset()):
    def ejecutar(motor, train, validation, test, spec, presupuesto, *, candidate, split_plan_digest,
                 dataset, pliegue, repeticion, margen_segundos=None, **_):
        condicion = candidate.rsplit("-c5a-", 1)[-1]
        llamadas.append({"motor": motor.nombre, "condicion": condicion, "pliegue": pliegue,
                         "wall_seconds": presupuesto.wall_seconds, "hilos": presupuesto.hilos,
                         "seed": presupuesto.seed, "margen": margen_segundos})
        if (motor.nombre, condicion, pliegue) in falla:
            return SimpleNamespace(
                estado="failed", informe=None, recursos=None, config_efectiva=None, traza="tb",
                motivo_del_estado=motivo_real("intento_agoto_el_tiempo", valor=52.4),
                engine_version=None, pipeline_digest=None)
        return SimpleNamespace(
            estado="completed",
            informe={"metrics": [{"metric_id": m, "value": _VALOR[motor.nombre]}
                                 for m in ("auroc", "accuracy")]},
            recursos={"wall_seconds": 1.0, "cpu_seconds": 1.0}, traza=None, motivo_del_estado=None,
            config_efectiva=dict(_CONFIG_TABM), engine_version="1.1.0+falso",
            pipeline_digest=f"pd-{motor.nombre}")
    return ejecutar


@pytest.fixture
def _sin_anclaje_ni_digest_pesado(monkeypatch):
    """El digest del entorno (hash de ~260 ficheros) no es lo que se prueba aquí."""
    monkeypatch.setattr(sonda, "componentes_del_digest_del_entorno", lambda: {"x": "y"})


def _correr(monkeypatch, salida, ejecutor, *argv):
    monkeypatch.setattr(sonda, "ejecutar_intento_aislado", ejecutor)
    sonda.main([*argv, "--salida", str(salida)])


def test_main_solo_un_conjunto_mide_lo_que_dice_con_los_presupuestos_del_studio(
        tmp_path, monkeypatch, _sin_anclaje_ni_digest_pesado, estudio):
    salida = tmp_path / "solo.json"
    llamadas: list = []
    _correr(monkeypatch, salida, _ejecutor_falso(llamadas), "--solo", "dresses-sales")
    # 2 pliegues x (4 S + 2 curva) + 4 reajustes (pliegue 0) = 16
    assert len(llamadas) == 16
    s = [c for c in llamadas if c["condicion"] == "S"]
    assert len(s) == 8
    for c in llamadas:
        assert c["hilos"] == 1 and c["margen"] == sonda.MARGEN_S
    tabm_s = [c for c in s if c["motor"] == TABM]
    assert {c["wall_seconds"] for c in tabm_s} == {estudio["segundos_por_intento"][TABM]}
    assert {c["wall_seconds"] for c in s if c["motor"] == LGBM} == \
        {estudio["segundos_por_intento"][LGBM]}
    assert sorted({c["wall_seconds"] for c in llamadas if c["condicion"].startswith("S+")}) == [60.0, 120.0]
    assert {c["wall_seconds"] for c in llamadas if c["condicion"] == "reajuste"
            and c["motor"] == TABM} == {16.0}
    assert {c["seed"] for c in llamadas if c["condicion"] == "reajuste"} == {0}

    payload = json.loads(salida.read_text())
    assert payload["tipo_de_ejecucion"] == "solo" and payload["parcial"] is False
    assert payload["condiciones_del_studio"]["segundos_por_intento"][TABM] == pytest.approx(22.4)
    assert payload["regla"]["digest"] == sonda.digest_de_la_regla()
    assert payload["procedencia"]["repositorios"]["matrixaistudio"]["commit"]
    assert len(payload["resultados"]) == 16 and payload["n_reusados"] == 0
    v = payload["veredicto"]
    assert v["completa"]["numero"] == "2/2" and v["compite"]["cumplidos_de_la_cartera"] == "1/1"
    assert set(v) >= {"completa", "aprende", "compite", "cabe", "cumple_las_cuatro"}
    assert "dresses-sales" in payload["cuadro_por_conjunto"]
    assert payload["digest_resultados_crudos"] and payload["digest_solo_de_resultados"]
    # los pesos NO viajan, solo su tamaño
    assert all("pesos" not in r for r in payload["resultados"])


def test_main_humo_es_un_pliegue_sin_curva_y_con_reajuste(tmp_path, monkeypatch,
                                                           _sin_anclaje_ni_digest_pesado):
    salida = tmp_path / "humo.json"
    llamadas: list = []
    _correr(monkeypatch, salida, _ejecutor_falso(llamadas), "--humo")
    assert [c["condicion"] for c in llamadas] == ["S"] * 4 + ["reajuste"] * 4
    assert {c["pliegue"] for c in llamadas} == {0}
    assert json.loads(salida.read_text())["tipo_de_ejecucion"] == "humo"


def test_main_relanzar_reusa_y_un_fallo_se_reintenta_y_se_apunta(
        tmp_path, monkeypatch, _sin_anclaje_ni_digest_pesado):
    salida = tmp_path / "solo.json"
    primera: list = []
    _correr(monkeypatch, salida, _ejecutor_falso(primera, falla={(TABM, "S", 1)}),
            "--solo", "dresses-sales")
    p1 = json.loads(salida.read_text())
    assert p1["veredicto"]["completa"]["numero"] == "1/2"
    assert p1["veredicto"]["completa"]["intentos_que_no_completan"][0]["clasificacion"] == \
        "agotado_por_el_techo"
    segunda: list = []
    _correr(monkeypatch, salida, _ejecutor_falso(segunda), "--solo", "dresses-sales")
    # el único intento que se vuelve a medir es el fallido: lo bueno se reusa, el `failed` NO
    assert [(c["motor"], c["condicion"], c["pliegue"]) for c in segunda] == [(TABM, "S", 1)]
    p2 = json.loads(salida.read_text())
    assert p2["n_reusados"] == 15 and p2["veredicto"]["completa"]["numero"] == "2/2"
    assert p2["n_intentos_reintentados"] == 1
    assert p2["intentos_reintentados"][0]["clasificacion_del_fallo"] == "agotado_por_el_techo"
    assert p2["intentos_reintentados"][0]["registro_sustituido"]["estado"] == "failed"


def test_main_forzar_ignora_el_cache_entero(tmp_path, monkeypatch, _sin_anclaje_ni_digest_pesado):
    salida = tmp_path / "solo.json"
    _correr(monkeypatch, salida, _ejecutor_falso([]), "--solo", "dresses-sales")
    otra: list = []
    _correr(monkeypatch, salida, _ejecutor_falso(otra), "--solo", "dresses-sales", "--forzar")
    assert len(otra) == 16
    sin: list = []
    _correr(monkeypatch, salida, _ejecutor_falso(sin), "--solo", "dresses-sales")
    assert sin == []


def test_main_no_pisa_el_resultado_real_con_una_prueba(monkeypatch):
    monkeypatch.setattr(sonda, "ejecutar_intento_aislado", _ejecutor_falso([]))
    with pytest.raises(SystemExit, match="no escribe en el resultado real"):
        sonda.main(["--solo", "dresses-sales", "--salida", str(sonda.RUTA_DEL_RESULTADO)])


# ---------------------------------------------------------------------------
# 10. --estimar Y LA ORDEN DE ENCOLADO
# ---------------------------------------------------------------------------

def _medida(**cambios):
    m = {"wall_s": 20.0, "wall_s_ligero": 5.0, "carga_s": 10.0, "preparacion_s": 1.0,
         "parado_por_plazo": True, "estado": "completed"}
    m.update(cambios)
    return m


def test_la_estimacion_se_puede_comprobar_a_mano(estudio):
    e = sonda.estimar_desde_medidas({"a": _medida()}, estudio)
    # por pliegue: 2*20 + 2*5 + curva + 4*1 ; curva (parado por plazo): para 60 s, 20 + 0,75*(60-22,4)
    # = 48,2 ; para 120 s, 20 + 0,75*97,6 = 93,2 -> 141,4 ; por pliegue = 40 + 10 + 141,4 + 4 = 195,4
    # reajuste: 2*min(20, 16+30) + 2*min(5, 1,333+30) = 40 + 10 = 50 ; total = 10 + 2*195,4 + 50 = 450,8
    assert e["total_s"] == pytest.approx(450.8, abs=0.1)
    assert e["procesos"] == 1 and e["total_horas_cota"] >= e["total_horas"]


def test_si_tabm_converge_en_s_la_curva_no_cuesta_mas(estudio):
    parado = sonda.estimar_desde_medidas({"a": _medida(parado_por_plazo=True)}, estudio)
    converge = sonda.estimar_desde_medidas({"a": _medida(parado_por_plazo=False)}, estudio)
    assert converge["total_s"] < parado["total_s"]


def test_la_orden_de_encolado_lleva_la_regla_y_los_tres_commits(monkeypatch):
    monkeypatch.setattr(c3p, "_commits_de_ahora", lambda: {
        "matrixAI": {"sha": "aaaaaaa", "sin_commitear": False},
        "matrixai-engines": {"sha": "bbbbbbb", "sin_commitear": False}})
    monkeypatch.setattr(c3p, "fin_de_la_suite_nocturna_medido",
                        lambda: {"medido": False, "desde": "01:45", "fuente": "x", "motivo": "x"})
    orden = sonda.para_encolar(9700, memoria="8G", digest_de_la_regla_="d" * 64)
    texto = orden["de_dia"]["encolar"]
    assert "~/encolar.sh 01-119-c5a" in texto and " 8G matrixAI " in texto
    assert f"--regla-registrada {'d' * 64}" in texto and "ESTIMADA_S=9700" in texto
    assert "sonda_studio_119_c5a.py" in texto and "COMMITS=matrixAI=aaaaaaa,matrixai-engines=bbbbbbb," \
        "matrixaistudio=" in texto
    assert " 13300 " in texto  # el tope: max(9700 x 1,25, 9700 + 3600), redondeado a 100 s


def test_estimar_con_motor_falso_escribe_un_fichero_de_estimacion(
        tmp_path, monkeypatch, _sin_anclaje_ni_digest_pesado):
    salida = tmp_path / "estimar.json"
    llamadas: list = []
    monkeypatch.setattr(c3p, "_commits_de_ahora", lambda: {
        "matrixAI": {"sha": "a", "sin_commitear": False},
        "matrixai-engines": {"sha": "b", "sin_commitear": False}})
    _correr(monkeypatch, salida, _ejecutor_falso(llamadas), "--estimar",
            "--estimar-conjuntos", "dresses-sales,climate-model-simulation-crashes")
    # un TabM y un lightgbm por conjunto, con el motor falso: nada real
    assert [c["motor"] for c in llamadas] == [TABM, LGBM, TABM, LGBM]
    assert all(c["hilos"] == 1 for c in llamadas)
    payload = json.loads(salida.read_text())
    assert payload["tipo_de_ejecucion"] == "estimar" and len(payload["conjuntos_medidos"]) == 2
    assert payload["estimacion"]["total_horas"] > 0 and "para_encolar" in payload

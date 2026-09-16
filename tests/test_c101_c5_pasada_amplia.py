# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C5 — la validación amplia: que el guion mida LOS 40 y que la regla de
cierre use la métrica de CADA TAREA.

Lo que estas pruebas vigilan, y por qué cada una:

1. **Que lo ya commiteado no se mueva.** `aplicar_regla_de_cierre` tenía que
   crecer para aceptar una métrica por dataset, y los artefactos de 101-C3 ya
   escritos cuadran con su `digest_canonico`. Sin los parámetros nuevos, la
   función devuelve el MISMO diccionario que antes — clave a clave, sin una de
   más.
2. **Que el denominador no encoja en silencio.** Es el defecto que C5 existe
   para cerrar: con una sola métrica sobre las tres tareas, 20 de los 40
   datasets se caen de la cuenta y el veredicto sigue diciendo CUMPLE.
3. **Que la derivación de los 40 diga lo mismo que la lista a mano de C3.**
   Cambiar un dato escrito por uno calculado sin comprobar que dicen lo mismo
   es cambiarlo, no derivarlo.
4. **Que los ARFF de los 8 sellados existan y se puedan leer.** Un sellado
   ilegible descubierto a las 40 horas de pasada es exactamente lo que este
   paso existe para evitar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ / "benchmarks" / "fase0") not in sys.path:
    sys.path.insert(0, str(RAIZ / "benchmarks" / "fase0"))

from benchmarks.fase0.protocolo import (ProtocoloExploratorio,  # noqa: E402
                                        aplicar_regla_de_cierre,
                                        estabilidad_del_ganador,
                                        metrica_del_dataset)

RUTA_PROTOCOLO = RAIZ / "benchmarks" / "fase0" / "protocolo_exploratorio.json"
ARFF_DIR = Path.home() / "fase0_openml_datos" / "arff"


@pytest.fixture(scope="module")
def protocolo() -> ProtocoloExploratorio:
    return ProtocoloExploratorio.cargar(RUTA_PROTOCOLO)


def _registros(dataset: str, metrica: str, valores: dict[str, float],
               n_pliegues: int = 5, estado: str = "completed") -> list[dict]:
    """Registros crudos como los que escribe la pasada: un motor por clave de
    `valores`, `n_pliegues` pliegues de la repetición 0, todos con el mismo
    número. Sirve para razonar sobre el denominador sin depender de que los
    motores reales den un valor u otro."""
    salida = []
    for pliegue in range(n_pliegues):
        for motor, valor in valores.items():
            registro = {"dataset": dataset, "motor": motor, "estado": estado,
                        "repeticion": 0, "pliegue": pliegue}
            registro[metrica] = valor
            salida.append(registro)
    return salida


# ---------------------------------------------------------------------------
# 1. Lo ya commiteado no se mueve
# ---------------------------------------------------------------------------

def test_sin_los_parametros_nuevos_el_veredicto_es_identico(protocolo):
    """LA PRUEBA QUE PROTEGE LA EVIDENCIA YA ESCRITA.

    `aplicar_regla_de_cierre` creció dos parámetros. Los tres JSON de 101-C3
    commiteados cuadran con su `digest_canonico`, que se calcula sobre un
    documento que INCLUYE este veredicto: una clave nueva añadida sin querer
    haría que una re-pasada de C3 ya no reprodujera su propio digest, y eso no
    se nota hasta que alguien intenta reproducirlo.

    Se comprueba el conjunto EXACTO de claves, no que las de antes sigan
    estando: `set(a) <= set(b)` lo pasa un diccionario con diez claves de más.
    """
    resultados = (_registros("kc2", "auroc", {"lightgbm": 0.90, "catboost": 0.80})
                  + _registros("sick", "auroc", {"lightgbm": 0.95, "catboost": 0.99}))
    viejo = aplicar_regla_de_cierre(resultados, protocolo.regla_de_cierre,
                                    motor="lightgbm")
    esperadas = {
        "motor", "metrica", "puntos_exigidos",
        "intervalos_de_los_que_NO_cumplen_que_cruzan_el_liston",
        "n_medidas_sin_pliegue_declarado", "fraccion_minima", "datasets",
        "cumplidos", "fraccion", "cumple_la_regla", "aciertos_por_ser_el_mejor",
        "datasets_que_puede_perder_sin_incumplir",
        "datasets_que_le_faltan_para_cumplir", "definicion_de_mejor", "detalle"}
    assert set(viejo) == esperadas, (
        f"la llamada SIN los parametros nuevos tiene que devolver exactamente las "
        f"claves de siempre. Sobran {sorted(set(viejo) - esperadas)}, faltan "
        f"{sorted(esperadas - set(viejo))}")
    claves_detalle = {
        "dataset", "cumple", "perdido_por_fallo", "mejor", "distancia_en_puntos",
        "segundo", "ventaja_sobre_el_segundo_en_puntos", "intervalo_de_la_distancia",
        "motores_que_compitieron"}
    for entrada in viejo["detalle"]:
        assert set(entrada) == claves_detalle, (
            f"el detalle de {entrada['dataset']} tiene claves de mas o de menos: "
            f"{sorted(set(entrada) ^ claves_detalle)}")


def test_los_tres_artefactos_de_c3_siguen_cuadrando_con_su_digest():
    """Los JSON de 101-C3 son evidencia sellada. Si un cambio en `protocolo.py`
    los rompiera, se notaría aquí y no al publicarlos.

    No recomputa el veredicto: comprueba que el documento guardado sigue
    cuadrando con el digest que lleva dentro, que es lo único que ata ese
    fichero a sí mismo.
    """
    from matrixai.estudio.validacion import digest_canonico
    directorio = RAIZ / "benchmarks" / "fase0"
    ficheros = sorted(directorio.glob("pasada_exploratoria_101_c3*.json"))
    assert ficheros, "no hay artefactos de C3 que comprobar; la prueba no tendria dientes"
    for ruta in ficheros:
        payload = json.loads(ruta.read_text(encoding="utf-8"))
        guardado = payload.pop("digest_resultados_crudos", None)
        assert guardado is not None, f"{ruta.name} no lleva digest"
        assert digest_canonico(payload) == guardado, (
            f"{ruta.name} ya NO cuadra con su propio digest: algo ha reescrito "
            f"evidencia sellada")


# ---------------------------------------------------------------------------
# 2. El denominador que encogía solo
# ---------------------------------------------------------------------------

def test_una_sola_metrica_sobre_tres_tareas_PIERDE_datasets_del_denominador(protocolo):
    """EL DEFECTO, ESCRITO COMO PRUEBA. Documenta lo medido el 2026-09-14: con
    `metrica="auroc"` sobre datasets de las tres tareas, los de multiclase y
    regresión no aparecen ni para bien ni para mal.

    Está aquí en positivo —afirmando el comportamiento viejo— para que si
    alguien «arregla» la ruta por omisión, esta prueba se ponga roja y obligue
    a mirar los artefactos de C3 que dependen de ella.
    """
    resultados = (_registros("kc2", "auroc", {"lightgbm": 0.90, "catboost": 0.80})
                  + _registros("yeast", "accuracy", {"lightgbm": 0.60, "catboost": 0.70})
                  + _registros("Moneyball", "r2", {"lightgbm": 0.88, "catboost": 0.90}))
    v = aplicar_regla_de_cierre(resultados, protocolo.regla_de_cierre, motor="lightgbm")
    assert v["datasets"] == 1, (
        "tres datasets entran y con una sola metrica solo se cuenta uno: si esto "
        "cambia, el veredicto de C3 cambia con ello")
    assert v["cumple_la_regla"] is True
    assert [d["dataset"] for d in v["detalle"]] == ["kc2"]


def test_con_la_metrica_de_cada_tarea_los_tres_datasets_CUENTAN(protocolo):
    """La reparación: el mismo caso, con el mapa por tarea, cuenta los tres."""
    mapa = {"kc2": "auroc", "yeast": "accuracy", "Moneyball": "r2"}
    resultados = (_registros("kc2", "auroc", {"lightgbm": 0.90, "catboost": 0.80})
                  + _registros("yeast", "accuracy", {"lightgbm": 0.60, "catboost": 0.70})
                  + _registros("Moneyball", "r2", {"lightgbm": 0.88, "catboost": 0.90}))
    v = aplicar_regla_de_cierre(resultados, protocolo.regla_de_cierre,
                                motor="lightgbm", metrica_por_dataset=mapa,
                                datasets_exigidos=sorted(mapa))
    assert v["datasets"] == 3, "los tres datasets tienen que entrar en el denominador"
    assert {d["dataset"] for d in v["detalle"]} == {"kc2", "yeast", "Moneyball"}
    # lightgbm gana kc2 (distancia 0), pierde yeast por 10 puntos y Moneyball por 2.
    por_ds = {d["dataset"]: d for d in v["detalle"]}
    assert por_ds["kc2"]["cumple"] is True
    assert por_ds["yeast"]["cumple"] is False
    assert por_ds["yeast"]["distancia_en_puntos"] == pytest.approx(10.0)
    assert por_ds["Moneyball"]["distancia_en_puntos"] == pytest.approx(2.0)
    assert v["cumplidos"] == 2 and v["fraccion"] == pytest.approx(2 / 3)
    assert v["cumple_la_regla"] is False
    # y cada entrada dice CON QUE se midio, que es lo que permite leerla
    assert por_ds["Moneyball"]["metrica"] == "r2"
    assert por_ds["yeast"]["metrica"] == "accuracy"


def test_un_fallo_ajeno_ya_no_mueve_el_denominador(protocolo):
    """La otra mitad del defecto, y la que lo hacía indetectable: con una sola
    métrica, un dataset de otra tarea entraba en el denominador SOLO si alguien
    se había caído en él. El denominador dependía de una avería."""
    base = (_registros("kc2", "auroc", {"lightgbm": 0.90, "catboost": 0.80})
            + _registros("Moneyball", "r2", {"lightgbm": 0.88, "catboost": 0.90}))
    sin_fallo = aplicar_regla_de_cierre(base, protocolo.regla_de_cierre, motor="lightgbm")
    con_fallo = aplicar_regla_de_cierre(
        base + [{"dataset": "Moneyball", "motor": "lightgbm", "estado": "failed",
                 "repeticion": 0, "pliegue": 9}],
        protocolo.regla_de_cierre, motor="lightgbm")
    assert sin_fallo["datasets"] == 1 and con_fallo["datasets"] == 2, (
        "comportamiento viejo: el denominador cambia segun haya fallo o no")

    mapa = {"kc2": "auroc", "Moneyball": "r2"}
    exigidos = sorted(mapa)
    a = aplicar_regla_de_cierre(base, protocolo.regla_de_cierre, motor="lightgbm",
                                metrica_por_dataset=mapa, datasets_exigidos=exigidos)
    b = aplicar_regla_de_cierre(
        base + [{"dataset": "Moneyball", "motor": "lightgbm", "estado": "failed",
                 "repeticion": 0, "pliegue": 9}],
        protocolo.regla_de_cierre, motor="lightgbm",
        metrica_por_dataset=mapa, datasets_exigidos=exigidos)
    assert a["datasets"] == b["datasets"] == 2, (
        "con el mapa por tarea, el denominador son los datasets declarados y no "
        "depende de que alguien se haya caido")


def test_un_dataset_exigido_sin_una_sola_medida_APARECE_diciendo_por_que(protocolo):
    """Un aserto negativo lo pasa un `{}`: sin `datasets_exigidos`, un dataset
    en el que NINGÚN motor produjo la métrica sigue cayéndose de la cuenta,
    porque no aparece en ningún registro con valor."""
    mapa = {"kc2": "auroc", "diamonds": "r2"}
    resultados = _registros("kc2", "auroc", {"lightgbm": 0.90, "catboost": 0.80})
    v = aplicar_regla_de_cierre(resultados, protocolo.regla_de_cierre, motor="lightgbm",
                                metrica_por_dataset=mapa, datasets_exigidos=sorted(mapa))
    assert v["datasets"] == 2, "el dataset exigido y no medido tiene que contar"
    por_ds = {d["dataset"]: d for d in v["detalle"]}
    assert por_ds["diamonds"]["cumple"] is False
    assert por_ds["diamonds"]["sin_medida"], (
        "tiene que decir POR QUE no hay numero, no solo que no cumple")
    assert "diamonds" in v["datasets_exigidos_que_no_se_midieron"]
    assert por_ds["kc2"]["sin_medida"] is None


def test_la_frase_de_lectura_dice_que_cada_tarea_se_midio_con_lo_suyo(protocolo):
    """La advertencia se REDACTA con los números medidos. Un «32/40 CUMPLE» de
    tres tareas medidas cada una con su métrica y uno de tres tareas medidas
    todas con AUROC se escriben igual si nadie lo dice."""
    from benchmarks.fase0.protocolo import veredicto_con_su_alcance
    mapa = {"kc2": "auroc", "yeast": "accuracy", "Moneyball": "r2"}
    resultados = (_registros("kc2", "auroc", {"lightgbm": 0.90, "catboost": 0.80})
                  + _registros("yeast", "accuracy", {"lightgbm": 0.60, "catboost": 0.70})
                  + _registros("Moneyball", "r2", {"lightgbm": 0.88, "catboost": 0.90}))
    v = veredicto_con_su_alcance(
        protocolo, resultados, motor="lightgbm",
        motores_declarados=["baseline", "lightgbm", "catboost"],
        datasets_declarados=sorted(mapa), metrica_por_dataset=mapa,
        datasets_exigidos=sorted(mapa))
    frase = v["como_hay_que_leer_este_numero"]
    assert "metrica de SU tarea" in frase
    for metric_id in ("auroc", "accuracy", "r2"):
        assert metric_id in frase, f"la frase no nombra {metric_id}: {frase}"
    assert v["metrica_por_dataset"] == mapa


def test_la_estabilidad_del_ganador_tambien_lee_la_metrica_de_cada_tarea():
    """Pasar el mapa al veredicto y no a la estabilidad daría dos bloques del
    mismo objeto medidos con métricas distintas, presentados como si dijeran lo
    mismo."""
    mapa = {"kc2": "auroc", "Moneyball": "r2"}
    resultados = (_registros("kc2", "auroc", {"lightgbm": 0.90, "catboost": 0.80})
                  + _registros("Moneyball", "r2", {"lightgbm": 0.88, "catboost": 0.90}))
    sin_mapa = estabilidad_del_ganador(resultados)
    con_mapa = estabilidad_del_ganador(resultados, metrica_por_dataset=mapa)
    assert sin_mapa["n_datasets"] == 1, "con auroc para todos, Moneyball no se ve"
    assert con_mapa["n_datasets"] == 2
    assert con_mapa["metrica_por_dataset"] == mapa
    por_ds = {d["dataset"]: d for d in con_mapa["detalle"]}
    assert por_ds["Moneyball"]["mejor_por_media"] == "catboost"


def test_metrica_del_dataset_distingue_las_tres_ausencias():
    """Sin mapa, la de siempre. Con mapa, la suya. Y un dataset que el mapa no
    nombra devuelve `None` — que el llamante declara, no se traga. Un valor
    ausente no es un cero ni el que a uno le convenga."""
    assert metrica_del_dataset("lo_que_sea", "auroc", None) == "auroc"
    assert metrica_del_dataset("kc2", "auroc", {"kc2": "r2"}) == "r2"
    assert metrica_del_dataset("otro", "auroc", {"kc2": "r2"}) is None


# ---------------------------------------------------------------------------
# 3. Los 40, derivados y no escritos a mano
# ---------------------------------------------------------------------------

def test_la_pasada_declara_los_40_del_protocolo_con_los_8_sellados(protocolo):
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    datasets = c5.datasets_de_la_pasada(protocolo)
    assert len(datasets) == 40
    assert {d.nombre for d in datasets} == {d.nombre for d in protocolo.datasets}
    sellados = [d.nombre for d in datasets if d.sellado]
    assert len(sellados) == 8, f"los sellados tienen que entrar en C5: {sellados}"
    # y los data_id sellados son los que la auditoria del 101 dejo por escrito
    assert sorted(d.data_id for d in datasets if d.sellado) == [
        3, 6, 46, 201, 1050, 1462, 40900, 42225]
    # las tres tareas y los tres cubos, con las cuentas del protocolo
    import collections
    por_tarea = collections.Counter(d.tarea for d in datasets)
    por_cubo = collections.Counter(d.cubo for d in datasets)
    assert dict(por_tarea) == {"binary_classification": 20,
                               "multiclass_classification": 10, "regression": 10}
    assert dict(por_cubo) == {"pequeno": 15, "mediano": 15, "grande": 10}


def test_el_criterio_del_subconjunto_dice_que_NO_hay_recorte():
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    criterio = c5.CRITERIO_DEL_SUBCONJUNTO
    assert "40" in criterio and "SELLADOS" in criterio.upper()
    # y no se queda en una promesa: la lista se deriva, no se escribe
    assert "deriva" in criterio


def test_la_metrica_de_cierre_sale_del_protocolo_y_cambia_con_la_tarea(protocolo):
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    datasets = c5.datasets_de_la_pasada(protocolo)
    mapa = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    assert len(mapa) == 40
    por_tarea = {d.tarea: mapa[d.nombre] for d in datasets}
    assert por_tarea["binary_classification"] == "auroc"
    assert por_tarea["multiclass_classification"] == "accuracy"
    assert por_tarea["regression"] == "r2"
    # las tres metricas del catalogo de 105-C1 existen de verdad
    from matrixai.estudio.metricas import REGISTRO
    for metric_id in set(mapa.values()):
        assert metric_id in REGISTRO, (
            f"{metric_id} no esta en el catalogo de metricas: la pasada mediria "
            f"con un id que nadie calcula")


def test_una_metrica_registrada_que_no_se_sabe_traducir_PARA_la_pasada(protocolo):
    """Un valor de repuesto aquí es el denominador que encoge solo, por otro
    camino: el dataset se quedaría sin métrica y se caería de la cuenta."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    from dataclasses import replace
    regla_rara = replace(
        protocolo.regla_de_cierre,
        metrica_por_tarea={**protocolo.regla_de_cierre.metrica_por_tarea,
                           "regression": "una_metrica_que_nadie_ha_implementado"})
    otro = replace(protocolo, regla_de_cierre=regla_rara)
    datasets = c5.datasets_de_la_pasada(protocolo)
    with pytest.raises(SystemExit) as exc:
        c5.metrica_de_cierre_por_dataset(otro, datasets)
    assert "una_metrica_que_nadie_ha_implementado" in str(exc.value)


def test_la_regresion_NO_pide_ni_clases_ni_clase_positiva(protocolo):
    """`ProblemSpec` levanta si a una regresión se le pasan clases. Que esta
    prueba exista es lo que impide que alguien «unifique» las tres ramas."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    datasets = {d.nombre: d for d in c5.datasets_de_la_pasada(protocolo)}

    regresion = datasets["Moneyball"]
    regresion.clases, regresion.positiva = None, None
    spec = c5.problem_spec_de(regresion, "RS", ("a", "b"))
    assert spec.task == "regression" and spec.classes is None
    assert spec.positive_label is None

    multiclase = datasets["yeast"]
    multiclase.clases, multiclase.positiva = ("CYT", "ERL", "EXC"), "ERL"
    spec = c5.problem_spec_de(multiclase, "class", ("a", "b"))
    assert spec.task == "multiclass_classification"
    assert spec.positive_label is None, (
        "en multiclase no hay clase positiva; poner la minoritaria haria que el "
        "informe publicara sensibilidad/especificidad de esa contra el resto")

    binaria = datasets["kc2"]
    binaria.clases, binaria.positiva = ("no", "yes"), "yes"
    spec = c5.problem_spec_de(binaria, "problems", ("a", "b"))
    assert spec.positive_label == "yes"


def test_las_repeticiones_del_cubo_grande_salen_del_protocolo(protocolo):
    """El cubo grande corre 1 repetición y no 3, y ese número está registrado
    (con su motivo) en `DisenoDeParticion`, no escrito en el guion."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    datasets = c5.datasets_de_la_pasada(protocolo)
    plan = c5.plan_de_la_pasada(datasets, protocolo)
    assert plan["por_cubo"]["grande"]["repeticiones"] == 1
    assert plan["por_cubo"]["pequeno"]["repeticiones"] == 3
    assert plan["por_cubo"]["mediano"]["repeticiones"] == 3
    # la cuenta entera, a mano: 15x5x3x7 + 15x5x3x7 + 10x5x1x7
    assert plan["por_cubo"]["pequeno"]["n_intentos"] == 15 * 5 * 3 * 7
    assert plan["por_cubo"]["mediano"]["n_intentos"] == 15 * 5 * 3 * 7
    assert plan["por_cubo"]["grande"]["n_intentos"] == 10 * 5 * 1 * 7
    assert plan["n_intentos"] == 3500
    # y los topes de pared son los registrados, en segundos
    assert plan["por_cubo"]["pequeno"]["wall_seconds_por_intento"] == 120.0
    assert plan["por_cubo"]["mediano"]["wall_seconds_por_intento"] == 300.0
    assert plan["por_cubo"]["grande"]["wall_seconds_por_intento"] == 600.0
    # la cota de peor caso, tambien a mano: (1575*120 + 1575*300 + 350*600)/3600
    assert plan["horas_de_reloj_cota_peor_caso"] == pytest.approx(
        (1575 * 120 + 1575 * 300 + 350 * 600) / 3600.0, abs=0.01)


def test_la_regresion_NO_se_estratifica_y_la_clasificacion_SI():
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    assert c5.ESTRATIFICACION["regression"] is False
    assert c5.ESTRATIFICACION["binary_classification"] is True
    assert c5.ESTRATIFICACION["multiclass_classification"] is True
    # y el motivo viaja con los NUMEROS medidos, no como una opinion
    criterio = c5.CRITERIO_DE_LA_ESTRATIFICACION
    assert "Moneyball" in criterio and "house_prices_nominal" in criterio
    assert "13,84" in criterio and "407" in criterio


# ---------------------------------------------------------------------------
# 4. Los datos: los 40 ARFF, los 8 sellados incluidos
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARFF_DIR.exists(), reason="sin los ARFF descargados")
def test_los_40_arff_estan_y_su_sha256_es_el_REGISTRADO(protocolo):
    """No basta con que el fichero exista: el protocolo registró el sha256 de
    cada ARFF antes de medir nada, y un dataset sustituido con el mismo nombre
    es un dataset distinto midiendo lo mismo."""
    import hashlib
    faltan, distintos = [], []
    for ds in protocolo.datasets:
        ruta = ARFF_DIR / f"{ds.data_id}.arff"
        if not ruta.exists():
            faltan.append(f"{ds.nombre} ({ds.data_id})")
            continue
        huella = hashlib.sha256()
        with ruta.open("rb") as fichero:
            for trozo in iter(lambda: fichero.read(1 << 20), b""):
                huella.update(trozo)
        if huella.hexdigest() != ds.sha256_arff:
            distintos.append(f"{ds.nombre}: {huella.hexdigest()[:12]} != "
                             f"{ds.sha256_arff[:12]}")
    assert not faltan, f"faltan ARFF, y 8 de ellos son sellados: {faltan}"
    assert not distintos, f"ARFF que no son los registrados: {distintos}"


@pytest.mark.skipif(not ARFF_DIR.exists(), reason="sin los ARFF descargados")
def test_los_8_sellados_se_LEEN_y_traen_su_objetivo_declarado(protocolo):
    """LA PRUEBA QUE JUSTIFICA ESTE PASO. Los ocho sellados no se han abierto
    nunca a propósito, así que «se leen» era una suposición heredada, no un
    dato — y descubrirlo a las 40 horas de pasada cuesta las 40 horas."""
    import lector_arff
    sellados = [d for d in protocolo.datasets if d.sellado]
    assert len(sellados) == 8
    for ds in sellados:
        leido = lector_arff.cargar(ARFF_DIR / f"{ds.data_id}.arff",
                                   objetivo_declarado=ds.columna_objetivo,
                                   n_columnas_declaradas=ds.n_columnas)
        assert leido.objetivo == ds.columna_objetivo, (
            f"{ds.nombre}: el lector devuelve el objetivo {leido.objetivo!r} y el "
            f"catalogo registro {ds.columna_objetivo!r}")
        assert len(leido.filas) == ds.n_filas, (
            f"{ds.nombre}: {len(leido.filas)} filas leidas frente a {ds.n_filas} "
            f"registradas")
        con_objetivo = [f for f in leido.filas if f[leido.objetivo] is not None]
        assert con_objetivo, f"{ds.nombre}: ninguna fila trae objetivo"


@pytest.mark.skipif(not ARFF_DIR.exists(), reason="sin los ARFF descargados")
def test_la_minoria_derivada_reproduce_los_doce_pares_de_c3(protocolo):
    """LA COMPROBACIÓN QUE CONVIERTE «derivar» EN «derivar», y no en «cambiar».

    C3 lleva sus doce clases positivas escritas a mano. C5 las calcula. Que las
    dos formas digan lo mismo NO se puede suponer: si no coincidieran, C5
    estaría midiendo otra cosa que C3 y la comparación entre las dos pasadas
    no valdría, sin que nada lo dijera.
    """
    import lector_arff
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    from benchmarks.fase0.pasada_exploratoria_101_c3 import DATASETS as DOCE

    por_id = {d.data_id: d for d in protocolo.datasets}
    discrepancias = []
    for data_id, nombre, _cubo, positiva, negativa in DOCE:
        entrada = por_id[data_id]
        leido = lector_arff.cargar(ARFF_DIR / f"{data_id}.arff",
                                   objetivo_declarado=entrada.columna_objetivo,
                                   n_columnas_declaradas=entrada.n_columnas)
        filas = [f for f in leido.filas if f[leido.objetivo] is not None]
        clases, medida = c5.clase_positiva_medida(filas, leido.objetivo)
        # el ORDEN tambien: en tres de los doce (climate, PhishingWebsites,
        # Internet-Advertisements) ordenar alfabeticamente da otro orden que el
        # que C3 declaro, y dos pasadas que se van a comparar no pueden
        # diferir en `spec.classes` sin que nadie lo haya medido.
        if medida != positiva or clases != (negativa, positiva):
            discrepancias.append(
                f"{nombre}: a mano ({negativa}, {positiva}), medido {clases} "
                f"con positiva {medida}")
    assert not discrepancias, (
        "la derivacion NO reproduce la lista de C3, asi que C5 mediria otra cosa: "
        + "; ".join(discrepancias))


def test_la_clase_positiva_es_la_MINORITARIA_y_desempata_por_nombre():
    """La convención de C3, literal. Y el desempate por nombre y no por orden
    de aparición: el orden de aparición depende de cómo esté ordenado el ARFF,
    así que la «positiva» podría cambiar al reordenar filas sin que ningún dato
    cambiara."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    filas = [{"y": "no"}] * 9 + [{"y": "si"}]
    clases, positiva = c5.clase_positiva_medida(filas, "y")
    # (negativa, positiva) = (mayoritaria, minoritaria), la convencion de C3
    assert clases == ("no", "si") and positiva == "si"
    # y con la mayoritaria ALFABETICAMENTE POSTERIOR, el orden NO es el alfabetico
    zeta = [{"y": "zeta"}] * 9 + [{"y": "alfa"}]
    clases_z, positiva_z = c5.clase_positiva_medida(zeta, "y")
    assert positiva_z == "alfa"
    assert clases_z == ("zeta", "alfa"), (
        "en binaria el orden es (negativa, positiva) como en C3, no el alfabetico: "
        "en tres de los doce de C3 los dos ordenes difieren")
    # en MULTICLASE no hay convencion de C3 que copiar: alfabetico, determinista
    multi = [{"y": "c"}] * 5 + [{"y": "a"}] * 3 + [{"y": "b"}] * 1
    clases_m, positiva_m = c5.clase_positiva_medida(multi, "y")
    assert clases_m == ("a", "b", "c") and positiva_m == "b"
    # empate: gana el nombre menor, no el que aparece primero
    empate = [{"y": "zeta"}, {"y": "alfa"}]
    _clases, positiva_empate = c5.clase_positiva_medida(empate, "y")
    assert positiva_empate == "alfa"
    # las filas sin objetivo no cuentan
    con_nulos = [{"y": None}] * 50 + [{"y": "a"}] * 3 + [{"y": "b"}] * 7
    _c, p = c5.clase_positiva_medida(con_nulos, "y")
    assert p == "a"


# ---------------------------------------------------------------------------
# 5. Lo que el artefacto tiene que llevar DENTRO del sello
# ---------------------------------------------------------------------------

def test_las_medidas_que_no_se_pueden_dar_se_declaran_con_su_motivo():
    """El protocolo exige seis medidas `siempre`. Dos no existen en el camino
    de medición. Declararlas a cero sería peor que callarlas, y callarlas es
    peor que decir por qué faltan."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    faltan = c5.MEDIDAS_SIEMPRE_QUE_NO_SE_PUEDEN_DAR
    assert set(faltan) == {"tiempo_de_prediccion", "rss_pico_mb"}
    for medida, motivo in faltan.items():
        assert len(motivo) > 60, f"{medida}: el motivo no dice nada"
    # las otras cuatro SI se dan, y hay que poder comprobarlo
    siempre = json.loads(RUTA_PROTOCOLO.read_text())["metricas_por_tarea"]["siempre"]
    assert set(faltan) < set(siempre), (
        "lo que se declara ausente tiene que ser de la lista registrada, no de otra")


def test_todas_las_metricas_del_informe_llegan_al_registro():
    """C3 sacaba `auroc` y `accuracy` de un informe que ya traía doce. El hueco
    está en el cableado, catorce veces de catorce."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    informe = {"metrics": [{"metric_id": "auroc", "value": 0.9},
                           {"metric_id": "brier_score", "value": 0.08},
                           {"metric_id": "ppv", "value": None}]}
    metricas = c5.metricas_del_informe(informe)
    assert metricas == {"auroc": 0.9, "brier_score": 0.08, "ppv": None}
    assert "ppv" in metricas, (
        "una metrica indefinida se guarda como None; omitirla la volveria "
        "indistinguible de una que no se pidio")
    assert c5.metricas_del_informe(None) == {}


# ---------------------------------------------------------------------------
# 6. La cuenta de coste, y el artefacto COMPUESTO CON EL CÓDIGO QUE CORRE
# ---------------------------------------------------------------------------

def test_la_estimacion_se_calcula_a_mano_y_declara_que_es_una_extrapolacion(protocolo):
    """La aritmética, contrastada a mano. Y que diga que extrapola: 28 de los
    40 datasets no los ha corrido nadie nunca, y una previsión que no lo dice
    se lee como una medida."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    datasets = c5.datasets_de_la_pasada(protocolo)
    plan = c5.plan_de_la_pasada(datasets, protocolo)
    est = c5.estimacion_anclada_en_lo_medido(plan)
    esperado = (1575 * 5.40 + 1575 * 13.88 + 350 * 60.0) / 3600.0
    assert est["horas_previstas"] == pytest.approx(esperado, abs=0.01)
    assert est["horas_de_cota"] == plan["horas_de_reloj_cota_peor_caso"]
    assert est["horas_previstas"] < est["horas_de_cota"], (
        "una prevision por encima de la cota seria una cota mal calculada")
    # cada ancla dice DE DONDE sale; una sin procedencia es un numero a mano
    for cubo, entrada in est["por_cubo"].items():
        assert len(entrada["de_donde_sale_el_ancla"]) > 40, cubo
    assert "EXTRAPOLACION" in " ".join(est).upper() or est["esto_es_una_EXTRAPOLACION"]
    assert "sellados" in est["esto_es_una_EXTRAPOLACION"]
    # y que avise de lo unico que puede romperla por arriba: la densa en grande
    assert "600" in est["lo_que_puede_hacerla_fallar_por_arriba"]
    assert "denso" in est["lo_que_puede_hacerla_fallar_por_arriba"]


def test_el_ancla_del_cubo_grande_dice_que_NO_hay_pasada_de_ese_cubo():
    """Un ancla del cubo grande presentada como las otras dos sería media
    verdad tranquilizadora: las de pequeño y mediano son medias de 630 intentos
    reales, la de grande son dos sondas de un pliegue."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    _valor, de_donde = c5.ANCLAS_MEDIDAS_SEGUNDOS_POR_INTENTO["grande"]
    assert "NO hay pasada de cubo grande" in de_donde
    for cubo in ("pequeno", "mediano"):
        _v, d = c5.ANCLAS_MEDIDAS_SEGUNDOS_POR_INTENTO[cubo]
        assert "media real" in d and "630" in d


def test_el_artefacto_COMPUESTO_lleva_el_alcance_DENTRO_de_su_digest(protocolo, tmp_path):
    """PROBAR EL ARTEFACTO NO ES PROBAR EL CÓDIGO QUE LO PRODUCE: esto compone
    uno NUEVO con `_componer_y_guardar`, que es la función que corre de verdad,
    en vez de leer un JSON ya escrito —que cuadra con su digest aunque el
    código que lo escribe se haya roto—.

    Y comprueba las dos mitades: que el digest cuadra, y que TOCAR el alcance
    lo rompe. Un aserto negativo lo pasa un `{}`.
    """
    from matrixai.estudio.validacion import digest_canonico
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5

    datasets = [d for d in c5.datasets_de_la_pasada(protocolo)
                if d.nombre in ("kc2", "yeast", "Moneyball")]
    for ds in datasets:
        ds.objetivo = "y"
        if ds.tarea != "regression":
            ds.clases, ds.positiva = ("a", "b"), "a"
    mapa = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    resultados = []
    for ds in datasets:
        resultados += _registros(ds.nombre, mapa[ds.nombre],
                                 {"lightgbm": 0.9, "catboost": 0.8, "baseline": 0.5})
    plan = c5.plan_de_la_pasada(datasets, protocolo)
    ruta = tmp_path / "compuesto.json"
    salida = c5._componer_y_guardar(
        resultados, {"procedencia_id": "prueba", "anclable": False, "avisos": ["de prueba"]},
        {}, ruta, datasets=datasets, plan=plan, protocolo=protocolo,
        metrica_por_dataset=mapa, particiones={}, subconjunto=None,
        total_wall_s=1.0, reusados=0, parcial=False)

    escrito = json.loads(ruta.read_text(encoding="utf-8"))
    assert escrito["digest_resultados_crudos"] == salida["digest_resultados_crudos"]
    assert escrito["corte"] == "101-C5"

    # MITAD 1: el documento cuadra con su propio digest
    cuerpo = dict(escrito)
    guardado = cuerpo.pop("digest_resultados_crudos")
    assert digest_canonico(cuerpo) == guardado

    # MITAD 2: y tocar el alcance lo ROMPE — o sea que esta dentro del sello
    manipulado = json.loads(json.dumps(cuerpo))
    manipulado["alcance_y_veredicto"]["alcance"]["datasets"]["criterio_del_subconjunto"] = (
        "los 40, de verdad, palabra")
    assert digest_canonico(manipulado) != guardado, (
        "el alcance esta FUERA del sello: se podria reescribir sin que el fichero "
        "dejara de cuadrar consigo mismo, y un alcance reescribible no declara nada")

    # el alcance dice lo que hay que saber para leer el numero
    alcance = escrito["alcance_y_veredicto"]["alcance"]
    assert alcance["protocolo"]["digest_sha256"] == protocolo.digest()
    assert alcance["datasets"]["n_del_protocolo"] == 40
    # y la metrica de cada tarea viaja en el mismo objeto que el veredicto
    assert escrito["alcance_y_veredicto"]["metrica_de_cierre_por_dataset"] == mapa
    por_motor = escrito["alcance_y_veredicto"]["por_motor"]
    assert "baseline" not in por_motor, "el baseline no compite, la regla lo excluye"
    assert por_motor["lightgbm"]["datasets"] == 3, (
        "los tres datasets, uno por tarea, tienen que estar en el denominador")


def test_un_subconjunto_de_prueba_se_declara_COMO_TAL_en_el_artefacto(protocolo, tmp_path):
    """`--solo` existe para probar el guion. Un fichero salido de `--solo` que
    no se distinga de la pasada de C5 es exactamente la media verdad que
    convierte una prueba en una evidencia."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    datasets = [d for d in c5.datasets_de_la_pasada(protocolo) if d.nombre == "kc2"]
    mapa = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    resultados = _registros("kc2", "auroc", {"lightgbm": 0.9, "catboost": 0.8})
    ruta = tmp_path / "solo.json"
    escrito = c5._componer_y_guardar(
        resultados, {"procedencia_id": "p", "anclable": False, "avisos": []}, {}, ruta,
        datasets=datasets, plan=c5.plan_de_la_pasada(datasets, protocolo),
        protocolo=protocolo, metrica_por_dataset=mapa, particiones={},
        subconjunto=["kc2"], total_wall_s=1.0, reusados=0, parcial=False)
    assert escrito["es_subconjunto_de_prueba"] is True
    assert escrito["subconjunto_pedido"] == ["kc2"]
    criterio = escrito["alcance_y_veredicto"]["alcance"]["datasets"]["criterio_del_subconjunto"]
    assert "SUBCONJUNTO DE PRUEBA" in criterio and "no la pasada de C5" in criterio
    assert "1 de los 40" in criterio
    # y el alcance sigue contando los 40 del protocolo, o sea que se ve el recorte
    assert escrito["alcance_y_veredicto"]["alcance"]["datasets"]["n_del_protocolo"] == 40
    assert len(escrito["alcance_y_veredicto"]["alcance"]["datasets"]["que_faltan"]) == 39


def test_el_guardia_PARA_si_falta_un_arff(protocolo, tmp_path, monkeypatch):
    """Un ARFF que falta descubierto a las 40 horas de pasada cuesta las 40
    horas. El guardia lo mira antes de medir nada — y por eso tiene que estar
    probado que PARA, no que existe."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    datasets = c5.datasets_de_la_pasada(protocolo)
    monkeypatch.setattr(c5, "ARFF_DIR", tmp_path / "no_existe")
    with pytest.raises(SystemExit) as exc:
        c5._exigir_que_LA_PASADA_QUEPA(datasets, protocolo)
    assert "faltan 40 ARFF" in str(exc.value)
    assert "decision de Roberto" in str(exc.value), (
        "bajar datasets es decision de Roberto (cuota de OpenML y gigas), no de "
        "la pasada: el mensaje tiene que decirlo")


def test_una_metrica_que_se_llame_como_un_campo_del_registro_PARA_la_pasada():
    """Las métricas se aplanan encima del registro para que la regla de cierre
    pueda leer `r.get(metrica)` sin conocer la forma anidada. Una que se
    llamara como un campo lo pisaría: un `estado` sobrescrito por un número
    convierte un fallo en un intento completado, y un fallo cuenta como
    dataset perdido."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    registro = {"dataset": "kc2", "estado": "failed", "wall_s": 1.0}
    # el caso normal: ninguna de las quince del catalogo colisiona
    c5.aplanar_metricas_en_el_registro(registro, {"auroc": 0.9, "r2": None})
    assert registro["auroc"] == 0.9 and registro["estado"] == "failed"
    assert registro["r2"] is None
    # y el dia que una colisione, PARA en vez de pisar
    with pytest.raises(SystemExit) as exc:
        c5.aplanar_metricas_en_el_registro({"estado": "failed"}, {"estado": 0.99})
    assert "estado" in str(exc.value) and "silencio" in str(exc.value)


def test_ninguna_metrica_del_catalogo_choca_HOY_con_un_campo_del_registro():
    """El guardia de arriba es para el futuro; esto comprueba el presente
    contra el catálogo REAL, no contra una lista escrita a mano."""
    from matrixai.estudio.metricas import REGISTRO
    campos_del_registro = {
        "dataset", "data_id", "cubo", "tarea", "sellado", "motor", "repeticion",
        "pliegue", "estado", "semilla", "presupuesto_wall_s", "configuracion",
        "metrica_de_cierre", "wall_s", "metricas", "tiempo_de_ajuste",
        "cpu_segundos", "motivo", "traza", "entorno_digest", "motor_digest",
        "procedencia_id", "reusado"}
    chocan = sorted(set(REGISTRO) & campos_del_registro)
    assert not chocan, (
        f"estas metricas del catalogo pisarian un campo del registro: {chocan}")


# ---------------------------------------------------------------------------
# 7. La caché por intento, que es lo que evita repetir la pasada entera
# ---------------------------------------------------------------------------

def test_la_cache_reusa_lo_que_midio_EL_MISMO_codigo_y_NO_lo_demas(protocolo, tmp_path):
    """LA CACHÉ EXISTE PARA QUE UN FALLO A MITAD NO CUESTE LA PASADA ENTERA, y
    por eso no se puede dar por buena: una caché inerte parece una caché que
    funciona y no encontró nada que reusar (pasó de verdad el 2026-09-12 — 0
    reusables de 720, porque el fichero previo no guardaba los digests).

    Se comprueba sobre un artefacto COMPUESTO con `_componer_y_guardar` y
    releído con `_cargar_cache`, que son las dos funciones que corren de
    verdad, no sobre un diccionario inventado a mano.

    Las dos mitades, porque un `assert reusable` lo pasa un `_reusable` que
    devuelva siempre `True`:
      · lo medido con el MISMO código y el MISMO presupuesto se reusa;
      · y un cambio en cualquiera de los dos lo invalida.
    """
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    from benchmarks.fase0.pasada_exploratoria_101_c3 import _cargar_cache, _reusable

    entorno = c5._digest_entorno()
    digest_motor = "aaaabbbbccccdddd"
    datasets = [d for d in c5.datasets_de_la_pasada(protocolo) if d.nombre == "kc2"]
    for d in datasets:
        d.objetivo, d.clases, d.positiva = "problems", ("no", "yes"), "yes"
    resultados = [{
        "dataset": "kc2", "motor": "lightgbm", "repeticion": r, "pliegue": f,
        "estado": "completed", "auroc": 0.9, "presupuesto_wall_s": 120.0,
        "entorno_digest": entorno, "motor_digest": digest_motor,
        "procedencia_id": "p", "reusado": False}
        for r in range(3) for f in range(5)]
    ruta = tmp_path / "con_cache.json"
    c5._componer_y_guardar(
        resultados, {"procedencia_id": "p", "anclable": False, "avisos": []}, {}, ruta,
        datasets=datasets, plan=c5.plan_de_la_pasada(datasets, protocolo),
        protocolo=protocolo,
        metrica_por_dataset=c5.metrica_de_cierre_por_dataset(protocolo, datasets),
        particiones={}, subconjunto=["kc2"], total_wall_s=1.0, reusados=0, parcial=False)

    cache, payload = _cargar_cache(ruta)
    assert len(cache) == 15, (
        "el artefacto escrito tiene que poder releerse como cache: 3 repeticiones "
        "x 5 pliegues")
    previo = cache[("kc2", "lightgbm", 0, 0)]

    # MITAD 1: mismo codigo, mismo presupuesto -> se reusa
    assert _reusable(previo, entorno, digest_motor, 120.0) is True

    # MITAD 2: y cada cosa que cambia lo INVALIDA, una por una
    assert _reusable(previo, "otro_entorno0000", digest_motor, 120.0) is False, (
        "un cambio en el codigo compartido tiene que invalidar TODOS los intentos")
    assert _reusable(previo, entorno, "otro_motor00000", 120.0) is False, (
        "un cambio en el fichero de UN motor tiene que invalidar solo los suyos")
    assert _reusable(previo, entorno, digest_motor, 300.0) is False, (
        "mover `presupuesto.minutos_por_cubo` no toca ningun fichero de codigo: sin "
        "el presupuesto en la clave, la cache reusaria intentos medidos con OTRO tope")
    assert _reusable(None, entorno, digest_motor, 120.0) is False

    # y la procedencia del fichero previo se puede leer, que es lo que evita
    # firmar como medido hoy un numero reusado de hace cinco dias
    from benchmarks.fase0.pasada_exploratoria_101_c3 import procedencia_declarada
    assert procedencia_declarada(payload)["estado"] in ("anclable", "no_anclable")


def test_el_digest_de_entorno_de_C5_incluye_su_propio_fichero(protocolo):
    """Si el guion de C5 no entrara en su propio digest de entorno, cambiar
    cómo parte los datos o cómo compone el `ProblemSpec` NO invalidaría la
    caché — y una re-pasada «limpia» reusaría exactamente los intentos que se
    acaban de reparar. Es el agujero que ya se cerró dos veces en C3."""
    from benchmarks.fase0 import pasada_amplia_101_c5 as c5
    ficheros = {p.name for p in c5._FICHEROS_COMPARTIDOS}
    assert "pasada_amplia_101_c5.py" in ficheros
    assert "pasada_exploratoria_101_c3.py" in ficheros, (
        "C5 importa de C3 la preparacion, la lectura y la procedencia: un cambio "
        "alli cambia lo que C5 mide")
    for imprescindible in ("harness.py", "subproceso.py", "particiones.py",
                           "preparacion.py", "lector_arff.py",
                           "particion_por_diseno.py"):
        assert imprescindible in ficheros, imprescindible
    # y el digest CAMBIA de verdad si cambia uno de esos ficheros
    assert c5._digest_entorno() != c3_digest_con_un_fichero_cambiado(c5)


def c3_digest_con_un_fichero_cambiado(c5):
    """Recalcula el digest de entorno sustituyendo el contenido de un fichero
    por otro. Sin esto, la prueba de arriba comprobaría que la lista tiene los
    nombres, no que el digest los MIRA."""
    import hashlib
    from benchmarks.fase0.pasada_exploratoria_101_c3 import _digest_fichero
    trozos = []
    for i, fichero in enumerate(c5._FICHEROS_COMPARTIDOS):
        trozos.append("0" * 16 if i == 0 else _digest_fichero(fichero))
    return hashlib.sha256("".join(trozos).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# HALLAZGO M1, REPARADO EL 2026-09-16: 3.500 no eran 3.500.
#
# El artefacto llevaba `plan.n_intentos = 3500` y `n_intentos = 3479` uno al
# lado del otro **y nada que los reconciliara**. El motivo estaba dentro, en
# `particion_por_dataset.yeast.limites` —su clase minoritaria deja 4 eventos y
# el quinto pliegue no se puede formar—, pero habia que ir a buscarlo sabiendo
# ya que existia. Medido: 3.479 + 7 motores x 1 pliegue x 3 repeticiones = 3.500.
# ---------------------------------------------------------------------------

import pasada_amplia_101_c5 as _c5  # noqa: E402

_RUTA_AMPLIA = RAIZ / "benchmarks" / "fase0" / "pasada_amplia_101_c5_resultado.json"


def _amplia() -> dict:
    if not _RUTA_AMPLIA.exists():
        pytest.skip("la pasada amplia no esta en este arbol")
    return json.loads(_RUTA_AMPLIA.read_text(encoding="utf-8"))


def test_el_hueco_entre_el_plan_y_lo_medido_queda_EXPLICADO():
    d = _amplia()
    r = _c5.reconciliar_el_plan_con_lo_medido(
        d["plan"], d["particion_por_dataset"], d["resultados"], n_motores=7)
    assert (r["plan"], r["medidos"], r["hueco"]) == (3500, 3479, 21)
    assert r["cuadra"] is True
    assert r["resto_sin_explicar"] == 0
    assert [e["dataset"] for e in r["por_dataset"]] == ["yeast"]
    assert r["por_dataset"][0]["intentos_que_explica"] == 21


def test_el_motivo_MEDIDO_viaja_con_la_reconciliacion():
    """No basta con «faltan 21»: tiene que decir POR QUE, y con el numero.

    `limites` trae la medida real —4 eventos en la clase minoritaria— y sin
    ella «yeast dio 4 pliegues» es una afirmacion sin respaldo dentro del
    propio artefacto.
    """
    d = _amplia()
    r = _c5.reconciliar_el_plan_con_lo_medido(
        d["plan"], d["particion_por_dataset"], d["resultados"], n_motores=7)
    limites = r["por_dataset"][0]["limites"]
    assert limites, "sin `limites` la explicacion no tiene respaldo"
    assert limites[0]["clave"] == "pliegues_reducidos_por_eventos"
    assert limites[0]["medida"]["eventos_clase_minoritaria"] == 4


def test_CUADRA_es_una_conclusion_y_no_un_adorno():
    """La mitad sin la que esto seria un sello de goma.

    Se inventa un hueco que ninguna particion justifica. Si `cuadra` saliera
    `true` igualmente, el campo diria «todo explicado» sobre 21 intentos que
    nadie explica — que es justo peor que no tener campo.
    """
    d = _amplia()
    plan_inflado = dict(d["plan"], n_intentos=d["plan"]["n_intentos"] + 100)
    r = _c5.reconciliar_el_plan_con_lo_medido(
        plan_inflado, d["particion_por_dataset"], d["resultados"], n_motores=7)
    assert r["cuadra"] is False
    assert r["resto_sin_explicar"] == 100


def test_sin_n_intentos_en_el_plan_NO_se_inventa_una_reconciliacion():
    d = _amplia()
    plan_mudo = {k: v for k, v in d["plan"].items() if k != "n_intentos"}
    r = _c5.reconciliar_el_plan_con_lo_medido(
        plan_mudo, d["particion_por_dataset"], d["resultados"], n_motores=7)
    assert r["plan"] is None
    assert "cuadra" not in r, "sin plan no hay veredicto que dar"
    assert "nada que reconciliar" in r["motivo"]


def test_un_dataset_con_su_particion_COMPLETA_no_aparece_en_el_hueco():
    """Solo los recortados explican algo: si apareciera uno intacto, el
    recuento sumaria intentos que nunca faltaron."""
    d = _amplia()
    r = _c5.reconciliar_el_plan_con_lo_medido(
        d["plan"], d["particion_por_dataset"], d["resultados"], n_motores=7)
    nombrados = {e["dataset"] for e in r["por_dataset"]}
    for nombre, particion in d["particion_por_dataset"].items():
        if particion.get("n_pliegues_pedidos") == particion.get("n_pliegues_obtenidos"):
            assert nombre not in nombrados


# ---------------------------------------------------------------------------
# HALLAZGO A3, REPARADO EL 2026-09-16: el punto de control fallaba justo en el
# cubo para el que se escribio.
#
# `guardar(parcial=True)` colgaba del bucle de REPETICIONES, y el protocolo
# registra `repeticiones_grande = 1`: en el cubo grande cada repeticion ES el
# dataset entero, asi que los 10 datasets mas caros tenian UN solo guardado, al
# final. Su propio comentario decia que existia para que morir a mitad no
# costase todo lo hecho, y era exactamente lo que pasaba.
#
# La nota lo daba por ACEPTADO —«una perdida garantizada de 1 h contra una
# probable de 2 h»— y esa aceptacion se tomo **sin medir el coste de guardar**.
# Medido el 2026-09-16 sobre los 3.479 resultados: **0,90 s** (0,67 el veredicto
# con su bootstrap, 0,04 la dispersion, 0,19 serializar 5,2 MB). El intercambio
# real era 36 s contra hasta hora y media.
#
# Se lee del CODIGO con `ast` y no con una expresion regular: lo que hay que
# comprobar es en que BUCLE esta la llamada, y eso una regex no lo ve.
# ---------------------------------------------------------------------------

import ast as _ast  # noqa: E402


def _cuerpo_de_main() -> _ast.FunctionDef:
    fuente = (RAIZ / "benchmarks" / "fase0" / "pasada_amplia_101_c5.py").read_text(
        encoding="utf-8")
    arbol = _ast.parse(fuente)
    for nodo in _ast.walk(arbol):
        if isinstance(nodo, _ast.FunctionDef) and nodo.name == "main":
            return nodo
    raise AssertionError("no se encuentra `main()` en el guion de la pasada")


def _profundidad_de_los_guardados() -> set[int]:
    """Cuantos `for` anidados envuelven a cada `guardar(parcial=True)`.

    En `main()` los bucles son: datasets (1) -> repeticiones (2) -> pliegues (3).
    Un guardado a profundidad 2 solo corre al acabar cada repeticion; a
    profundidad 3, tras cada pliegue.
    """
    profundidades: set[int] = set()

    def recorrer(nodo, nivel: int) -> None:
        for hijo in _ast.iter_child_nodes(nodo):
            siguiente = nivel + 1 if isinstance(hijo, (_ast.For, _ast.AsyncFor)) else nivel
            if (isinstance(hijo, _ast.Call)
                    and isinstance(hijo.func, _ast.Name) and hijo.func.id == "guardar"
                    and any(k.arg == "parcial"
                            and getattr(k.value, "value", None) is True
                            for k in hijo.keywords)):
                profundidades.add(nivel)
            recorrer(hijo, siguiente)

    recorrer(_cuerpo_de_main(), 0)
    return profundidades


def test_hay_un_punto_de_control_al_nivel_del_PLIEGUE_y_no_solo_de_la_repeticion():
    """El defecto que esta prueba existe para impedir que vuelva."""
    profundidades = _profundidad_de_los_guardados()
    assert 3 in profundidades, (
        f"todos los `guardar(parcial=True)` estan a profundidad {sorted(profundidades)}: "
        "con `repeticiones_grande = 1`, eso deja los 10 datasets del cubo grande "
        "con un solo guardado al final del dataset entero")


def test_el_guardado_de_cada_repeticion_SIGUE_ahi_como_suelo():
    """El de reloj es un anadido, no un sustituto: si el pliegue durase menos
    que el tope, sin este no habria ningun guardado garantizado por vuelta."""
    assert 2 in _profundidad_de_los_guardados()


def test_el_punto_de_control_del_pliegue_va_por_RELOJ_y_no_en_cada_vuelta():
    """Bajarlo a cada pliegue a secas daria 15 guardados por dataset del cubo
    pequeno —que entero dura dos minutos—: un 11 % de sobrecoste donde no hay
    nada que proteger. Con tope de tiempo, la granularidad la pone el coste
    real de lo que se mide."""
    fuente = (RAIZ / "benchmarks" / "fase0" / "pasada_amplia_101_c5.py").read_text(
        encoding="utf-8")
    assert "SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL" in fuente
    assert "ultimo_punto_de_control" in fuente
    tope = _c5.__dict__.get("SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL")
    if tope is None:  # es local a `main()`, se lee del fuente
        import re
        m = re.search(r"SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL\s*=\s*([\d.]+)", fuente)
        assert m, "la constante no declara su valor"
        tope = float(m.group(1))
    # 0,90 s por guardado medido: con 60 s de tope, el sobrecoste queda por
    # debajo del 1,5 % aunque cada pliegue durase justo un tope entero.
    assert 10.0 <= tope <= 300.0, (
        f"un tope de {tope}s no tiene sentido: por debajo de 10 s el guardado "
        "(0,90 s medidos) empieza a pesar, y por encima de 300 s deja de "
        "proteger al cubo grande, que es para lo que se puso")

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C3 — LA PASADA SE MIDE CON EL PRESUPUESTO REGISTRADO, Y DECLARA
CUÁNTAS CONFIGURACIONES CORRIÓ.

Las dos desviaciones que la auditoría del 2026-09-13 encontró entre la pasada
de siete motores y el protocolo sellado, y que bloqueaban mover la cartera:

1. **El presupuesto aplicado era el 40 % del registrado.** El script llevaba
   `WALL_SECONDS = 120.0` GLOBAL; el protocolo registra
   `presupuesto.minutos_por_cubo` y al **mediano le da 5 minutos (300 s)**.
   Seis de los doce datasets de la pasada son medianos. El ÚNICO fallo de los
   1.260 intentos fue `matrixai.dense.torch_cpu` sobre
   `Internet-Advertisements` (mediano) matado a los **156,9 s** — 120 de
   presupuesto más el margen de 30 del subproceso—, o sea **por debajo de su
   presupuesto registrado**. Y la regla de cierre dice con todas las letras
   que «un fallo (timeout/crash) cuenta como dataset perdido para ese motor»:
   el recorte de presupuesto no hacía la medición más barata, la hacía OTRA.

2. **Corrió 1 de las 2 configuraciones registradas por motor** (7 de 13
   contando el dummy, que registra una) y el bloque `alcance` —que YA declara
   el recorte de datasets y el de motores— se callaba éste. Es el eje que más
   pesa, porque la `definicion_de_mejor` registrada habla de «la mejor media
   de los AJUSTES».

   **Este fichero NO prueba que se corran las dos**, porque no se corren: el
   protocolo declara CUÁNTAS son y no CUÁLES, los espacios de búsqueda que el
   anexo C §2.3 exige pre-registrar con sha256 no están en ninguna parte del
   árbol, y ninguno de los siete adaptadores implementa búsqueda interna.
   Inventarlos habría sido elegirlos después de ver los números. Lo que se
   prueba es que el recorte esté DECLARADO y sea derivable de los registros,
   que es lo que permite leer el 9/12 sabiendo qué se midió.

Y una tercera, que no bloqueaba: el veredicto se publicaba sin su
incertidumbre. La ventaja de catboost sobre lightgbm en `pc1` son 2,926
puntos con IC95 emparejado que **cruza el listón de 2,0**: el veredicto es
correcto e inequívoco, pero sin el intervalo «no cumple» y «no cumple, y con
estos 15 pliegues no se distingue de cumplir» se leen igual.

NINGUNA de las tres toca la regla de cierre. Eso lo comprueba, byte a byte,
`test_c101_c1_protocolo_fase0.py::test_la_REGLA_DE_CIERRE_no_se_movio_en_la_
re_firma`, y aquí se vuelve a comprobar que el VEREDICTO tampoco se movió al
añadirle el intervalo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.fase0 import pasada_exploratoria_101_c3 as pasada
from benchmarks.fase0.protocolo import (ProtocoloExploratorio,
                                        _intervalo_pareado,
                                        aplicar_regla_de_cierre,
                                        calcular_coste,
                                        veredicto_con_su_alcance)

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
_PROTOCOLO = _FASE0 / "protocolo_exploratorio.json"
#: La pasada de los SIETE motores, la que la auditoría revisó. Sin `skipTest`
#: si falta: está en el árbol, y un salto silencioso ante un fichero ausente
#: es un banco de pruebas sin dientes.
_JSON_SIETE = _FASE0 / "pasada_exploratoria_101_c3_siete_motores_20260913.json"

#: El tope de pared al que murió el único fallo de los 1.260 intentos. Escrito
#: a mano: es un HECHO medido, y un número calculado aquí no anclaría nada.
WALL_DEL_UNICO_FALLO_2026_09_13 = 156.949


def _protocolo() -> ProtocoloExploratorio:
    return ProtocoloExploratorio.cargar(_PROTOCOLO)


def _crudos() -> list[dict]:
    return json.loads(_JSON_SIETE.read_text(encoding="utf-8"))["resultados"]


# ---------------------------------------------------------------------------
# DESVIACIÓN 1 — el presupuesto sale del protocolo, por cubo
# ---------------------------------------------------------------------------

def test_el_presupuesto_de_pared_SALE_del_protocolo_y_es_POR_CUBO():
    """El número no se escribe aquí: se lee del protocolo registrado y se
    compara con lo que la pasada aplicaría. Si mañana alguien re-firma el
    protocolo con otros minutos, esto sigue cuadrando — lo que no puede pasar
    es que los dos se separen."""
    protocolo = _protocolo()
    registrados = protocolo.presupuesto.minutos_por_cubo
    for cubo, minutos in registrados.items():
        assert pasada.wall_seconds_del_cubo(cubo, protocolo) == float(minutos) * 60.0, (
            f"el presupuesto que la pasada aplicaria al cubo {cubo!r} no es el "
            f"registrado ({minutos} min)")
    # Y los tres NO son el mismo número: un `minutos_por_cubo` que diera lo
    # mismo para 500 filas y para 100.000 sería la constante global otra vez,
    # solo que escrita en el protocolo.
    assert len(set(registrados.values())) == 3


def test_el_MEDIANO_no_se_mide_con_el_presupuesto_del_PEQUENO():
    """El accidente exacto, con su número: la pasada de siete motores aplicó
    120 s a los seis datasets medianos y el protocolo les da 300.

    El aserto que importa es el último: con el presupuesto registrado, el
    intento que murió a los 156,9 s **no habría muerto**. Un `assert
    mediano > pequeno` solo lo pasaría cualquier par de números distintos."""
    protocolo = _protocolo()
    pequeno = pasada.wall_seconds_del_cubo("pequeno", protocolo)
    mediano = pasada.wall_seconds_del_cubo("mediano", protocolo)
    assert mediano > pequeno
    assert mediano == 300.0, "el protocolo registra 5 minutos para el cubo mediano"
    from matrixai_engines.subproceso import MARGEN_POR_DEFECTO_SEGUNDOS
    assert mediano + MARGEN_POR_DEFECTO_SEGUNDOS > WALL_DEL_UNICO_FALLO_2026_09_13, (
        "con el presupuesto REGISTRADO del cubo mediano, el unico fallo de los "
        "1.260 intentos (muerto a 156,9 s) seguiria muriendo: entonces la "
        "correccion no corrige nada")


def test_el_BUCLE_de_la_pasada_le_da_a_cada_intento_el_presupuesto_de_SU_cubo(tmp_path,
                                                                              monkeypatch):
    """EL CABLEADO, medido corriendo `main()` de verdad — no leyendo el código.

    El hueco está en el cableado catorce veces de cada catorce en este
    proyecto: que `wall_seconds_del_cubo()` devuelva 300 no dice que el
    `Presupuesto` que llega al motor lleve 300. Aquí se sustituye el
    ejecutor de intentos por uno que apunta lo que recibe —los motores no
    corren, que es lo caro— y se conducen DOS datasets reales, uno de cada
    cubo, con 2 pliegues y 1 repetición: 28 intentos y unos segundos.
    """
    vistos: list[tuple] = []

    class _Intento:
        estado = "completed"
        informe = {"metrics": [{"metric_id": "auroc", "value": 0.5}]}
        motivo_del_estado = None

    def _falso(motor, train, validation, test, spec, presupuesto, *, candidate, **kw):
        vistos.append((kw["dataset"], motor.nombre, presupuesto.wall_seconds, candidate))
        return _Intento()

    monkeypatch.setattr(pasada, "ejecutar_intento_aislado", _falso)
    monkeypatch.setattr(pasada, "DATASETS", [(1063, "kc2", "pequeno", "yes", "no"),
                                             (40983, "wilt", "mediano", "2", "1")])
    monkeypatch.setattr(pasada, "FOLDS", 2)
    monkeypatch.setattr(pasada, "REPETICIONES_PEQUENO_MEDIANO", 1)
    salida = tmp_path / "salida.json"
    monkeypatch.setattr(sys, "argv", ["pasada", "--forzar", "--salida", str(salida)])
    pasada.main()

    por_dataset = {}
    for dataset, _motor, wall, _candidate in vistos:
        por_dataset.setdefault(dataset, set()).add(wall)
    assert por_dataset == {"kc2": {120.0}, "wilt": {300.0}}, (
        "el presupuesto que llega al Presupuesto de cada intento no es el del "
        "cubo de su dataset")

    payload = json.loads(salida.read_text(encoding="utf-8"))
    # Y lo que se le dio a cada intento queda ESCRITO en su registro: un
    # `failed` por tope de pared no se puede leer sin saber contra qué tope.
    por_registro = {r["dataset"]: r["presupuesto_wall_s"] for r in payload["resultados"]}
    assert por_registro == {"kc2": 120.0, "wilt": 300.0}


def test_el_JSON_declara_el_presupuesto_POR_CUBO_con_el_digest_de_donde_sale(tmp_path):
    """La cabecera del resultado decía `wall_seconds_por_intento: 120.0`: un
    número solo para dos cubos distintos, sin decir contra qué se comparaba.
    Ahora declara los minutos registrados, el cubo a cubo aplicado, y el
    digest del protocolo del que salen — un presupuesto sin procedencia es
    otro número escrito a mano.

    **SE COMPRUEBA SOBRE EL PAYLOAD COMPUESTO, no sobre la función que lo
    fabrica, y esto no es un detalle de estilo: la primera versión de esta
    prueba llamaba a `presupuesto_declarado_de_la_pasada()` a pelo, y el
    sabotaje que ponía `"presupuesto": {}` en el JSON de salida la dejó
    VERDE. Probar la función no es probar el producto.** Se compone con el
    MISMO código que corre la pasada, sin pasar el bloque, que es el camino
    que el sabotaje rompía.
    """
    protocolo = _protocolo()
    payload = pasada._componer_y_guardar(
        [], {"procedencia_id": "p1", "anclable": True}, {},
        tmp_path / "salida.json", total_wall_s=1.0, reusados=0, parcial=False)
    bloque = payload["presupuesto"]
    assert bloque["protocolo_digest_sha256"] == protocolo.digest()
    assert bloque["minutos_por_cubo_registrados"] == dict(
        protocolo.presupuesto.minutos_por_cubo)
    cubos_de_la_pasada = {cubo for _i, _n, cubo, _p, _g in pasada.DATASETS}
    assert set(bloque["wall_seconds_por_cubo"]) == cubos_de_la_pasada
    for cubo, segundos in bloque["wall_seconds_por_cubo"].items():
        assert segundos == float(protocolo.presupuesto.minutos_por_cubo[cubo]) * 60.0
    # Y la declaración va DENTRO del sello, como el alcance y por el mismo
    # motivo: un presupuesto que se puede reescribir sin que el fichero deje
    # de cuadrar consigo mismo no declara nada.
    from matrixai.estudio.validacion import digest_canonico
    guardado = dict(payload)
    digest = guardado.pop("digest_resultados_crudos")
    assert digest_canonico(guardado) == digest
    sin_presupuesto = dict(guardado)
    sin_presupuesto.pop("presupuesto")
    assert digest_canonico(sin_presupuesto) != digest


def test_el_GUARDIA_para_antes_de_medir_si_el_presupuesto_no_es_el_registrado(monkeypatch):
    """La otra mitad del guardia, y la que de verdad protege: sin ella lo
    pasaría un `return` vacío.

    Se simula la desviación REAL —aplicar el presupuesto del pequeño a todos
    los cubos— y se comprueba que la pasada se para ANTES de medir nada."""
    monkeypatch.setattr(pasada, "wall_seconds_del_cubo",
                        lambda cubo, protocolo=None: 120.0)
    with pytest.raises(SystemExit) as excinfo:
        pasada._exigir_que_el_PRESUPUESTO_sea_EL_REGISTRADO()
    assert "300.0 s" in str(excinfo.value) and "mediano" in str(excinfo.value)


def test_el_guardia_para_si_el_protocolo_registrado_no_se_puede_LEER(monkeypatch, tmp_path):
    """Sin protocolo no hay presupuesto registrado, y seguir con uno de
    repuesto sería volver a la constante global por otro camino. Un
    presupuesto a medias tranquiliza igual que uno falso."""
    monkeypatch.setattr(pasada, "RUTA_DEL_PROTOCOLO", tmp_path / "no_existe.json")
    with pytest.raises(SystemExit) as excinfo:
        pasada._exigir_que_el_PRESUPUESTO_sea_EL_REGISTRADO()
    assert "no se puede leer el protocolo registrado" in str(excinfo.value)


def test_el_guardia_del_presupuesto_se_LLAMA_de_verdad_antes_de_medir(tmp_path,
                                                                     monkeypatch):
    """Que la función exista y nadie la llame es exactamente el defecto que
    esto repara — el hueco de cableado número quince fue justo eso con
    `reserva_segura()`. Se comprueba por COMPORTAMIENTO, no leyendo el
    código: si el guardia del presupuesto levanta, `main()` no llega a medir.

    **LOS SEGUROS DE ABAJO NO SON DECORACIÓN, Y COSTARON UN SUSTO.** La
    primera versión de esta prueba se fiaba de que el guardia levantase, y
    dejaba `argv` sin `--salida` y la lista real de doce datasets. Al
    SABOTEAR el cableado —quitar la llamada al guardia, que es justo lo que
    esta prueba tiene que cazar— `main()` siguió adelante y arrancó la pasada
    de verdad, con los siete motores, apuntando al JSON de evidencia que
    escribe al terminar cada dataset. Se mató a tiempo y la evidencia quedó
    intacta, pero la lección va aquí escrita: una prueba cuyo sabotaje es
    «no pares» tiene que ser inofensiva TAMBIÉN cuando no para.
    """
    llamado: list[int] = []

    def _explota() -> None:
        llamado.append(1)
        raise SystemExit("guardia del presupuesto")

    class _Intento:
        estado = "completed"
        informe = {"metrics": [{"metric_id": "auroc", "value": 0.5}]}
        motivo_del_estado = None

    monkeypatch.setattr(pasada, "_exigir_que_el_PRESUPUESTO_sea_EL_REGISTRADO", _explota)
    # Los seguros: si el guardia NO levanta, esto no mide nada real ni escribe
    # en la evidencia — un dataset pequeño, dos pliegues, una repetición,
    # ningún motor de verdad y la salida a un temporal.
    monkeypatch.setattr(pasada, "ejecutar_intento_aislado", lambda *a, **k: _Intento())
    monkeypatch.setattr(pasada, "DATASETS", [(1063, "kc2", "pequeno", "yes", "no")])
    monkeypatch.setattr(pasada, "FOLDS", 2)
    monkeypatch.setattr(pasada, "REPETICIONES_PEQUENO_MEDIANO", 1)
    monkeypatch.setattr(sys, "argv", ["pasada", "--forzar", "--salida",
                                      str(tmp_path / "salida.json")])
    with pytest.raises(SystemExit):
        pasada.main()
    assert llamado, "main() midio sin preguntar por el presupuesto registrado"


def test_el_CACHE_no_reusa_un_intento_medido_con_OTRO_presupuesto():
    """Desde que el tope de pared sale del protocolo y no de una constante de
    este fichero, mover `minutos_por_cubo` NO toca ningún fichero de código:
    el `entorno_digest` no cambia y el caché reusaría tal cual intentos
    medidos con otro tope. Un intento que falló por tiempo con 120 s puede
    completar con 300 — que es justo la corrección que abrió este agujero."""
    previo = {"entorno_digest": "E", "motor_digest": "M", "presupuesto_wall_s": 120.0}
    assert pasada._reusable(previo, "E", "M", 120.0)
    assert not pasada._reusable(previo, "E", "M", 300.0), (
        "el cache reusa un intento medido con 120 s para una pasada de 300 s")
    # Y uno de antes de que el campo existiera NO es uno con el presupuesto de
    # hoy: un valor ausente no es un cero ni el que a uno le conviene.
    assert not pasada._reusable({"entorno_digest": "E", "motor_digest": "M"}, "E", "M", 120.0)


# ---------------------------------------------------------------------------
# DESVIACIÓN 2 — el recorte de configuraciones, declarado
# ---------------------------------------------------------------------------

def test_el_modelo_de_COSTE_del_protocolo_solo_cuadra_con_las_13_configuraciones():
    """La comprobación que convierte «el protocolo pide 2 por motor» de una
    lectura en un hecho derivado: su propio `coste_calculado.total_ejecuciones`
    (6.500, guardado en el JSON registrado) sale de 13 configuraciones. Con
    una por motor darían 3.500."""
    protocolo = _protocolo()
    payload = json.loads(_PROTOCOLO.read_text(encoding="utf-8"))
    assert sum(m.configuraciones for m in protocolo.motores) == 13
    assert calcular_coste(protocolo, cpus=8).total_ejecuciones == \
        payload["coste_calculado"]["total_ejecuciones"] == 6500

    import dataclasses
    una_sola = dataclasses.replace(
        protocolo, motores=tuple(dataclasses.replace(m, configuraciones=1)
                                 for m in protocolo.motores))
    assert calcular_coste(una_sola, cpus=8).total_ejecuciones == 3500, (
        "con UNA configuracion por motor el coste registrado tambien cuadraria, "
        "y entonces el 6.500 no diria nada sobre cuantas configuraciones se "
        "prometieron medir")


def test_el_ALCANCE_declara_el_recorte_de_configuraciones_por_motor():
    """El bloque declaraba el recorte de datasets y el de motores y se callaba
    éste. Y no basta un total: «7 de 13» no dice si el recorte cayó repartido
    o entero sobre un motor."""
    protocolo = _protocolo()
    v = veredicto_con_su_alcance(
        protocolo, _crudos(), motor="lightgbm",
        motores_declarados=pasada.nombres_de_los_motores_de_la_pasada(),
        datasets_declarados=[n for _i, n, _c, _p, _g in pasada.DATASETS],
        configuraciones_declaradas=pasada.configuraciones_de_la_pasada(),
        criterio_de_las_configuraciones=pasada.CRITERIO_DE_LAS_CONFIGURACIONES)
    bloque = v["alcance"]["configuraciones"]
    assert bloque["n_del_protocolo"] == 13
    assert bloque["n_que_corrieron"] == 7
    # Una por motor, y al dummy no le falta ninguna: registra una sola.
    assert bloque["que_faltan_por_motor"]["dummy"] == 0
    assert all(bloque["que_faltan_por_motor"][m] == 1
               for m in ("lightgbm", "catboost", "xgboost", "sklearn.hgb",
                         "sklearn.logreg", "matrixai.dense.torch_cpu"))
    # El criterio VIAJA al artefacto: hasta hoy vivía en el docstring del
    # script, o sea en el código, y quien leía el resultado no lo tenía. Y no
    # basta con que el campo exista con cualquier contenido — eso lo pasa una
    # cadena vacía: tiene que decir POR QUÉ solo corrió una, que es lo que
    # impide leer el recorte como un descuido.
    criterio = bloque["criterio_de_las_configuraciones"]
    assert "ESPACIOS DE BUSQUEDA" in criterio and "sha256" in criterio


def test_la_LECTURA_del_veredicto_nombra_el_recorte_de_configuraciones():
    """Es lo único que ve quien lee el JSON por encima, y la frase se REDACTA
    con los números medidos: una frase guardada deja de ser verdad en cuanto
    cambian los números que la sostenían y sigue sonando razonable."""
    v = veredicto_con_su_alcance(
        _protocolo(), _crudos(), motor="lightgbm",
        motores_declarados=pasada.nombres_de_los_motores_de_la_pasada(),
        datasets_declarados=[n for _i, n, _c, _p, _g in pasada.DATASETS],
        configuraciones_declaradas=pasada.configuraciones_de_la_pasada(),
        criterio_de_las_configuraciones=pasada.CRITERIO_DE_LAS_CONFIGURACIONES)
    lectura = v["como_hay_que_leer_este_numero"]
    assert "Configuraciones: 7 de 13" in lectura
    # Y por qué importa, no solo cuántas: la regla mide «la mejor media de los
    # AJUSTES», y con una familia de ajuste por motor eso no es lo que dice.
    assert "AJUSTES" in lectura and "familia" in lectura


def test_las_configuraciones_OBSERVADAS_salen_de_los_registros_no_de_la_declaracion():
    """El par que caza «tocaron la lista y no regeneraron el JSON». Se le
    mete un registro con una configuración que NADIE declaró: tiene que
    aparecer en lo observado y no en lo declarado."""
    protocolo = _protocolo()
    crudos = [dict(r) for r in _crudos()[:40]]
    for r in crudos:
        r["configuracion"] = "default"
    intruso = dict(crudos[0], configuracion="busqueda-inventada")
    v = veredicto_con_su_alcance(
        protocolo, crudos + [intruso], motor="lightgbm",
        motores_declarados=pasada.nombres_de_los_motores_de_la_pasada(),
        datasets_declarados=[n for _i, n, _c, _p, _g in pasada.DATASETS],
        configuraciones_declaradas=pasada.configuraciones_de_la_pasada())
    bloque = v["alcance"]["configuraciones"]
    assert bloque["observable_en_los_registros"] is True
    observadas = {c for v_ in bloque["observadas_en_los_resultados"].values() for c in v_}
    assert "busqueda-inventada" in observadas
    declaradas = {c for v_ in bloque["declaradas_por_la_pasada"].values() for c in v_}
    assert "busqueda-inventada" not in declaradas


def test_unos_registros_SIN_el_campo_no_dicen_que_corrieran_CERO_configuraciones():
    """Los JSON de antes del 2026-09-13 no traen `configuracion`. De ahí no se
    sigue que corrieran cero: se sigue que no se puede saber cuáles corrieron.
    Un aserto negativo lo pasa un diccionario vacío, así que se exige que el
    motivo esté ESCRITO."""
    bloque = veredicto_con_su_alcance(
        _protocolo(), _crudos(), motor="lightgbm",
        motores_declarados=pasada.nombres_de_los_motores_de_la_pasada(),
        datasets_declarados=[n for _i, n, _c, _p, _g in pasada.DATASETS],
        configuraciones_declaradas=pasada.configuraciones_de_la_pasada(),
    )["alcance"]["configuraciones"]
    assert bloque["observable_en_los_registros"] is False
    assert bloque["observadas_en_los_resultados"] == {}
    assert "no se puede saber" in bloque["por_que_no_es_observable"]


def test_cada_intento_GRABA_su_configuracion(tmp_path, monkeypatch):
    """Sin el campo en el registro, «cuántas configuraciones corrieron» no se
    puede DERIVAR y hay que creerse la declaración — que es justo lo que el
    bloque de alcance no hace ni con los motores ni con los datasets."""
    class _Intento:
        estado = "completed"
        informe = {"metrics": [{"metric_id": "auroc", "value": 0.5}]}
        motivo_del_estado = None

    monkeypatch.setattr(pasada, "ejecutar_intento_aislado",
                        lambda *a, **k: _Intento())
    monkeypatch.setattr(pasada, "DATASETS", [(1063, "kc2", "pequeno", "yes", "no")])
    monkeypatch.setattr(pasada, "FOLDS", 2)
    monkeypatch.setattr(pasada, "REPETICIONES_PEQUENO_MEDIANO", 1)
    salida = tmp_path / "salida.json"
    monkeypatch.setattr(sys, "argv", ["pasada", "--forzar", "--salida", str(salida)])
    pasada.main()

    payload = json.loads(salida.read_text(encoding="utf-8"))
    assert payload["resultados"]
    assert {r["configuracion"] for r in payload["resultados"]} == {
        pasada.CONFIGURACION_UNICA}
    bloque = payload["alcance_y_veredicto"]["alcance"]["configuraciones"]
    assert bloque["observable_en_los_registros"] is True
    assert bloque["n_que_corrieron"] == 7


# ---------------------------------------------------------------------------
# LA TERCERA — el veredicto con su incertidumbre
# ---------------------------------------------------------------------------

def test_el_intervalo_de_pc1_CRUZA_el_liston_y_el_JSON_lo_dice():
    """El caso que la auditoría midió: catboost aventaja a lightgbm en `pc1`
    por 2,926 puntos, y el IC95 emparejado de esa diferencia CRUZA el listón
    de 2,0. El veredicto es correcto e inequívoco; publicarlo sin el
    intervalo lo hace parecer holgado y no lo es."""
    protocolo = _protocolo()
    r = aplicar_regla_de_cierre(_crudos(), protocolo.regla_de_cierre, motor="lightgbm")
    pc1 = next(d for d in r["detalle"] if d["dataset"] == "pc1")
    assert pc1["cumple"] is False
    assert round(pc1["distancia_en_puntos"], 3) == 2.926
    iv = pc1["intervalo_de_la_distancia"]
    assert iv["contra"] == "catboost"
    assert iv["n_pliegues_emparejados"] == 15
    assert iv["bajo"] < protocolo.regla_de_cierre.puntos < iv["alto"]
    assert iv["cruza_el_liston"] is True
    # Y el recuento de portada: de los TRES datasets que lightgbm pierde, dos
    # tienen el listón dentro de su intervalo. Sin este número hay que abrir el
    # detalle uno a uno para enterarse.
    assert r["intervalos_de_los_que_NO_cumplen_que_cruzan_el_liston"] == 2


def test_hay_un_incumplido_que_NO_cruza_el_liston():
    """La otra mitad, sin la cual `cruza_el_liston` podría estar puesto a
    `True` siempre y las dos lecturas seguirían sin distinguirse. `diabetes`
    es el incumplido claro de lightgbm: 3,575 puntos con el intervalo entero
    por encima de 2,0."""
    protocolo = _protocolo()
    r = aplicar_regla_de_cierre(_crudos(), protocolo.regla_de_cierre, motor="lightgbm")
    diabetes = next(d for d in r["detalle"] if d["dataset"] == "diabetes")
    assert diabetes["cumple"] is False
    iv = diabetes["intervalo_de_la_distancia"]
    assert iv["cruza_el_liston"] is False
    assert iv["bajo"] > protocolo.regla_de_cierre.puntos


def test_el_intervalo_NO_movio_el_veredicto_de_NINGUN_motor():
    """Lo no negociable: la regla registrada decide, y este añadido solo la
    acompaña. Se re-deriva sobre la evidencia de los 1.260 intentos, que no se
    ha tocado, y se compara con lo que el artefacto ya llevaba guardado —
    incluido el 12/12 de catboost, que es el que hoy discute la cartera."""
    protocolo = _protocolo()
    payload = json.loads(_JSON_SIETE.read_text(encoding="utf-8"))
    guardados = payload["alcance_y_veredicto"]["por_motor"]
    assert len(guardados) == 6
    for motor, guardado in guardados.items():
        r = aplicar_regla_de_cierre(payload["resultados"], protocolo.regla_de_cierre,
                                    motor=motor)
        assert (r["cumplidos"], r["datasets"], r["cumple_la_regla"]) == (
            guardado["cumplidos"], guardado["datasets"], guardado["cumple_la_regla"]), (
            f"el veredicto de {motor} se movio al añadirle el intervalo")
    assert guardados["lightgbm"]["cumplidos"] == 9
    assert guardados["catboost"]["cumple_la_regla"] is True


def test_sin_pliegue_emparejable_NO_se_inventa_un_intervalo():
    """Un intervalo donde no hay datos que emparejar se lee igual que uno
    medido, y es peor que ninguno. Con un solo pliegue común no hay nada que
    remuestrear."""
    assert _intervalo_pareado([1.0]) is None
    assert _intervalo_pareado([]) is None
    registros = [
        {"dataset": "d", "motor": "a", "estado": "completed", "auroc": 0.80,
         "repeticion": 0, "pliegue": 0},
        {"dataset": "d", "motor": "b", "estado": "completed", "auroc": 0.90,
         "repeticion": 0, "pliegue": 0},
    ]
    protocolo = _protocolo()
    r = aplicar_regla_de_cierre(registros, protocolo.regla_de_cierre, motor="a")
    assert r["detalle"][0]["intervalo_de_la_distancia"] is None
    assert r["detalle"][0]["distancia_en_puntos"] is not None


def test_unos_registros_sin_pliegue_declarado_se_CUENTAN_no_se_ignoran():
    """Un JSON cuyos registros no digan repetición/pliegue no puede dar
    intervalos, y eso tiene que verse: si el recuento no estuviera, «ningún
    intervalo» se leería como «todos empatados» en vez de como «no se pudo
    emparejar»."""
    registros = [
        {"dataset": "d", "motor": "a", "estado": "completed", "auroc": 0.80},
        {"dataset": "d", "motor": "b", "estado": "completed", "auroc": 0.90},
    ]
    r = aplicar_regla_de_cierre(registros, _protocolo().regla_de_cierre, motor="a")
    assert r["n_medidas_sin_pliegue_declarado"] == 2
    assert r["detalle"][0]["intervalo_de_la_distancia"] is None


def test_el_intervalo_es_EMPAREJADO_y_no_dos_intervalos_sueltos():
    """La diferencia que este módulo existe para no perder. Dos motores
    correlacionados —el segundo siempre 3 puntos por debajo del primero, con
    los dos subiendo y bajando a la vez entre pliegues— tienen una diferencia
    CONSTANTE: emparejado, el intervalo es un punto. Restando dos intervalos
    sueltos saldría anchísimo, porque cada uno recoge la variación entre
    pliegues que la resta cancela."""
    a = [0.60, 0.70, 0.80, 0.90, 0.95]
    b = [x + 0.03 for x in a]
    registros = []
    for i, (va, vb) in enumerate(zip(a, b)):
        registros.append({"dataset": "d", "motor": "a", "estado": "completed",
                          "auroc": va, "repeticion": 0, "pliegue": i})
        registros.append({"dataset": "d", "motor": "b", "estado": "completed",
                          "auroc": vb, "repeticion": 0, "pliegue": i})
    r = aplicar_regla_de_cierre(registros, _protocolo().regla_de_cierre, motor="a")
    iv = r["detalle"][0]["intervalo_de_la_distancia"]
    assert iv["n_pliegues_emparejados"] == 5
    assert abs(iv["alto"] - iv["bajo"]) < 1e-9, (
        "el intervalo de una diferencia constante no es un punto: no se esta "
        "emparejando por pliegue, se estan restando dos muestras sueltas")
    assert abs(iv["bajo"] - 3.0) < 1e-9


def test_el_intervalo_es_REPRODUCIBLE_bit_a_bit():
    """Un intervalo que cambia de límites cada vez que se recalcula no se
    puede citar: «el que salió» sería el que a uno le conviniera."""
    protocolo = _protocolo()
    crudos = _crudos()
    primero = aplicar_regla_de_cierre(crudos, protocolo.regla_de_cierre, motor="lightgbm")
    segundo = aplicar_regla_de_cierre(crudos, protocolo.regla_de_cierre, motor="lightgbm")
    assert [d["intervalo_de_la_distancia"] for d in primero["detalle"]] == \
           [d["intervalo_de_la_distancia"] for d in segundo["detalle"]]


def test_el_percentil_es_el_de_105_C2_y_no_una_segunda_implementacion():
    """«Nunca una segunda implementación», que es lo que el propio protocolo
    escribe en `metricas_por_tarea.intervalos`. Se comprueba por
    COMPORTAMIENTO: si `_percentil` de 105-C2 se rompiera, este intervalo
    tendría que romperse con él."""
    import matrixai.estudio.incertidumbre as inc
    original = inc._percentil
    try:
        inc._percentil = lambda ordenados, proporcion: 42.0
        assert _intervalo_pareado([1.0, 2.0, 3.0]) == (42.0, 42.0), (
            "el intervalo no pasa por `_percentil` de 105-C2: tiene su propia copia")
    finally:
        inc._percentil = original
    assert _intervalo_pareado([1.0, 2.0, 3.0]) != (42.0, 42.0)

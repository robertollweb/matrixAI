# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""111-C1 — LO QUE IMPIDE QUE UN TICK VERDE MIENTA EN CI.

Estas pruebas nacen de una frase del encargo: **un tick verde cuando la
verificación no pudo comprobar tres de las cuatro evidencias convierte «no se
midió» en «está bien» a la vista de un equipo entero, y en CI nadie abre el
registro de un trabajo verde.**

El módulo se escribió con eso delante y quedó **sin una sola prueba**: el
agente que lo escribió se quedó sin sesión justo al terminarlo. Código sin
prueba no se commitea, así que aquí está lo que lo sostiene.

CONVENCIÓN DEL FICHERO: funciones `test_*` sueltas, como el resto de
`tests/test_c1*`. Nada de clases `XxxTest`, que pytest NO recoge.
"""
from __future__ import annotations

import pytest

from matrixai.ci.verify_action import (
    ALCANCES,
    REQUERIDOS_POR_DEFECTO,
    SALIDAS,
    PeticionDeEngines,
    alcances_del_informe,
    anotaciones,
    decidir,
    resumen_markdown,
)


def _informe(**estados: str) -> dict:
    """Un informe de `verify` con el estado que se le pida por alcance.

    La forma es la REAL (`stages`, un diccionario por nombre), medida sobre
    el módulo y no supuesta: mi primera versión inventó `scopes` como lista y
    los cuatro alcances salieron `MISSING` — lo que habría hecho pasar tres
    pruebas por el motivo equivocado.
    """
    return {"stages": {n: {"status": estados.get(n, "PASS"),
                           "reason": (None if estados.get(n, "PASS") == "PASS"
                                      else f"motivo de {n}")}
                       for n in ALCANCES}}


# ---------------------------------------------------------------------------
# La regla que sostiene todo lo demás
# ---------------------------------------------------------------------------

def test_NUNCA_mas_verde_que_verify_aunque_no_se_exija_nada():
    """`require: none` NO puede convertir un paquete imposible de comprobar
    en un tick verde.

    Es la regla que este corte existe para tener. Sin ella, quien no quiera
    ver rojos escribe `require: none` en su flujo y **la acción deja de
    medir sin dejar de parecer que mide**.
    """
    v = decidir(_informe(manifest="NOT_RUN", R1="NOT_RUN", training="NOT_RUN",
                         R3="NOT_RUN"),
                codigo_de_verify=SALIDAS["sin_realizar"], requeridos=())
    assert v.codigo != SALIDAS["ok"], "con `require: none` y verify en 3, salió VERDE"
    assert v.nota and "never reports greener than verify" in v.nota


def test_y_LA_OTRA_MITAD_un_paquete_entero_SI_sale_verde():
    """Sin esto, una acción que siempre saliera roja pasaría la de arriba."""
    v = decidir(_informe(), codigo_de_verify=SALIDAS["ok"])
    assert v.codigo == SALIDAS["ok"]
    assert v.titulo == "VERIFIED"
    assert v.fallidos == () and v.sin_realizar == ()


def test_require_solo_puede_APRETAR_nunca_aflojar():
    """Exigir de más pone rojo lo que estaba verde; no exigir NO pone verde
    lo que estaba rojo."""
    parcial = _informe(training="NOT_RUN", R3="NOT_RUN")
    # No exigidos: verde, porque `verify` no vio nada mal.
    suelto = decidir(parcial, codigo_de_verify=SALIDAS["ok"], requeridos=("manifest", "R1"))
    assert suelto.codigo == SALIDAS["ok"]
    # Exigidos: rojo. `require` APRIETA.
    apretado = decidir(parcial, codigo_de_verify=SALIDAS["ok"], requeridos=ALCANCES)
    assert apretado.codigo == SALIDAS["sin_realizar"]
    assert set(apretado.exigidos_sin_realizar) == {"training", "R3"}
    # Y un FALLO no se afloja quitándolo de los exigidos.
    roto = decidir(_informe(manifest="FAIL"), codigo_de_verify=SALIDAS["fallo"], requeridos=())
    assert roto.codigo == SALIDAS["fallo"], "un FAIL salió verde al no exigirlo"


# ---------------------------------------------------------------------------
# Lo que no se realizó se VE, verde o rojo
# ---------------------------------------------------------------------------

def test_los_NO_REALIZADOS_salen_en_el_resumen_AUNQUE_el_trabajo_sea_verde():
    """El caso peligroso no es el rojo: es el VERDE con evidencias sin medir.

    Ahí es donde «no se midió» se lee como «está bien», y es justo cuando
    nadie abre el registro.
    """
    v = decidir(_informe(training="NOT_RUN", R3="INCOMPARABLE"),
                codigo_de_verify=SALIDAS["ok"], requeridos=("manifest", "R1"))
    assert v.codigo == SALIDAS["ok"], "el montaje de esta prueba ya no es el verde"
    texto = resumen_markdown(v, paquete="p.zip", orden=ALCANCES,
                             engines=PeticionDeEngines(pedido=False, mirado_en=("x",)))
    assert "training" in texto and "R3" in texto
    assert "motivo de training" in texto, "el motivo literal no viaja al resumen"
    assert "motivo de R3" in texto


def test_y_llevan_ANOTACION_que_es_lo_que_se_ve_sin_abrir_el_registro():
    v = decidir(_informe(training="NOT_RUN"), codigo_de_verify=SALIDAS["ok"],
                requeridos=("manifest", "R1"))
    marcas = anotaciones(v)
    assert any("::warning" in m and "training" in m for m in marcas), marcas


def test_EL_TITULAR_Y_EL_SEMAFORO_dicen_lo_mismo():
    """Un titular tranquilizador sobre un trabajo rojo es peor que ninguno.

    Lo cazó el propio autor del módulo: con `require: none`, un paquete con
    **0 de 4 alcances realizados** salía titulado «PARTIAL» —que se lee como
    «verde con matices»— mientras el trabajo se iba en rojo con un 3.
    """
    v = decidir(_informe(manifest="NOT_RUN", R1="NOT_RUN", training="NOT_RUN",
                         R3="NOT_RUN"),
                codigo_de_verify=SALIDAS["sin_realizar"], requeridos=())
    assert v.codigo != SALIDAS["ok"]
    assert v.titulo == "NOT VERIFIED", (
        f"código {v.codigo} (rojo) con el titular «{v.titulo}»")


# ---------------------------------------------------------------------------
# Ausencia de veredicto NO es un aprobado
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estado", ["NOT_RUN", "INCOMPARABLE"])
def test_NOT_RUN_e_INCOMPARABLE_no_son_aprobados(estado):
    """Son ausencia de veredicto por motivos distintos, y ninguno es un PASS."""
    alcances = alcances_del_informe(_informe(R1=estado), requeridos=("R1",))
    r1 = next(a for a in alcances if a.nombre == "R1")
    assert not r1.realizado
    assert decidir(_informe(R1=estado), codigo_de_verify=SALIDAS["ok"],
                   requeridos=("R1",)).codigo == SALIDAS["sin_realizar"]


def test_un_alcance_AUSENTE_del_informe_no_se_da_por_bueno():
    """Un informe que no trae un alcance no está diciendo que salió bien."""
    v = decidir({"stages": {"manifest": {"status": "PASS"}}},
                codigo_de_verify=SALIDAS["ok"], requeridos=ALCANCES)
    assert v.codigo == SALIDAS["sin_realizar"]
    assert set(v.exigidos_sin_realizar) >= {"R1", "training", "R3"}


def test_lo_exigido_POR_DEFECTO_es_lo_que_una_maquina_ajena_puede_realizar():
    """`training` y `R3` fuera por MEDIDA, no por olvido: `R3` solo sabe
    comparar entornos idénticos y un runner de GitHub no iguala nunca el de
    otra máquina. Exigirlos pondría en rojo a los paquetes honestos, y un
    rojo que no es un fallo enseña a poner `continue-on-error`."""
    assert REQUERIDOS_POR_DEFECTO == ("manifest", "R1")
    assert set(ALCANCES) - set(REQUERIDOS_POR_DEFECTO) == {"training", "R3"}


# ---------------------------------------------------------------------------
# El YAML y el módulo son DOS SITIOS: que no divergan
# ---------------------------------------------------------------------------

def _action_yml() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1]
            / ".github" / "actions" / "matrixai-verify" / "action.yml").read_text(encoding="utf-8")


def test_el_YAML_no_pasa_NINGUNA_bandera_que_el_modulo_no_acepte():
    """*Dos sitios declarando lo mismo acaban divergiendo*, y aquí divergir
    significa que la acción **revienta en CI y en ningún otro sitio**: el
    YAML solo se ejecuta dentro de GitHub, donde no hay suite que lo mire.

    Por eso la comprobación va al revés de lo intuitivo: no se prueba el
    YAML corriéndolo —no se puede—, se prueba que **lo que escribe encaja
    con lo que el módulo declara**.
    """
    import re
    from matrixai.ci import verify_action

    fuente = verify_action.__doc__ and ""  # el módulo se lee por su parser real
    import argparse
    import inspect
    codigo = inspect.getsource(verify_action.main)
    aceptadas = set(re.findall(r'p\.add_argument\("(--[a-z-]+)"', codigo))
    assert aceptadas, "no se pudo leer ninguna bandera del módulo: la prueba no mide nada"

    usadas = set(re.findall(r"^\s*--([a-z-]+)\s", _action_yml(), re.M))
    usadas = {f"--{u}" for u in usadas}
    assert usadas, "el YAML no pasa ninguna bandera: ¿sigue llamando al módulo?"
    assert usadas <= aceptadas, (
        f"el YAML pasa banderas que el módulo no acepta: {sorted(usadas - aceptadas)}")


def test_y_el_YAML_no_inventa_un_VALOR_POR_DEFECTO_distinto_del_modulo():
    """Un defecto distinto en cada sitio es peor que no tener defecto: quien
    lea el YAML creerá una cosa y quien lea el módulo, otra — y la que manda
    es la del YAML, que es la que nadie prueba."""
    # Con el PARSER de YAML, no con una regex: mi primera versión usaba una,
    # era codiciosa entre bloques y se comía el primer carácter de cada valor
    # (`manifest,R1` salía `anifest,R1`). Habría dado un rojo que no era del
    # producto. Y de paso esto lee el fichero como lo lee GitHub.
    yaml = pytest.importorskip("yaml")
    accion = yaml.safe_load(_action_yml())
    entradas = accion["inputs"]
    assert entradas["require"]["default"] == ",".join(REQUERIDOS_POR_DEFECTO), (
        f"el YAML exige por defecto {entradas['require']['default']!r} y el módulo "
        f"{','.join(REQUERIDOS_POR_DEFECTO)!r}")
    # `package` es la única sin defecto, y tiene que seguir siendo obligatoria:
    # un defecto ahí verificaría un paquete que nadie pidió verificar.
    assert entradas["package"]["required"] is True
    assert "default" not in entradas["package"]


def test_el_YAML_SIGUE_SIENDO_TONTO():
    """No decide nada. Si crece, es que algo se escapó al único sitio donde
    no hay forma de probarlo.

    El techo es generoso a propósito —no es un límite de estilo— pero un
    `if`, un `case` o una tubería de `jq` dentro del YAML sí serían la señal
    de que la decisión se ha mudado allí."""
    yml = _action_yml()
    for senal in ("if [", "case ", "| jq", "&&"):
        cuerpo = yml.split("runs:", 1)[1]
        assert senal not in cuerpo, (
            f"el YAML ha empezado a DECIDIR (encontrado {senal!r}): eso vive en "
            "`verify_action.py`, donde la suite lo prueba")
    assert yml.count("shell: bash") <= 3

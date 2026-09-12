"""101-C3 — el TERCER defecto de cableado de la pasada exploratoria de Fase 0.

QUÉ SE MIDIÓ (2026-09-12, sobre `benchmarks/fase0/pasada_exploratoria_101_c3_
resultado.json` del 2026-09-07). 15 intentos perdidos, los 15 del motor
`matrixai.dense.torch_cpu`, los 15 en el dataset `pc4`, los 15 con el mismo
mensaje: `IndexError: list index out of range`. `pc4` quedaba fuera de la regla
de cierre de 101-C1 («un fallo = dataset perdido para ese motor») y LightGBM
figuraba ahí como mejor SIN RIVAL.

LA CADENA COMPLETA, medida eslabón a eslabón y no supuesta:

1. `matrixai/training/dataset_project.py` fabrica desde el CSV un prompt que
   incrusta los NOMBRES DE LAS COLUMNAS en un bloque `FEATURES:`.
2. `pc4` tiene `CYCLOMATIC_COMPLEXITY`, `DESIGN_COMPLEXITY`,
   `ESSENTIAL_COMPLEXITY` y `NORMALIZED_CYLOMATIC_COMPLEXITY`. Dentro de esos
   nombres vive la subcadena `complex`, que está en `_COMPOSITE_HINTS`.
   `_prompt_wants_composite` barría el prompt ENTERO -> True.
3. -> `CompositeNetworkGenerator(force_residual=True)` -> un `.mxai` con
   `BLOCK res1` y `architecture_decision = {"kind": "residual",
   "source": "prompt_hints"}`. Nadie había pedido nada: lo dijo una columna.
4. El parser devuelve `NetworkSpec(kind="composite_network", layers=[], ...)`:
   el cuerpo vive en `top_layers`/`blocks`, NO en `layers`.
5. `dense_forward` iteraba `network.layers` -> CERO vueltas -> devolvía el
   VECTOR DE ENTRADA tal cual como si fuera la salida de la red: 37 números
   para una red de 2 unidades de salida (medido).
6. El motor denso de `matrixai-engines` los tomaba por probabilidades:
   `etiquetas[salida.index(max(salida))]` con 2 etiquetas y un índice hasta 36
   -> `IndexError` sin una sola palabra sobre la red.

De los 12 datasets de la pasada, `pc4` es el ÚNICO cuyos nombres de columna
chocan con la lista de hints (comprobados los 12) — y es exactamente el único
cuyos 15 intentos densos se perdieron.
"""
from __future__ import annotations

import pytest

from matrixai.forward import DenseForwardError, dense_forward, dense_forward_trace
from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai.parameters.network_params import build_network_parameter_set
from matrixai.parser import parse_text
from matrixai.playground import _prompt_is_sequence, _prompt_wants_composite

# Los nombres REALES de las columnas de `pc4` que dispararon el fallo.
_COLUMNAS_PC4_CULPABLES = (
    "cyclomatic_complexity", "design_complexity", "essential_complexity",
    "normalized_cylomatic_complexity",
)

_MXAI_COMPUESTA = """PROJECT P

VECTOR Entrada[3]
  a: Scalar
  b: Scalar
  c: Scalar
END

NETWORK Red
  INPUT Entrada
  LAYER Dense units=4 activation=relu
  BLOCK res1
    LAYER Dense units=4 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT salida: ProbabilityMap[no, si]
END

GRAPH
  Entrada -> Red
END
"""

_MXAI_DENSA_PLANA = """PROJECT P

VECTOR Entrada[3]
  a: Scalar
  b: Scalar
  c: Scalar
END

NETWORK Red
  INPUT Entrada
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT salida: ProbabilityMap[no, si]
END

GRAPH
  Entrada -> Red
END
"""


def _prompt_desde_columnas(*columnas: str) -> str:
    """El MISMO molde que `dataset_project.py` fabrica desde un CSV."""
    lineas = "\n".join(f"  {c}: Scalar en [0.0, 1.0]" for c in columnas)
    return "clasificar\nFEATURES:\n" + lineas + "\nSALIDA: y: ProbabilityMap[no, si]\n"


def _parametros_de(programa):
    """Un `ParameterSet` real para una red PARSEADA: sus capas traen
    `input_shape=[]` hasta que la inferencia de formas las resuelve, igual que
    hace el playground antes de entrenar."""
    from matrixai.types import check_network_types

    red = programa.networks[0]
    vector_map = {v.name: v for v in programa.vectors}
    tipos = check_network_types(red, vector_map)
    capas = tipos.resolved_layers if tipos.resolved_layers else red.layers
    ps = build_network_parameter_set(red, capas, "h", seed=7)
    return ps, [0.5] * len(vector_map[red.input].fields)


def _red_densa_plana(*capas: tuple[int, str], input_dim: int = 3) -> NetworkSpec:
    layers, dim = [], input_dim
    for i, (units, activacion) in enumerate(capas, start=1):
        layers.append(DenseLayerSpec(index=i, units=units, activation=activacion,
                                     input_shape=[dim], output_shape=[units]))
        dim = units
    return NetworkSpec(name="Red", input="Entrada", layers=layers, output="y",
                       output_type_str="Scalar")


# ---------------------------------------------------------------------------
# Mitad 1 — el forward denso se planta ante una red que no entiende, en vez de
# devolver la entrada disfrazada de salida.
# ---------------------------------------------------------------------------

def test_la_red_compuesta_deja_layers_vacio_por_construccion():
    """La premisa de todo lo demás, medida y no supuesta."""
    red = parse_text(_MXAI_COMPUESTA).networks[0]
    assert red.kind == "composite_network"
    assert red.layers == [], "el cuerpo vive en top_layers/blocks, no en layers"
    assert red.top_layers and red.blocks


def test_dense_forward_rechaza_una_red_compuesta():
    red = parse_text(_MXAI_COMPUESTA).networks[0]
    with pytest.raises(DenseForwardError) as exc:
        dense_forward(red, build_network_parameter_set(
            _red_densa_plana((2, "softmax")), [], "h"), [0.1, 0.2, 0.3])
    mensaje = str(exc.value)
    assert "Red" in mensaje and "composite_forward" in mensaje, (
        "el error tiene que nombrar la red y el forward que SÍ le toca")


def test_dense_forward_ya_no_devuelve_la_entrada_disfrazada_de_prediccion():
    """EL FALLO EXACTO DE `pc4`: 37 entradas saliendo por una red de 2 salidas.

    Sin el rechazo, esto devolvía `entrada` intacta y el error sólo aparecía
    cien líneas más allá, en el que llamaba, como un `IndexError` mudo.
    """
    red = parse_text(_MXAI_COMPUESTA).networks[0]
    entrada = [0.1, 0.2, 0.3]
    ps = build_network_parameter_set(_red_densa_plana((2, "softmax")), [], "h")
    try:
        salida = dense_forward(red, ps, entrada)
    except DenseForwardError:
        return
    pytest.fail(f"devolvió {salida!r} en vez de negarse; ¿es la entrada? "
                f"{salida == entrada}")


def test_dense_forward_rechaza_una_red_sin_capas():
    with pytest.raises(DenseForwardError, match="no layers"):
        dense_forward(_red_densa_plana(), build_network_parameter_set(
            _red_densa_plana((2, "softmax")), [], "h"), [0.1, 0.2, 0.3])


def test_dense_forward_rechaza_un_bloque_transformer_por_su_nombre():
    """Espejo del guardián que `composite_forward` ya tenía para lo mismo."""
    red = _red_densa_plana((2, "softmax"))
    red = NetworkSpec(name=red.name, input=red.input, layers=red.layers,
                      output=red.output, output_type_str=red.output_type_str,
                      kind="composite_network", transformer_blocks=[object()])
    with pytest.raises(DenseForwardError, match="transformer_network_forward"):
        dense_forward(red, build_network_parameter_set(
            _red_densa_plana((2, "softmax")), [], "h"), [0.1, 0.2, 0.3])


def test_la_otra_mitad_una_red_densa_plana_sigue_pasando_igual():
    """El rechazo no puede haberse comido el camino bueno."""
    red = _red_densa_plana((4, "relu"), (2, "softmax"))
    ps = build_network_parameter_set(red, red.layers, "h", seed=7)
    salida = dense_forward(red, ps, [0.1, 0.2, 0.3])
    assert len(salida) == 2, "la salida es la de la ÚLTIMA capa, no la entrada"
    assert salida != [0.1, 0.2, 0.3]
    assert abs(sum(salida) - 1.0) < 1e-9
    assert dense_forward_trace(red, ps, [0.1, 0.2, 0.3]).output == salida


def test_la_otra_mitad_una_red_densa_del_parser_sigue_pasando_igual():
    programa = parse_text(_MXAI_DENSA_PLANA)
    red = programa.networks[0]
    assert red.kind == "dense_network"
    ps, entrada = _parametros_de(programa)
    assert len(dense_forward(red, ps, entrada)) == 2


# ---------------------------------------------------------------------------
# Mitad 2 — los hints de arquitectura leen lo que se PIDIÓ, no los nombres de
# las columnas que describen los datos.
# ---------------------------------------------------------------------------

def test_una_columna_llamada_complexity_no_pide_una_red_residual():
    """EL CASO `pc4`, con sus nombres de columna reales."""
    prompt = _prompt_desde_columnas(*_COLUMNAS_PC4_CULPABLES)
    assert "complex" in prompt.lower(), "la subcadena sigue ahí: el barrido era el fallo"
    assert not _prompt_wants_composite(prompt)


def test_una_columna_llamada_temporal_no_enruta_a_secuencias():
    prompt = _prompt_desde_columnas("temporal_id", "precio")
    assert "temporal" in prompt.lower()
    assert not _prompt_is_sequence(prompt)


@pytest.mark.parametrize("prosa", [
    "red con bloques residuales y LayerNorm",
    "a complex nonlinear model",
    "quiero una red profunda con dropout",
])
def test_la_otra_mitad_la_prosa_que_pide_residual_la_sigue_pidiendo(prosa):
    assert _prompt_wants_composite(prosa)
    # Y la sigue pidiendo aunque venga acompañada de un bloque de campos.
    assert _prompt_wants_composite(prosa + "\nFEATURES:\n  precio: Scalar en [0.0, 1.0]\n")


@pytest.mark.parametrize("prosa", [
    "predecir una serie temporal de ventas",
    "modelo de sequence para el histórico",
])
def test_la_otra_mitad_la_prosa_de_secuencia_la_sigue_pidiendo(prosa):
    assert _prompt_is_sequence(prosa)
    assert _prompt_is_sequence(prosa + "\nFEATURES:\n  precio: Scalar en [0.0, 1.0]\n")


def test_un_prompt_normal_sigue_sin_pedir_nada_de_esto():
    assert not _prompt_wants_composite("clasificar spam de email")
    assert not _prompt_is_sequence("clasificar pacientes tabulares")


# ---------------------------------------------------------------------------
# Y por el PRODUCTO, no sólo por la función: el mismo CSV que `pc4` le daba al
# motor denso ya no sale con una red que su forward no sabe ejecutar.
# ---------------------------------------------------------------------------

def test_un_csv_con_columnas_de_complejidad_genera_una_red_densa_ejecutable():
    from matrixai.training.dataset_project import generate_project_from_dataset

    columnas = list(_COLUMNAS_PC4_CULPABLES) + ["loc_total", "branch_count"]
    filas = [",".join(columnas + ["c"])]
    for i in range(60):
        valores = [f"{(i * 7 + j) % 97 / 97:.3f}" for j in range(len(columnas))]
        filas.append(",".join(valores + ["TRUE" if i % 4 == 0 else "FALSE"]))
    proyecto = generate_project_from_dataset(
        "\n".join(filas) + "\n", "c", locale="es", column_type_overrides={"c": "categorical"})

    programa = parse_text(proyecto["mxai"])
    red = programa.networks[0]
    assert red.kind == "dense_network", (
        "un nombre de columna no puede elegir la arquitectura; "
        f"architecture_decision={proyecto.get('architecture_decision')}")
    # Y el forward denso la ejecuta de verdad: salida de la ÚLTIMA capa, no la
    # entrada devuelta tal cual (que es lo que rompía `pc4`).
    ps, entrada = _parametros_de(programa)
    salida = dense_forward(red, ps, entrada)
    assert len(salida) == red.layers[-1].units
    assert len(salida) < len(entrada) and salida != entrada[:len(salida)]

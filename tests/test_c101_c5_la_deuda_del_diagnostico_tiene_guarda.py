"""LA ÚLTIMA HEURÍSTICA SIN DECLARACIÓN, Y SU FECHA DE CADUCIDAD (2026-09-16).

**De dónde viene.** El 2026-09-15 se reparó en la raíz que un nivel DECLARADO no
es un dato ausente: `_NULL_TOKENS` incluye `"none"` y `house_prices_nominal`
declara `None` en su cabecera ARFF como categoría válida —«sin revestimiento de
mampostería», **864 de 1.460 filas**—, así que el 59 % de una columna buena se
leía como vacía. Costó parar y relanzar una medición de 17 horas. La reparación
introdujo `tokens_de_ausencia`: quien PRODUCE el CSV declara cómo marca la
ausencia, y con declaración la heurística no se aplica.

**Lo que quedó fuera a propósito, y está bien que quedara.**
`matrixai/training/diagnostico.py` sigue usando la heurística siempre. Se
declaró como DEUDA y no se reparó, con este motivo: *«hoy no diverge de nada
porque nadie le pasa tokens y no está en el camino del denso; ampliaba
superficie sin defecto medido detrás»*. Esa decisión es correcta — y **medida**
hoy: `diagnosticar_csv` ni siquiera ACEPTA `tokens_de_ausencia`, así que no
puede propagar una declaración aunque exista.

**PERO NADIE VIGILABA SU CONDICIÓN DE CADUCIDAD, y ese es el hueco que este
fichero cierra.** El día que alguien añada el parámetro a `diagnosticar_csv` —o
que el Studio empiece a declarar— hay que hilarlo hasta las TRES llamadas
internas a `_is_null`. Si se añade el parámetro y se olvida una, el diagnóstico
volvería a leer un nivel declarado como ausente **en silencio**, que es
exactamente el defecto que costó la medición. Una deuda declarada sin guarda es
una nota que envejece: dice «hoy no diverge» y nadie vuelve a comprobar el
«hoy».

**Está escrito para morir.** Cuando el hueco se cierre, estas pruebas se ponen
rojas y eso es el aviso, no un estorbo: la reescritura será exigir lo contrario
—que las tres llamadas SÍ reciban la declaración—.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

_RUTA = Path(__file__).resolve().parent.parent / "matrixai" / "training" / "diagnostico.py"


def _llamadas_a_is_null() -> list[ast.Call]:
    arbol = ast.parse(_RUTA.read_text(encoding="utf-8"))
    return [n for n in ast.walk(arbol)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_is_null"]


def test_la_funcion_que_sabe_de_declaraciones_SIGUE_aceptandolas():
    """El otro lado de la deuda: si `_is_null` dejara de aceptar la declaración,
    esta deuda no sería deuda, sería un callejón sin salida."""
    from matrixai.training.dataset_analysis import _is_null

    assert "tokens_de_ausencia" in inspect.signature(_is_null).parameters


def test_el_diagnostico_NO_acepta_la_declaracion_todavia():
    """LA PRUEBA ESCRITA PARA MORIR.

    Mientras `diagnosticar_csv` no acepte `tokens_de_ausencia`, la heurística de
    dentro no puede divergir de ninguna declaración: no hay declaración que
    contradecir. El día que se añada, esto se pone rojo — y lo que toca entonces
    es hilarlo hasta las tres llamadas de abajo, no borrar la prueba.
    """
    from matrixai.training.diagnostico import diagnosticar_csv

    parametros = inspect.signature(diagnosticar_csv).parameters
    assert "tokens_de_ausencia" not in parametros, (
        "`diagnosticar_csv` ya acepta la declaración de ausencia. ESTA PRUEBA HA "
        "CUMPLIDO SU FUNCIÓN: ahora hay que comprobar que llega hasta las TRES "
        "llamadas internas a `_is_null` (ver la prueba de abajo) y reescribir "
        "las dos para exigir lo contrario. Si se añade el parámetro y se olvida "
        "una llamada, el diagnóstico vuelve a leer un nivel declarado como "
        "ausente EN SILENCIO — el defecto que costó una medición de 17 horas.")


def test_las_TRES_llamadas_internas_siguen_usando_la_heuristica():
    """El hecho, leído del código y no supuesto.

    Se cuenta cuántas son a propósito: si mañana aparece una cuarta, el número
    cambia y alguien tiene que mirar si también necesita la declaración. Un
    `all(...)` sin recuento dejaría entrar una llamada nueva sin ruido.
    """
    llamadas = _llamadas_a_is_null()
    assert len(llamadas) == 3, (
        f"`diagnostico.py` tiene ahora {len(llamadas)} llamadas a `_is_null` y no "
        "3. Si es una nueva, mira si necesita la declaración de ausencia; si se "
        "ha ido una, actualiza este número. El recuento existe para que un "
        "cambio aquí no pase desapercibido.")
    for llamada in llamadas:
        assert not llamada.keywords, (
            f"la llamada de la línea {llamada.lineno} ya pasa argumentos con "
            "nombre: comprueba si uno es `tokens_de_ausencia` y, si lo es, "
            "reescribe estas pruebas — el hueco se está cerrando.")
        assert len(llamada.args) == 1, (
            f"la llamada de la línea {llamada.lineno} pasa "
            f"{len(llamada.args)} argumentos posicionales: el segundo de "
            "`_is_null` es `tokens_de_ausencia`, así que el hueco se está "
            "cerrando por la puerta de atrás y estas pruebas tienen que "
            "reescribirse con él.")

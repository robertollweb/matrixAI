# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Dos contaminantes del camino de LECTURA de la pasada de Fase 0, medidos el
2026-09-13 y reparados antes de volver a medir.

**1. El `?` del estándar ARFF no llegaba como faltante a NINGÚN motor.**
`pasada_exploratoria_101_c3.py` tenía su propia copia de `cargar_arff` que
hacía `None if texto == ""`, y el marcador de ausencia del formato ARFF es
`"?"`, no la cadena vacía. `scipy` lo entrega literal en las columnas
nominales (en las numéricas ya sale `NaN`), así que el `"?"` entraba en
`ajustar_preparacion` como UNA CATEGORÍA MÁS: ni se imputaba ni encendía el
indicador de faltante, y eso vale para los cuatro motores. Medido sobre los
doce de la pasada: `sick` trae **150 de sus 3.772 filas con `sex == "?"`**.

Y no era un descuido aislado, era la firma de la duplicación: el lector bueno
(`lector_arff.py`, 101-C5) ya lo hacía bien desde el 09-12. Dos sitios
declarando lo mismo acabaron divergiendo, así que la reparación es que quede
UNO.

**2. Cinco de los cuarenta traen una columna identificadora que el catálogo
no cuenta**, y metida al modelo es una fuga: a `splice` le entraba
`Instance_name`, **3.178 valores distintos para 3.190 filas**. Ninguno de los
cinco está entre los doce ya medidos, así que lo publicado no se apoya en eso
— pero la pasada de los cuarenta sí lo haría.

El criterio NO es «muchos niveles»: eso tiraría `KDDCup09_appetency.Var200`,
una categórica legítima de **15.415 niveles**. Es que sobre las FILAS los dos
números no se rozan (0,9962 frente a 0,3083), y aun así el umbral no se elige
aquí: **cuántas** sobran lo dice el catálogo registrado y **cuál** lo dice el
detector de identificadores del núcleo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
_FASE0 = _RAIZ / "benchmarks" / "fase0"
if str(_FASE0) not in sys.path:
    sys.path.insert(0, str(_FASE0))

from benchmarks.fase0.lector_arff import (  # noqa: E402
    cargar, columnas_que_el_nucleo_llama_identificador)

DATOS = Path("/home/deployer/fase0_openml_datos/arff")
RUTA_PROTOCOLO = _FASE0 / "protocolo_exploratorio.json"
PROTOCOLO = json.loads(RUTA_PROTOCOLO.read_text(encoding="utf-8"))
POR_NOMBRE = {e["nombre"]: e for e in PROTOCOLO["datasets"]}

#: Los doce de la pasada exploratoria del 101-C3 — los que YA se midieron y
#: sobre los que se apoya el veredicto publicado.
LOS_DOCE = ("kc2", "climate-model-simulation-crashes", "diabetes", "breast-w",
            "pc4", "pc1", "sick", "jm1", "ozone-level-8hr", "PhishingWebsites",
            "Internet-Advertisements", "wilt")

con_los_arff = pytest.mark.skipif(
    not DATOS.is_dir(), reason="los ARFF del protocolo no están descargados")


def _cargar(nombre, con_catalogo=True):
    e = POR_NOMBRE[nombre]
    return cargar(DATOS / f"{e['data_id']}.arff", e["columna_objetivo"],
                  e["n_columnas"] if con_catalogo else None)


def _arff(tmp_path, cuerpo, nombre="min.arff"):
    ruta = tmp_path / nombre
    ruta.write_text(cuerpo, encoding="utf-8")
    return ruta


# --------------------------------------------------------------------------
# 1. El `?` es un faltante
# --------------------------------------------------------------------------

def test_el_interrogante_de_un_nominal_llega_como_FALTANTE(tmp_path):
    """El caso más pequeño que dispara el defecto: tres filas y un nominal.

    `scipy` devuelve la cadena `"?"`, no `""` ni `None`. Es la línea exacta
    que la copia de la pasada tenía mal.
    """
    ruta = _arff(tmp_path, "@relation r\n@attribute sexo {F,M}\n"
                           "@attribute edad NUMERIC\n@attribute clase {si,no}\n"
                           "@data\nF,30,si\n?,?,no\nM,40,si\n")
    leido = cargar(ruta, "clase")
    assert leido.filas[1]["sexo"] is None, "el '?' nominal no llegó como faltante"
    assert leido.filas[1]["edad"] is None, "el '?' numérico no llegó como faltante"
    # La mitad POSITIVA: no se ha convertido en faltante todo lo demás.
    assert leido.filas[0]["sexo"] == "F"
    assert leido.filas[2]["edad"] == 40.0


@con_los_arff
def test_sick_pierde_sus_150_interrogantes_por_el_camino_de_la_PASADA():
    """Probar la función no es probar el producto: esto entra por donde entra
    la pasada de verdad, `pasada_exploratoria_101_c3.cargar_arff`.

    150 de 3.772 filas de `sick` traen `sex == "?"`, y `sick` es uno de los
    DOCE ya medidos.
    """
    pasada = pytest.importorskip("pasada_exploratoria_101_c3")
    filas, objetivo = pasada.cargar_arff(38)
    assert objetivo == "Class"
    assert len(filas) == 3772
    assert sum(1 for f in filas if f["sex"] is None) == 150
    # Un aserto negativo lo pasa una lista vacía: la mitad positiva es que las
    # otras 3.622 siguen trayendo su valor.
    assert sum(1 for f in filas if f["sex"] in ("F", "M")) == 3622
    hay_interrogantes = [(f["row_id"], c) for f in filas for c, v in f.items() if v == "?"]
    assert hay_interrogantes == [], f"quedan '?' como valor: {hay_interrogantes[:5]}"


@con_los_arff
@pytest.mark.parametrize("admite", [False, True])
def test_el_interrogante_no_llega_como_CATEGORIA_a_ningun_motor(admite):
    """El defecto no era del motor denso: afecta a los cuatro, porque el `"?"`
    se colaba antes, en `ajustar_preparacion`.

    `admite` recorre las dos capacidades que distinguen a los motores
    (`admite_categoricas`/`admite_faltantes`): en las dos, la fila ausente
    tiene que salir con el centinela de FALTANTE del núcleo, no con `"?"`.
    """
    from matrixai.training.preparacion import (CATEGORIA_FALTANTE, ajustar_preparacion,
                                               transformar_fila)
    pasada = pytest.importorskip("pasada_exploratoria_101_c3")
    filas, objetivo = pasada.cargar_arff(38)
    predictores = tuple(k for k in filas[0] if k not in ("row_id", objetivo))
    sin_dato = [f for f in filas if f["sex"] is None][:60]
    con_dato = [f for f in filas if f["sex"] is not None][:240]
    assert len(sin_dato) == 60 and len(con_dato) == 240, (
        "no hay filas sin dato en `sex`: el '?' no llegó como faltante y esta "
        "prueba se quedaría sin caso que mirar")

    politica = ajustar_preparacion(sin_dato + con_dato, objetivo=objetivo,
                                   columnas=predictores, admite_categoricas=admite,
                                   admite_faltantes=admite)
    transformada = transformar_fila(sin_dato[0], politica)
    assert transformada["sex"] == CATEGORIA_FALTANTE
    # La mitad positiva: una fila CON dato sigue trayendo su categoría.
    assert transformar_fila(con_dato[0], politica)["sex"] in ("F", "M")
    # Y `"?"` ha dejado de ser una categoría conocida de la columna.
    propuesta = next(p for p in politica.columnas if p.columna == "sex")
    assert "?" not in propuesta.categorias_conocidas
    assert set(propuesta.categorias_conocidas) == {"F", "M"}


# --------------------------------------------------------------------------
# 2. La columna identificadora que el catálogo dice que sobra
# --------------------------------------------------------------------------

@con_los_arff
@pytest.mark.parametrize("nombre,columna,distintos,filas", [
    ("splice", "Instance_name", 3178, 3190),
    ("house_prices_nominal", "Id", 1460, 1460),
    ("house_sales", "id", 21436, 21613),
])
def test_la_columna_identificadora_SOBRANTE_se_excluye_y_se_dice_por_que(
        nombre, columna, distintos, filas):
    """Los tres casos donde el catálogo declara una columna menos que el ARFF
    y el núcleo señala cuál. `Allstate_Claims_Severity` es el cuarto y se deja
    fuera de la prueba por coste (188.318 filas); el barrido de los cuarenta
    lo cubre.
    """
    leido = _cargar(nombre)
    assert columna in leido.columnas_excluidas
    assert columna not in leido.filas[0], "la columna sigue en las filas"
    motivo = leido.motivos_de_exclusion[columna]
    assert str(distintos) in motivo and str(filas) in motivo, (
        f"el motivo no trae los números con los que se decidió: {motivo}")
    assert leido.avisos == []


@con_los_arff
def test_tras_la_exclusion_los_CUARENTA_cuadran_con_su_catalogo():
    """La comprobación que de verdad cierra el hallazgo, y la que impide que
    esto se convierta en una poda a ojo: después de excluir, TODOS los
    datasets traen exactamente las columnas que su catálogo declara — ni uno
    de más (quedaría un identificador dentro) ni uno de menos (se habría
    tirado una columna legítima).

    Solo se leen las cabeceras: los 254 MB de los cuarenta ficheros no hacen
    falta para contar `@attribute`, y las columnas STRING se descuentan porque
    el lector las quita.
    """
    import re
    atributo = re.compile(r"^\s*@attribute\s+(?P<n>'[^']*'|\"[^\"]*\"|\S+)\s+(?P<t>.+?)\s*$",
                          re.IGNORECASE)
    descuadran = []
    for entrada in PROTOCOLO["datasets"]:
        ruta = DATOS / f"{entrada['data_id']}.arff"
        if not ruta.exists():
            continue
        tipos = []
        with open(ruta, encoding="utf-8", errors="replace") as fichero:
            for linea in fichero:
                if linea.strip().lower().startswith("@data"):
                    break
                casado = atributo.match(linea)
                if casado and linea.strip().lower().startswith("@attribute"):
                    tipos.append(casado.group("t").strip().lower())
        trae = len([t for t in tipos if t != "string"])
        if trae != entrada["n_columnas"]:
            descuadran.append((entrada["nombre"], trae, entrada["n_columnas"]))
    assert descuadran == [("splice", 62, 61), ("house_prices_nominal", 81, 80),
                          ("Allstate_Claims_Severity", 132, 131),
                          ("house_sales", 23, 22)], (
        f"cambió el conjunto de datasets con columna sobrante: {descuadran}")


@con_los_arff
def test_una_categorica_LEGITIMA_de_alta_cardinalidad_NO_se_excluye():
    """La otra mitad, y la que impide que esto se coma una columna de verdad.

    `Amazon_employee_access.RESOURCE` declara **7.518 niveles** para 32.769
    filas (0,2294) — muchos niveles y ningún identificador. Su catálogo cuadra
    con el fichero, así que no sobra nada y no se toca nada.
    """
    leido = _cargar("Amazon_employee_access")
    assert leido.columnas_excluidas == []
    assert "RESOURCE" in leido.filas[0]
    assert len({f["RESOURCE"] for f in leido.filas}) > 7000


def test_una_categorica_cardinal_CONVIVIENDO_con_un_id_sobrevive(tmp_path):
    """El caso duro, y el que `Amazon_employee_access` NO cubre: ahí el
    catálogo cuadra, así que la criba ni se ejecuta (medido saboteando el
    criterio: la prueba de arriba seguía verde).

    Aquí sobra una columna Y hay una categórica cardinal en la misma tabla:
    300 valores distintos en 1.000 filas (0,30 — la proporción de
    `KDDCup09_appetency.Var200`, 15.415/50.000 = 0,3083). Se va el id y se
    queda ella.
    """
    cuerpo = ["@relation r", "@attribute fila_id NUMERIC",
              "@attribute cardinal STRING_NO", "@attribute clase {si,no}", "@data"]
    cuerpo[2] = "@attribute cardinal {" + ",".join(f"v{i}" for i in range(300)) + "}"
    cuerpo += [f"{i},v{i % 300},{'si' if i % 2 else 'no'}" for i in range(1000)]
    ruta = _arff(tmp_path, "\n".join(cuerpo) + "\n")

    leido = cargar(ruta, "clase", 2)
    assert leido.columnas_excluidas == ["fila_id"], f"avisos={leido.avisos}"
    assert "cardinal" in leido.filas[0]
    assert len({f["cardinal"] for f in leido.filas}) == 300


@con_los_arff
@pytest.mark.parametrize("nombre", LOS_DOCE)
def test_los_DOCE_ya_medidos_no_pierden_ninguna_columna(nombre):
    """Lo publicado no se re-dibuja por esta reparación: ninguno de los doce
    tiene columna sobrante, así que ninguno pierde nada."""
    leido = _cargar(nombre)
    assert leido.columnas_excluidas == []
    assert leido.avisos == []
    assert len(leido.filas[0]) - 1 == POR_NOMBRE[nombre]["n_columnas"]


def test_sin_catalogo_NO_se_excluye_nada(tmp_path):
    """El criterio entero se apoya en una declaración previa. Sin ella esto
    sería justo la heurística que no se quiere, así que no hace nada."""
    cuerpo = ["@relation r", "@attribute fila_id NUMERIC", "@attribute x NUMERIC",
              "@attribute clase {si,no}", "@data"]
    cuerpo += [f"{i},{i % 3},{'si' if i % 2 else 'no'}" for i in range(40)]
    ruta = _arff(tmp_path, "\n".join(cuerpo) + "\n")

    sin_catalogo = cargar(ruta, "clase")
    assert sin_catalogo.columnas_excluidas == []
    assert "fila_id" in sin_catalogo.filas[0]

    # La mitad positiva: CON el catálogo (declara 2 de las 3) sí se excluye.
    con_catalogo = cargar(ruta, "clase", 2)
    assert con_catalogo.columnas_excluidas == ["fila_id"]
    assert "fila_id" not in con_catalogo.filas[0]


def test_si_las_dos_senales_no_coinciden_NO_se_adivina(tmp_path):
    """Sobra una columna y el núcleo señala DOS identificadores: cuál de las
    dos sobra no se decide a ojo. No se excluye nada y se dice en voz alta.

    Elegir una sería exactamente «la lectura que da el resultado que
    conviene», con la agravante de que nadie se enteraría.
    """
    cuerpo = ["@relation r", "@attribute fila_id NUMERIC", "@attribute otro_id NUMERIC",
              "@attribute x NUMERIC", "@attribute clase {si,no}", "@data"]
    cuerpo += [f"{i},{1000 + i},{i % 3},{'si' if i % 2 else 'no'}" for i in range(40)]
    ruta = _arff(tmp_path, "\n".join(cuerpo) + "\n")

    leido = cargar(ruta, "clase", 3)          # 4 atributos, el catálogo dice 3
    assert leido.columnas_excluidas == []
    assert "fila_id" in leido.filas[0] and "otro_id" in leido.filas[0]
    assert len(leido.avisos) == 1
    assert "no se adivina" in leido.avisos[0]
    assert "fila_id" in leido.avisos[0] and "otro_id" in leido.avisos[0]


def test_el_OBJETIVO_nunca_se_excluye_por_parecer_identificador(tmp_path):
    """Un objetivo casi único (una regresión sobre un precio, por ejemplo) no
    puede caerse por la puerta de los identificadores: sin objetivo no hay
    nada que medir, y el fallo sería mudo.
    """
    cuerpo = ["@relation r", "@attribute fila_id NUMERIC", "@attribute x NUMERIC",
              "@attribute precio NUMERIC", "@data"]
    cuerpo += [f"{i},{i % 3},{10000 + i}" for i in range(40)]
    ruta = _arff(tmp_path, "\n".join(cuerpo) + "\n")

    leido = cargar(ruta, "precio", 2)
    assert leido.columnas_excluidas == ["fila_id"], (
        f"se excluyó otra cosa: {leido.columnas_excluidas} / avisos={leido.avisos}")
    assert "precio" in leido.filas[0]


def test_faltan_columnas_en_vez_de_sobrar_tambien_se_declara(tmp_path):
    """El descuadre al revés — el catálogo declara MÁS columnas de las que el
    fichero trae — no es «nada que excluir»: es que uno de los dos miente, y
    callarlo dejaría pasar un ARFF que no es el que se registró.
    """
    cuerpo = ["@relation r", "@attribute x NUMERIC", "@attribute clase {si,no}", "@data"]
    cuerpo += [f"{i % 3},{'si' if i % 2 else 'no'}" for i in range(40)]
    ruta = _arff(tmp_path, "\n".join(cuerpo) + "\n")

    leido = cargar(ruta, "clase", 5)
    assert leido.columnas_excluidas == []
    assert len(leido.avisos) == 1 and "FALTAN 3" in leido.avisos[0]


@con_los_arff
def test_el_objetivo_es_el_DECLARADO_no_el_ultimo_atributo():
    """Salió al cablear el lector único, y afecta a la pasada de los cuarenta:
    en CINCO datasets el objetivo que el catálogo declara no es el último
    `@attribute` — `Moneyball` (RS), `APSFailure` (class), `diamonds` (price),
    `house_sales` (price) y `okcupid-stem` (job). La copia que la pasada tenía
    usaba `nombres[-1]`, así que habría entrenado contra otra columna.

    Ninguno de los cinco está entre los doce ya medidos.
    """
    assert _cargar("house_sales").objetivo == "price"
    assert _cargar("okcupid-stem").objetivo == "job"
    # La mitad positiva: donde el declarado SÍ es el último, sigue siendo ése.
    assert _cargar("sick").objetivo == "Class"


@con_los_arff
def test_la_PASADA_pide_el_objetivo_declarado_y_no_el_ultimo():
    """Esta prueba existe porque la de arriba SALIÓ VERDE con el cableado
    saboteado (2026-09-13): probaba el lector, que recibe el objetivo como
    parámetro, y no que la pasada se lo PASE. El hueco está en el cableado.

    `house_sales` declara `price`, que es su SEGUNDO atributo; el último es
    `date_day`.
    """
    pasada = pytest.importorskip("pasada_exploratoria_101_c3")
    _filas, objetivo = pasada.cargar_arff(42731)
    assert objetivo == "price"


# --------------------------------------------------------------------------
# La línea que explica por qué NO hace lo obvio
# --------------------------------------------------------------------------

def test_el_PREFILTRO_no_puede_cambiar_la_respuesta_del_nucleo():
    """`columnas_que_el_nucleo_llama_identificador` no pregunta por todas las
    columnas: descarta antes las que no llegan a `_IDENTIFIER_UNIQUE_RATIO`,
    para no convertir en CSV las 132 columnas de `Allstate_Claims_Severity`.

    Eso solo vale si esa condición es NECESARIA para el núcleo — si lo fuera
    solo «casi», el atajo estaría decidiendo por su cuenta. Se mide con los
    dos lados delante: justo por debajo del ratio el núcleo NUNCA dice
    identificador, y justo por encima sí.
    """
    from matrixai.training.dataset_analysis import (_IDENTIFIER_UNIQUE_RATIO,
                                                    analyze_dataset_csv)
    import csv
    import io

    def veredicto(valores):
        buffer = io.StringIO()
        escritor = csv.writer(buffer)
        escritor.writerow(["c"])
        for v in valores:
            escritor.writerow([v])
        return analyze_dataset_csv(buffer.getvalue())["columns"]["c"]["type"]

    # 1.000 filas: 980 distintas es 0,98 (justo el umbral), 979 se queda corto.
    justo = [str(i) for i in range(980)] + ["0"] * 20
    corto = [str(i) for i in range(970)] + ["0"] * 30
    assert len(set(justo)) / len(justo) >= _IDENTIFIER_UNIQUE_RATIO
    assert len(set(corto)) / len(corto) < _IDENTIFIER_UNIQUE_RATIO

    assert veredicto(justo) == "identifier"
    assert veredicto(corto) != "identifier", (
        "el núcleo llama identificador a una columna por debajo de su propio "
        "umbral de unicidad: el prefiltro dejaría de ser inocuo")


def test_quien_decide_es_el_NUCLEO_y_no_la_unicidad(tmp_path):
    """La unicidad sola NO basta, y por eso se pregunta al núcleo en vez de
    quedarse con el prefiltro.

    `medida` son 200 valores DECIMALES todos distintos —una medida de dominio,
    como el sueldo que el núcleo documenta en `_IDENTIFIER_RUN_DENSITY`—: pasa
    el prefiltro con unicidad 1,0 y aun así el núcleo NO la llama
    identificador. `fila_id` sí. `repetida` ni siquiera llega.
    """
    filas = [{"fila_id": float(i), "medida": 15000.5 + i * 3.7,
              "repetida": str(i % 4)} for i in range(200)]
    medidas = columnas_que_el_nucleo_llama_identificador(
        filas, ["fila_id", "medida", "repetida"])
    assert set(medidas) == {"fila_id"}, (
        "el núcleo no fue quien decidió: con la unicidad sola, `medida` (200 "
        "valores distintos de 200) también saldría identificador")
    assert medidas["fila_id"]["unique_ratio"] == 1.0


# --------------------------------------------------------------------------
# El cableado: que la pasada lo use y lo declare
# --------------------------------------------------------------------------

def test_la_pasada_NO_tiene_su_propia_copia_del_lector():
    """El hueco está en el cableado catorce veces de cada catorce, y este lo
    fue: el lector bueno existía desde el 09-12 y esta pasada seguía con el
    suyo. Que exista `lector_arff` no sirve de nada si la pasada no lo llama.
    """
    fuente = (_FASE0 / "pasada_exploratoria_101_c3.py").read_text(encoding="utf-8")
    assert "lector_arff.cargar(" in fuente
    assert "arff.loadarff" not in fuente, "la pasada volvió a leer el ARFF por su cuenta"


def test_el_lector_invalida_el_CACHE_de_la_pasada():
    """Un caché que no ve el arreglo es peor que no tener caché: daría números
    nuevos con datos viejos y nadie lo notaría. Si el lector —o el módulo del
    núcleo que decide qué es un identificador— cambia, los 720 intentos tienen
    que dejar de ser reusables.
    """
    pasada = pytest.importorskip("pasada_exploratoria_101_c3")
    compartidos = {Path(p).name for p in pasada._FICHEROS_COMPARTIDOS}
    assert "lector_arff.py" in compartidos
    assert "dataset_analysis.py" in compartidos


@con_los_arff
def test_la_pasada_DECLARA_en_su_salida_que_columna_quito_y_por_que():
    """Una exclusión silenciosa es un dato que desaparece sin rastro. Quien
    lea el JSON tiene que ver que `splice` entró SIN `Instance_name`, y el
    número con el que se decidió."""
    pasada = pytest.importorskip("pasada_exploratoria_101_c3")
    pasada.LECTURA_DECLARADA.clear()
    pasada.cargar_arff(46)          # splice
    declarado = pasada.LECTURA_DECLARADA["46"]
    assert declarado["nombre"] == "splice"
    assert declarado["columnas_excluidas"] == ["Instance_name"]
    assert "3178" in declarado["motivos_de_exclusion"]["Instance_name"]
    assert declarado["catalogo_leido"] is True

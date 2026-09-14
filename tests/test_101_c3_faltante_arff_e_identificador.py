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
import re
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


#: UNA sola declaración de cómo se lee un `@attribute` en este fichero. Había
#: dos —la de `test_tras_la_exclusion…` y la que traje el 2026-09-14 para los
#: niveles— y son la misma línea: exactamente el defecto del que trata este
#: fichero, cometido dentro de él.
_ATRIBUTO = re.compile(r"^\s*@attribute\s+(?P<n>'[^']*'|\"[^\"]*\"|\S+)\s+(?P<t>.+?)\s*$",
                       re.IGNORECASE)


def _atributos_declarados(ruta: Path) -> list[tuple[str, str]]:
    """`[(nombre, tipo)]` de la CABECERA, en orden y sin tocar los datos."""
    fuera: list[tuple[str, str]] = []
    with open(ruta, encoding="utf-8", errors="replace") as fichero:
        for linea in fichero:
            if linea.strip().lower().startswith("@data"):
                break
            if not linea.strip().lower().startswith("@attribute"):
                continue
            casado = _ATRIBUTO.match(linea)
            if casado is None:
                continue
            nombre = casado.group("n").strip()
            if len(nombre) >= 2 and nombre[0] == nombre[-1] and nombre[0] in "'\"":
                nombre = nombre[1:-1]
            fuera.append((nombre, casado.group("t").strip()))
    return fuera


def _niveles_declarados(ruta: Path) -> dict[str, list[str]]:
    """`{columna: niveles}` de cada nominal, desde `_atributos_declarados`."""
    return {nombre: [v.strip().strip("\'\"") for v in tipo[1:-1].split(",")]
            for nombre, tipo in _atributos_declarados(ruta)
            if tipo.startswith("{") and tipo.endswith("}")}


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
    descuadran = []
    for entrada in PROTOCOLO["datasets"]:
        ruta = DATOS / f"{entrada['data_id']}.arff"
        if not ruta.exists():
            continue
        tipos = [tipo.lower() for _nombre, tipo in _atributos_declarados(ruta)]
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


# --------------------------------------------------------------------------
# 3. LO QUE HACE SEGURO CONVERTIR TODOS LOS `?` — medido el 2026-09-14
# --------------------------------------------------------------------------
# `lector_arff.cargar` hace `None if texto in ("", "?") else texto` SIN mirar
# de qué columna viene: vale para un predictor nominal, para el objetivo y
# para una columna que declarase `?` entre sus niveles. Esa decisión es
# correcta —`?` es el marcador de ausencia del estándar ARFF— pero su
# INOCUIDAD sobre este protocolo se apoya en dos hechos que nadie había
# medido, y una línea que explica por qué no hace lo obvio necesita una prueba
# con su nombre:
#
#   · ninguno de los cuarenta declara `?` como nivel legítimo de un nominal,
#     así que la conversión no destruye ninguna categoría real;
#   · ninguno de los cuarenta trae `?` en su columna objetivo, así que no hay
#     ni una fila que pierda su etiqueta por el camino.
#
# El día que entre un dataset que rompa cualquiera de los dos, esto se pone
# rojo antes de que nadie mida nada — que es justo cuando hay que enterarse,
# porque el segundo caso no da error: la fila se queda sin objetivo y la
# pasada la descarta EN SILENCIO (`filas_con_objetivo`, `particiones_base`).

#: Un campo que es exactamente `?` (con los espacios de alineación que algunos
#: ARFF meten). Validado el 2026-09-14 contra un troceador que respeta las
#: comillas SIMPLES de ARFF: los dos dan el MISMO número en los cuarenta
#: ficheros. El atajo importa porque el exacto tarda minutos y este 8,6 s.
_CAMPO_INTERROGANTE = re.compile(rb"(?<=[,\n])[ \t]*\?[ \t]*(?=[,\n\r])")


def _celdas_interrogante(ruta: Path) -> int:
    datos = ruta.read_bytes()
    return len(_CAMPO_INTERROGANTE.findall(b"\n" + datos[datos.lower().find(b"@data"):]))


@con_los_arff
def test_ningun_ARFF_del_protocolo_declara_el_interrogante_como_NIVEL_legitimo():
    """Lo que hace inocuo convertir TODOS los `?`: ninguno es una categoría.

    Si un dataset declarase `{sí,no,?}`, ese `?` sería un valor con
    significado y el lector lo estaría borrando — y en silencio, porque la
    celda saldría como ausente igual que las de verdad.
    """
    declaran, nominales, ficheros = [], 0, 0
    for entrada in PROTOCOLO["datasets"]:
        ruta = DATOS / f"{entrada['data_id']}.arff"
        if not ruta.exists():
            continue
        ficheros += 1
        for columna, niveles in _niveles_declarados(ruta).items():
            nominales += 1
            if "?" in niveles:
                declaran.append((entrada["nombre"], columna))
    # La mitad positiva, porque `declaran == []` lo pasaría un barrido que no
    # hubiera mirado nada: el 2026-09-14 son 40 ficheros y 2.017 columnas
    # nominales, MEDIDAS — el primer número que escribí aquí fue a ojo y era
    # falso, y lo cazó este mismo aserto.
    assert ficheros == 40, f"no se barrieron los cuarenta, sino {ficheros}"
    assert nominales == 2017, f"cambió cuántas columnas nominales hay: {nominales}"
    assert declaran == [], (
        "algún ARFF declara `?` como nivel legítimo y el lector lo está "
        f"convirtiendo en ausente: {declaran}")


def test_un_nivel_declarado_como_interrogante_SE_PIERDE_y_asi_se_ve(tmp_path):
    """La otra mitad, que es la que impide que la de arriba pase por vacía: si
    un ARFF SÍ lo declarase, esto es exactamente lo que pasaría."""
    ruta = _arff(tmp_path, "@relation r\n"
                           "@attribute respuesta {si,no,?}\n"
                           "@attribute clase {a,b}\n"
                           "@data\n"
                           "si,a\n?,b\nno,a\n")
    leido = cargar(ruta, "clase")
    assert [f["respuesta"] for f in leido.filas] == ["si", None, "no"], (
        "el `?` declarado como nivel tiene que salir como ausente: es la "
        "consecuencia que la prueba de los cuarenta da por no ocurrida")


@con_los_arff
def test_ningun_OBJETIVO_de_los_cuarenta_trae_interrogante():
    """El segundo hecho: nadie pierde su etiqueta al convertir los `?`.

    Importa porque no da error. `particiones_base` se queda con
    `f[objetivo] is not None`, así que una fila cuyo objetivo fuera `?`
    desaparecería de la medición sin que nada lo dijera.
    """
    con_interrogante = []
    for entrada in PROTOCOLO["datasets"]:
        ruta = DATOS / f"{entrada['data_id']}.arff"
        if not ruta.exists():
            continue
        objetivo = entrada["columna_objetivo"]
        leido = _niveles_declarados(ruta)
        if objetivo in leido and "?" in leido[objetivo]:
            con_interrogante.append((entrada["nombre"], "declarado"))
    assert con_interrogante == []
    # La medida de verdad: el lector, sobre cuatro de los diez que SÍ traen `?`.
    # Con su cuenta de filas delante: `sin_etiqueta == []` lo pasaría también
    # una lectura que no hubiera devuelto ni una fila.
    for nombre, filas_esperadas in (("sick", 3772), ("adult", 48842),
                                    ("house_prices_nominal", 1460), ("Moneyball", 1232)):
        leido = _cargar(nombre)
        assert len(leido.filas) == filas_esperadas, (
            f"{nombre}: el lector devolvió {len(leido.filas)} filas, no "
            f"{filas_esperadas} — el aserto de abajo no estaría mirando nada")
        assert any(v is None for f in leido.filas for v in f.values()), (
            f"{nombre} trae `?` y ni una celda salió ausente: la conversión no corrió")
        sin_etiqueta = [f["row_id"] for f in leido.filas if f[leido.objetivo] is None]
        assert sin_etiqueta == [], (
            f"{nombre}: {len(sin_etiqueta)} filas se quedaron sin objetivo y la "
            "pasada las descartaría en silencio")


def test_un_interrogante_en_el_OBJETIVO_deja_la_fila_sin_etiqueta(tmp_path):
    """La mitad positiva de la anterior: el día que ocurra, esto es el efecto.

    No se «arregla» aquí —convertirlo es lo correcto— pero queda escrito que
    el precio es una fila que la pasada tira sin decirlo.
    """
    ruta = _arff(tmp_path, "@relation r\n"
                           "@attribute x numeric\n"
                           "@attribute clase {a,b}\n"
                           "@data\n"
                           "1,a\n2,?\n3,b\n")
    leido = cargar(ruta, "clase")
    assert [f["clase"] for f in leido.filas] == ["a", None, "b"]
    assert len([f for f in leido.filas if f["clase"] is not None]) == 2, (
        "la fila del medio se pierde para la pasada, y nada lo declara")


@con_los_arff
def test_tiene_faltantes_es_ALGUNA_celda_y_NO_el_uno_por_ciento_del_anexo():
    """`tiene_faltantes` del catálogo y la exigencia del anexo NO son lo mismo.

    El anexo C §2.2 pide «≥ 10 con faltantes (**> 1 % de celdas**)», y
    `generar_protocolo.verificar_cobertura` cuenta `tiene_faltantes`, que sale
    de `NumberOfMissingValues` de OpenML: ALGUNA celda ausente, sin umbral.

    Medido el 2026-09-14 sobre los cuarenta ARFF: con «alguna» salen **10**
    —y el catálogo acierta en los diez, ni uno de más ni de menos—; con el
    «> 1 %» del paréntesis salen **7**. Es el mismo patrón que el
    `alta_cardinalidad` del 09-13: el anexo pedía dos cosas y el código
    comprueba una.

    Esta prueba NO decide cuál de las dos lecturas vale —cambiarlo movería la
    selección de los cuarenta y con ella el protocolo registrado, que no es
    una decisión que se tome dentro de un test—: fija los DOS números para que
    nadie los descubra otra vez desde cero, y se pone roja si alguno se mueve.
    """
    con_alguna, por_encima_del_uno_por_ciento, declaran = [], [], []
    total_celdas = 0
    for entrada in PROTOCOLO["datasets"]:
        ruta = DATOS / f"{entrada['data_id']}.arff"
        if not ruta.exists():
            pytest.skip("faltan ARFF: el recuento sería parcial y mentiría")
        celdas = _celdas_interrogante(ruta)
        total_celdas += celdas
        if entrada["tiene_faltantes"]:
            declaran.append(entrada["nombre"])
        if celdas:
            con_alguna.append(entrada["nombre"])
            if celdas / (entrada["n_filas"] * (entrada["n_columnas"] + 1)) > 0.01:
                por_encima_del_uno_por_ciento.append(entrada["nombre"])

    assert sorted(declaran) == sorted(con_alguna), (
        "`tiene_faltantes` ya no coincide con «tiene alguna celda `?`»: "
        f"declara {sorted(declaran)}, mide {sorted(con_alguna)}")
    assert len(con_alguna) == 10
    assert len(por_encima_del_uno_por_ciento) == 7, (
        "cambió cuántos pasan el «> 1 % de celdas» del anexo: "
        f"{sorted(por_encima_del_uno_por_ciento)}")
    assert len(por_encima_del_uno_por_ciento) < len(declaran), (
        "si los dos criterios ya coinciden, esta prueba sobra y el aviso "
        "de su nombre también")
    # Y la MAGNITUD, no solo los conjuntos. Va aquí porque un sabotaje del
    # 2026-09-14 salió VERDE sin ella: cambiar el contador por uno que cuenta
    # cualquier `?` del fichero —también dentro de un texto entrecomillado— no
    # movía ni un nombre de las listas de arriba, así que la prueba no
    # distinguía un contador exacto de uno tosco. Este número sí.
    assert total_celdas == 9_319_291, (
        f"cambió cuántas celdas ausentes traen los cuarenta: {total_celdas:,}")

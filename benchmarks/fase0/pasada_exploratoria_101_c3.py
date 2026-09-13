#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C3 — la pasada exploratoria real: «ejecutar el protocolo corto y
guardar resultados crudos bajo su hash, con costes reales y comparación de
los candidatos» (texto literal del criterio de terminado).

12 datasets reales de OpenML, no sellados, clasificación binaria, del
protocolo ya registrado (`protocolo_exploratorio.json`, 101-C1) — cubren
numéricas puras, categóricas de alta cardinalidad, faltantes y
desbalanceo, que es literalmente lo que el contrato 101 pide para la
exploración («8-12 datasets externos con casos numéricos/categóricos,
faltantes y desbalanceo»). Ninguno de los 8 sellados se toca.

DISEÑO DE PARTICIÓN: el YA REGISTRADO en el protocolo (`DisenoDeParticion`,
101-C1) — 5 pliegues x 3 repeticiones para pequeño/mediano — no uno
inventado aquí. Cambiar folds/repeticiones sin registrar un protocolo
nuevo sería precisamente lo que `protocolo.py` prohíbe por diseño.

UNA SOLA CONFIGURACIÓN POR MOTOR, Y AHORA DENTRO DEL ARTEFACTO. El motivo
entero está en `CRITERIO_DE_LAS_CONFIGURACIONES`, aquí abajo, y desde el
2026-09-13 viaja al JSON dentro del bloque `alcance` — no se repite en este
docstring a propósito: dos sitios declarando lo mismo acaban divergiendo, y
el que se quedó viejo la última vez fue precisamente éste (decía «los 4
adaptadores» cuando ya eran siete).

MINORÍA = POSITIVA, CONVENCIÓN EXPLÍCITA. Para AUROC hace falta una clase
"positiva" declarada; sin leer la semántica de cada dataset uno por uno,
la convención aplicada aquí es que la clase MINORITARIA es la positiva
(el caso de interés en un problema desbalanceado) — documentado por
dataset abajo, no adivinado fila a fila.

PREPARACIÓN POR MOTOR, NO UNA SOLA VEZ POR DATASET. `ajustar_preparacion`
(103-C3) recibe `admite_categoricas`/`admite_faltantes` de la
`Capacidades` de CADA motor — un motor que no soporta nativamente
categóricas/faltantes recibe datos imputados/codificados; uno que sí,
recibe los valores crudos (`None` en faltantes, para que su tratamiento
nativo los vea). Se ajusta sobre train de cada pliegue, nunca sobre
validation/test.

CACHÉ POR INTENTO, PARA NO RELANZAR LOS 80 MINUTOS ENTEROS POR UN
CAMBIO PEQUEÑO (pedido explícito de Roberto, 2026-09-07: «el tiempo es
oro»). Cada intento se reusa del resultado anterior si su digest de
código no cambió: un digest de los ficheros COMPARTIDOS (harness,
subproceso, particiones, preparación, este mismo script — cambiar
cualquiera invalida TODO) y un digest del fichero del MOTOR concreto
(cambiar `densa.py` invalida solo sus intentos, no los de los otros
tres). Determinismo verificado antes de confiar en él (mismo motor +
misma semilla + mismos datos + mismo código -> mismo resultado, ya
medido en `test_determinismo_misma_semilla_mismo_digest` de 102-C2).
`--forzar` en la línea de comandos ignora el caché entero, para la
validación final antes de cerrar el corte.

PROCEDENCIA EN LA SALIDA (2026-09-12). Hasta hoy este JSON guardaba los
números y NADA de dónde salieron. Ahora lleva un bloque `procedencia` con
los commits de los DOS repositorios —y si su árbol estaba SUCIO al medir—,
las versiones de las bibliotecas realmente cargadas, los digests de código
que el script ya calculaba, y el sha256 de los 12 ARFF. Sin eso, un número
medido hace un mes no se puede volver a atar a nada: medido el 09-12, el
`pipeline_digest` de 101-C4 (`b888d0b1…`, del 09-09) ya NO reproduce —hoy
sale `eeea0cdf…`— y no hay forma de decir cuál de los 33 commits que
`matrixai-engines` acumuló desde entonces lo movió.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
_RAIZ_DE_ENGINES = _RAIZ_DEL_CORE.parent / "matrixai-engines" / "src"
for ruta in (_RAIZ_DEL_CORE, _RAIZ_DE_ENGINES):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from matrixai.estudio import ProblemSpec  # noqa: E402
from matrixai.estudio.validacion import digest_canonico  # noqa: E402
from matrixai.training.particion_por_diseno import proponer_particion  # noqa: E402
from matrixai.training.preparacion import (ajustar_preparacion,  # noqa: E402
                                           tipar_columnas_numericas, transformar_fila)

from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.motores.baseline import MotorBaseline  # noqa: E402
from matrixai_engines.motores.lineal import MotorLineal  # noqa: E402
from matrixai_engines.motores.arbol_lightgbm import MotorArbolLightGBM  # noqa: E402
from matrixai_engines.motores.densa import MotorDensaPropia  # noqa: E402
from matrixai_engines.motores.arbol_sklearn_hgb import MotorArbolHGB  # noqa: E402
from matrixai_engines.motores.arbol_xgboost import MotorArbolXGBoost  # noqa: E402
from matrixai_engines.motores.arbol_catboost import MotorArbolCatBoost  # noqa: E402
from matrixai_engines.procedencia import versiones_de_bibliotecas  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import lector_arff  # noqa: E402
from protocolo import (ProtocoloExploratorio,  # noqa: E402
                       veredicto_con_su_alcance)

ARFF_DIR = Path.home() / "fase0_openml_datos" / "arff"

# (data_id, nombre, cubo, positiva, negativa) -- los 12, no sellados,
# binarios, pequeño/mediano, del protocolo registrado. Minoría = positiva.
#
# EL CRITERIO, escrito para que viaje AL JSON. Hasta el 2026-09-12 esta
# explicación vivía solo en el comentario de arriba, o sea en el código: quien
# leía el resultado no la tenía delante. Verificado el 2026-09-13 que los 12
# son EXACTAMENTE los 12 del protocolo que cumplen el criterio — ni uno de
# propina ni uno menos.
CRITERIO_DEL_SUBCONJUNTO = (
    "los 12 datasets del protocolo que son binary_classification, de cubo "
    "pequeno o mediano, y NO sellados (los sellados se reservan para la "
    "confirmatoria). Quedan fuera las 10 de multiclase, las 10 de regresion y "
    "todo el cubo grande: de esta pasada no se sigue NADA sobre esas.")

DATASETS = [
    (1063, "kc2", "pequeno", "yes", "no"),
    (40994, "climate-model-simulation-crashes", "pequeno", "0", "1"),
    (37, "diabetes", "pequeno", "tested_positive", "tested_negative"),
    (15, "breast-w", "pequeno", "malignant", "benign"),
    (1049, "pc4", "pequeno", "TRUE", "FALSE"),
    (1068, "pc1", "pequeno", "true", "false"),
    (38, "sick", "mediano", "sick", "negative"),
    (1053, "jm1", "mediano", "true", "false"),
    (1487, "ozone-level-8hr", "mediano", "2", "1"),
    (4534, "PhishingWebsites", "mediano", "-1", "1"),
    (40978, "Internet-Advertisements", "mediano", "ad", "noad"),
    (40983, "wilt", "mediano", "2", "1"),
]

# Diseño de partición YA REGISTRADO en protocolo_exploratorio.json (101-C1),
# no inventado aquí.
FOLDS = 5
REPETICIONES_PEQUENO_MEDIANO = 3
SEMILLAS = (0, 1, 2)

#: EL PROTOCOLO REGISTRADO, cargado UNA vez y desde un solo sitio.
#:
#: Antes lo cargaba solo `_alcance_y_veredicto()`, al final, para redactar el
#: veredicto — o sea que el fichero que declara con hash lo que se iba a medir
#: no tenía ni voz ni voto MIENTRAS se medía. De ahí salió la desviación que
#: esto repara.
RUTA_DEL_PROTOCOLO = Path(__file__).resolve().parent / "protocolo_exploratorio.json"


def protocolo_registrado() -> ProtocoloExploratorio:
    """El protocolo con el que esta pasada se mide. Sin caché a propósito:
    se lee del disco cada vez que se pregunta, que son unas pocas veces por
    pasada y ninguna dentro del bucle caliente. Un valor cacheado en un
    módulo es exactamente lo que permite que el fichero cambie y el proceso
    siga usando el de antes sin enterarse."""
    return ProtocoloExploratorio.cargar(RUTA_DEL_PROTOCOLO)


def wall_seconds_del_cubo(cubo: str, protocolo: ProtocoloExploratorio | None = None) -> float:
    """El presupuesto de pared de UN intento, EN SEGUNDOS, sacado del
    `presupuesto.minutos_por_cubo` del protocolo REGISTRADO.

    **EL ACCIDENTE QUE ESTO CIERRA (medido el 2026-09-13).** Aquí había una
    constante global, `WALL_SECONDS = 120.0`, con el comentario «generoso
    frente a lo medido». El protocolo registrado declara 2 min al cubo
    pequeño, **5 al mediano** y 10 al grande; seis de los doce datasets de
    esta pasada son medianos, así que la pasada aplicaba el **40 %** del
    presupuesto que tenía registrado para la mitad de sus datasets.

    No es un detalle de configuración: el ÚNICO fallo de los 1.260 intentos
    de la pasada de siete motores fue `matrixai.dense.torch_cpu` sobre
    `Internet-Advertisements` (mediano) matado desde fuera a **156,9 s** —
    120 de presupuesto más el margen del subproceso—, o sea **por debajo de
    los 300 s que el protocolo le da a su cubo**. Y un fallo no es un número
    peor: la regla de cierre registrada dice, con todas las letras, que «un
    fallo (timeout/crash) cuenta como dataset perdido para ese motor». El
    presupuesto recortado no hacía la medición más barata, la hacía OTRA.

    Que salga del protocolo y no de una constante es el punto entero: un
    número escrito aquí se puede mover sin que ningún digest se entere, y
    `minutos_por_cubo` está dentro de `a_json()`, o sea dentro del hash
    registrado antes de medir.
    """
    protocolo = protocolo if protocolo is not None else protocolo_registrado()
    minutos = protocolo.presupuesto.minutos_por_cubo[cubo]
    return float(minutos) * 60.0

#: LA ÚNICA CONFIGURACIÓN QUE ESTA PASADA CORRE POR MOTOR, con su nombre.
#:
#: Estaba escrita a mitad de la llamada a `ejecutar_intento_aislado`
#: (`candidate=f"{motor.nombre}-default"`), que es donde nadie la encuentra, y
#: no llegaba a los registros: el artefacto no tenía un solo campo del que se
#: pudiera leer CUÁNTAS configuraciones corrieron.
CONFIGURACION_UNICA = "default"

#: EL CRITERIO DEL RECORTE DE CONFIGURACIONES, escrito para que viaje AL JSON.
#:
#: Hasta el 2026-09-13 esto vivía SOLO en el docstring de este módulo — o sea,
#: en el código— y el bloque `alcance` del resultado declaraba el recorte de
#: datasets y el de motores y se callaba éste. Es el mismo defecto que la
#: auditoría del 09-12 clasificó GRAVE para los motores, sobre el eje que más
#: pesa: la regla de cierre registrada mide «la mejor media de los AJUSTES».
CRITERIO_DE_LAS_CONFIGURACIONES = (
    "UNA configuracion por motor (la `-default`) de las DOS que el protocolo "
    "registra por motor (13 en total contando el dummy, que registra una). El "
    "protocolo declara CUANTAS son (`Motor.configuraciones`) y su modelo de "
    "coste solo cuadra con 13, pero NO declara CUALES: el anexo C 2.3 las "
    "nombra —«defaults» y «busqueda bajo presupuesto»— y su propia medida "
    "anti-favoritismo numero 1 exige que los ESPACIOS DE BUSQUEDA esten "
    "registrados con sha256 ANTES de la primera pasada. No lo estan en "
    "`protocolo_exploratorio.json` ni en ningun otro sitio del arbol, y "
    "ninguno de los siete adaptadores implementa busqueda interna: son de "
    "hiperparametros FIJOS. Inventar aqui los espacios seria elegirlos "
    "DESPUES de ver los numeros, que es exactamente lo que el pre-registro "
    "existe para impedir. Se declara, no se esconde: de esta pasada no se "
    "sigue nada sobre «el rival en su mejor version».")

#: Las configuraciones que esta pasada corre, POR MOTOR. Se compone de
#: `motores_de_la_pasada()` y no de una lista copiada: una lista copiada es la
#: que acaba divergiendo de la que de verdad corre.
def configuraciones_de_la_pasada() -> dict[str, list[str]]:
    return {nombre: [CONFIGURACION_UNICA]
            for nombre in nombres_de_los_motores_de_la_pasada()}


#: Los hilos que pide cada intento. Estaba escrito a mano en la línea del
#: `Presupuesto`, donde nadie lo encuentra.
HILOS_POR_INTENTO = 4

#: Los procesos que esta pasada lanza a la vez. UNO: es secuencial.
PROCESOS_A_LA_VEZ = 1


def _exigir_que_la_reserva_QUEPA() -> None:
    """El guardia que faltaba, y sin el cual `reserva_segura()` era una
    función que nadie llamaba.

    Re-auditoría del 2026-09-12: el commit que la creó se titula «la reserva
    de concurrencia deja de pedir 24 hilos sobre 8 CPUs», y eso describía una
    FUNCIÓN, no un guardia. Medido: cero llamantes fuera de los tests y de
    `calcular_coste()`, que solo INFORMA. El lanzador real llevaba `hilos=4`
    escrito a mano y nadie preguntaba nunca cuántos procesos caben. Es el
    hueco de cableado número quince de este proyecto.

    Aquí no se DECIDE el reparto —eso está en el protocolo registrado con
    hash, y cambiarlo es decisión de Roberto porque cambia su digest—: se
    comprueba que lo que esta pasada va a pedir de verdad cabe en la máquina
    donde se está ejecutando, y se para antes de empezar si no.

    Parar antes es el punto. Una pasada de más de una hora que satura la
    máquina no se nota hasta que los intentos empiezan a fallar por tope de
    pared, y entonces lo que se pierde no es tiempo: es la medición, porque un
    fallo cuenta como dataset perdido para ese motor.

    **Y ESE PÁRRAFO DESCRIBÍA EL ACCIDENTE QUE ESTE GUARDIA NO VIGILABA
    (2026-09-13).** «Los intentos empiezan a fallar por tope de pared» tiene
    DOS causas, no una: que la máquina no dé abasto —la que esto miraba— y que
    el tope de pared aplicado no sea el registrado —la que pasó—. El guardia
    cubría la mitad de recursos de su propio docstring y se callaba la del
    presupuesto: la pasada de siete motores aplicó 120 s a todos los cubos
    teniendo 300 s registrados para el mediano, y el único fallo de los 1.260
    intentos fue exactamente eso, un tope de pared a 156,9 s sobre un dataset
    mediano.

    Por eso ahora comprueba las dos cosas antes de medir nada: que la reserva
    de CPU quepa, y que el presupuesto de pared que se va a aplicar a cada
    cubo sea el del protocolo registrado y no otro. Ninguna de las dos DECIDE
    nada — las dos leen lo registrado y paran si lo que se va a pedir no
    coincide.
    """
    from benchmarks.fase0.protocolo import cpus_disponibles, reserva_segura

    _exigir_que_el_PRESUPUESTO_sea_EL_REGISTRADO()

    caben = reserva_segura(HILOS_POR_INTENTO)
    if PROCESOS_A_LA_VEZ > caben:
        raise SystemExit(
            f"esta pasada pediría {PROCESOS_A_LA_VEZ} procesos x "
            f"{HILOS_POR_INTENTO} hilos = {PROCESOS_A_LA_VEZ * HILOS_POR_INTENTO} "
            f"hilos, y en esta máquina ({cpus_disponibles()} CPUs disponibles) "
            f"solo caben {caben} procesos de ese tamaño. Bajar los procesos o "
            f"los hilos por intento antes de volver a lanzarla")


def _exigir_que_el_PRESUPUESTO_sea_EL_REGISTRADO() -> None:
    """Que el tope de pared que esta pasada va a aplicar a CADA dataset salga
    del protocolo registrado, y que el protocolo se pueda leer.

    Tres cosas, y las tres paran la pasada antes de medir nada:

    1. **El protocolo tiene que cargarse.** Si no, no hay presupuesto
       registrado que aplicar, y seguir con un número de repuesto sería
       volver a la constante global por otro camino. Un presupuesto a medias
       tranquiliza igual que uno falso.
    2. **Cada cubo que esta pasada va a tocar tiene que estar registrado**,
       y con un presupuesto positivo. Un `KeyError` a los veinte minutos de
       pasada es la misma pérdida que un tope de pared.
    3. **Lo que se va a pedir tiene que ser lo registrado**: se comprueba
       llamando a `wall_seconds_del_cubo()`, la MISMA función que el bucle
       usa, no recalculando el número aquí. Dos sitios calculando el
       presupuesto acaban divergiendo, y el que divergiría es el que mide.
    """
    try:
        protocolo = protocolo_registrado()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"no se puede leer el protocolo registrado ({RUTA_DEL_PROTOCOLO}): "
            f"{type(exc).__name__}: {exc}. Sin él no hay presupuesto de pared "
            f"que aplicar, y medir con uno de repuesto es medir otra cosa") from exc

    registrados = protocolo.presupuesto.minutos_por_cubo
    for cubo in sorted({cubo for _id, _n, cubo, _p, _neg in DATASETS}):
        if cubo not in registrados:
            raise SystemExit(
                f"el cubo {cubo!r} lo usan datasets de esta pasada y el protocolo "
                f"registrado ({protocolo.version_protocolo}, digest "
                f"{protocolo.digest()[:16]}) no le da presupuesto: "
                f"{sorted(registrados)}")
        aplicado = wall_seconds_del_cubo(cubo, protocolo)
        esperado = float(registrados[cubo]) * 60.0
        if aplicado != esperado or aplicado <= 0.0:
            raise SystemExit(
                f"el presupuesto que esta pasada aplicaría al cubo {cubo!r} son "
                f"{aplicado} s y el protocolo registrado declara {esperado} s "
                f"({registrados[cubo]} min). Medir con un tope de pared distinto "
                f"del registrado convierte un fallo por tiempo en un dataset "
                f"perdido que el protocolo no pedía perder")


_DIR_ENGINES = _RAIZ_DE_ENGINES / "matrixai_engines"
# Compartidos: un cambio en cualquiera de estos invalida TODO el caché,
# porque afecta a los 4 motores por igual (o a la propia lógica de esta
# pasada).
_FICHEROS_COMPARTIDOS = (
    _DIR_ENGINES / "harness.py", _DIR_ENGINES / "subproceso.py",
    _DIR_ENGINES / "particiones.py", _DIR_ENGINES / "motor.py",
    _DIR_ENGINES / "capacidades.py", _DIR_ENGINES / "errores.py",
    _DIR_ENGINES / "textos.py",
    _RAIZ_DEL_CORE / "matrixai" / "training" / "particion_por_diseno.py",
    _RAIZ_DEL_CORE / "matrixai" / "training" / "preparacion.py",
    # Añadidos el 2026-09-12, y no por simetría: los TRES ficheros de abajo
    # son donde vivían los tres defectos de cableado que costaron los 36
    # intentos fallidos de la pasada del 09-07, y ninguno invalidaba el caché.
    #
    #   · `dataset_project.py` fabrica el prompt desde el CSV y normaliza las
    #     etiquetas: ahí estaba `-1` y `1` declarados «la misma etiqueta», que
    #     costó los 15 intentos de `PhishingWebsites`.
    #   · `playground.py` decide la arquitectura: ahí estaba el barrido que
    #     leía los nombres de columna, y por el que `pc4` entrenaba una red
    #     residual que nadie pidió (15 intentos más).
    #   · `dense_forward.py` devolvía el vector de ENTRADA como si fuera la
    #     salida cuando la red no era plana.
    #
    # Sin ellos, una re-pasada SIN `--forzar` reusaría del caché exactamente
    # los intentos que se acaban de reparar, y la medición «limpia» traería
    # los fallos viejos dentro. Un caché que no ve el arreglo es peor que no
    # tener caché: da un número nuevo con datos viejos y nadie lo nota.
    _RAIZ_DEL_CORE / "matrixai" / "training" / "dataset_project.py",
    _RAIZ_DEL_CORE / "matrixai" / "playground.py",
    _RAIZ_DEL_CORE / "matrixai" / "forward" / "dense_forward.py",
    # 2026-09-13: el lector de ARFF, desde que esta pasada dejo de tener su
    # propia copia. Sin el, reparar el `"?"` del estandar ARFF o la exclusion
    # de la columna identificadora NO invalidaria el cache, y una re-pasada
    # «limpia» reusaria exactamente los intentos que se acaban de reparar. Es
    # el mismo agujero que se cerro con los tres ficheros de arriba.
    _RAIZ_DEL_CORE / "benchmarks" / "fase0" / "lector_arff.py",
    # Y el modulo del NUCLEO que declara que es un identificador: la exclusion
    # de la columna sobrante se decide con su criterio, asi que cambiarlo
    # cambia que columnas ve el modelo.
    _RAIZ_DEL_CORE / "matrixai" / "training" / "dataset_analysis.py",
    Path(__file__),
)
# Por motor: un cambio SOLO invalida los intentos de ESE motor.
_FICHERO_POR_MOTOR = {
    "baseline": _DIR_ENGINES / "motores" / "baseline.py",
    "sklearn.lineal": _DIR_ENGINES / "motores" / "lineal.py",
    "lightgbm": _DIR_ENGINES / "motores" / "arbol_lightgbm.py",
    "matrixai.dense.torch_cpu": _DIR_ENGINES / "motores" / "densa.py",
    "sklearn.hgb": _DIR_ENGINES / "motores" / "arbol_sklearn_hgb.py",
    "xgboost": _DIR_ENGINES / "motores" / "arbol_xgboost.py",
    "catboost": _DIR_ENGINES / "motores" / "arbol_catboost.py",
}


def motores_de_la_pasada() -> list:
    """Los motores que esta pasada corre — UN solo sitio.

    Estaban instanciados a mitad de `main()`, donde nadie que leyera el
    resultado los veía, y de ahí salió el hallazgo GRAVE del 2026-09-12: el
    protocolo registrado pide SIETE motores, esta pasada corría CUATRO, y el
    recorte no estaba declarado en ninguna parte — ni en el script ni en el
    JSON. Los tres que faltaban (`sklearn.hgb`, `xgboost`, `catboost`) son los
    rivales DIRECTOS de un GBM, así que «lightgbm gana» se estaba leyendo con
    los tres rivales que más le aprietan fuera de la foto.

    **YA SON SIETE (2026-09-13).** Los tres se construyeron ese día
    (`9cfbe87` y `a8295bd`) precisamente para que esa acotación dejara de
    hacer falta. Lo que se sabe de ellos hasta ahora, medido sobre seis ARFF
    reales a un pliegue y una semilla —que no es un veredicto y no se presenta
    como tal—: xgboost gana a lightgbm en 2 de 6 y catboost en 4 de 6. O sea
    que esta pasada puede muy bien **desmentir** la promoción que la cartera
    declara hoy, y eso es exactamente para lo que se corre.

    Sacarlo a una función es lo que permite que `_componer_y_guardar` escriba
    la lista REAL en el JSON sin copiarla: una lista copiada es la que acaba
    divergiendo de la que de verdad corre.
    """
    return [MotorBaseline(), MotorLineal(), MotorArbolLightGBM(), MotorDensaPropia(),
            MotorArbolHGB(), MotorArbolXGBoost(), MotorArbolCatBoost()]


def nombres_de_los_motores_de_la_pasada() -> list[str]:
    """Sus nombres, preguntados a los objetos: si mañana un motor cambia de
    `nombre`, esto cambia con él y el JSON no miente."""
    return [m.nombre for m in motores_de_la_pasada()]


def _digest_fichero(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]


def _digest_entorno() -> str:
    return hashlib.sha256("".join(_digest_fichero(f) for f in _FICHEROS_COMPARTIDOS)
                         .encode()).hexdigest()[:16]


# --- Procedencia: lo que hace falta para volver a ATAR un numero medido ------
#
# Deuda cerrada el 2026-09-12. Hasta hoy este JSON guardaba los numeros y NADA
# de donde salieron: ni el commit de los dos repositorios, ni las versiones de
# las bibliotecas, ni el sha256 de los ARFF de entrada, ni siquiera los digests
# de codigo que este mismo script YA calculaba para su cache.
#
# Medido ese dia sobre el fichero commiteado del 09-07: los 720 registros
# traian `entorno_digest=None` y `motor_digest=None` -- 720 de 720 -- asi que
# `_reusable()` daba REUSABLES 0 DE 720 y una re-pasada repetia los 74,8 min
# enteros. Un cache inerte que ademas no lo dice parece un cache que funciona y
# no encontro nada que reusar; por eso `main()` lo declara en voz alta.

_RUTAS_DE_REPOSITORIO = {
    "matrixAI": _RAIZ_DEL_CORE,
    "matrixai-engines": _RAIZ_DE_ENGINES.parent,
}

#: Lo que se dice de un JSON de medicion escrito ANTES de que existiera el
#: bloque de procedencia. NO es un hueco por rellenar: los ficheros del
#: 2026-09-07 (la pasada y el caso "sick") y del 09-09 (el caso "adult") se
#: midieron sin registrar commit ni versiones, y escribirles hoy los de hoy
#: seria fabricar justo la procedencia que les falta.
SIN_PROCEDENCIA = ("sin procedencia: medido antes de que este bloque existiera; "
                   "el commit y las versiones de entonces no se registraron y no "
                   "se pueden reconstruir sin inventarlos")


def _sha256_de(ruta: Path) -> str:
    """El sha256 COMPLETO del fichero de datos, sin truncar.

    Los digests de codigo de aqui arriba van a 16 hex porque se comparan entre
    si dentro de una misma pasada; este viaja fuera y es la unica prueba de que
    los datos de entrada eran LOS MISMOS, asi que va entero (mismo criterio que
    `digest_canonico`: «una huella corta es comoda de leer y no es una prueba»).
    Se lee a trozos: `adult` (1590.arff) son 5,7 MB y los sellados grandes pasan
    de 30."""
    huella = hashlib.sha256()
    with ruta.open("rb") as fichero:
        for trozo in iter(lambda: fichero.read(1 << 20), b""):
            huella.update(trozo)
    return huella.hexdigest()


def _estado_del_repositorio(raiz: Path) -> dict:
    """El commit Y si el arbol estaba sucio. Los dos, siempre.

    Un commit a secas es una coartada: si el arbol tiene cambios sin commitear,
    ese sha NO identifica el codigo que produjo los numeros, y publicarlo solo
    invita a un `git checkout` y a creer que se reprodujo lo mismo. Un digest de
    un arbol sucio presentado como limpio es peor que no tener digest.

    Cuando git no puede responder (no esta instalado, la ruta no es un
    repositorio, la llamada se cuelga) el commit va a `None` CON su motivo:
    nunca una cadena de relleno ni un «desconocido» mudo."""
    try:
        cabeza = subprocess.run(("git", "-C", str(raiz), "rev-parse", "HEAD"),
                                capture_output=True, text=True, timeout=30)
        estado = subprocess.run(("git", "-C", str(raiz), "status", "--porcelain"),
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:  # git ausente o colgado
        return {"commit": None, "arbol_sucio": None,
                "motivo": f"no se pudo preguntar a git: {type(exc).__name__}: {exc}"}
    if cabeza.returncode != 0 or estado.returncode != 0:
        fallo = (cabeza.stderr + estado.stderr).strip() or "git devolvio un codigo de error"
        return {"commit": None, "arbol_sucio": None, "motivo": fallo}
    lineas = [l for l in estado.stdout.splitlines() if l.strip()]
    # `--porcelain` ya respeta .gitignore, asi que `documentacion/` y los
    # `__pycache__` no cuentan. Los sin seguimiento SI cuentan: un modulo nuevo
    # sin commitear puede tapar a otro y cambiar lo que se mide.
    modificados = sorted(l[3:] for l in lineas if not l.startswith("??"))
    sin_seguimiento = sorted(l[3:] for l in lineas if l.startswith("??"))
    return {"commit": cabeza.stdout.strip(),
            "arbol_sucio": bool(modificados or sin_seguimiento),
            "ficheros_modificados": modificados,
            "ficheros_sin_seguimiento": sin_seguimiento,
            "motivo": None}


def procedencia_de_la_medicion(*, digests_de_codigo: dict,
                               datos_de_entrada: dict[str, Path]) -> dict:
    """El bloque que convierte un numero medido en un numero ANCLABLE.

    Lo minimo para volver a atar una medicion dentro de un mes: los commits de
    los DOS repositorios (el core y `matrixai-engines`, que derivo 33 commits
    entre el 09-07 y el 09-12 y por eso el `pipeline_digest` de 101-C4 ya no
    reproduce), las versiones de las bibliotecas REALMENTE cargadas
    (`versiones_de_bibliotecas()` de 102-C1, que ya existia y nadie llamaba
    desde aqui), los digests de codigo que el script ya calculaba, y el sha256
    de los ARFF de entrada.

    `anclable` es una CONCLUSION, no un deseo: basta un arbol sucio, un git que
    no responde o un ARFF que no esta para que valga `False`, y el motivo
    concreto va en `avisos`, en texto. Declarar lo que PASO, no lo que se pidio.
    """
    repositorios = {nombre: _estado_del_repositorio(raiz)
                    for nombre, raiz in _RUTAS_DE_REPOSITORIO.items()}
    avisos: list[str] = []
    for nombre, estado in repositorios.items():
        if estado["commit"] is None:
            avisos.append(f"{nombre}: SIN COMMIT ({estado['motivo']}) -- esta medicion "
                          f"no se puede atar a una version del codigo")
        elif estado["arbol_sucio"]:
            avisos.append(
                f"{nombre}: ARBOL SUCIO al medir ({len(estado['ficheros_modificados'])} "
                f"modificados, {len(estado['ficheros_sin_seguimiento'])} sin seguimiento) "
                f"-- el commit {estado['commit'][:12]} NO identifica el codigo que produjo "
                f"estos numeros")

    datos: dict[str, dict] = {}
    for nombre, ruta in sorted(datos_de_entrada.items()):
        if not ruta.exists():
            datos[nombre] = {"ruta": str(ruta), "sha256": None,
                             "motivo": "el fichero no existe al sellar la procedencia"}
            avisos.append(f"datos de entrada «{nombre}»: {ruta} no existe, sin sha256")
            continue
        datos[nombre] = {"ruta": str(ruta), "sha256": _sha256_de(ruta),
                         "bytes": ruta.stat().st_size, "motivo": None}

    bloque = {
        "medido": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repositorios": repositorios,
        "versiones_de_bibliotecas": versiones_de_bibliotecas(),
        "digests_de_codigo": dict(digests_de_codigo),
        "datos_de_entrada": datos,
        "anclable": not avisos,
        "avisos": avisos,
    }
    # El id se calcula sobre TODO lo anterior, avisos incluidos: dos pasadas del
    # mismo commit con el arbol sucio de formas distintas no son la misma
    # procedencia y no pueden compartir identificador.
    bloque["procedencia_id"] = digest_canonico(bloque)
    return bloque


def procedencia_declarada(payload: dict) -> dict:
    """Que se puede anclar de un JSON de medicion YA ESCRITO.

    Los tres ficheros commiteados antes del 2026-09-12
    (`pasada_exploratoria_101_c3_resultado.json` del 09-07,
    `informe_101_c4_caso_sick.json` del 09-07, `informe_101_c4_caso_adult.json`
    del 09-09) caen en `sin_procedencia`, y ahi se quedan: sus numeros son de
    esas fechas y ponerles el commit de hoy seria fabricar procedencia. Ademas
    los tres verifican hoy su `digest_canonico` (comprobado el 2026-09-12) y
    anadirles una clave romperia lo unico que si los ata.

    Un bloque presente pero con `anclable=False` NO es lo mismo que ninguno:
    ese si dice por que no se ata. Media verdad tranquilizadora tambien es media
    limpieza, asi que los dos estados se distinguen por nombre."""
    bloque = payload.get("procedencia")
    if not isinstance(bloque, dict):
        return {"estado": "sin_procedencia", "anclable": False, "explicacion": SIN_PROCEDENCIA}
    if bloque.get("anclable") is True:
        return {"estado": "anclable", "anclable": True,
                "explicacion": "procedencia completa: commits de arbol limpio, versiones "
                               "de bibliotecas y sha256 de los datos de entrada"}
    avisos = bloque.get("avisos") or ["procedencia incompleta, y sin motivo declarado"]
    return {"estado": "no_anclable", "anclable": False, "explicacion": "; ".join(avisos)}


def _procedencias_citadas(resultados: list[dict], procedencia: dict,
                          previas: dict) -> tuple[dict, int]:
    """Las procedencias que el fichero de salida tiene que llevar, y cuantos
    intentos no tienen ninguna.

    Un fichero escrito hoy con 700 intentos REUSADOS del cache no se midio hoy:
    atribuirles el commit de hoy seria mentir sobre 700 de 720 numeros. Cada
    registro cita SU procedencia (`procedencia_id`) y el fichero lleva todas las
    citadas. Los reusados de un fichero anterior al 2026-09-12 no citan ninguna
    y se cuentan aparte, en vez de heredar la de hoy en silencio."""
    citadas = {procedencia["procedencia_id"]: procedencia}
    sin_ninguna = 0
    for registro in resultados:
        identificador = registro.get("procedencia_id")
        if identificador is None:
            sin_ninguna += 1
        elif identificador not in citadas:
            citadas[identificador] = previas.get(identificador) or {
                "procedencia_id": identificador, "anclable": False,
                "avisos": ["el fichero anterior citaba esta procedencia y no la traia"]}
    return citadas, sin_ninguna


def _reusable(previo: dict | None, entorno_digest: str, motor_digest: str,
              presupuesto_wall_s: float | None = None) -> bool:
    """Un registro previo vale si lo midio EL MISMO codigo Y con EL MISMO
    presupuesto de pared. Vive en una funcion propia desde el 2026-09-12 para
    poder medirlo sin lanzar los 720 intentos: sobre el JSON del 09-07 devuelve
    False 720 veces, porque aquel fichero se escribio antes de que los digests
    se guardaran.

    **EL PRESUPUESTO ENTRA EN LA CLAVE DESDE EL 2026-09-13, y no por
    simetria.** Desde que el tope de pared sale del protocolo registrado y no
    de una constante de este fichero, mover `presupuesto.minutos_por_cubo` NO
    toca ningun fichero de codigo: el `entorno_digest` no cambia y el caché
    reusaria tal cual intentos medidos con OTRO tope. Un intento que se
    completó con 120 s y uno que se completó con 300 s no son el mismo
    intento, y el que falló por tiempo con 120 s puede completar con 300 —
    que es justo la correccion que abrio este agujero. Es el mismo hueco que
    ya se cerro dos veces en este fichero: un caché que no ve el cambio da un
    numero nuevo con datos viejos y nadie lo nota.

    Un registro sin `presupuesto_wall_s` (los de antes del 2026-09-13) NO es
    un registro con el presupuesto de hoy: se descarta. Un valor ausente no es
    un cero, ni el que a uno le conviene.
    """
    if previo is None:
        return False
    if not (previo.get("entorno_digest") == entorno_digest
            and previo.get("motor_digest") == motor_digest):
        return False
    if presupuesto_wall_s is None:
        return True
    return previo.get("presupuesto_wall_s") == presupuesto_wall_s


def _cargar_cache(ruta: Path) -> tuple[dict[tuple, dict], dict]:
    """Los registros previos Y el fichero entero: desde el 2026-09-12 la salida
    lleva las procedencias de las pasadas que midieron cada intento, y un
    intento que se reusa tiene que seguir citando LA SUYA, no la de hoy."""
    if not ruta.exists():
        return {}, {}
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    return ({(r["dataset"], r["motor"], r["repeticion"], r["pliegue"]): r
            for r in payload.get("resultados", [])}, payload)


#: Lo que el catálogo registrado dice de cada dataset, por `data_id`. Se lee
#: una vez y se cachea: `cargar_arff` lo necesita en cada dataset y el fichero
#: no cambia a mitad de una pasada.
_CATALOGO_POR_DATA_ID: dict[int, dict] | None = None


def _catalogo_registrado() -> dict[int, dict]:
    """El catálogo del protocolo registrado, indexado por `data_id`.

    Si no se puede leer devuelve vacío y `cargar_arff` sigue leyendo los ARFF
    sin la parte que depende de él —el objetivo declarado y la columna
    identificadora sobrante—, PERO lo dice: un catálogo que no se pudo leer no
    es un catálogo que diga «nada que excluir».
    """
    global _CATALOGO_POR_DATA_ID
    if _CATALOGO_POR_DATA_ID is None:
        ruta = Path(__file__).resolve().parent / "protocolo_exploratorio.json"
        try:
            payload = json.loads(ruta.read_text(encoding="utf-8"))
            _CATALOGO_POR_DATA_ID = {d["data_id"]: d for d in payload["datasets"]}
        except (OSError, ValueError, KeyError) as exc:
            print(f"AVISO DE CATALOGO: no se pudo leer {ruta}: "
                 f"{type(exc).__name__}: {exc} -- se lee sin objetivo declarado "
                 f"ni exclusion de identificadores", flush=True)
            _CATALOGO_POR_DATA_ID = {}
    return _CATALOGO_POR_DATA_ID


#: Lo que cada dataset declaró al leerse: normalizaciones, columnas excluidas
#: con su motivo, y avisos. Viaja al JSON de salida — una columna que se cae
#: sin dejar rastro es un dato que desaparece.
LECTURA_DECLARADA: dict[str, dict] = {}


def cargar_arff(data_id: int) -> tuple[list[dict], str]:
    """El ARFF, leído por el lector UNICO (`lector_arff.cargar`).

    ESTE FICHERO TENIA SU PROPIA COPIA, y las dos divergieron — que es
    exactamente lo que pasa cuando dos sitios declaran lo mismo. La de aquí
    hacia `None if texto == ""`, y el marcador de ausencia del ESTANDAR ARFF
    es `"?"`, no la cadena vacia. `scipy` lo entrega literal en las columnas
    nominales (en las numericas ya sale `NaN`), asi que el `"?"` llegaba al
    nucleo como UNA CATEGORIA MAS: ni se imputaba ni encendia el indicador de
    faltante, y eso vale para los CUATRO motores, no solo para el denso.

    Medido el 2026-09-13 sobre los doce de esta pasada: `sick` trae 150 de sus
    3.772 filas con `sex == "?"`. Es el unico de los doce afectado, y el unico
    numero que se mueve es el de `sklearn.lineal` en `sick` (AUROC 0,9640138 ->
    0,9638602 en el pliegue 0 de la repeticion 0). El veredicto de la regla de
    cierre NO se mueve: `sick` ya estaba a 3,53 puntos del mejor para ese
    motor, y el liston son 2,0.

    El lector unico trae ademas lo que esta copia nunca tuvo: las tres
    normalizaciones que hacen legibles cuatro de los cuarenta, el objetivo
    DECLARADO por el catalogo (en cinco de los cuarenta no es el ultimo
    atributo, y `nombres[-1]` entrenaba contra otra columna), y la exclusion de
    la columna identificadora que el catalogo dice que sobra.
    """
    entrada = _catalogo_registrado().get(data_id)
    leido = lector_arff.cargar(
        ARFF_DIR / f"{data_id}.arff",
        objetivo_declarado=entrada.get("columna_objetivo") if entrada else None,
        n_columnas_declaradas=entrada.get("n_columnas") if entrada else None)
    LECTURA_DECLARADA[str(data_id)] = {
        "nombre": entrada["nombre"] if entrada else None,
        "objetivo": leido.objetivo,
        "normalizaciones": list(leido.normalizaciones),
        "columnas_excluidas": list(leido.columnas_excluidas),
        "motivos_de_exclusion": dict(leido.motivos_de_exclusion),
        "avisos": list(leido.avisos),
        "catalogo_leido": entrada is not None,
    }
    for columna in leido.columnas_excluidas:
        print(f"  COLUMNA EXCLUIDA en data_id={data_id}: {columna} -- "
             f"{leido.motivos_de_exclusion.get(columna, 'sin motivo declarado')}", flush=True)
    for aviso in leido.avisos:
        print(f"  AVISO DE LECTURA en data_id={data_id}: {aviso}", flush=True)
    return leido.filas, leido.objetivo


def preparar_para_motor(train_filas: list[dict], todas_filas: list[dict], objetivo: str,
                        predictores: tuple[str, ...], motor) -> list[dict]:
    """`ajustar_preparacion` SOLO sobre train de este pliegue; `transformar_
    fila` sobre TODAS (train/validation/test) con esa misma política."""
    capacidades = motor.capabilities()
    politica = ajustar_preparacion(train_filas, objetivo=objetivo, columnas=predictores,
                                   admite_categoricas=capacidades.admite_categoricas,
                                   admite_faltantes=capacidades.admite_faltantes)
    resultado = []
    for fila in todas_filas:
        transformada = transformar_fila(fila, politica)
        transformada["row_id"] = fila["row_id"]
        transformada[objetivo] = fila[objetivo]
        resultado.append(transformada)
    return resultado


def particiones_base(data_id: int, nombre_ds: str, cubo: str, positiva: str, negativa: str):
    filas, objetivo = cargar_arff(data_id)
    filas_con_objetivo = [f for f in filas if f[objetivo] is not None]
    predictores = tuple(k for k in filas[0] if k not in ("row_id", objetivo))

    # SIN ESTO, 6 intentos perdidos en `Internet-Advertisements` — medido en la
    # pasada del 2026-09-07 y diagnosticado el 09-12. Un ARFF entrega como
    # TEXTO sus columnas nominales, aunque sus valores sean `"0"`/`"1"`.
    # `ajustar_preparacion` no parsea texto (por diseño del núcleo: que una
    # columna «parezca» numérica no es serlo), así que las ve categóricas y
    # mete el centinela `__desconocida__` cuando una categoría no aparecía en
    # train. Los motores SÍ parsean, ven la columna numérica, y
    # `float("__desconocida__")` revienta dentro del subproceso aislado.
    #
    # El dataset entero se perdía para la regla de cierre, que cuenta un fallo
    # como dataset perdido — y el veredicto «lightgbm 9/12 = 0,750 < 0,800» se
    # apoyaba en parte en eso, con una distancia REAL de 1,15 puntos, dentro
    # del margen de 2. El Studio ya tenía esta reparación desde el 09; este
    # camino nunca la recibió, y por eso la función vive ahora en el núcleo en
    # vez de en una tercera copia.
    tipar_columnas_numericas(filas_con_objetivo, predictores)

    propuesta = proponer_particion(
        filas_con_objetivo, plan_id=f"101c3-{nombre_ds}", observation_id_field="row_id",
        split_type="iid", seed=0, test_fraction=0.2, folds=FOLDS,
        repeats=REPETICIONES_PEQUENO_MEDIANO, objetivo=objetivo)
    if not propuesta.es_viable:
        raise RuntimeError(f"{nombre_ds}: particion no viable, bloqueos={propuesta.bloqueos}")

    por_id = {f["row_id"]: f for f in filas_con_objetivo}
    spec = ProblemSpec(problem_id=f"101c3-{nombre_ds}", target=objetivo,
                       task="binary_classification", observation_unit="fila",
                       classes=(negativa, positiva), positive_label=positiva,
                       predictors=predictores)
    return por_id, propuesta, spec, objetivo, predictores


def main() -> None:
    _exigir_que_la_reserva_QUEPA()
    parser = argparse.ArgumentParser()
    parser.add_argument("--forzar", action="store_true",
                       help="ignora el caché entero, re-ejecuta los 720 intentos")
    # Añadido el 2026-09-12 para poder MEDIR el cableado de la procedencia sin
    # relanzar los 720 intentos ni pisar el JSON commiteado: sin esto, el único
    # modo de comprobar que el bloque llega de verdad al fichero es una lectura
    # del código, y el hueco está en el CABLEADO catorce veces de cada catorce.
    parser.add_argument("--salida", default=None,
                       help="ruta del JSON de salida (por omisión, el de este directorio)")
    args = parser.parse_args()

    ruta_salida = (Path(args.salida) if args.salida
                  else Path(__file__).resolve().parent / "pasada_exploratoria_101_c3_resultado.json")
    cache_previo, payload_previo = ({}, {}) if args.forzar else _cargar_cache(ruta_salida)
    entorno_digest = _digest_entorno()
    digest_por_motor = {nombre: _digest_fichero(ruta) for nombre, ruta in _FICHERO_POR_MOTOR.items()}

    # La procedencia se sella ANTES de medir: describe el arbol con el que se
    # va a medir, no el que quede al acabar.
    procedencia = procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "por_motor": digest_por_motor},
        datos_de_entrada={nombre: ARFF_DIR / f"{data_id}.arff"
                          for data_id, nombre, _cubo, _pos, _neg in DATASETS})
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    if payload_previo:
        declarada = procedencia_declarada(payload_previo)
        print(f"cache previo: {len(cache_previo)} registros, procedencia "
             f"{declarada['estado']} -- {declarada['explicacion']}", flush=True)

    motores = motores_de_la_pasada()
    # UNA lectura del protocolo para toda la pasada, y se pasa a mano a
    # `wall_seconds_del_cubo`: releerlo 1.260 veces desde el disco no cambiaría
    # el número pero sí permitiría que cambiara A MITAD de pasada, y entonces
    # los intentos de antes y los de después se habrían medido con presupuestos
    # distintos sin que nada lo dijera.
    protocolo = protocolo_registrado()
    wall_por_cubo = {cubo: wall_seconds_del_cubo(cubo, protocolo)
                     for _id, _n, cubo, _p, _neg in DATASETS}
    presupuesto_declarado = presupuesto_declarado_de_la_pasada(protocolo)
    print(f"presupuesto de pared, del protocolo {protocolo.version_protocolo} "
         f"(digest {protocolo.digest()[:16]}): "
         + ", ".join(f"{c}={v:.0f}s" for c, v in sorted(wall_por_cubo.items())),
         flush=True)
    resultados = []
    reusados = 0
    inicio_total = time.perf_counter()

    for data_id, nombre_ds, cubo, positiva, negativa in DATASETS:
        por_id, propuesta, spec, objetivo, predictores = particiones_base(
            data_id, nombre_ds, cubo, positiva, negativa)
        test_ids = propuesta.plan.observaciones_del_rol("test")
        n_total = len(por_id)
        wall_seconds = wall_por_cubo[cubo]
        print(f"\n=== {nombre_ds} (data_id={data_id}, n={n_total}, "
             f"test={len(test_ids)}, cubo={cubo}, presupuesto={wall_seconds:.0f}s) ===",
             flush=True)

        for repeticion in range(REPETICIONES_PEQUENO_MEDIANO):
            for pliegue_i in range(FOLDS):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
                if pliegue is None:
                    continue
                for motor in motores:
                    clave = (nombre_ds, motor.nombre, repeticion, pliegue_i)
                    previo = cache_previo.get(clave)
                    if _reusable(previo, entorno_digest, digest_por_motor[motor.nombre],
                                 wall_seconds):
                        # `setdefault`, no la de hoy: un intento reusado de un
                        # fichero anterior al 2026-09-12 no tiene procedencia, y
                        # ponerle la de esta pasada seria firmar como medido hoy
                        # un numero de hace cinco dias.
                        registro = dict(previo, reusado=True)
                        registro.setdefault("procedencia_id", None)
                        resultados.append(registro)
                        reusados += 1
                        continue

                    filas_train_crudas = [por_id[i] for i in pliegue.entrena]
                    filas_val_crudas = [por_id[i] for i in pliegue.valida]
                    filas_test_crudas = [por_id[i] for i in test_ids]

                    transformadas = preparar_para_motor(
                        filas_train_crudas, filas_train_crudas + filas_val_crudas + filas_test_crudas,
                        objetivo, predictores, motor)
                    n_train, n_val = len(filas_train_crudas), len(filas_val_crudas)
                    filas_train_t = transformadas[:n_train]
                    filas_val_t = transformadas[n_train:n_train + n_val]
                    filas_test_t = transformadas[n_train + n_val:]

                    train = Particion.desde_filas(filas_train_t, row_id_field="row_id",
                                                  target_field=objetivo)
                    validation = Particion.desde_filas(filas_val_t, row_id_field="row_id",
                                                        target_field=objetivo)
                    test = Particion.desde_filas(filas_test_t, row_id_field="row_id",
                                                 target_field=objetivo)

                    presupuesto = Presupuesto(wall_seconds=wall_seconds, hilos=HILOS_POR_INTENTO,
                                              seed=SEMILLAS[repeticion])
                    t0 = time.perf_counter()
                    intento = ejecutar_intento_aislado(
                        motor, train, validation, test, spec, presupuesto,
                        candidate=f"{motor.nombre}-{CONFIGURACION_UNICA}",
                        split_plan_digest=propuesta.plan.digest(),
                        dataset=nombre_ds, pliegue=pliegue_i, repeticion=repeticion)
                    transcurrido = time.perf_counter() - t0

                    auroc = accuracy = None
                    if intento.informe is not None:
                        for m in intento.informe.get("metrics", []):
                            if m["metric_id"] == "auroc":
                                auroc = m["value"]
                            elif m["metric_id"] == "accuracy":
                                accuracy = m["value"]

                    registro = {
                        "dataset": nombre_ds, "cubo": cubo, "motor": motor.nombre,
                        "repeticion": repeticion, "pliegue": pliegue_i, "estado": intento.estado,
                        # EL PRESUPUESTO QUE DE VERDAD SE LE DIO A ESTE INTENTO,
                        # por intento y no solo en la cabecera: un `failed` por
                        # tope de pared no se puede leer sin saber contra que
                        # tope se midio, y la cabecera dice el de la pasada
                        # ENTERA. Ademas es lo que hace que el caché note un
                        # cambio de presupuesto que no toca ningun codigo.
                        "presupuesto_wall_s": wall_seconds,
                        # LA CONFIGURACION DE ESTE INTENTO. Sin ella, «cuantas
                        # configuraciones corrieron» no se puede DERIVAR de los
                        # registros y hay que creerse la declaracion — que es
                        # justo lo que el bloque de alcance no hace con los
                        # motores ni con los datasets.
                        "configuracion": CONFIGURACION_UNICA,
                        "wall_s": round(transcurrido, 3), "auroc": auroc, "accuracy": accuracy,
                        "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
                        "entorno_digest": entorno_digest, "motor_digest": digest_por_motor[motor.nombre],
                        "procedencia_id": procedencia["procedencia_id"],
                        "reusado": False,
                    }
                    resultados.append(registro)
                print(f"  rep={repeticion} pliegue={pliegue_i}: "
                     f"{sum(1 for r in resultados if r['dataset']==nombre_ds and r['repeticion']==repeticion and r['pliegue']==pliegue_i and r['estado']=='completed')}/{len(motores)} completed",
                     flush=True)

        # AL TERMINAR CADA DATASET, no solo al final. Cuesta menos de un
        # segundo y es la diferencia entre perder una hora de máquina o los
        # minutos del dataset en curso.
        _componer_y_guardar(resultados, procedencia, payload_previo, ruta_salida,
                            total_wall_s=time.perf_counter() - inicio_total,
                            reusados=reusados, parcial=True,
                            presupuesto=presupuesto_declarado)

    total = time.perf_counter() - inicio_total
    print(f"\n=== total: {total:.1f}s ({total/60:.1f} min), {len(resultados)} intentos "
         f"({reusados} reusados del caché, {len(resultados) - reusados} ejecutados) ===")
    if cache_previo and reusados == 0:
        # El fallo que cerro esta deuda: 0 reusados se lee como «no habia nada
        # que reusar» cuando en realidad el fichero previo no guardaba los
        # digests. Medido el 2026-09-12 sobre el JSON del 09-07: 0 de 720.
        print(f"AVISO DE CACHE: {len(cache_previo)} registros previos y NINGUNO reusable "
             f"-- {procedencia_declarada(payload_previo)['explicacion']}")

    salida = _componer_y_guardar(
        resultados, procedencia, payload_previo, ruta_salida,
        total_wall_s=total, reusados=reusados, parcial=False,
        presupuesto=presupuesto_declarado)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")


def _alcance_y_veredicto(resultados) -> dict:
    """El veredicto de la regla de cierre CON su alcance pegado, para cada
    motor que compite.

    Para CADA motor y no solo para lightgbm: dar el veredicto de uno solo
    vuelve a ser elegir qué se enseña, que es la misma familia de defecto que
    esto repara. El baseline queda fuera porque la propia regla lo excluye del
    cálculo de «mejor» — no compite, es la referencia.

    Si el protocolo no se puede leer, esto NO inventa un alcance: deja dicho
    que no lo pudo determinar. Un alcance a medias tranquiliza igual que uno
    falso.
    """
    ruta_protocolo = Path(__file__).resolve().parent / "protocolo_exploratorio.json"
    try:
        protocolo = ProtocoloExploratorio.cargar(ruta_protocolo)
    except (OSError, ValueError) as exc:
        return {"no_se_pudo_determinar_el_alcance": f"{type(exc).__name__}: {exc}"}

    nombres_motores = nombres_de_los_motores_de_la_pasada()
    nombres_datasets = [nombre for _id, nombre, _c, _p, _n in DATASETS]
    por_motor = {}
    for nombre in nombres_motores:
        if nombre == "baseline":
            continue
        por_motor[nombre] = veredicto_con_su_alcance(
            protocolo, resultados, motor=nombre,
            motores_declarados=nombres_motores,
            datasets_declarados=nombres_datasets,
            criterio_del_subconjunto=CRITERIO_DEL_SUBCONJUNTO,
            configuraciones_declaradas=configuraciones_de_la_pasada(),
            criterio_de_las_configuraciones=CRITERIO_DE_LAS_CONFIGURACIONES)
    # El alcance es el MISMO para los tres, así que se declara una vez arriba y
    # no tres veces dentro: dos sitios declarando lo mismo acaban divergiendo.
    alcance = next(iter(por_motor.values()))["alcance"] if por_motor else {}
    for v in por_motor.values():
        v.pop("alcance", None)
    return {"alcance": alcance, "por_motor": por_motor}


def presupuesto_declarado_de_la_pasada(protocolo: ProtocoloExploratorio | None = None) -> dict:
    """El presupuesto de pared que esta pasada aplica, CUBO POR CUBO, y de
    dónde sale. Va al JSON de salida y DENTRO de su digest.

    Hasta el 2026-09-13 la cabecera del resultado decía
    `"wall_seconds_por_intento": 120.0`: un número solo, para doce datasets de
    dos cubos distintos, sin decir contra qué se comparaba. Se podía leer
    entero sin enterarse de que el protocolo registrado le daba 300 s al
    mediano — que es exactamente lo que pasó, y le costó un intento a la
    pasada de siete motores.

    Lleva el digest del protocolo del que salen los minutos: un presupuesto
    sin su procedencia es otro número escrito a mano.
    """
    protocolo = protocolo if protocolo is not None else protocolo_registrado()
    cubos = sorted({cubo for _id, _n, cubo, _p, _neg in DATASETS})
    return {
        "wall_seconds_por_cubo": {c: wall_seconds_del_cubo(c, protocolo) for c in cubos},
        "minutos_por_cubo_registrados": dict(protocolo.presupuesto.minutos_por_cubo),
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "hilos_por_intento": HILOS_POR_INTENTO,
        "procesos_a_la_vez": PROCESOS_A_LA_VEZ,
        "procesos_en_paralelo_registrados": protocolo.presupuesto.procesos_en_paralelo,
    }


def _componer_y_guardar(resultados, procedencia, payload_previo, ruta_salida, *,
                        total_wall_s, reusados, parcial, presupuesto=None):
    """Compone el JSON y lo escribe. UN solo sitio, y se llama también a
    mitad de la pasada.

    **Por qué existe, medido el 2026-09-12**: la pasada solo escribía al
    terminar. Murió en `Internet-Advertisements` —3.279 filas por 1.558
    columnas, el más pesado de los doce— tras **162 pliegues completados y una
    hora larga de máquina**, y no dejó NADA. Ni los 162 buenos, ni el motivo.
    Con el caché arreglado ese trabajo era reaprovechable, pero solo si está
    escrito en alguna parte.

    Guardar solo al final convierte cualquier muerte en una pasada entera
    perdida, y una pasada de C5 durará bastante más que ésta.

    `parcial` no es decoración: un fichero a medias que no lo diga se lee como
    una pasada completa con datasets que faltan, que es peor que no tenerlo.
    """
    procedencias, sin_procedencia = _procedencias_citadas(
        resultados, procedencia, (payload_previo.get("procedencias") or {}))

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "procedencia": procedencia,
        "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial,
        # EL PRESUPUESTO, POR CUBO Y CON SU PROCEDENCIA. Sustituye al
        # `wall_seconds_por_intento: 120.0` de una sola cifra, que era la
        # desviacion respecto del protocolo escrita en la cabecera del propio
        # artefacto sin que nada la contradijera.
        "presupuesto": presupuesto if presupuesto is not None
                       else presupuesto_declarado_de_la_pasada(),
        "procesos_en_paralelo": PROCESOS_A_LA_VEZ,
        "folds": FOLDS, "repeticiones": REPETICIONES_PEQUENO_MEDIANO,
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados),
        "n_reusados": reusados,
        # QUE SE LEYO DE CADA ARFF, y que se dejo fuera. Una columna excluida
        # sin rastro es un dato que desaparece: quien lea estos numeros tiene
        # que poder ver que `splice` entro SIN `Instance_name` y por que. Va
        # dentro del digest, como el alcance y por el mismo motivo — una
        # declaracion que se puede reescribir en silencio no declara nada.
        "lectura_de_los_datos": dict(LECTURA_DECLARADA),
        "resultados": resultados,
    }
    # EL ALCANCE VIAJA CON EL NÚMERO, y va DENTRO del digest.
    #
    # Reparación del hallazgo GRAVE del 2026-09-12: el protocolo pide 40
    # datasets y 7 motores, esta pasada mide 12 y 4, y el JSON no lo decía en
    # ninguna parte. Cualquiera podía leer «lightgbm 10/12 = 0,833 CUMPLE» sin
    # enterarse de que compitió contra TRES motores y no contra seis.
    #
    # Dentro del digest a propósito: si el alcance quedara fuera del sello, se
    # podría editar sin que el fichero dejara de cuadrar consigo mismo, y un
    # alcance que se puede reescribir en silencio no declara nada.
    salida["alcance_y_veredicto"] = _alcance_y_veredicto(resultados)
    salida["digest_resultados_crudos"] = digest_canonico(salida)
    # Escritura ATÓMICA: a un temporal y luego `replace`. Sin esto, morir a
    # mitad de escribir dejaría un JSON truncado, y un fichero corrupto es
    # peor que ninguno — el caché lo leería y fallaría sin decir por qué.
    temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta_salida)
    return salida


if __name__ == "__main__":
    main()

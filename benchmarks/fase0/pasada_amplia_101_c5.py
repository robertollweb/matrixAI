#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C5 — la validación AMPLIA: «congelar versión del selector y ejecutar el
protocolo amplio, INCLUYENDO RESERVADOS» (texto literal del criterio).

LOS 40 DEL PROTOCOLO, LOS 8 SELLADOS INCLUIDOS, LAS TRES TAREAS, LOS TRES
CUBOS. Eso es lo único que separa esta pasada de la exploratoria de 101-C3, y
por eso este fichero NO la reescribe: importa de ella todo lo que ya está
auditado —procedencia, caché por intento con digest de código, lectura única
de ARFF, preparación por motor, los siete motores, la escritura atómica— y
añade SOLO lo que el alcance amplio obliga a cambiar. Dos sitios declarando lo
mismo acaban divergiendo, y aquí el que divergiría es el que mide.

QUÉ CAMBIA, Y POR QUÉ CADA COSA (todo medido el 2026-09-14, no supuesto):

1. **LOS DATASETS SALEN DEL PROTOCOLO, NO DE UNA LISTA A MANO.** La pasada de
   C3 lleva sus doce escritos en una tupla, con su clase positiva al lado.
   Cuarenta escritos a mano son cuarenta ocasiones de teclear mal un nombre de
   clase, y un nombre de clase mal tecleado no falla: cambia en silencio qué
   es «positivo». Aquí la lista se DERIVA de `protocolo_exploratorio.json` y
   la clase positiva se MIDE (la minoritaria, la misma convención que C3).
   Verificado antes de confiar en ello: la derivación reproduce EXACTAMENTE
   los doce pares escritos a mano de C3, 12 de 12
   (`test_la_minoria_derivada_reproduce_los_doce_pares_de_c3`).

2. **LA MÉTRICA DE CIERRE CAMBIA CON LA TAREA, y hasta hoy no cambiaba.**
   `regla_de_cierre.metrica_por_tarea` del protocolo registrado lo dice desde
   el primer día —AUROC en binaria, accuracy/F1-macro en multiclase, R² en
   regresión— y `aplicar_regla_de_cierre` recibía UNA métrica para todos los
   datasets. Medido sobre resultados mezclados de las tres tareas: con
   `metrica="auroc"`, los datasets de multiclase y regresión **desaparecen del
   denominador sin decirlo** (3 datasets entran, 1 sale contado), y basta que
   uno de ellos tenga un intento fallido para que **reaparezca contado como
   perdido**. O sea que el denominador de «cumple en el 80 %» dependía de si
   un motor ajeno se había caído. Ver `METRICA_DE_CIERRE_POR_TAREA` y el
   parámetro `metrica_por_dataset` de `aplicar_regla_de_cierre`.

3. **LAS REPETICIONES SALEN DEL PROTOCOLO, POR CUBO.** C3 tiene
   `REPETICIONES_PEQUENO_MEDIANO = 3` escrito como constante porque nunca tocó
   el cubo grande. El protocolo registra 3 para pequeño/mediano y **1 para
   grande** (`particion.repeticiones_para`), con su motivo escrito en
   `DisenoDeParticion`. Un número escrito aquí se podría mover sin que ningún
   digest se entere.

4. **LA REGRESIÓN NO SE ESTRATIFICA POR EL OBJETIVO, y esto es un HALLAZGO,
   no una preferencia.** `particiones_base` de C3 pasa `objetivo=objetivo` a
   `proponer_particion` siempre, que es lo correcto para una clase. Con un
   objetivo CONTINUO cada valor distinto se convierte en un estrato propio, y
   lo medido el 2026-09-14 sobre los datasets reales del protocolo es esto:

     · `Moneyball` (374 valores distintos en 1.232 filas): los pliegues bajan
       de **5 a 2** —`Limite(pliegues_reducidos_por_eventos, minoria=1)`— y el
       pliegue que sale tiene **entrena=407 frente a valida=593**: se entrena
       con MENOS datos de los que se valida. Todos los valores únicos caen
       siempre en el mismo cubo, así que el sesgo no es aleatorio, es fijo.
     · `house_prices_nominal` (663 valores distintos): el test sale al
       **13,84 %** en vez del 20 % pedido, porque `round(1 * 0,2) = 0` deja
       FUERA de test a toda fila cuyo valor de objetivo sea único. El test
       deja de ser una muestra y pasa a ser «las filas cuyo precio se repite».

   Por eso aquí la estratificación se pide solo para las tareas de
   clasificación. NO es una decisión libre de este fichero: el protocolo
   registra «5 folds estratificados», y estratificar por un continuo no es
   estratificar. Va declarado dentro del sello (`ESTRATIFICACION`) y está en
   la lista de lo que Roberto tiene que confirmar.

5. **LOS `Limite` DE LA PARTICIÓN SE GUARDAN.** C3 los tira: lee
   `propuesta.es_viable` y nunca mira `propuesta.limites`. Con 12 binarias con
   eventos de sobra nunca saltó ninguno (1.260 = 12 x 15 x 7 exacto). Con los
   40 salta: `yeast` baja a 4 pliegues (su clase minoritaria tiene 4 casos) y
   `wine_quality` también. Sin guardarlos, la pasada produce menos intentos de
   los declarados y **nada dice por qué** — el recuento final sale más bajo y
   se lee como intentos perdidos.

6. **SE GUARDAN TODAS LAS MÉTRICAS DEL INFORME, no `auroc` y `accuracy`.** El
   protocolo exige seis secundarias en binaria, la calibración y seis medidas
   `siempre`; el artefacto de C3 guarda dos números. `evaluar()` (105-C1) ya
   las calcula TODAS y las tiraba el llamante — el hueco está en el cableado,
   otra vez. Aquí cada intento guarda el mapa entero, y además
   `tiempo_de_ajuste` y `cpu_segundos` del bloque `recursos`, que también
   existían y nadie leía. Las dos que NO se pueden dar se declaran ausentes
   con su motivo en `MEDIDAS_SIEMPRE_QUE_NO_SE_PUEDEN_DAR`: un valor ausente
   no es un cero.

LO QUE ESTA PASADA SIGUE SIN SER. Una configuración por motor, igual que C3 y
por el mismo motivo ya escrito en `CRITERIO_DE_LAS_CONFIGURACIONES`: los
espacios de búsqueda no están registrados con sha256 antes de medir, e
inventarlos aquí sería elegirlos después de ver los números. Se declara, no se
esconde, y viaja dentro del sello.

CÓMO SE LANZA — y esta pasada NO se lanza sola, ni «un ratito para probar»:

    python3 benchmarks/fase0/pasada_amplia_101_c5.py --estimar     # solo cuentas
    python3 benchmarks/fase0/pasada_amplia_101_c5.py --solo kc2,balance-scale \\
        --salida /tmp/prueba.json                                  # subconjunto diminuto
    python3 benchmarks/fase0/pasada_amplia_101_c5.py               # LA PASADA ENTERA
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
_RAIZ_DE_ENGINES = _RAIZ_DEL_CORE.parent / "matrixai-engines" / "src"
for _ruta in (_RAIZ_DEL_CORE, _RAIZ_DE_ENGINES, str(Path(__file__).resolve().parent)):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

from matrixai.estudio import ProblemSpec  # noqa: E402
from matrixai.estudio.validacion import digest_canonico  # noqa: E402
from matrixai.training.particion_por_diseno import proponer_particion  # noqa: E402
from matrixai.training.preparacion import tipar_columnas_numericas  # noqa: E402

from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

from protocolo import (ESTADOS_QUE_CUENTAN_COMO_MEDIDA,  # noqa: E402
                       ProtocoloExploratorio,
                       estabilidad_del_ganador,
                       veredicto_con_su_alcance)

# LO QUE SE IMPORTA DE LA PASADA DE C3, Y NO SE COPIA. Todo esto está auditado
# y su evidencia commiteada; reescribirlo aquí sería crear el segundo sitio que
# acaba divergiendo del primero.
import pasada_exploratoria_101_c3 as c3  # noqa: E402
from pasada_exploratoria_101_c3 import (ARFF_DIR,  # noqa: E402
                                        CONFIGURACION_UNICA,
                                        CRITERIO_DE_LAS_CONFIGURACIONES,
                                        HILOS_POR_INTENTO,
                                        PROCESOS_A_LA_VEZ,
                                        _cargar_cache,
                                        _digest_fichero,
                                        _procedencias_citadas,
                                        _reusable,
                                        cargar_arff,
                                        configuraciones_de_la_pasada,
                                        motores_de_la_pasada,
                                        nombres_de_los_motores_de_la_pasada,
                                        preparar_para_motor,
                                        procedencia_de_la_medicion,
                                        procedencia_declarada,
                                        protocolo_registrado,
                                        wall_seconds_del_cubo)

#: EL PROTOCOLO con el que esta pasada se mide. El MISMO fichero registrado y
#: firmado de 101-C1, leído por la MISMA función que lo lee en C3.
RUTA_DEL_PROTOCOLO = c3.RUTA_DEL_PROTOCOLO

#: EL CRITERIO DEL SUBCONJUNTO, escrito para que viaje AL JSON. En C5 no hay
#: subconjunto: es el protocolo entero, que es exactamente lo que el criterio
#: de terminado pide («incluyendo reservados»). Se declara igual que en C3 —un
#: alcance que dice «todo» y no se puede comprobar contra nada declara lo mismo
#: que uno que no está—, y `alcance_de_una_pasada` lo contrasta contra la lista
#: real del protocolo, que es lo que lo convierte en comprobable.
CRITERIO_DEL_SUBCONJUNTO = (
    "LOS 40 datasets del protocolo registrado, sin recorte: las tres tareas "
    "(20 binarias, 10 multiclase, 10 de regresion), los tres cubos de tamano "
    "(15 pequeno, 15 mediano, 10 grande) y LOS 8 SELLADOS, que hasta esta "
    "pasada no se habian tocado nunca a proposito. La lista NO esta escrita "
    "en este fichero: se deriva de `protocolo_exploratorio.json` y se "
    "contrasta contra el, de modo que un dataset que faltara se veria en "
    "`alcance.datasets.que_faltan` en vez de desaparecer sin ruido.")

#: DE LA MÉTRICA QUE EL PROTOCOLO REGISTRA AL `metric_id` QUE EL CATÁLOGO
#: CALCULA — y ninguno de los tres se elige aquí.
#:
#: `regla_de_cierre.metrica_por_tarea` está DENTRO del digest registrado antes
#: de medir (invariante 1), así que lo que se decide en este mapa no es qué
#: métrica gobierna —eso ya está firmado— sino con qué `metric_id` del catálogo
#: de 105-C1 se calcula. Dos de los tres son literales:
#:
#:   · `"AUROC"` -> `auroc`, el mismo que C3 usó en sus doce.
#:   · `"R2"` -> `r2`. Es la métrica que la regla nombra, y no `rmse` (que el
#:     protocolo declara PRIMARIA para regresión en `metricas_por_tarea`):
#:     la regla de cierre mide distancias con `max` de la media, o sea exige
#:     que MÁS sea MEJOR, y el RMSE es al revés. Usar el primario aquí daría
#:     el peor motor por ganador en todos los datasets de regresión.
#:
#: El tercero NO es literal y por eso está aquí arriba, con su nombre y su
#: motivo, y no resuelto a mitad de una función:
#:
#:   · `"accuracy_o_f1_macro"` nombra DOS métricas del catálogo (`accuracy` y
#:     `macro_f1`) y el protocolo no dice cuál. Elegir después de ver los
#:     números es exactamente lo que el pre-registro existe para impedir, así
#:     que se fija AHORA y por escrito: gobierna `accuracy`, que es la que la
#:     cadena nombra primero. Las dos se calculan y se guardan en cada intento,
#:     así que la otra lectura se puede sacar del artefacto SIN volver a correr
#:     nada — pero la que decide el veredicto es una sola y está escrita antes
#:     de medir. Queda en la lista de lo que Roberto tiene que confirmar.
METRICA_DE_CIERRE_POR_TAREA = {
    "AUROC": "auroc",
    "accuracy_o_f1_macro": "accuracy",
    "R2": "r2",
}

#: LA OTRA LECTURA DE MULTICLASE, la que el protocolo también nombraba. Se
#: guarda en el artefacto al lado del veredicto que gobierna, para que «y con
#: F1-macro qué habría salido» se pueda responder sin relanzar 40 h de máquina.
#: No es el veredicto: es la sensibilidad del veredicto a la mitad que la
#: cadena registrada dejó sin decidir.
METRICA_ALTERNATIVA_DE_MULTICLASE = "macro_f1"

#: SI SE ESTRATIFICA POR EL OBJETIVO, POR TAREA. Medido, no preferido — el
#: punto 4 del docstring de arriba trae los dos números que lo obligan.
ESTRATIFICACION = {
    "binary_classification": True,
    "multiclass_classification": True,
    "regression": False,
}

#: POR QUÉ la regresión no se estratifica, escrito para que viaje AL JSON.
CRITERIO_DE_LA_ESTRATIFICACION = (
    "El protocolo registra «5 folds estratificados x 3 repeticiones». La "
    "estratificacion reparte CLASES; `proponer_particion` la aplica cuando se "
    "le pasa `objetivo`, y con un objetivo CONTINUO cada valor distinto se "
    "vuelve un estrato de una sola fila. Medido el 2026-09-14 sobre los "
    "datasets reales del protocolo: en `Moneyball` (374 valores distintos) los "
    "pliegues bajan de 5 a 2 y el pliegue sale con entrena=407 frente a "
    "valida=593 (se entrena con menos de lo que se valida, y siempre los "
    "mismos porque los valores unicos caen todos en el mismo cubo); en "
    "`house_prices_nominal` (663 valores distintos) el test sale al 13,84 % en "
    "vez del 20 % pedido, porque round(1*0,2)=0 deja fuera de test toda fila "
    "cuyo valor de objetivo sea unico — el test deja de ser una muestra y pasa "
    "a ser «las filas cuyo precio se repite». Por eso aqui se estratifica solo "
    "en las dos tareas de clasificacion. NO se sustituye por una "
    "estratificacion en cuantiles: eso seria un diseno de particion distinto "
    "del registrado, y un diseno nuevo se registra, no se improvisa al medir.")

#: LAS MEDIDAS QUE EL PROTOCOLO PIDE SIEMPRE Y ESTA MÁQUINA NO PUEDE DAR, con
#: su motivo. `metricas_por_tarea.siempre` registra seis; cuatro se cablean en
#: esta pasada (`tiempo_de_ajuste` y `cpu_segundos` del bloque `recursos` que
#: ya existía y nadie leía; `tasa_de_fallo` y `dispersion_entre_semillas` se
#: derivan de los registros). Las dos de abajo NO existen en el camino de
#: medición y declararlas a cero sería peor que callarlas.
MEDIDAS_SIEMPRE_QUE_NO_SE_PUEDEN_DAR = {
    "tiempo_de_prediccion": (
        "el harness llama a `motor.predict` dentro de `_muestra_de_test` sin "
        "cronometrarlo, y `Recursos` (matrixai-engines/particiones.py) solo "
        "tiene `wall_seconds` y `cpu_seconds` del AJUSTE. Darlo exige tocar "
        "`matrixai-engines`, que no es territorio de este fichero."),
    "rss_pico_mb": (
        "el hijo de `ejecutar_intento_aislado` no devuelve su `ru_maxrss`: el "
        "padre recibe el `Intento` por la cola y nada mas. Se puede medir "
        "(`resource.getrusage(RUSAGE_CHILDREN)`) pero atribuir a UN intento el "
        "pico acumulado de todos los hijos seria un numero falso, y uno falso "
        "es peor que uno ausente."),
}

def donde_vive_cada_medida_siempre(protocolo: ProtocoloExploratorio,
                                   resultados: list) -> dict:
    """UN MAPA DE CADA MEDIDA QUE EL PROTOCOLO EXIGE A DÓNDE VIVE EN EL ARTEFACTO.

    `metricas_por_tarea.siempre` registra seis nombres. Hasta el 2026-09-16 dos
    salían con su nombre en cada registro, dos se declaraban imposibles, y las
    otras dos —`tasa_de_fallo` y `dispersion_entre_semillas`— **no aparecían
    con ese nombre en ningún sitio**: el puente vivía solo en un comentario de
    este fuente. Quien comparara el protocolo con el JSON —una persona o un
    comprobador— las veía FALTAR. Y medido: `tasa_de_fallo` no estaba en
    ninguna parte, ni con otro nombre; lo único cercano era un booleano por
    dataset, `perdido_por_fallo`.

    **Se construye DESDE la lista del protocolo, no desde una lista propia.** Si
    el protocolo registra un séptimo nombre y nadie le da casa, aparece aquí con
    `estado: "SIN_CASA"` en vez de desaparecer — y la prueba que acompaña a
    esta función no admite ninguno. Una lista escrita a mano se quedaría con
    los seis de hoy.

    `tasa_de_fallo` se DERIVA aquí, de `estado`, con la MISMA lista blanca que
    usa la regla de cierre (`ESTADOS_QUE_CUENTAN_COMO_MEDIDA`): dos criterios
    distintos de «fallo» darían una tasa que no cuadra con los datasets que la
    regla cuenta como perdidos.
    """
    siempre = list((protocolo.metricas_por_tarea or {}).get("siempre") or [])
    por_motor: dict[str, dict] = {}
    for r in resultados:
        m = por_motor.setdefault(r["motor"], {"intentos": 0, "fallidos": 0})
        m["intentos"] += 1
        if r.get("estado") not in ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
            m["fallidos"] += 1
    for m in por_motor.values():
        m["tasa"] = (m["fallidos"] / m["intentos"]) if m["intentos"] else None

    casas = {
        "tiempo_de_ajuste": {
            "estado": "guardada", "donde": "resultados[].tiempo_de_ajuste"},
        "cpu_segundos": {
            "estado": "guardada", "donde": "resultados[].cpu_segundos"},
        "dispersion_entre_semillas": {
            "estado": "derivada",
            "donde": ("alcance_y_veredicto.por_motor.<motor>.dispersion."
                      "sd_entre_semillas_mediana / sd_entre_semillas_maxima, y por "
                      "dataset en .por_dataset.<dataset>.sd_entre_semillas")},
        "tasa_de_fallo": {
            "estado": "derivada",
            "donde": "aqui mismo, en `valor_por_motor`",
            "como": ("intentos cuyo `estado` no esta en "
                     "ESTADOS_QUE_CUENTAN_COMO_MEDIDA, sobre el total del motor"),
            "valor_por_motor": dict(sorted(por_motor.items()))},
    }
    for nombre, motivo in MEDIDAS_SIEMPRE_QUE_NO_SE_PUEDEN_DAR.items():
        casas[nombre] = {"estado": "imposible", "motivo": motivo}

    return {nombre: casas.get(nombre, {
                "estado": "SIN_CASA",
                "motivo": ("el protocolo la registra y esta pasada no dice donde "
                           "vive: ni guardada, ni derivada, ni declarada imposible")})
            for nombre in siempre}


#: Los ficheros compartidos cuyo cambio invalida TODO el caché: los de C3 (que
#: ya incluyen el harness, el subproceso, las particiones, la preparación, el
#: lector de ARFF y el propio script de C3) MÁS éste. Se compone de la tupla de
#: C3 en vez de copiarla: si allí se añade un fichero, aquí entra solo.
_FICHEROS_COMPARTIDOS = c3._FICHEROS_COMPARTIDOS + (Path(__file__).resolve(),)

#: El presupuesto de pared que recibe un dataset cuyo cubo no está registrado.
#: No hay ninguno: la comprobación de abajo para la pasada antes de medir nada.
_SIN_PRESUPUESTO = object()


def _digest_entorno() -> str:
    """El digest de código de ESTA pasada. Misma fórmula que la de C3 y sobre
    la misma lista más este fichero, así que un cambio aquí invalida el caché
    de C5 y NO el de C3 — que es lo correcto: los intentos de C3 los midió otro
    código."""
    import hashlib
    return hashlib.sha256("".join(_digest_fichero(f) for f in _FICHEROS_COMPARTIDOS)
                          .encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LOS 40, DERIVADOS DEL PROTOCOLO
# ---------------------------------------------------------------------------

class DatasetDeLaPasada:
    """Un dataset del protocolo con lo que esta pasada necesita saber de él,
    y NADA escrito a mano: el `data_id`, el cubo y la tarea salen del catálogo
    registrado; las clases y la positiva se MIDEN sobre el ARFF.

    Es una clase y no una tupla porque una tupla de siete posiciones es donde
    se cuela un `positiva`/`negativa` cambiado de orden sin que nada chille.
    """

    __slots__ = ("data_id", "nombre", "cubo", "tarea", "sellado", "clases",
                 "positiva", "objetivo", "n_filas_declaradas")

    def __init__(self, entrada: dict):
        self.data_id = entrada["data_id"]
        self.nombre = entrada["nombre"]
        self.cubo = entrada["cubo_de_tamano"]
        self.tarea = entrada["tarea"]
        self.sellado = bool(entrada["sellado"])
        self.n_filas_declaradas = entrada["n_filas"]
        self.clases: tuple[str, ...] | None = None
        self.positiva: str | None = None
        self.objetivo: str | None = None

    def a_json(self) -> dict:
        return {"data_id": self.data_id, "nombre": self.nombre, "cubo": self.cubo,
                "tarea": self.tarea, "sellado": self.sellado, "objetivo": self.objetivo,
                "n_clases": len(self.clases) if self.clases else None,
                "clase_positiva": self.positiva,
                "estratificado_por_el_objetivo": ESTRATIFICACION[self.tarea]}


def datasets_de_la_pasada(protocolo: ProtocoloExploratorio | None = None,
                          ) -> list[DatasetDeLaPasada]:
    """LOS 40 del protocolo registrado, en orden estable y SIN filtro.

    El orden es por cubo (pequeño primero) y dentro del cubo por filas: la
    pasada gasta primero lo barato, así que si muere a las tres horas ya ha
    dejado medidos los baratos en vez de estar todavía con el primer grande.
    No cambia ningún número: el orden de los datasets no entra en ninguna
    métrica ni en ninguna semilla (cada partición se siembra con su `plan_id`).
    """
    protocolo = protocolo if protocolo is not None else protocolo_registrado()
    orden_de_cubo = {"pequeno": 0, "mediano": 1, "grande": 2}
    entradas = [d.a_json() for d in protocolo.datasets]
    entradas.sort(key=lambda e: (orden_de_cubo[e["cubo_de_tamano"]], e["n_filas"],
                                 e["data_id"]))
    return [DatasetDeLaPasada(e) for e in entradas]


def clase_positiva_medida(filas, objetivo: str) -> tuple[tuple[str, ...], str | None]:
    """Las clases QUE HAY en el ARFF y cuál es la positiva, MEDIDAS.

    La convención es la de C3, literal: **la minoritaria es la positiva** (el
    caso de interés en un problema desbalanceado). Allí estaba escrita a mano
    dataset por dataset; aquí se calcula, y la prueba
    `test_la_minoria_derivada_reproduce_los_doce_pares_de_c3` comprueba que la
    cuenta da exactamente los doce pares que C3 tecleó — 12 de 12 medido el
    2026-09-14. Sin esa comprobación, derivar sería cambiar un dato escrito por
    uno calculado sin saber si dicen lo mismo.

    El desempate es por NOMBRE cuando dos clases empatan en recuento, y no por
    el orden en que aparecen en el fichero: el orden de aparición depende de
    cómo esté ordenado el ARFF, o sea que la «positiva» podría cambiar al
    reordenar filas sin que ningún dato cambiara.

    **Y EL ORDEN DE LA TUPLA ES EL DE C3, `(negativa, positiva)`, A PROPÓSITO.**
    Ordenarlas alfabéticamente habría sido lo natural aquí, y en tres de los
    doce de C3 (`climate-model-simulation-crashes`, `PhishingWebsites`,
    `Internet-Advertisements`) da un orden DISTINTO del que C3 declaró. Con la
    misma `positive_label`, que `spec.classes` vaya en otro orden no debería
    cambiar ninguna métrica — pero «no debería» no es una medida, y la pregunta
    que importa es si los números de C5 se pueden poner al lado de los de C3.
    Copiando la convención de C3 la pregunta no hay que contestarla: el orden
    es el mismo y las dos pasadas son comparables sin asteriscos. En multiclase
    no hay convención de C3 que copiar y se ordenan alfabéticamente, que es
    determinista y no depende de cómo esté ordenado el fichero.
    """
    conteos: dict[str, int] = {}
    for fila in filas:
        valor = fila[objetivo]
        if valor is None:
            continue
        conteos[str(valor)] = conteos.get(str(valor), 0) + 1
    nombres = sorted(conteos)
    if len(nombres) < 2:
        return tuple(nombres), None
    positiva = min(nombres, key=lambda c: (conteos[c], c))
    if len(nombres) > 2:
        return tuple(nombres), positiva
    negativa = next(c for c in nombres if c != positiva)
    return (negativa, positiva), positiva


# ---------------------------------------------------------------------------
# LOS GUARDIAS, QUE PARAN ANTES DE MEDIR NADA
# ---------------------------------------------------------------------------

def metrica_de_cierre_por_dataset(protocolo: ProtocoloExploratorio,
                                  datasets) -> dict[str, str]:
    """`nombre del dataset -> metric_id` con el que la regla de cierre lo mide.

    Sale de `regla_de_cierre.metrica_por_tarea` del protocolo REGISTRADO
    pasado por `METRICA_DE_CIERRE_POR_TAREA`. Si el protocolo nombrara una
    métrica que este mapa no sabe traducir, esto PARA: seguir con `auroc` de
    repuesto es exactamente el defecto que C5 viene a cerrar —el dataset se
    caería del denominador sin decirlo— y un veredicto sobre un denominador
    silenciosamente recortado se lee igual de bien que uno correcto.
    """
    registrada = protocolo.regla_de_cierre.metrica_por_tarea
    por_dataset: dict[str, str] = {}
    for ds in datasets:
        nombre_registrado = registrada.get(ds.tarea)
        if nombre_registrado is None:
            raise SystemExit(
                f"el protocolo registrado ({protocolo.version_protocolo}, digest "
                f"{protocolo.digest()[:16]}) no da metrica de cierre para la tarea "
                f"{ds.tarea!r}, que es la de {ds.nombre!r}. Sin metrica no hay "
                f"veredicto que dar, y darlo con la de otra tarea seria medir otra cosa")
        metric_id = METRICA_DE_CIERRE_POR_TAREA.get(nombre_registrado)
        if metric_id is None:
            raise SystemExit(
                f"el protocolo registra la metrica {nombre_registrado!r} para la tarea "
                f"{ds.tarea!r} ({ds.nombre!r}) y este fichero no sabe con que "
                f"`metric_id` del catalogo de 105-C1 se calcula. Conocidas: "
                f"{sorted(METRICA_DE_CIERRE_POR_TAREA)}. Anadirla es una decision "
                f"que se toma ANTES de medir y por escrito, no un valor de repuesto")
        por_dataset[ds.nombre] = metric_id
    return por_dataset


def _exigir_que_LA_PASADA_QUEPA(datasets, protocolo: ProtocoloExploratorio) -> None:
    """Lo mismo que el guardia de C3 y sobre los MISMOS invariantes, pero
    preguntando por los cubos y las tareas que ESTA pasada va a tocar — que son
    los tres cubos y las tres tareas, y no los dos cubos de una sola tarea.

    Las cuatro cosas paran la pasada antes de medir nada, porque las cuatro
    cuestan lo mismo descubiertas ahora que a las cuarenta horas, y a las
    cuarenta horas cuestan además las cuarenta horas.
    """
    from protocolo import cpus_disponibles, reserva_segura

    # 1. La reserva de CPU tiene que caber en esta máquina.
    caben = reserva_segura(HILOS_POR_INTENTO)
    if PROCESOS_A_LA_VEZ > caben:
        raise SystemExit(
            f"esta pasada pediria {PROCESOS_A_LA_VEZ} procesos x {HILOS_POR_INTENTO} "
            f"hilos y en esta maquina ({cpus_disponibles()} CPUs) caben {caben}")

    # 2. Cada cubo que se va a tocar tiene su presupuesto REGISTRADO, y el que
    #    se va a aplicar es ese y no otro. Se comprueba llamando a la MISMA
    #    funcion que el bucle usa, no recalculando el numero aqui.
    registrados = protocolo.presupuesto.minutos_por_cubo
    for cubo in sorted({d.cubo for d in datasets}):
        if cubo not in registrados:
            raise SystemExit(
                f"el cubo {cubo!r} lo usan datasets de esta pasada y el protocolo "
                f"registrado no le da presupuesto: {sorted(registrados)}")
        aplicado = wall_seconds_del_cubo(cubo, protocolo)
        esperado = float(registrados[cubo]) * 60.0
        if aplicado != esperado or aplicado <= 0.0:
            raise SystemExit(
                f"el presupuesto que se aplicaria al cubo {cubo!r} son {aplicado} s y "
                f"el registrado son {esperado} s. Medir con un tope distinto del "
                f"registrado convierte un fallo por tiempo en un dataset perdido que "
                f"el protocolo no pedia perder")

    # 3. Cada cubo tiene sus repeticiones registradas (y `repeticiones_para`
    #    levanta si el cubo es desconocido, que es lo que se quiere).
    for cubo in sorted({d.cubo for d in datasets}):
        if protocolo.particion.repeticiones_para(cubo) < 1:
            raise SystemExit(f"el cubo {cubo!r} no tiene repeticiones registradas")

    # 4. Cada tarea que se va a tocar tiene metrica de cierre traducible. Es la
    #    comprobacion que evita el denominador recortado en silencio, y por eso
    #    se hace ANTES y no al redactar el veredicto: al redactar el veredicto
    #    los datasets ya se han caido.
    metrica_de_cierre_por_dataset(protocolo, datasets)

    # 5. Los ARFF tienen que ESTAR. Ocho de los cuarenta no se han abierto
    #    nunca en ninguna pasada (son los sellados), asi que «existe» no es una
    #    suposicion que se pueda heredar de C3.
    faltan = [d.nombre for d in datasets if not (ARFF_DIR / f"{d.data_id}.arff").exists()]
    if faltan:
        raise SystemExit(
            f"faltan {len(faltan)} ARFF en {ARFF_DIR}: {', '.join(faltan)}. Bajarlos "
            f"es una decision de Roberto (cuota de OpenML y gigas), no de esta pasada")


# ---------------------------------------------------------------------------
# LA CUENTA DE LO QUE VA A COSTAR
# ---------------------------------------------------------------------------

def plan_de_la_pasada(datasets, protocolo: ProtocoloExploratorio) -> dict:
    """Cuántos intentos, de qué cubo y de qué tarea, y qué cota de reloj tienen.

    La COTA es el peor caso literal (cada intento agota su presupuesto), la
    misma definición que `calcular_coste` del protocolo usa y por el mismo
    motivo: es lo único que ningún intento puede superar sin contarse como
    fallo. No es la previsión — la previsión sale de lo MEDIDO y va aparte, en
    `--estimar`, para que no se confundan.
    """
    motores = nombres_de_los_motores_de_la_pasada()
    por_cubo: dict[str, dict] = {}
    por_tarea: dict[str, int] = {}
    total = 0
    segundos_cota = 0.0
    for ds in datasets:
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        intentos = protocolo.particion.folds * repeticiones * len(motores)
        wall = wall_seconds_del_cubo(ds.cubo, protocolo)
        total += intentos
        segundos_cota += intentos * wall
        entrada = por_cubo.setdefault(ds.cubo, {
            "n_datasets": 0, "folds": protocolo.particion.folds,
            "repeticiones": repeticiones, "wall_seconds_por_intento": wall,
            "n_intentos": 0, "horas_cota": 0.0})
        entrada["n_datasets"] += 1
        entrada["n_intentos"] += intentos
        entrada["horas_cota"] += intentos * wall / 3600.0
        por_tarea[ds.tarea] = por_tarea.get(ds.tarea, 0) + intentos
    for entrada in por_cubo.values():
        entrada["horas_cota"] = round(entrada["horas_cota"], 2)
    return {
        "n_datasets": len(datasets),
        "n_motores": len(motores),
        "configuraciones_por_motor": 1,
        "n_intentos": total,
        "por_cubo": por_cubo,
        "intentos_por_tarea": por_tarea,
        "horas_de_reloj_cota_peor_caso": round(segundos_cota / 3600.0, 2),
        "procesos_a_la_vez": PROCESOS_A_LA_VEZ,
        "que_es_la_cota": (
            "peor caso literal: cada uno de los intentos agota el presupuesto de "
            "pared de su cubo. No es la prevision — la mayoria termina mucho antes, "
            "y lo MEDIDO va en `--estimar`."),
    }


#: SEGUNDOS POR INTENTO **MEDIDOS**, con de dónde sale cada número.
#:
#: Ninguno es una estimación: los dos primeros son la media real de los 1.260
#: intentos de la pasada de siete motores de 101-C3
#: (`pasada_exploratoria_101_c3_conforme_20260914.json`, 630 intentos por cubo);
#: los otros salen de las sondas del 2026-09-14, un pliegue y una semilla, que
#: es poco y por eso se dice.
#:
#: **LO QUE ESTO NO ES.** No es una predicción del coste de la pasada: es una
#: ancla para saber si la cota de peor caso (242 h) está cerca o lejos de lo
#: que va a pasar, que es la pregunta que hay que contestar antes de reservar
#: la máquina. Los datasets que nadie ha corrido nunca —los 8 sellados, los 10
#: de regresión, los 10 de multiclase y 9 de los 10 del cubo grande— entran con
#: el ancla de su cubo, y eso es extrapolar. Se declara extrapolado.
ANCLAS_MEDIDAS_SEGUNDOS_POR_INTENTO = {
    "pequeno": (5.40, "media real de los 630 intentos de cubo pequeno de la pasada "
                      "de 7 motores de 101-C3 (6 datasets binarios)"),
    "mediano": (13.88, "media real de los 630 intentos de cubo mediano de la misma "
                       "pasada; la arrastra Internet-Advertisements (3.279 x 1.559), "
                       "que sale a 34,16 s por intento"),
    "grande": (60.0, "NO hay pasada de cubo grande. Sondas del 2026-09-14, un "
                     "pliegue: `adult` (48.842 x 15) da 4,4-11,1 s por motor, y "
                     "`KDDCup09_appetency` (50.000 x 231) da 18,2 s el baseline y "
                     "206-264 s los cuatro arboles. Se toma 60 s como ancla de un "
                     "cubo que va de 9 a 264 s segun el ancho del dataset: es un "
                     "numero con una dispersion de 30x dentro, no una media"),
}


def estimacion_anclada_en_lo_medido(plan: dict) -> dict:
    """La previsión, con su aritmética delante y su incertidumbre dicha.

    Dos números, porque responden a preguntas distintas y confundirlos es lo
    que convierte una cota en una promesa:

    * `horas_de_reloj_cota_peor_caso` (en `plan`) es lo que NINGÚN intento
      puede superar sin contarse como fallo. 242 h.
    * esto es lo que se espera de verdad, anclado en intentos ya medidos.

    La distancia entre los dos es el margen que tiene la pasada antes de que un
    tope de pared empiece a convertir intentos en datasets perdidos — que es la
    pregunta operativa, no el total.
    """
    por_cubo = {}
    segundos = 0.0
    for cubo, entrada in plan["por_cubo"].items():
        ancla, de_donde = ANCLAS_MEDIDAS_SEGUNDOS_POR_INTENTO[cubo]
        suyos = entrada["n_intentos"] * ancla
        segundos += suyos
        por_cubo[cubo] = {
            "n_intentos": entrada["n_intentos"],
            "segundos_por_intento_medidos": ancla,
            "de_donde_sale_el_ancla": de_donde,
            "horas": round(suyos / 3600.0, 2),
            "fraccion_del_tope_del_cubo": round(
                ancla / entrada["wall_seconds_por_intento"], 3),
        }
    return {
        "horas_previstas": round(segundos / 3600.0, 2),
        "horas_de_cota": plan["horas_de_reloj_cota_peor_caso"],
        "por_cubo": por_cubo,
        "esto_es_una_EXTRAPOLACION": (
            "las anclas de pequeno y mediano son medias reales de 630 intentos cada "
            "una, pero de datasets BINARIOS; la de grande sale de sondas de un solo "
            "pliegue. Los 8 sellados, los 10 de regresion, los 10 de multiclase y 9 "
            "de los 10 del cubo grande no los ha corrido nadie nunca: entran con el "
            "ancla de su cubo. La prevision puede quedarse corta y no invalida nada "
            "— la cota de peor caso sigue siendo la cota."),
        "lo_que_puede_hacerla_fallar_por_arriba": (
            "el motor denso tarda 5,7x lo que los arboles en el cubo mediano de C3 "
            "(47,56 s frente a 8,36 s de media). Aplicado a los 206-264 s que los "
            "arboles tardan en `KDDCup09_appetency`, la densa saldria por encima de "
            "los 600 s de tope de su cubo, o sea FALLO — y la regla de cierre "
            "registrada cuenta un fallo como dataset perdido para ese motor. Esa es "
            "la unica parte de esta cuenta que hay que mirar antes de lanzar."),
    }


# ---------------------------------------------------------------------------
# UN DATASET: leerlo, partirlo, y decir su ProblemSpec
# ---------------------------------------------------------------------------

def particiones_base(ds: DatasetDeLaPasada, protocolo: ProtocoloExploratorio):
    """Lo mismo que `particiones_base` de C3 pero para las TRES tareas, y
    declarando los `Limite` que C3 tira.

    Devuelve además el diccionario `particion_declarada`, que es lo que hace
    legible el recuento final: cuántos pliegues se pidieron, cuántos salieron,
    y por qué si no coinciden.
    """
    filas, objetivo = cargar_arff(ds.data_id)
    ds.objetivo = objetivo
    filas_con_objetivo = [f for f in filas if f[objetivo] is not None]
    predictores = tuple(k for k in filas[0] if k not in ("row_id", objetivo))
    tipar_columnas_numericas(filas_con_objetivo, predictores)

    if ds.tarea == "regression":
        ds.clases, ds.positiva = None, None
    else:
        ds.clases, ds.positiva = clase_positiva_medida(filas_con_objetivo, objetivo)

    folds = protocolo.particion.folds
    repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
    propuesta = proponer_particion(
        filas_con_objetivo, plan_id=f"101c5-{ds.nombre}", observation_id_field="row_id",
        split_type="iid", seed=0, test_fraction=0.2, folds=folds, repeats=repeticiones,
        # LA LINEA DEL PUNTO 4 DEL DOCSTRING. `None` en regresion no es
        # «se me olvidó»: estratificar por un continuo bajó los pliegues de 5
        # a 2 y dejó el test en el 13,84 %, las dos cosas medidas.
        objetivo=(objetivo if ESTRATIFICACION[ds.tarea] else None))
    if not propuesta.es_viable:
        raise RuntimeError(f"{ds.nombre}: particion no viable, "
                           f"bloqueos={[b.clave for b in propuesta.bloqueos]}")

    pliegues_reales = sorted({(p.repeticion, p.pliegue) for p in propuesta.pliegues.pliegues})
    particion_declarada = {
        "plan_digest": propuesta.plan.digest(),
        "estratificado_por_el_objetivo": ESTRATIFICACION[ds.tarea],
        "folds_pedidos": folds,
        "repeticiones_pedidas": repeticiones,
        "folds_obtenidos": propuesta.pliegues.folds,
        "n_pliegues_pedidos": folds * repeticiones,
        "n_pliegues_obtenidos": len(pliegues_reales),
        # LOS `Limite` QUE C3 TIRA. Sin esto, «salieron menos intentos de los
        # declarados» no tiene respuesta dentro del artefacto.
        "limites": [{"clave": l.clave, "campo": l.campo, "medida": dict(l.medida or {})}
                    for l in propuesta.limites],
        "n_filas_con_objetivo": len(filas_con_objetivo),
        "n_filas_leidas": len(filas),
        "n_test": len(propuesta.plan.observaciones_del_rol("test")),
        "n_predictores": len(predictores),
    }

    por_id = {f["row_id"]: f for f in filas_con_objetivo}
    spec = problem_spec_de(ds, objetivo, predictores)
    return por_id, propuesta, spec, objetivo, predictores, particion_declarada


def problem_spec_de(ds: DatasetDeLaPasada, objetivo: str,
                    predictores: tuple[str, ...]) -> ProblemSpec:
    """El `ProblemSpec` de cada tarea, con lo que el esquema EXIGE de cada una
    y nada más:

    * binaria: dos clases y clase positiva declarada — sin ella, sensibilidad,
      PPV y el umbral quedan sin definir y elegirla por orden alfabético es
      inventarse la mitad del problema (`esquemas.py` lo rechaza, no es una
      recomendación).
    * multiclase: tres clases o más y **`positive_label=None`**. No es que no
      se sepa: es que en multiclase no hay una, y poner la minoritaria haría
      que el informe calculara sensibilidad/especificidad «de esa contra el
      resto» y las publicara como si describieran el problema.
    * regresión: NI clases NI positiva. El esquema levanta si se le pasan, que
      es la comprobación que convierte esto en algo que no se puede olvidar.
    """
    if ds.tarea == "regression":
        return ProblemSpec(problem_id=f"101c5-{ds.nombre}", target=objetivo,
                           task="regression", observation_unit="fila",
                           predictors=predictores)
    return ProblemSpec(problem_id=f"101c5-{ds.nombre}", target=objetivo,
                       task=ds.tarea, observation_unit="fila",
                       classes=ds.clases,
                       positive_label=(ds.positiva if ds.tarea == "binary_classification"
                                       else None),
                       predictors=predictores)


def metricas_del_informe(informe: dict | None) -> dict:
    """TODAS las métricas que el informe trae, por su `metric_id`.

    C3 sacaba dos (`auroc` y `accuracy`) de una lista que ya venía entera:
    `evaluar()` (105-C1) recorre el catálogo de la tarea y devuelve lo que la
    muestra sostiene. Las seis secundarias de binaria, la calibración y las
    tres de regresión estaban ahí y las tiraba el llamante — el hueco está en
    el cableado, que en este proyecto ya van catorce.

    Una métrica con valor `None` se guarda como `None` y no se omite: el
    catálogo la devuelve así cuando es indefinida (un PPV sin ningún predicho
    positivo, por ejemplo), y omitirla la volvería indistinguible de una que no
    se pidió.
    """
    if not informe:
        return {}
    return {m["metric_id"]: m.get("value") for m in informe.get("metrics", [])}

def reconciliar_los_procesos_con_lo_registrado(usados: int, presupuesto: dict) -> dict:
    """POR QUE LA PASADA CORRE CON UN PROCESO Y EL PROTOCOLO REGISTRA DOS.

    El artefacto llevaba `procesos_en_paralelo: 1` y
    `presupuesto.procesos_en_paralelo_registrados: 2` **uno al lado del otro**,
    sin una frase que dijera que es deliberado ni por que. Misma familia que el
    «3.500 contra 3.479»: dos numeros verdaderos juntos y ningun puente.

    **LA DIRECCION SE CALCULA, NO SE ESCRIBE.** Hoy la desviacion es
    conservadora —menos procesos = menos competencia por la maquina = topes de
    reloj mas faciles de cumplir, no mas dificiles—, asi que no ensucia el
    veredicto. Pero una frase guardada diciendo «es conservadora» seguiria ahi,
    sonando razonable, el dia que alguien suba los procesos por encima de lo
    registrado — y ENTONCES la desviacion apretaria los topes y si podria
    costar datasets. Por eso `conservadora` es una conclusion.
    """
    registrados = presupuesto.get("procesos_en_paralelo_registrados")
    if registrados is None:
        return {"registrados": None, "usados": usados,
                "motivo": "el presupuesto no registra procesos; no hay nada que reconciliar"}
    conservadora = usados <= registrados
    return {
        "registrados": registrados,
        "usados": usados,
        "coincide": usados == registrados,
        "conservadora_para_el_veredicto": conservadora,
        "por_que": (
            "MENOS procesos que los registrados: menos competencia por la maquina, "
            "o sea topes de reloj de pared mas faciles de cumplir, no mas dificiles. "
            "La desviacion no puede inflar el recuento de datasets cumplidos."
            if usados < registrados else
            "los procesos coinciden con lo registrado."
            if usados == registrados else
            "MAS procesos que los registrados: mas competencia por la maquina, o sea "
            "topes de reloj MAS DIFICILES de cumplir. Un intento que se pasa del tope "
            "cuenta como dataset perdido, asi que esta desviacion SI puede mover el "
            "veredicto, y en la direccion de perjudicar a los motores lentos."),
        "donde_se_decide": (
            "`PROCESOS_A_LA_VEZ` en `benchmarks/fase0/pasada_exploratoria_101_c3.py`. "
            "El protocolo registra el reparto; esta constante dice lo que la maquina "
            "de casa aguanta."),
    }


def reconciliar_el_plan_con_lo_medido(plan: dict, particiones: dict,
                                      resultados: list, n_motores: int) -> dict:
    """POR QUE `n_intentos` NO ES EL DEL PLAN — hallazgo M1, 2026-09-16.

    El artefacto lleva `plan.n_intentos = 3500` y `n_intentos = 3479` uno al
    lado del otro **y nada que los reconcilie**. El motivo esta dentro, en
    `particion_por_dataset.<ds>.limites`, pero hay que ir a buscarlo dataset a
    dataset sabiendo ya que existe. Quien no lo sepa lee dos numeros distintos
    para lo mismo y se queda con la duda — o peor, con la sospecha.

    **NO SE ESCRIBE LA FRASE: SE CALCULA.** Una explicacion redactada y
    guardada deja de ser verdad en cuanto cambian los numeros que la sostenian
    y sigue sonando razonable. Aqui se suma lo que cada particion recortada
    explica y **se compara con el hueco de verdad**.

    **Y `cuadra` es una CONCLUSION, no un deseo.** Si lo explicado no cubre el
    hueco entero queda un `resto_sin_explicar` distinto de cero, y eso es
    exactamente lo que hay que ver: un intento que falta y que ninguna
    particion justifica no es un detalle de recuento, es una medicion que se
    perdio sin que nadie lo dijera.
    """
    esperados = plan.get("n_intentos")
    medidos = len(resultados)
    if esperados is None:
        return {"plan": None, "medidos": medidos,
                "motivo": "el plan no declara `n_intentos`; no hay nada que reconciliar"}
    explicado: list[dict] = []
    for nombre, particion in sorted(particiones.items()):
        pedidos = particion.get("n_pliegues_pedidos")
        obtenidos = particion.get("n_pliegues_obtenidos")
        if pedidos is None or obtenidos is None or pedidos == obtenidos:
            continue
        explicado.append({
            "dataset": nombre,
            "pliegues_pedidos": pedidos,
            "pliegues_obtenidos": obtenidos,
            "intentos_que_explica": (pedidos - obtenidos) * n_motores,
            "limites": particion.get("limites"),
        })
    suma = sum(e["intentos_que_explica"] for e in explicado)
    hueco = esperados - medidos
    return {
        "plan": esperados,
        "medidos": medidos,
        "hueco": hueco,
        "explicado_por_particiones_recortadas": suma,
        "resto_sin_explicar": hueco - suma,
        "cuadra": hueco == suma,
        "por_dataset": explicado,
        "como_se_lee": (
            "`plan.n_intentos` es lo que se pedia ANTES de particionar; "
            "`n_intentos` es lo que se pudo medir. La diferencia se explica "
            "dataset a dataset con los pliegues que su clase minoritaria no "
            "dio. Si `cuadra` es false, queda un hueco que ninguna particion "
            "justifica y eso SI es un problema."),
    }



#: Los campos del registro que NO son métricas. Aplanar las métricas encima del
#: registro es lo que permite que `aplicar_regla_de_cierre` lea `r.get(metrica)`
#: sin conocer la forma anidada — pero una métrica que se llamara como un campo
#: lo PISARÍA, y un `estado` sobrescrito por un número convierte un fallo en un
#: intento completado sin que nada chille. Hoy ninguna de las quince del
#: catálogo colisiona; el guardia está para el día que se añada una.
def aplanar_metricas_en_el_registro(registro: dict, metricas: dict) -> dict:
    """Las métricas como claves de primer nivel del registro, y PARA si alguna
    pisaría un campo que ya existe."""
    chocan = sorted(set(metricas) & set(registro))
    if chocan:
        raise SystemExit(
            f"la(s) metrica(s) {chocan} del catalogo se llama(n) igual que un campo "
            f"del registro de esta pasada. Aplanarlas encima lo sobrescribiria en "
            f"silencio — un `estado` pisado por un numero convierte un fallo en un "
            f"intento completado. Renombrar el campo del registro antes de medir")
    registro.update(metricas)
    return registro


# ---------------------------------------------------------------------------
# LA PASADA
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true",
                        help="ignora el cache entero y re-ejecuta TODOS los intentos")
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON de salida (por omision, el de este directorio)")
    parser.add_argument("--estimar", action="store_true",
                        help="imprime la cuenta de coste con su aritmetica y NO mide nada")
    parser.add_argument("--solo", default=None,
                        help="nombres de dataset separados por coma: corre SOLO esos. "
                             "Para probar el guion, nunca para medir — lo que salga "
                             "queda declarado como subconjunto en el propio artefacto")
    args = parser.parse_args(argv)

    protocolo = protocolo_registrado()
    datasets = datasets_de_la_pasada(protocolo)
    subconjunto = None
    if args.solo:
        pedidos = [n.strip() for n in args.solo.split(",") if n.strip()]
        conocidos = {d.nombre for d in datasets}
        desconocidos = [n for n in pedidos if n not in conocidos]
        if desconocidos:
            raise SystemExit(f"--solo nombra datasets que no estan en el protocolo: "
                             f"{desconocidos}")
        datasets = [d for d in datasets if d.nombre in pedidos]
        subconjunto = pedidos

    _exigir_que_LA_PASADA_QUEPA(datasets, protocolo)
    plan = plan_de_la_pasada(datasets, protocolo)
    metrica_por_dataset = metrica_de_cierre_por_dataset(protocolo, datasets)

    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]}")
    print(f"datasets: {plan['n_datasets']} ({sum(1 for d in datasets if d.sellado)} sellados), "
          f"motores: {plan['n_motores']}, intentos: {plan['n_intentos']}")
    for cubo, e in sorted(plan["por_cubo"].items()):
        print(f"  cubo {cubo:8}: {e['n_datasets']:>2} datasets x {e['folds']} pliegues x "
              f"{e['repeticiones']} rep x {plan['n_motores']} motores = {e['n_intentos']:>5} "
              f"intentos, {e['wall_seconds_por_intento']:.0f}s de tope -> cota "
              f"{e['horas_cota']:.2f} h")
    print(f"  COTA de peor caso: {plan['horas_de_reloj_cota_peor_caso']:.2f} h "
          f"({plan['horas_de_reloj_cota_peor_caso']/24:.2f} dias), {PROCESOS_A_LA_VEZ} proceso(s)")
    print(f"  metrica de cierre por tarea: "
          + ", ".join(f"{t}={METRICA_DE_CIERRE_POR_TAREA[m]}"
                      for t, m in sorted(protocolo.regla_de_cierre.metrica_por_tarea.items())))
    if args.estimar:
        estimacion = estimacion_anclada_en_lo_medido(plan)
        print(f"\n  PREVISION anclada en intentos ya medidos: "
              f"{estimacion['horas_previstas']:.2f} h "
              f"(la cota de peor caso son {estimacion['horas_de_cota']:.2f} h)")
        for cubo, e in sorted(estimacion["por_cubo"].items()):
            print(f"    {cubo:8}: {e['n_intentos']:>5} x {e['segundos_por_intento_medidos']:>6.2f}s "
                  f"= {e['horas']:>6.2f} h  ({e['fraccion_del_tope_del_cubo']*100:.1f} % del tope)")
            print(f"              ancla: {e['de_donde_sale_el_ancla']}")
        print(f"\n  ES UNA EXTRAPOLACION: {estimacion['esto_es_una_EXTRAPOLACION']}")
        print(f"\n  LO QUE HAY QUE MIRAR ANTES DE LANZAR: "
              f"{estimacion['lo_que_puede_hacerla_fallar_por_arriba']}")
        return

    ruta_salida = (Path(args.salida) if args.salida else
                   Path(__file__).resolve().parent / "pasada_amplia_101_c5_resultado.json")
    cache_previo, payload_previo = ({}, {}) if args.forzar else _cargar_cache(ruta_salida)
    entorno_digest = _digest_entorno()
    digest_por_motor = {n: _digest_fichero(r) for n, r in c3._FICHERO_POR_MOTOR.items()}

    procedencia = procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "por_motor": digest_por_motor},
        datos_de_entrada={d.nombre: ARFF_DIR / f"{d.data_id}.arff" for d in datasets})
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    if payload_previo:
        declarada = procedencia_declarada(payload_previo)
        print(f"cache previo: {len(cache_previo)} registros, procedencia "
              f"{declarada['estado']} -- {declarada['explicacion']}", flush=True)

    motores = motores_de_la_pasada()
    resultados: list[dict] = []
    particiones_declaradas: dict[str, dict] = {}
    reusados = 0
    inicio = time.perf_counter()

    #: CADA CUANTO, COMO MUCHO, se escribe un punto de control — 2026-09-16.
    #:
    #: El guardado colgaba del bucle de REPETICIONES, y el protocolo registra
    #: `repeticiones_grande = 1`: en el cubo grande cada repeticion ES el
    #: dataset entero, asi que los 10 datasets mas caros tenian UN solo
    #: guardado, al final. Morir a la hora y veinte costaba la hora y veinte —
    #: justo lo que el comentario de abajo dice evitar (hallazgo A3).
    #:
    #: **Se acepto sin medir el coste, y medido no habia nada que aceptar**: un
    #: guardado completo tarda **0,90 s** sobre los 3.479 resultados (0,67 s el
    #: veredicto con su bootstrap, 0,04 s la dispersion, 0,19 s serializar 5,2
    #: MB). El intercambio real no era «1 h garantizada contra 2 h probables»:
    #: era **36 s contra hasta hora y media**.
    #:
    #: **Va por RELOJ y no por vuelta de bucle**, que es lo que lo hace gratis
    #: en los dos extremos: bajarlo al nivel de pliegue a secas daria 15
    #: guardados por dataset del cubo pequeno —donde el dataset entero dura dos
    #: minutos— y eso si seria un 11 % de sobrecoste. Con un tope de tiempo, la
    #: granularidad la pone el coste real de lo que se esta midiendo.
    SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0
    ultimo_punto_de_control = [time.perf_counter()]

    def guardar(parcial: bool) -> dict:
        ultimo_punto_de_control[0] = time.perf_counter()
        return _componer_y_guardar(
            resultados, procedencia, payload_previo, ruta_salida, datasets=datasets,
            plan=plan, protocolo=protocolo, metrica_por_dataset=metrica_por_dataset,
            particiones=particiones_declaradas, subconjunto=subconjunto,
            total_wall_s=time.perf_counter() - inicio, reusados=reusados, parcial=parcial)

    for ds in datasets:
        por_id, propuesta, spec, objetivo, predictores, declarada = particiones_base(
            ds, protocolo)
        particiones_declaradas[ds.nombre] = declarada
        test_ids = propuesta.plan.observaciones_del_rol("test")
        wall_seconds = wall_seconds_del_cubo(ds.cubo, protocolo)
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        print(f"\n=== {ds.nombre} (data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"sellado={'SI' if ds.sellado else 'no'}, n={len(por_id)}, "
              f"test={len(test_ids)}, tope={wall_seconds:.0f}s, "
              f"metrica={metrica_por_dataset[ds.nombre]}) ===", flush=True)
        if declarada["limites"]:
            print(f"  LIMITE DE PARTICION: {declarada['limites']} -- pliegues "
                  f"{declarada['n_pliegues_obtenidos']} de {declarada['n_pliegues_pedidos']}",
                  flush=True)

        for repeticion in range(repeticiones):
            for pliegue_i in range(protocolo.particion.folds):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion,
                                                        pliegue=pliegue_i)
                if pliegue is None:
                    continue
                for motor in motores:
                    clave = (ds.nombre, motor.nombre, repeticion, pliegue_i)
                    previo = cache_previo.get(clave)
                    if _reusable(previo, entorno_digest, digest_por_motor[motor.nombre],
                                 wall_seconds):
                        registro = dict(previo, reusado=True)
                        registro.setdefault("procedencia_id", None)
                        resultados.append(registro)
                        reusados += 1
                        continue

                    crudas_train = [por_id[i] for i in pliegue.entrena]
                    crudas_val = [por_id[i] for i in pliegue.valida]
                    crudas_test = [por_id[i] for i in test_ids]
                    transformadas = preparar_para_motor(
                        crudas_train, crudas_train + crudas_val + crudas_test,
                        objetivo, predictores, motor)
                    n_tr, n_va = len(crudas_train), len(crudas_val)
                    hacer = lambda xs: Particion.desde_filas(  # noqa: E731
                        xs, row_id_field="row_id", target_field=objetivo)

                    presupuesto = Presupuesto(
                        wall_seconds=wall_seconds, hilos=HILOS_POR_INTENTO,
                        seed=protocolo.particion.semillas[repeticion])
                    t0 = time.perf_counter()
                    intento = ejecutar_intento_aislado(
                        motor, hacer(transformadas[:n_tr]),
                        hacer(transformadas[n_tr:n_tr + n_va]),
                        hacer(transformadas[n_tr + n_va:]),
                        spec, presupuesto,
                        candidate=f"{motor.nombre}-{CONFIGURACION_UNICA}",
                        split_plan_digest=propuesta.plan.digest(),
                        dataset=ds.nombre, pliegue=pliegue_i, repeticion=repeticion)
                    transcurrido = time.perf_counter() - t0

                    metricas = metricas_del_informe(intento.informe)
                    recursos = intento.recursos or {}
                    registro = {
                        "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                        "tarea": ds.tarea, "sellado": ds.sellado,
                        "motor": motor.nombre, "repeticion": repeticion,
                        "pliegue": pliegue_i, "estado": intento.estado,
                        "semilla": protocolo.particion.semillas[repeticion],
                        "presupuesto_wall_s": wall_seconds,
                        "configuracion": CONFIGURACION_UNICA,
                        "metrica_de_cierre": metrica_por_dataset[ds.nombre],
                        "wall_s": round(transcurrido, 3),
                        # TODAS las metricas del informe, no dos. Y ademas
                        # aplanadas arriba, porque `aplicar_regla_de_cierre`
                        # lee `r.get(metrica)` del registro plano — dejarlas
                        # solo anidadas seria construir el dato y no cablearlo.
                        "metricas": metricas,
                        # Del bloque `recursos`, que ya existia y nadie leia.
                        "tiempo_de_ajuste": recursos.get("wall_seconds"),
                        "cpu_segundos": recursos.get("cpu_seconds"),
                        "motivo": (intento.motivo_del_estado["es"]
                                   if intento.motivo_del_estado else None),
                        "traza": intento.traza,
                        "entorno_digest": entorno_digest,
                        "motor_digest": digest_por_motor[motor.nombre],
                        "procedencia_id": procedencia["procedencia_id"],
                        "reusado": False,
                    }
                    aplanar_metricas_en_el_registro(registro, metricas)
                    resultados.append(registro)
                completados = sum(
                    1 for r in resultados
                    if r["dataset"] == ds.nombre and r["repeticion"] == repeticion
                    and r["pliegue"] == pliegue_i and r["estado"] == "completed")
                print(f"  rep={repeticion} pliegue={pliegue_i}: "
                      f"{completados}/{len(motores)} completed", flush=True)
                # Y TAMBIEN AQUI SI HA PASADO EL TIEMPO. Sin esto, el cubo
                # grande —una sola repeticion por dataset— solo guardaba al
                # terminar el dataset entero. Ver la constante de arriba: el
                # guardado cuesta 0,90 s, asi que el tope de 60 s lo deja por
                # debajo del 1,5 % aunque cada pliegue durase justo un minuto.
                if (time.perf_counter() - ultimo_punto_de_control[0]
                        >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL):
                    guardar(parcial=True)
            # AL TERMINAR CADA REPETICION, siempre — es el SUELO, no el techo.
            # C3 guardaba por dataset porque su dataset mas caro eran doce
            # minutos; aqui un dataset del cubo grande puede ser hora y media.
            guardar(parcial=True)

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados} reusados, {len(resultados)-reusados} ejecutados) ===")
    if cache_previo and reusados == 0:
        print(f"AVISO DE CACHE: {len(cache_previo)} registros previos y NINGUNO reusable "
              f"-- {procedencia_declarada(payload_previo)['explicacion']}")
    salida = guardar(parcial=False)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")


def _alcance_y_veredicto(resultados, datasets, protocolo, metrica_por_dataset,
                         subconjunto) -> dict:
    """El veredicto de la regla de cierre CON su alcance pegado, para cada
    motor que compite — y ahora con LA MÉTRICA DE CADA TAREA.

    La diferencia con el de C3 no es de forma: allí `veredicto_con_su_alcance`
    recibía `metrica="auroc"` y aquí recibe el mapa por dataset. Con una sola
    métrica sobre las tres tareas, los 20 datasets que no son binarios se caen
    del denominador sin decirlo — medido, ver el punto 2 del docstring.
    """
    nombres_motores = nombres_de_los_motores_de_la_pasada()
    nombres_datasets = [d.nombre for d in datasets]
    criterio = CRITERIO_DEL_SUBCONJUNTO
    if subconjunto is not None:
        criterio = (f"SUBCONJUNTO DE PRUEBA, no la pasada de C5: se corrio con --solo "
                    f"{','.join(subconjunto)}, o sea {len(subconjunto)} de los 40 del "
                    f"protocolo. De este fichero no se sigue NADA sobre el resto. "
                    f"El criterio de la pasada completa seria: {CRITERIO_DEL_SUBCONJUNTO}")
    por_motor = {}
    for nombre in nombres_motores:
        if nombre == "baseline":
            continue
        por_motor[nombre] = veredicto_con_su_alcance(
            protocolo, resultados, motor=nombre,
            motores_declarados=nombres_motores,
            datasets_declarados=nombres_datasets,
            criterio_del_subconjunto=criterio,
            configuraciones_declaradas=configuraciones_de_la_pasada(),
            criterio_de_las_configuraciones=CRITERIO_DE_LAS_CONFIGURACIONES,
            metrica_por_dataset=metrica_por_dataset,
            datasets_exigidos=nombres_datasets)
    alcance = next(iter(por_motor.values()))["alcance"] if por_motor else {}
    for v in por_motor.values():
        v.pop("alcance", None)
    return {
        "alcance": alcance,
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "metrica_de_cierre_por_tarea_registrada":
            dict(protocolo.regla_de_cierre.metrica_por_tarea),
        "de_la_metrica_registrada_al_metric_id": dict(METRICA_DE_CIERRE_POR_TAREA),
        "estabilidad_del_ganador": estabilidad_del_ganador(
            resultados, metrica_por_dataset=metrica_por_dataset),
        "por_motor": por_motor,
    }


def presupuesto_declarado_de_la_pasada(datasets, protocolo) -> dict:
    """El presupuesto de pared que esta pasada aplica, CUBO POR CUBO, y de
    dónde sale — con el digest del protocolo del que salen los minutos."""
    cubos = sorted({d.cubo for d in datasets})
    return {
        "wall_seconds_por_cubo": {c: wall_seconds_del_cubo(c, protocolo) for c in cubos},
        "minutos_por_cubo_registrados": dict(protocolo.presupuesto.minutos_por_cubo),
        "repeticiones_por_cubo_registradas": {
            c: protocolo.particion.repeticiones_para(c) for c in cubos},
        "folds_registrados": protocolo.particion.folds,
        "semillas_registradas": list(protocolo.particion.semillas),
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "hilos_por_intento": HILOS_POR_INTENTO,
        "procesos_a_la_vez": PROCESOS_A_LA_VEZ,
        "procesos_en_paralelo_registrados": protocolo.presupuesto.procesos_en_paralelo,
    }


def _componer_y_guardar(resultados, procedencia, payload_previo, ruta_salida, *,
                        datasets, plan, protocolo, metrica_por_dataset, particiones,
                        subconjunto, total_wall_s, reusados, parcial) -> dict:
    """Compone el JSON y lo escribe, atómicamente. UN solo sitio, y se llama
    también a mitad de la pasada.

    El bloque `alcance_y_veredicto` se calcula ANTES del digest y va DENTRO de
    él, igual que en C3 y por el mismo motivo: un alcance que se puede
    reescribir sin que el fichero deje de cuadrar consigo mismo no declara
    nada. El alcance viaja con el número y dentro del sello.
    """
    import json

    procedencias, sin_procedencia = _procedencias_citadas(
        resultados, procedencia, (payload_previo.get("procedencias") or {}))

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "101-C5",
        "procedencia": procedencia,
        "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial,
        "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "plan": plan,
        "presupuesto": presupuesto_declarado_de_la_pasada(datasets, protocolo),
        "procesos_en_paralelo": PROCESOS_A_LA_VEZ,
        "datasets_declarados": [d.a_json() for d in datasets],
        # LA PARTICION DE CADA DATASET, con sus `Limite`. C3 los tiraba, y sin
        # ellos «salieron menos intentos de los declarados» no tiene respuesta
        # dentro del artefacto.
        "particion_por_dataset": dict(particiones),
        "criterio_de_la_estratificacion": CRITERIO_DE_LA_ESTRATIFICACION,
        "medidas_siempre_que_no_se_pueden_dar": dict(MEDIDAS_SIEMPRE_QUE_NO_SE_PUEDEN_DAR),
        # EL MAPA COMPLETO, desde la lista del protocolo: con esto el protocolo
        # se puede comprobar CONTRA EL ARTEFACTO sin leer este fuente.
        "donde_vive_cada_medida_siempre": donde_vive_cada_medida_siempre(
            protocolo, resultados),
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados),
        # LOS DOS NUMEROS, RECONCILIADOS. Ver la funcion: se calcula, no se
        # redacta, y `cuadra` es una conclusion.
        # LA OTRA PAREJA DE NUMEROS SIN PUENTE: `procesos_en_paralelo` dice 1 y
        # `presupuesto.procesos_en_paralelo_registrados` dice 2. Misma familia,
        # misma disciplina: la direccion de la desviacion se CALCULA.
        "por_que_los_procesos_no_son_los_registrados": (
            reconciliar_los_procesos_con_lo_registrado(
                PROCESOS_A_LA_VEZ,
                presupuesto_declarado_de_la_pasada(datasets, protocolo))),
        "por_que_n_intentos_no_es_el_del_plan": reconciliar_el_plan_con_lo_medido(
            plan, dict(particiones), resultados,
            n_motores=len(nombres_de_los_motores_de_la_pasada())),
        "n_reusados": reusados,
        "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
        "resultados": resultados,
    }
    salida["alcance_y_veredicto"] = _alcance_y_veredicto(
        resultados, datasets, protocolo, metrica_por_dataset, subconjunto)
    c3.sellar_la_salida(salida)
    temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta_salida)
    return salida


if __name__ == "__main__":
    main()

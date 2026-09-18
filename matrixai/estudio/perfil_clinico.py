# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""109-C2 — el perfil de la tarea cuando la tarea es CLÍNICA: la tabla de
umbrales con su incertidumbre, el PPV/NPV **a la prevalencia declarada**, la
curva de decisión (DCA) con sus dos referencias, y la ficha que se le enseña a
quien decide.

LO PRIMERO, Y NO ES LA ARITMÉTICA: **PPV Y NPV DEPENDEN DE LA PREVALENCIA, Y
SENSIBILIDAD Y ESPECIFICIDAD NO.** Un PPV medido sobre una muestra con un 40 %
de casos y enseñado a un equipo cuya población tiene un 2 % es un número que
miente sin equivocarse en ninguna cuenta: la división está bien hecha y la
conclusión que induce es falsa. Por eso este módulo **no publica PPV ni NPV si
nadie declara la prevalencia** — no los calcula con la de la muestra y se los
calla: `ValorDeMetrica` sin valor y con motivo, que es la forma que este
paquete ya tiene de decir «esto no se sabe» sin fabricar un cero. Cuando SÍ se
declara, el número viaja con la prevalencia pegada (`prevalencia_declarada`) y
al lado la de la muestra (`prevalencia_observada`), para que la distancia entre
las dos se vea en vez de adivinarse.

El transporte es Bayes: `PPV = sens·pr / (sens·pr + (1-esp)·(1-pr))` y
`NPV = esp·(1-pr) / (esp·(1-pr) + (1-sens)·pr)`. **Descansa en un supuesto que
se declara y no se esconde**: que la sensibilidad y la especificidad medidas
aquí valen también allí. Si el espectro de la población destino es otro —otro
punto del curso de la enfermedad, otro criterio de inclusión—, no valen, y
entonces el PPV transportado tampoco. La ficha lo escribe.

**«VALIDACIÓN INTERNA ÚNICAMENTE» VA ARRIBA**, no en una nota al pie: cambia
qué se puede hacer con todo lo que viene debajo, así que es lo primero que se
lee. `alcance_de_validacion()` lo DERIVA de lo que ya declaran 104-C0 y 103-C4
(la etiqueta de evidencia y el tipo de partición) en vez de abrir un campo
nuevo que alguien pudiera rellenar a mano distinto de los otros dos — dos
sitios declarando lo mismo acaban divergiendo. Y no hay forma de construir un
`PerfilClinico` sin ese dato: un dibujo afirma por omisión, y una ficha sin esa
línea afirma que el modelo se validó fuera.

**LOS SEGMENTOS LOS PREDEFINE EL EQUIPO.** Un subgrupo encontrado mirando los
datos y presentado igual que uno predefinido es el error clásico de este
terreno. `AnalisisDeSegmento` (105-C4) ya lleva `predefinido`; aquí se añade lo
que faltaba: **sin nadie que conste como quien los predefinió, ningún segmento
puede viajar marcado como predefinido** — se rechaza al construir el perfil, no
se avisa en una nota. Con equipo o sin él, los exploratorios se marcan.

LA CURVA DE DECISIÓN, Y SUS DOS REFERENCIAS. `beneficio neto = TP/n − FP/n ·
pt/(1−pt)`, la fórmula literal del criterio de terminado. Las dos referencias
no son adorno: **«a nadie» vale 0 por definición** (no se trata a ninguno: ni
TP ni FP) y **«a todos» sale de la MISMA fórmula** con los recuentos de tratar
a todo el mundo (`TP = positivos`, `FP = negativos`) — escribir para la
referencia una segunda fórmula equivalente sería la duplicación que acaba
divergiendo. Un modelo que no supera a las dos no aporta nada en ese umbral, y
eso se dice: `umbrales_sin_ventaja` lo enumera y la ficha lo escribe.

`pt` es a la vez el umbral con el que se decide y el precio al que se cambia un
falso positivo por un verdadero positivo — eso ES la curva de decisión, no una
simplificación. `pt = 1` no existe (el cambio divide por cero) y se rechaza en
vez de devolver un número enorme.

POR QUÉ UNA MUESTRA CON LA CLASE YA DECIDIDA SE RECHAZA AQUÍ. `matriz_de_
confusion` (105-C1) da prioridad a `predictions` sobre el umbral, y con razón:
lo que el modelo dijo manda. Pero una curva de decisión MUEVE el umbral, y
sobre etiquetas fijas devolvería la misma fila en todos los umbrales — una
curva plana perfectamente creíble y perfectamente falsa. Es un fallo de
cableado, así que se levanta excepción, no se calcula a medias.

EL CONTROL CONOCIDO-BUENO, Y LO QUE MIDE DE VERDAD. El criterio de terminado
pide que «invertir la probabilidad intercambie sens/esp y la prueba lo cace».
Medido (`test_c109_c2_perfil_clinico.py`), son DOS hechos distintos, y cada uno
vale en un umbral concreto que hay que decir:

* cambiar `p` por `1−p` **dejando la misma clase positiva y el MISMO umbral
  `t`** da `sens'(t) = 1 − sens(1−t)` y `esp'(t) = 1 − esp(1−t)`: complementa
  las métricas del umbral REFLEJADO, no las del mismo. Solo en `t = 0,5` —el
  punto fijo del reflejo— las dos cosas coinciden. En `t = 0,15`, medido: la
  directa da 1,0/0,333 y la invertida 0,750/0,0, que NO es su complemento
  (0,0/0,667) y SÍ el de la directa en 0,85.
* el INTERCAMBIO exacto aparece cuando la inversión se escribe entera: `1−p` es
  la probabilidad de la OTRA clase, así que la clase positiva pasa a ser la
  otra **y el umbral se refleja a `1−t`**; entonces `sens' = esp` y
  `esp' = sens`. Sin reflejar el umbral no hay intercambio, salvo otra vez en
  `t = 0,5`.

Las dos igualdades valen salvo por las filas que caen justo en el umbral (la
matriz de confusión decide con `>=`). Están probadas en 0,5 **y fuera de 0,5**:
un criterio comprobado solo en su punto fijo no está comprobado, y una nota
que dice «intercambia» sin decir en qué umbral se hereda igual que un dato
falso.

STDLIB PURO, como el resto del paquete, y lo que redacta el core se traduce en
el core: los motivos salen de `textos.py` y la ficha lleva sus dos redacciones
aquí mismo.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Sequence

from matrixai.estudio.calibracion import CurvaDeFiabilidad
from matrixai.estudio.comparaciones import ComparacionEmparejada
from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import ValorDeMetrica, _sobre
from matrixai.estudio.incertidumbre import intervalo, medir
from matrixai.estudio.metricas import (
    REGISTRO,
    EntradaNoMedible,
    Indefinida,
    Muestra,
    matriz_de_confusion,
)
from matrixai.estudio.segmentos import AnalisisDeSegmento
from matrixai.estudio.textos import IDIOMAS, motivo
from matrixai.estudio.validacion import (
    digest_canonico,
    exigir_booleano,
    exigir_entero,
    exigir_mapa,
    exigir_real,
    exigir_texto,
    exigir_texto_o_nulo,
)
from matrixai.estudio.vocabulario import (
    ETIQUETAS_DE_EVIDENCIA,
    TIPOS_DE_PARTICION,
    exigir_opcion,
)

__all__ = [
    "ALCANCES_DE_VALIDACION",
    "MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE",
    "BENEFICIO_NETO_NO_TRATAR",
    "PPV_A_PREVALENCIA",
    "NPV_A_PREVALENCIA",
    "VERSION_DEL_PERFIL",
    "CurvaDeDecision",
    "FilaDeUmbral",
    "PerfilClinico",
    "PuntoDeDecision",
    "TablaDeUmbrales",
    "alcance_de_validacion",
    "beneficio_neto",
    "curva_de_decision",
    "ficha_del_perfil",
    "tabla_de_umbrales",
]

#: La versión de fórmula de lo que ESTE módulo calcula: el transporte a
#: prevalencia y el beneficio neto. Las de C1 siguen siendo las suyas.
VERSION_DEL_PERFIL = "1.0.0"

#: POR QUÉ UN PERFIL PUEDE NO TRAER LA COMPARACIÓN CON EL BASELINE (109-C3).
#: Cerrado, como todo vocabulario de esta casa: una clave suelta no tiene texto
#: en la ficha. Son los dos casos en que la pregunta no se puede plantear, no
#: dos maneras de perderla: si el ganador ES el baseline no hay «contra quién»,
#: y un baseline que no puntuó no tiene muestra que emparejar. Cuando SÍ se
#: pudo plantear pero las filas no casan, eso no es un motivo de ausencia: es
#: una comparación con veredicto `incomparable`, que dice por qué.
MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE = ("el_ganador_es_el_baseline",
                                           "el_baseline_no_puntuo")

#: Los `metric_id` del transporte. **No son `ppv`/`npv`**, que en el registro
#: de 105-C1 son los observados en la muestra: confundirlos es exactamente el
#: defecto que este corte existe para evitar, y dos nombres distintos lo hacen
#: imposible por construcción.
PPV_A_PREVALENCIA = "ppv_at_declared_prevalence"
NPV_A_PREVALENCIA = "npv_at_declared_prevalence"

#: El beneficio neto de no tratar a nadie: **0 por definición**, no medido.
#: Sin tratar a nadie no hay TP ni FP, y la fórmula da cero para cualquier
#: `pt` y cualquier muestra. Es constante a propósito: un cero medido y un
#: cero definido se leen igual y no son lo mismo.
BENEFICIO_NETO_NO_TRATAR = 0.0

#: Qué alcance tiene lo que se midió. Tokens en inglés, como el resto de los
#: vocabularios que viajan en JSON.
#:
#: * `external` — cohorte externa (103-C4): otra fuente de datos.
#: * `temporal` — misma fuente, periodo distinto: el test reservado se separó
#:   por tiempo.
#: * `internal_only` — todo lo demás, **incluida la prueba reservada interna**.
#:   Una partición aleatoria del mismo dataset sigue siendo validación interna,
#:   por muy reservada que estuviera esa partición.
ALCANCES_DE_VALIDACION = ("internal_only", "temporal", "external")

#: Los tipos de partición (103-C4) que SEPARAN POR TIEMPO. Vive una vez y lo
#: leen dos sitios: `alcance_de_validacion()`, para decidir si cabe hablar de
#: validación temporal, y la ficha, para no negar una separación temporal que
#: existe cuando la evidencia no llega a sostenerla — que es justo lo que la
#: ficha escribía en 6 de las 24 combinaciones (auditoría del 2026-09-15).
_DISENOS_CON_SEPARACION_TEMPORAL = ("temporal", "groups_and_time")


def alcance_de_validacion(*, evidencia: str, diseno: str) -> str:
    """El alcance, DERIVADO de lo que ya se declaró en otro sitio.

    No hay un campo «alcance» que rellenar a mano: sale de la etiqueta de
    evidencia (104-C0) y del tipo de partición (103-C4), que son los dos
    documentos donde esa decisión ya vive. Validación temporal o externa
    **solo si existe la cohorte**; todo lo demás es interna únicamente.
    """
    exigir_opcion(evidencia, "evidencia", ETIQUETAS_DE_EVIDENCIA)
    exigir_opcion(diseno, "diseno", TIPOS_DE_PARTICION)
    if evidencia == "external_validation":
        return "external"
    if evidencia in ("independent_test", "repeated_test_use") \
            and diseno in _DISENOS_CON_SEPARACION_TEMPORAL:
        return "temporal"
    return "internal_only"


# ---------------------------------------------------------------------------
# El beneficio neto y la curva de decisión
# ---------------------------------------------------------------------------

def _exigir_pt(valor: Any, campo: str) -> float:
    pt = exigir_real(valor, campo, minimo=0.0, maximo=1.0)
    if pt == 1.0:
        raise EntradaNoMedible("umbral_de_probabilidad_uno", campo=campo)
    return pt


def beneficio_neto(*, tp: int, fp: int, n: int, umbral_de_probabilidad: float) -> float:
    """`TP/n − FP/n · pt/(1−pt)`, la fórmula literal del criterio de terminado.

    Vive UNA vez y la usan las dos filas de la curva: la del modelo y la de
    «tratar a todos», que es esta misma cuenta con los recuentos de tratar a
    todo el mundo. La tercera referencia, «a nadie», no pasa por aquí porque
    no es una medición: es `BENEFICIO_NETO_NO_TRATAR`, cero por definición.
    """
    exigir_entero(tp, "tp", minimo=0)
    exigir_entero(fp, "fp", minimo=0)
    exigir_entero(n, "n", minimo=1)
    pt = _exigir_pt(umbral_de_probabilidad, "umbral_de_probabilidad")
    if tp + fp > n:
        raise EsquemaInvalido("fuera_de_rango", campo="tp+fp", minimo=0, maximo=n,
                              valor=repr(tp + fp))
    return tp / n - (fp / n) * (pt / (1.0 - pt))


@dataclass(frozen=True)
class PuntoDeDecision:
    """Una fila de la curva: el beneficio neto del modelo en `pt` y el de las
    dos referencias, para que la comparación no dependa de quien la pinte."""

    umbral_de_probabilidad: float
    n: int
    tp: int
    fp: int
    beneficio_neto: float
    beneficio_neto_tratar_a_todos: float

    def __post_init__(self) -> None:
        _exigir_pt(self.umbral_de_probabilidad, "umbral_de_probabilidad")
        exigir_entero(self.n, "n", minimo=1)
        exigir_entero(self.tp, "tp", minimo=0)
        exigir_entero(self.fp, "fp", minimo=0)
        exigir_real(self.beneficio_neto, "beneficio_neto")
        exigir_real(self.beneficio_neto_tratar_a_todos, "beneficio_neto_tratar_a_todos")

    @property
    def beneficio_neto_no_tratar(self) -> float:
        """Cero, por definición. Derivado y no guardado: un campo escribible
        dejaría que alguien publicase otra cosa llamándola «a nadie»."""
        return BENEFICIO_NETO_NO_TRATAR

    @property
    def supera_las_referencias(self) -> bool:
        """ESTRICTAMENTE mayor que las dos. Empatar con «tratar a todos» —lo
        que pasa por construcción en cuanto el umbral es tan bajo que el
        modelo trata a todo el mundo— no es aportar nada."""
        return (self.beneficio_neto > self.beneficio_neto_tratar_a_todos
                and self.beneficio_neto > self.beneficio_neto_no_tratar)

    def a_json(self) -> dict[str, Any]:
        return {"umbral_de_probabilidad": self.umbral_de_probabilidad, "n": self.n,
                "tp": self.tp, "fp": self.fp, "beneficio_neto": self.beneficio_neto,
                "beneficio_neto_tratar_a_todos": self.beneficio_neto_tratar_a_todos,
                "beneficio_neto_no_tratar": self.beneficio_neto_no_tratar,
                "supera_las_referencias": self.supera_las_referencias}


@dataclass(frozen=True)
class CurvaDeDecision:
    """La curva entera sobre una muestra (o sobre un segmento de ella).

    `puntos` vacío exactamente cuando hay `undefined_reason`: una curva que no
    se pudo medir no es una curva plana en cero.
    """

    puntos: tuple[PuntoDeDecision, ...]
    n: int
    prevalencia_observada: float | None = None
    segmento_id: str | None = None
    predefinido: bool | None = None
    undefined_reason: dict[str, str] | None = None
    formula_version: str = VERSION_DEL_PERFIL

    def __post_init__(self) -> None:
        object.__setattr__(self, "puntos", tuple(self.puntos))
        exigir_entero(self.n, "n", minimo=0)
        exigir_texto_o_nulo(self.segmento_id, "segmento_id")
        if self.prevalencia_observada is not None:
            exigir_real(self.prevalencia_observada, "prevalencia_observada",
                        minimo=0.0, maximo=1.0)
        if self.predefinido is not None:
            exigir_booleano(self.predefinido, "predefinido")
        if (self.segmento_id is None) != (self.predefinido is None):
            raise EsquemaInvalido("falta_campo", campo="segmento_id/predefinido")
        if bool(self.puntos) == (self.undefined_reason is not None):
            clave = ("metrica_con_valor_y_motivo" if self.puntos
                     else "metrica_sin_valor_ni_motivo")
            raise EsquemaInvalido(clave, campo="curva_de_decision")

    @property
    def umbrales_sin_ventaja(self) -> tuple[float, ...]:
        """Los `pt` en los que el modelo NO supera a las dos referencias. Es
        la mitad que un gráfico esconde y la ficha tiene que escribir."""
        return tuple(p.umbral_de_probabilidad for p in self.puntos
                     if not p.supera_las_referencias)

    @property
    def supera_en_algun_umbral(self) -> bool:
        return any(p.supera_las_referencias for p in self.puntos)

    def a_json(self) -> dict[str, Any]:
        return {"puntos": [p.a_json() for p in self.puntos], "n": self.n,
                "prevalencia_observada": self.prevalencia_observada,
                "segmento_id": self.segmento_id, "predefinido": self.predefinido,
                "umbrales_sin_ventaja": list(self.umbrales_sin_ventaja),
                "undefined_reason": (dict(self.undefined_reason)
                                     if self.undefined_reason else None),
                "formula_version": self.formula_version}

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def _umbrales_declarados(umbrales: Sequence[float], campo: str) -> tuple[float, ...]:
    valores = tuple(umbrales)
    if not valores:
        raise EntradaNoMedible("umbrales_no_declarados")
    if len(set(valores)) != len(valores):
        raise EntradaNoMedible("umbral_de_probabilidad_repetido", campo=campo)
    return valores


def curva_de_decision(muestra: Muestra, *, umbrales_de_probabilidad: Sequence[float],
                      segmento_id: str | None = None,
                      predefinido: bool | None = None) -> CurvaDeDecision:
    """La curva de decisión sobre `muestra`, en los `pt` que se declaren.

    Los umbrales los declara quien decide: aquí no hay un barrido por omisión
    de 0 a 1 ni un 0,5 de cortesía. Un umbral de probabilidad es una
    afirmación clínica —«a partir de aquí compensa tratar»— y fabricarla sería
    exactamente lo que el invariante 5 del 109 prohíbe.
    """
    if muestra.task != "binary_classification":
        raise EntradaNoMedible("metrica_de_otra_tarea", campo="curva_de_decision",
                               opciones=["binary_classification"], valor=muestra.task)
    if muestra.predictions is not None:
        raise EntradaNoMedible("dca_con_etiquetas_declaradas", campo="predictions")
    # Cada `pt` es una PROBABILIDAD, así que hace falta probabilidad. Una muestra
    # que además traiga `scores` vale igual: desde el 2026-09-17
    # `Muestra.puntuacion_del_positivo` corta sobre la probabilidad cuando la
    # hay, la misma escala que declara. Hasta entonces se rechazaba, porque el
    # corte ocurría —dentro de `matriz_de_confusion`— sobre la puntuación
    # cruda, y un `pt` comparado contra un logit da una curva creíble sobre un
    # eje que no es el suyo (medido el 2026-09-14).
    if muestra.probabilities is None:
        raise EntradaNoMedible("dca_sin_probabilidad_calibrada", campo="curva_de_decision")

    valores = _umbrales_declarados(umbrales_de_probabilidad, "umbrales_de_probabilidad")
    for pt in valores:
        _exigir_pt(pt, "umbrales_de_probabilidad")

    comun: dict[str, Any] = {"n": muestra.n, "segmento_id": segmento_id,
                             "predefinido": predefinido}
    if muestra.n == 0:
        return CurvaDeDecision(puntos=(), prevalencia_observada=None, **comun,
                               undefined_reason=motivo("sin_observaciones",
                                                       campo="curva_de_decision"))

    positivos = sum(1 for y in muestra.y_true if y == muestra.positive_label)
    negativos = muestra.n - positivos

    puntos = []
    for pt in valores:
        matriz = matriz_de_confusion(muestra, umbral=pt)
        puntos.append(PuntoDeDecision(
            umbral_de_probabilidad=pt, n=muestra.n, tp=matriz.tp, fp=matriz.fp,
            beneficio_neto=beneficio_neto(tp=matriz.tp, fp=matriz.fp, n=muestra.n,
                                          umbral_de_probabilidad=pt),
            # «Tratar a todos» es la MISMA fórmula con los recuentos de tratar
            # a todo el mundo: todos los positivos aciertan y todos los
            # negativos son falsos positivos.
            beneficio_neto_tratar_a_todos=beneficio_neto(
                tp=positivos, fp=negativos, n=muestra.n, umbral_de_probabilidad=pt)))

    return CurvaDeDecision(puntos=tuple(puntos), prevalencia_observada=positivos / muestra.n,
                           **comun)


# ---------------------------------------------------------------------------
# La tabla de umbrales: sens/esp con IC, y PPV/NPV a prevalencia declarada
# ---------------------------------------------------------------------------

def _transporte_a_prevalencia(prevalencia: float, cual: str
                              ) -> Callable[[Muestra, float | None], float | Indefinida]:
    """La fórmula de Bayes, con la forma que espera el remuestreo de 105-C2.

    Sensibilidad y especificidad NO se reimplementan aquí: se piden al
    registro único de 105-C1, que es donde viven sus fórmulas.
    """

    def formula(muestra: Muestra, umbral: float | None) -> float | Indefinida:
        sens = REGISTRO["sensitivity"].formula(muestra, umbral)
        if isinstance(sens, Indefinida):
            return Indefinida(motivo("transporte_sin_sensibilidad_ni_especificidad",
                                     campo=cual, valor="sensitivity"))
        esp = REGISTRO["specificity"].formula(muestra, umbral)
        if isinstance(esp, Indefinida):
            return Indefinida(motivo("transporte_sin_sensibilidad_ni_especificidad",
                                     campo=cual, valor="specificity"))
        if cual == PPV_A_PREVALENCIA:
            numerador = sens * prevalencia
            denominador = numerador + (1.0 - esp) * (1.0 - prevalencia)
        else:
            numerador = esp * (1.0 - prevalencia)
            denominador = numerador + (1.0 - sens) * prevalencia
        if denominador == 0.0:
            return Indefinida(motivo("transporte_con_denominador_cero", campo=cual))
        return numerador / denominador

    return formula


@dataclass(frozen=True)
class FilaDeUmbral:
    """Un umbral y lo que se puede decir en él.

    `ppv`/`npv` son SIEMPRE los transportados a la prevalencia declarada —y
    cuando no la hay, un `ValorDeMetrica` sin valor y con su motivo—. El PPV
    observado en la muestra no viaja en esta fila a propósito: publicarlo al
    lado invitaría a leerlo como el de la población, que es el defecto entero
    que este corte persigue. Quien lo quiera, lo pide a `calcular("ppv", …)`
    con los ojos abiertos.
    """

    umbral: float
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    sensibilidad: ValorDeMetrica
    especificidad: ValorDeMetrica
    ppv: ValorDeMetrica
    npv: ValorDeMetrica

    def __post_init__(self) -> None:
        exigir_real(self.umbral, "umbral")
        for campo in ("n", "tp", "fp", "tn", "fn"):
            exigir_entero(getattr(self, campo), campo, minimo=0)
        for campo in ("sensibilidad", "especificidad", "ppv", "npv"):
            if not isinstance(getattr(self, campo), ValorDeMetrica):
                raise EsquemaInvalido("no_es_mapa", campo=campo,
                                      valor=repr(getattr(self, campo)))

    def a_json(self) -> dict[str, Any]:
        return {"umbral": self.umbral, "n": self.n, "tp": self.tp, "fp": self.fp,
                "tn": self.tn, "fn": self.fn,
                "sensibilidad": self.sensibilidad.a_json(),
                "especificidad": self.especificidad.a_json(),
                "ppv": self.ppv.a_json(), "npv": self.npv.a_json()}


@dataclass(frozen=True)
class TablaDeUmbrales:
    """La tabla entera, con la prevalencia pegada a los números que dependen
    de ella y la de la muestra al lado, para que se vea la distancia."""

    filas: tuple[FilaDeUmbral, ...]
    prevalencia_declarada: float | None = None
    prevalencia_observada: float | None = None
    declarados_por: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "filas", tuple(self.filas))
        for fila in self.filas:
            if not isinstance(fila, FilaDeUmbral):
                raise EsquemaInvalido("no_es_mapa", campo="filas", valor=repr(fila))
        if self.prevalencia_declarada is not None:
            exigir_real(self.prevalencia_declarada, "prevalencia_declarada",
                        minimo=0.0, maximo=1.0)
        if self.prevalencia_observada is not None:
            exigir_real(self.prevalencia_observada, "prevalencia_observada",
                        minimo=0.0, maximo=1.0)
        exigir_texto_o_nulo(self.declarados_por, "declarados_por")

    @property
    def publica_ppv_y_npv(self) -> bool:
        return self.prevalencia_declarada is not None

    def a_json(self) -> dict[str, Any]:
        return {"filas": [f.a_json() for f in self.filas],
                "prevalencia_declarada": self.prevalencia_declarada,
                "prevalencia_observada": self.prevalencia_observada,
                "declarados_por": self.declarados_por,
                "formula_version": VERSION_DEL_PERFIL}

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def _sin_prevalencia(cual: str, muestra: Muestra) -> ValorDeMetrica:
    return ValorDeMetrica(
        metric_id=cual, formula_version=VERSION_DEL_PERFIL,
        undefined_reason=motivo("ppv_npv_sin_prevalencia_declarada"),
        n_observations=muestra.n, n_units=muestra.n_unidades)


def _transportado(cual: str, muestra: Muestra, *, prevalencia: float, umbral: float,
                  diseno: str, estimando: str, semilla: int, remuestras: int,
                  nivel: float) -> ValorDeMetrica:
    formula = _transporte_a_prevalencia(prevalencia, cual)
    resultado = formula(muestra, umbral) if muestra.n else Indefinida(
        motivo("sin_observaciones", campo=cual))
    comun = {"metric_id": cual, "formula_version": VERSION_DEL_PERFIL,
             "n_observations": muestra.n, "n_units": muestra.n_unidades}
    if isinstance(resultado, Indefinida):
        return ValorDeMetrica(undefined_reason=resultado.motivo, **comun)
    punto = ValorDeMetrica(value=float(resultado), **comun)
    ic = intervalo(cual, muestra, diseno=diseno, estimando=estimando, semilla=semilla,
                   remuestras=remuestras, nivel=nivel, umbral=umbral, valor=punto,
                   formula=formula)
    return ValorDeMetrica(value=punto.value, uncertainty=ic.a_json(), **comun)


def tabla_de_umbrales(muestra: Muestra, *, umbrales: Sequence[float], diseno: str,
                      estimando: str, semilla: int, prevalencia: float | None = None,
                      declarados_por: str | None = None, remuestras: int = 1000,
                      nivel: float = 0.95) -> TablaDeUmbrales:
    """Sensibilidad y especificidad con IC en cada umbral declarado, y PPV/NPV
    **a la prevalencia declarada** — o su ausencia dicha, si no la hay.

    `diseno`, `estimando` y `semilla` son obligatorios por lo mismo que en
    105-C2: el remuestreo no adivina el diseño. `prevalencia` no tiene valor
    por omisión y `None` NO significa «usa la de la muestra».
    """
    if muestra.task != "binary_classification":
        raise EntradaNoMedible("metrica_de_otra_tarea", campo="tabla_de_umbrales",
                               opciones=["binary_classification"], valor=muestra.task)
    if muestra.predictions is not None:
        raise EntradaNoMedible("dca_con_etiquetas_declaradas", campo="predictions")

    valores = _umbrales_declarados(umbrales, "umbrales")
    for umbral in valores:
        # El umbral vive en la escala en que se CORTA, que es la que la muestra
        # declara: con probabilidades —aunque traiga también `scores`—, en
        # [0,1], porque desde el 2026-09-17 `Muestra.puntuacion_del_positivo`
        # corta sobre ellas; solo con puntuaciones crudas, cualquier real.
        # Antes miraba `scores`, porque entonces mandaba `scores`.
        if muestra.probabilities is not None:
            exigir_real(umbral, "umbrales", minimo=0.0, maximo=1.0)
        else:
            exigir_real(umbral, "umbrales")
    if prevalencia is not None:
        exigir_real(prevalencia, "prevalencia", minimo=0.0, maximo=1.0)
    exigir_texto_o_nulo(declarados_por, "declarados_por")

    filas = []
    for umbral in valores:
        matriz = matriz_de_confusion(muestra, umbral=umbral)
        comun = {"diseno": diseno, "estimando": estimando, "semilla": semilla,
                 "remuestras": remuestras, "nivel": nivel, "umbral": umbral}
        if prevalencia is None:
            ppv = _sin_prevalencia(PPV_A_PREVALENCIA, muestra)
            npv = _sin_prevalencia(NPV_A_PREVALENCIA, muestra)
        else:
            ppv = _transportado(PPV_A_PREVALENCIA, muestra, prevalencia=prevalencia, **comun)
            npv = _transportado(NPV_A_PREVALENCIA, muestra, prevalencia=prevalencia, **comun)
        filas.append(FilaDeUmbral(
            umbral=umbral, n=muestra.n, tp=matriz.tp, fp=matriz.fp, tn=matriz.tn,
            fn=matriz.fn,
            sensibilidad=medir("sensitivity", muestra, **comun),
            especificidad=medir("specificity", muestra, **comun),
            ppv=ppv, npv=npv))

    observada = None
    if muestra.n:
        observada = sum(1 for y in muestra.y_true
                        if y == muestra.positive_label) / muestra.n
    return TablaDeUmbrales(filas=tuple(filas), prevalencia_declarada=prevalencia,
                           prevalencia_observada=observada,
                           declarados_por=declarados_por)


# ---------------------------------------------------------------------------
# El perfil
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerfilClinico:
    """El documento del corte: qué se midió, con qué alcance y qué se calla.

    `datos_sinteticos` no tiene valor por omisión a propósito. Atribuirle
    precisión clínica a datos sintéticos es el error más fácil de cometer aquí,
    y un campo con omisión `False` lo cometería por quien no lo mire.
    """

    ESQUEMA: ClassVar[str] = "matrixai.estudio.perfil_clinico"

    perfil_id: str
    evidencia: str
    diseno: str
    datos_sinteticos: bool
    tabla: TablaDeUmbrales
    curva: CurvaDeDecision
    calibracion: CurvaDeFiabilidad | None = None
    recalibracion: dict[str, Any] | None = None
    segmentos: tuple[AnalisisDeSegmento, ...] = ()
    curvas_por_segmento: tuple[CurvaDeDecision, ...] = ()
    segmentos_predefinidos_por: str | None = None
    politica_de_faltantes: dict[str, Any] | None = None
    #: 109-C3. La comparación EMPAREJADA del modelo contra el baseline sobre las
    #: mismas filas de este perfil, y quién era el baseline. O, si no se pudo
    #: plantear, por qué (`MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE`). Los tres
    #: en `None` significan «no consta», que no es lo mismo que «no se pudo».
    comparacion_con_el_baseline: ComparacionEmparejada | None = None
    baseline_comparado: str | None = None
    sin_comparacion_con_el_baseline: str | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.perfil_id, "perfil_id")
        exigir_opcion(self.evidencia, "evidencia", ETIQUETAS_DE_EVIDENCIA)
        exigir_opcion(self.diseno, "diseno", TIPOS_DE_PARTICION)
        exigir_booleano(self.datos_sinteticos, "datos_sinteticos")
        if not isinstance(self.tabla, TablaDeUmbrales):
            raise EsquemaInvalido("no_es_mapa", campo="tabla", valor=repr(self.tabla))
        if not isinstance(self.curva, CurvaDeDecision):
            raise EsquemaInvalido("no_es_mapa", campo="curva", valor=repr(self.curva))
        if self.curva.segmento_id is not None:
            raise EsquemaInvalido("falta_campo", campo="curva.segmento_id")
        self._exigir_tabla_y_curva_de_la_misma_muestra()
        self._exigir_una_comparacion_coherente()
        # Mapas LIBRES a propósito: sus productores les dan formas distintas (el
        # Studio, `RecalibracionLogistica.a_json()` y la política que midió
        # `ajustar_preparacion`; las pruebas del 109-C3, una política declarada
        # a mano) y aquí no se inventa un esquema que nadie más use. Lo que sí
        # se exige es lo que el propio perfil necesita para no mentir: que sea
        # un mapa —un objeto sin copiar revienta tarde, en `a_json()`, lejos de
        # quien lo pasó— y que no venga vacío, porque `{}` salía `null` en el
        # JSON y una sección muda en la ficha: dos documentos diciendo cosas
        # distintas del mismo campo. `None` es la forma de decir «no consta».
        for nombre in ("recalibracion", "politica_de_faltantes"):
            valor = getattr(self, nombre)
            if valor is None:
                continue
            mapa = exigir_mapa(valor, nombre)
            if not mapa:
                raise EsquemaInvalido("falta_campo", campo=nombre)
            object.__setattr__(self, nombre, mapa)
        object.__setattr__(self, "segmentos", tuple(self.segmentos))
        object.__setattr__(self, "curvas_por_segmento", tuple(self.curvas_por_segmento))
        exigir_texto_o_nulo(self.segmentos_predefinidos_por, "segmentos_predefinidos_por")
        if self.calibracion is not None and not isinstance(self.calibracion, CurvaDeFiabilidad):
            raise EsquemaInvalido("no_es_mapa", campo="calibracion",
                                  valor=repr(self.calibracion))

        # Sin equipo que conste como quien los predefinió, NINGÚN segmento
        # puede viajar marcado como predefinido: la decisión 5 del 109 dice
        # «sin declaración no hay subgrupos confirmatorios», y aquí se hace
        # cumplir en la construcción, no en una nota que alguien lea o no.
        for segmento in self.segmentos:
            if not isinstance(segmento, AnalisisDeSegmento):
                raise EsquemaInvalido("no_es_mapa", campo="segmentos", valor=repr(segmento))
            if segmento.predefinido and self.segmentos_predefinidos_por is None:
                raise EsquemaInvalido("segmento_confirmatorio_sin_equipo",
                                      campo=segmento.segmento_id)
        for curva in self.curvas_por_segmento:
            if not isinstance(curva, CurvaDeDecision):
                raise EsquemaInvalido("no_es_mapa", campo="curvas_por_segmento",
                                      valor=repr(curva))
            if curva.segmento_id is None:
                raise EsquemaInvalido("falta_campo", campo="curvas_por_segmento.segmento_id")
            if curva.predefinido and self.segmentos_predefinidos_por is None:
                raise EsquemaInvalido("segmento_confirmatorio_sin_equipo",
                                      campo=curva.segmento_id)

    def _exigir_una_comparacion_coherente(self) -> None:
        """La comparación con el baseline, o por qué no la hay: nunca las dos.

        Un perfil con comparación Y motivo de ausencia diría dos cosas del
        mismo hecho. Una comparación sin baseline nombrado no dice contra qué.
        Y la comparación tiene que remuestrear con el MISMO diseño que el resto
        del perfil: con otro, su intervalo trataría como independientes filas
        que la tabla trata como agrupadas o encadenadas en el tiempo, y el
        documento afirmaría dos incertidumbres distintas de la misma muestra.
        Lo que NO se puede comprobar aquí: que describa las mismas filas que la
        tabla. `ComparacionEmparejada` no guarda ni el `n` ni una huella de la
        muestra; eso lo garantiza el productor (en el Studio, la misma muestra
        de test que alimenta la tabla).
        """
        comparacion = self.comparacion_con_el_baseline
        if comparacion is None:
            if self.baseline_comparado is not None:
                raise EsquemaInvalido("falta_campo", campo="comparacion_con_el_baseline")
            if self.sin_comparacion_con_el_baseline is not None:
                exigir_opcion(self.sin_comparacion_con_el_baseline,
                              "sin_comparacion_con_el_baseline",
                              MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE)
            return
        if not isinstance(comparacion, ComparacionEmparejada):
            raise EsquemaInvalido("no_es_mapa", campo="comparacion_con_el_baseline",
                                  valor=repr(comparacion))
        if self.sin_comparacion_con_el_baseline is not None:
            raise EsquemaInvalido("metrica_con_valor_y_motivo",
                                  campo="comparacion_con_el_baseline")
        exigir_texto(self.baseline_comparado, "baseline_comparado")
        exigir_opcion(comparacion.diseno, "comparacion_con_el_baseline.diseno",
                      (self.diseno,))

    def _exigir_tabla_y_curva_de_la_misma_muestra(self) -> None:
        """La tabla y la curva describen la MISMA muestra, o no hay perfil.

        El documento afirma dos veces la prevalencia «observada en esta
        muestra», una de cada objeto: la tabla la guarda junto a la declarada
        (y la ficha la escribe ahí cuando hay declarada), y la curva la escribe
        encima del beneficio neto. Con tabla y curva de cohortes distintas el
        perfil se sellaba afirmando dos prevalencias contradictorias de «esta
        muestra» (medido el 2026-09-16: 0.4000 y 0.5000). Lo que se compara
        es lo que el documento AFIRMA de la muestra (`n` y prevalencia
        observada), no su identidad: ninguno de los dos objetos guarda una
        huella de las filas, así que dos cohortes con el mismo `n` y los mismos
        positivos no se distinguen aquí.
        """
        for fila in self.tabla.filas:
            if fila.n != self.curva.n:
                raise EsquemaInvalido("filas_desalineadas", campo="curva_de_decision",
                                      valor=self.curva.n, opciones=fila.n)
        de_la_tabla = self.tabla.prevalencia_observada
        de_la_curva = self.curva.prevalencia_observada
        if (de_la_tabla is None) != (de_la_curva is None):
            ausente = ("tabla_de_umbrales" if de_la_tabla is None
                       else "curva_de_decision")
            raise EsquemaInvalido("falta_campo", campo=f"{ausente}.prevalencia_observada")
        if de_la_tabla is not None and not math.isclose(de_la_tabla, de_la_curva,
                                                         rel_tol=0.0, abs_tol=1e-12):
            raise EsquemaInvalido("fuera_de_rango",
                                  campo="curva_de_decision.prevalencia_observada",
                                  minimo=de_la_tabla, maximo=de_la_tabla,
                                  valor=repr(de_la_curva))

    @property
    def alcance(self) -> str:
        """Derivado, nunca declarado a mano. Ver `alcance_de_validacion()`."""
        return alcance_de_validacion(evidencia=self.evidencia, diseno=self.diseno)

    @property
    def segmentos_exploratorios(self) -> tuple[str, ...]:
        return tuple(s.segmento_id for s in self.segmentos if s.es_exploratorio)

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "perfil_id": self.perfil_id, "evidencia": self.evidencia,
            "diseno": self.diseno, "alcance": self.alcance,
            "datos_sinteticos": self.datos_sinteticos,
            "tabla_de_umbrales": self.tabla.a_json(),
            "curva_de_decision": self.curva.a_json(),
            "calibracion": self.calibracion.a_json() if self.calibracion else None,
            "recalibracion": dict(self.recalibracion) if self.recalibracion else None,
            "segmentos": [s.a_json() for s in self.segmentos],
            "curvas_por_segmento": [c.a_json() for c in self.curvas_por_segmento],
            "segmentos_predefinidos_por": self.segmentos_predefinidos_por,
            "politica_de_faltantes": (dict(self.politica_de_faltantes)
                                      if self.politica_de_faltantes else None),
            "comparacion_con_el_baseline": (
                {"baseline": self.baseline_comparado,
                 **self.comparacion_con_el_baseline.a_json()}
                if self.comparacion_con_el_baseline is not None else None),
            "sin_comparacion_con_el_baseline": self.sin_comparacion_con_el_baseline,
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), None)

    def digest(self) -> str:
        return digest_canonico(self.a_json())


# ---------------------------------------------------------------------------
# La ficha: lo que lee una persona, en los dos idiomas
# ---------------------------------------------------------------------------

_T: dict[str, dict[str, str]] = {
    "es": {
        "titulo": "Perfil clínico de la tarea",
        "alcance_internal_only": "**VALIDACIÓN INTERNA ÚNICAMENTE.** No hay cohorte "
                                 "externa ni separación temporal: todo lo que sigue "
                                 "se midió dentro de la misma fuente de datos, así "
                                 "que no dice cómo se comportaría en otro sitio ni "
                                 "en otro periodo.",
        "alcance_internal_only_con_tiempo": "**VALIDACIÓN INTERNA ÚNICAMENTE.** No hay "
                                            "cohorte externa. La partición SÍ separa "
                                            "por tiempo, pero la etiqueta de evidencia "
                                            "no la sostiene como validación temporal: "
                                            "lo medido sobre el periodo reservado no "
                                            "consta como una prueba independiente. Todo "
                                            "lo que sigue se midió dentro de la misma "
                                            "fuente de datos, así que no sostiene cómo "
                                            "se comportaría en otro sitio ni en otro "
                                            "periodo.",
        "alcance_temporal": "**VALIDACIÓN TEMPORAL (misma fuente, periodo distinto).** "
                            "No hay cohorte externa: no dice cómo se comportaría en "
                            "otro centro.",
        "alcance_external": "**VALIDACIÓN EXTERNA** sobre una cohorte distinta de la "
                            "que desarrolló el modelo.",
        "aviso_sintetico": "**AVISO: los datos son SINTÉTICOS.** Nada de lo que hay "
                           "aquí abajo es precisión clínica: es una demostración "
                           "funcional del cálculo, y un modelo entrenado así no está "
                           "validado clínicamente.",
        "aviso_no_regulatorio": "Esto no es un producto sanitario ni una aprobación "
                                "regulatoria, y no sustituye al criterio de quien "
                                "firma el estudio.",
        "prevalencia": "Prevalencia",
        "prevalencia_declarada": "declarada para la población destino",
        "prevalencia_observada": "observada en esta muestra",
        "prevalencia_no_declarada": "**Prevalencia no declarada: no se publican PPV "
                                    "ni NPV.** El PPV de esta muestra describe a esta "
                                    "muestra; presentarlo como el de otra población "
                                    "sería un número exacto y falso.",
        "supuesto_transporte": "PPV y NPV se transportan con Bayes desde la "
                               "sensibilidad y la especificidad medidas aquí. Eso "
                               "supone que las dos valen igual en la población "
                               "destino; si su espectro es otro, no valen, y el "
                               "número transportado tampoco.",
        "umbrales": "Umbrales",
        "umbrales_declarados_por": "Umbrales declarados por",
        "no_declarado": "no declarado",
        "col_umbral": "umbral", "col_sens": "sensibilidad", "col_esp": "especificidad",
        "col_ppv": "PPV a prevalencia", "col_npv": "NPV a prevalencia",
        "col_recuentos": "TP/FP/TN/FN",
        "no_publicado": "no publicado",
        "ic": "IC", "sin_ic": "sin IC",
        "calibracion": "Calibración",
        "calibracion_ece": "Error de calibración esperado (ECE)",
        "calibracion_bins": "método de bins",
        "calibracion_no_medida": "No se ha medido la calibración. No medirla no la "
                                 "vuelve buena.",
        "recalibracion": "Recalibración que consta en el perfil",
        "recalibracion_no_consta": "No consta ninguna recalibración: este perfil no "
                                   "dice si las probabilidades se recalibraron antes "
                                   "de medir.",
        "comparacion": "Comparación con el baseline",
        "comparacion_que": "Emparejada sobre las mismas filas de este perfil: "
                           "`{metrica}` del modelo menos la del baseline "
                           "«{baseline}».",
        "comparacion_diferencia": "Diferencia (modelo − baseline)",
        "comparacion_mejora": "El modelo MEJORA al baseline: el intervalo entero "
                              "queda a su favor.",
        "comparacion_inferioridad": "El modelo es PEOR que el baseline: el "
                                    "intervalo entero queda en su contra.",
        "comparacion_equivalencia_practica": "Equivalencia práctica: el intervalo "
                                             "entero cabe dentro del margen "
                                             "declarado.",
        "comparacion_inconcluso": "Inconcluso: esta muestra no demuestra que el "
                                  "modelo mejore al baseline.",
        "comparacion_no_consta": "No consta ninguna comparación con el baseline en "
                                 "este perfil: no dice si el modelo aporta algo "
                                 "frente a la referencia más simple.",
        "sin_comparacion_el_ganador_es_el_baseline": "No hay comparación con el "
                                                     "baseline: el candidato ganador "
                                                     "ES el baseline, así que no hay "
                                                     "contra quién compararlo.",
        "sin_comparacion_el_baseline_no_puntuo": "No hay comparación con el "
                                                 "baseline: el baseline no llegó a "
                                                 "puntuar, así que no hay muestra "
                                                 "suya que emparejar.",
        "dca": "Beneficio neto (curva de decisión)",
        "dca_formula": "beneficio neto = TP/n − FP/n · pt/(1−pt); «a nadie» vale 0 "
                       "por definición.",
        "dca_prevalencia": "El beneficio neto se mide SOBRE ESTA MUESTRA, a su "
                           "prevalencia observada",
        "dca_no_transportado": "y **no está transportado a la prevalencia "
                               "declarada**: a diferencia del PPV, un beneficio neto "
                               "a otra prevalencia se mide sobre una cohorte de esa "
                               "población; no sale de convertir este con una fórmula.",
        "col_pt": "pt", "col_modelo": "modelo", "col_todos": "tratar a todos",
        "col_nadie": "a nadie", "col_supera": "¿supera a las dos?",
        "si": "sí", "no": "no",
        "dca_sin_ventaja": "**El modelo NO supera a las dos referencias en estos "
                           "umbrales**",
        "dca_nunca_supera": "**El modelo no supera a las dos referencias en ningún "
                            "umbral declarado**: en todos ellos se decide igual o "
                            "mejor sin él.",
        "dca_no_medida": "No se ha podido medir la curva de decisión",
        "dca_prevalencia_segmento": "Beneficio neto medido SOBRE ESTE SUBGRUPO, a su "
                                    "prevalencia observada",
        "segmentos": "Subgrupos",
        "segmentos_predefinidos_por": "Subgrupos predefinidos por",
        "segmentos_sin_equipo": "**Nadie ha predefinido subgrupos.** Sin equipo que "
                                "los declare de antemano no hay subgrupos "
                                "confirmatorios: lo que se mida aquí es exploratorio "
                                "y se lee como una pregunta, no como un hallazgo.",
        "segmentos_ninguno": "No se ha medido ningún subgrupo.",
        "exploratorio": "exploratorio",
        "predefinido": "predefinido",
        "cautela": "cautela",
        "faltantes": "Datos ausentes",
        "faltantes_no_declarada": "**No se declara ninguna política de datos "
                                  "ausentes.** Lo que se hizo con ellos no consta.",
    },
    "en": {
        "titulo": "Clinical task profile",
        "alcance_internal_only": "**INTERNAL VALIDATION ONLY.** There is no external "
                                 "cohort and no temporal split: everything below was "
                                 "measured inside one single data source, so it does "
                                 "not tell how it would behave elsewhere or at "
                                 "another time.",
        "alcance_internal_only_con_tiempo": "**INTERNAL VALIDATION ONLY.** There is no "
                                            "external cohort. The split DOES separate "
                                            "by time, but its evidence label does not "
                                            "support it as temporal validation: what "
                                            "was measured on the held-out period is "
                                            "not on record as an independent test. "
                                            "Everything below was measured inside one "
                                            "single data source, so it supports no "
                                            "claim about how it would behave elsewhere "
                                            "or at another time.",
        "alcance_temporal": "**TEMPORAL VALIDATION (same source, different period).** "
                            "There is no external cohort: it does not tell how it "
                            "would behave at another centre.",
        "alcance_external": "**EXTERNAL VALIDATION** on a cohort other than the one "
                            "this model was developed on.",
        "aviso_sintetico": "**WARNING: this data is SYNTHETIC.** Nothing below is "
                           "clinical accuracy: it is a functional demonstration of "
                           "this computation, and a model trained this way is not "
                           "clinically validated.",
        "aviso_no_regulatorio": "This is not a medical device nor a regulatory "
                                "approval, and it does not replace the judgement of "
                                "whoever signs this study.",
        "prevalencia": "Prevalence",
        "prevalencia_declarada": "declared, target population",
        "prevalencia_observada": "observed in this sample",
        "prevalencia_no_declarada": "**Prevalence not declared: PPV and NPV are not "
                                    "published.** This sample's PPV describes this "
                                    "sample; presenting it as another population's "
                                    "would be an exact, false number.",
        "supuesto_transporte": "PPV and NPV are transported via Bayes from "
                               "sensitivity and specificity as measured here. That "
                               "assumes both hold in target population; if its "
                               "spectrum differs, they do not, and neither does any "
                               "transported number.",
        "umbrales": "Thresholds",
        "umbrales_declarados_por": "Thresholds declared by",
        "no_declarado": "not declared",
        "col_umbral": "threshold", "col_sens": "sensitivity",
        "col_esp": "specificity",
        "col_ppv": "PPV at prevalence", "col_npv": "NPV at prevalence",
        "col_recuentos": "TP/FP/TN/FN",
        "no_publicado": "not published",
        "ic": "CI", "sin_ic": "no CI",
        "calibracion": "Calibration",
        "calibracion_ece": "Expected calibration error (ECE)",
        "calibracion_bins": "binning method",
        "calibracion_no_medida": "Calibration was not measured. Not measuring it does "
                                 "not make it good.",
        "recalibracion": "Recalibration on record in this profile",
        "recalibracion_no_consta": "No recalibration is on record: this profile does "
                                   "not say whether probabilities were recalibrated "
                                   "before measuring.",
        "comparacion": "Comparison against the baseline",
        "comparacion_que": "Paired on the same rows as this profile: the model's "
                           "`{metrica}` minus that of the baseline «{baseline}».",
        "comparacion_diferencia": "Difference (model − baseline)",
        "comparacion_mejora": "The model IMPROVES on the baseline: the whole "
                              "interval is in its favour.",
        "comparacion_inferioridad": "The model is WORSE than the baseline: the "
                                    "whole interval is against it.",
        "comparacion_equivalencia_practica": "Practical equivalence: the whole "
                                             "interval fits within the declared "
                                             "margin.",
        "comparacion_inconcluso": "Inconclusive: this sample does not show that the "
                                  "model improves on the baseline.",
        "comparacion_no_consta": "No comparison against the baseline is on record "
                                 "in this profile: it does not say whether the model "
                                 "adds anything over the simplest reference.",
        "sin_comparacion_el_ganador_es_el_baseline": "No comparison against the "
                                                     "baseline: the winning "
                                                     "candidate IS the baseline, so "
                                                     "there is nothing to compare "
                                                     "it against.",
        "sin_comparacion_el_baseline_no_puntuo": "No comparison against the "
                                                 "baseline: the baseline did not "
                                                 "score, so there is no sample of "
                                                 "its own to pair.",
        "dca": "Net benefit (decision curve)",
        "dca_formula": "net benefit = TP/n − FP/n · pt/(1−pt); treat-none is 0 by "
                       "definition.",
        "dca_prevalencia": "Net benefit is measured ON THIS SAMPLE, at its observed "
                           "prevalence",
        "dca_no_transportado": "and **it is not transported to declared "
                               "prevalence**: unlike PPV, net benefit at another "
                               "prevalence is measured on a cohort of that "
                               "population; it does not come out of converting this "
                               "one through a formula.",
        "col_pt": "pt", "col_modelo": "model", "col_todos": "treat all",
        "col_nadie": "treat none", "col_supera": "beats both?",
        "si": "yes", "no": "no",
        "dca_sin_ventaja": "**This model does NOT beat both references at these "
                           "thresholds**",
        "dca_nunca_supera": "**This model beats both references at no declared "
                            "threshold**: at every one of them, deciding without it "
                            "is as good or better.",
        "dca_no_medida": "Decision curve could not be measured",
        "dca_prevalencia_segmento": "Net benefit measured ON THIS SUBGROUP, at its "
                                    "observed prevalence",
        "segmentos": "Subgroups",
        "segmentos_predefinidos_por": "Subgroups predefined by",
        "segmentos_sin_equipo": "**Nobody predefined any subgroup.** Without a team "
                                "declaring them beforehand there are no confirmatory "
                                "subgroups: whatever is measured here is exploratory "
                                "and reads as a question, not as a finding.",
        "segmentos_ninguno": "No subgroup was measured.",
        "exploratorio": "exploratory",
        "predefinido": "predefined",
        "cautela": "caution",
        "faltantes": "Missing data",
        "faltantes_no_declarada": "**No missing-data policy is declared.** What was "
                                  "done with them is not on record.",
    },
}


def _t(locale: str) -> dict[str, str]:
    """El idioma pedido; el que no se sepa, inglés — igual que `ErrorDeEstudio`."""
    return _T["es"] if str(locale or "en").strip().lower() == "es" else _T["en"]


def _cifra(valor: float | None, *, decimales: int = 4) -> str:
    return "—" if valor is None else f"{valor:.{decimales}f}"


def _valor_con_ic(valor: ValorDeMetrica, textos: dict[str, str], idioma: str) -> str:
    """El número y su IC, o la marca de que no lo hay.

    La casilla NO lleva el motivo entero —una frase de tres líneas dentro de
    una celda hace la tabla ilegible— pero tampoco queda muda: dice «no
    publicado» y `_notas_de_lo_no_publicado()` escribe debajo de la tabla, una
    vez, el motivo de cada ausencia con las métricas que lo comparten. Media
    verdad tranquilizadora sería poner un guion y callar.
    """
    if valor.value is None:
        return textos["no_publicado"]
    ic = valor.uncertainty or {}
    if ic.get("ci_low") is None:
        return f"{_cifra(valor.value)} ({textos['sin_ic']})"
    nivel = int(round(float(ic.get("level", 0.95)) * 100))
    return (f"{_cifra(valor.value)} [{_cifra(ic['ci_low'])}–{_cifra(ic['ci_high'])}]"
            f" {textos['ic']} {nivel} %")


def _notas_de_lo_no_publicado(tabla: TablaDeUmbrales, textos: dict[str, str],
                              idioma: str) -> list[str]:
    """Una línea por MOTIVO distinto, con las métricas que lo comparten.

    Agrupado por motivo y no por fila: la misma ausencia repetida en cada
    umbral se leería como cinco problemas distintos.
    """
    notas: dict[str, list[str]] = {}
    for fila in tabla.filas:
        for valor in (fila.sensibilidad, fila.especificidad, fila.ppv, fila.npv):
            if valor.value is not None:
                continue
            texto = (valor.undefined_reason or {}).get(idioma) or textos["no_publicado"]
            ids = notas.setdefault(texto, [])
            if valor.metric_id not in ids:
                ids.append(valor.metric_id)
    return [f"- **{', '.join(ids)}**: {texto}" for texto, ids in notas.items()]


def ficha_del_perfil(perfil: PerfilClinico, *, locale: str = "en") -> str:
    """La ficha en Markdown, con el alcance de la validación ARRIBA.

    El orden no es estético: quien lee esto decide con ello, y «validación
    interna únicamente» cambia qué se puede hacer con todo lo demás. Por eso
    va antes que el primer número, y por eso `PerfilClinico` no se puede
    construir sin los dos campos de los que sale.
    """
    if not isinstance(perfil, PerfilClinico):
        raise EsquemaInvalido("no_es_mapa", campo="perfil", valor=repr(perfil))
    textos = _t(locale)
    idioma = "es" if textos is _T["es"] else "en"
    lineas: list[str] = [f"# {textos['titulo']}", ""]

    # --- arriba del todo: alcance, datos sintéticos y prevalencia ----------
    # La frase depende del DISEÑO además del alcance: con evidencia débil y una
    # partición temporal el alcance es interno —bien—, pero decir entonces «no
    # hay separación temporal» negaría una que existe.
    clave_de_alcance = "alcance_" + perfil.alcance
    if perfil.alcance == "internal_only" \
            and perfil.diseno in _DISENOS_CON_SEPARACION_TEMPORAL:
        clave_de_alcance = "alcance_internal_only_con_tiempo"
    lineas.append(f"> {textos[clave_de_alcance]}")
    if perfil.datos_sinteticos:
        lineas.append(">")
        lineas.append(f"> {textos['aviso_sintetico']}")
    lineas.append(">")
    if perfil.tabla.publica_ppv_y_npv:
        lineas.append(f"> {textos['prevalencia']}: "
                      f"{_cifra(perfil.tabla.prevalencia_declarada)} "
                      f"({textos['prevalencia_declarada']}) · "
                      f"{_cifra(perfil.tabla.prevalencia_observada)} "
                      f"({textos['prevalencia_observada']})")
        lineas.append(">")
        lineas.append(f"> {textos['supuesto_transporte']}")
    else:
        lineas.append(f"> {textos['prevalencia_no_declarada']}")
    lineas.append(">")
    lineas.append(f"> {textos['aviso_no_regulatorio']}")
    lineas.append("")

    # --- umbrales ----------------------------------------------------------
    lineas.append(f"## {textos['umbrales']}")
    lineas.append(f"- **{textos['umbrales_declarados_por']}**: "
                  f"{perfil.tabla.declarados_por or textos['no_declarado']}")
    lineas.append("")
    lineas.append(f"| {textos['col_umbral']} | {textos['col_sens']} | "
                  f"{textos['col_esp']} | {textos['col_ppv']} | {textos['col_npv']} | "
                  f"{textos['col_recuentos']} |")
    lineas.append("|---|---|---|---|---|---|")
    for fila in perfil.tabla.filas:
        lineas.append(
            f"| {_cifra(fila.umbral, decimales=3)} "
            f"| {_valor_con_ic(fila.sensibilidad, textos, idioma)} "
            f"| {_valor_con_ic(fila.especificidad, textos, idioma)} "
            f"| {_valor_con_ic(fila.ppv, textos, idioma)} "
            f"| {_valor_con_ic(fila.npv, textos, idioma)} "
            f"| {fila.tp}/{fila.fp}/{fila.tn}/{fila.fn} |")
    notas = _notas_de_lo_no_publicado(perfil.tabla, textos, idioma)
    if notas:
        lineas.append("")
        lineas.extend(notas)
    lineas.append("")

    # --- calibración -------------------------------------------------------
    lineas.append(f"## {textos['calibracion']}")
    if perfil.calibracion is None:
        lineas.append(textos["calibracion_no_medida"])
    elif perfil.calibracion.ece is None:
        lineas.append((perfil.calibracion.undefined_reason or {}).get(idioma, ""))
    else:
        lineas.append(f"- **{textos['calibracion_ece']}**: "
                      f"{_cifra(perfil.calibracion.ece)} "
                      f"({textos['calibracion_bins']}: "
                      f"{perfil.calibracion.metodo_de_bins})")
    # La recalibración cambia qué probabilidades se midieron en TODO el
    # documento; sellarla en el JSON y no escribirla aquí la escondía a quien
    # lee la ficha. Sin ella, se dice que no consta — no que no se hizo.
    lineas.append("")
    if perfil.recalibracion is None:
        lineas.append(textos["recalibracion_no_consta"])
    else:
        lineas.append(f"**{textos['recalibracion']}**:")
        lineas.extend(_lineas_de_un_mapa_libre(perfil.recalibracion, idioma))
    lineas.append("")

    # --- comparación con el baseline (109-C3) ------------------------------
    lineas.append(f"## {textos['comparacion']}")
    lineas.extend(_lineas_de_la_comparacion(perfil, textos, idioma))
    lineas.append("")

    # --- curva de decisión -------------------------------------------------
    lineas.append(f"## {textos['dca']}")
    lineas.append(textos["dca_formula"])
    # El beneficio neto TAMBIÉN depende de la prevalencia —`TP/n` y `FP/n` son
    # proporciones de ESTA muestra—, y a diferencia del PPV no se transporta
    # con una fórmula. Callarlo dejaría media verdad tranquilizadora justo en
    # la sección que un equipo mira para decidir.
    aviso = f"{textos['dca_prevalencia']}: {_cifra(perfil.curva.prevalencia_observada)}"
    if perfil.tabla.publica_ppv_y_npv:
        aviso += f", {textos['dca_no_transportado']}"
    else:
        aviso += "."
    lineas.append(aviso)
    lineas.append("")
    lineas.extend(_tabla_de_la_curva(perfil.curva, textos, idioma))

    for curva in perfil.curvas_por_segmento:
        marca = textos["predefinido"] if curva.predefinido else textos["exploratorio"]
        lineas.append("")
        lineas.append(f"### {curva.segmento_id} ({marca})")
        # La prevalencia DEL SUBGRUPO, no la de la muestra entera: el beneficio
        # neto de un subgrupo se lee a la suya, y callarla dejaba leerlo a la de
        # arriba, que es otra.
        lineas.append(f"{textos['dca_prevalencia_segmento']}: "
                      f"{_cifra(curva.prevalencia_observada)} (n={curva.n}).")
        lineas.append("")
        lineas.extend(_tabla_de_la_curva(curva, textos, idioma))
    lineas.append("")

    # --- subgrupos ---------------------------------------------------------
    lineas.append(f"## {textos['segmentos']}")
    if perfil.segmentos_predefinidos_por is None:
        lineas.append(textos["segmentos_sin_equipo"])
    else:
        lineas.append(f"- **{textos['segmentos_predefinidos_por']}**: "
                      f"{perfil.segmentos_predefinidos_por}")
    lineas.append("")
    if not perfil.segmentos:
        lineas.append(textos["segmentos_ninguno"])
    for segmento in perfil.segmentos:
        marca = textos["predefinido"] if segmento.predefinido else textos["exploratorio"]
        medida = segmento.metrica_segmento
        # Un guion mudo aquí dejaría sin explicar por qué el segmento no tiene
        # cifra (cero filas, una sola clase, un denominador vacío): el motivo
        # ya viene con el valor, y se escribe.
        escrito = (_cifra(medida.value) if medida.value is not None
                   else (medida.undefined_reason or {}).get(idioma, ""))
        detalle = f"{segmento.metric_id} {escrito} (n={segmento.n})"
        if segmento.motivo_de_cautela is not None:
            detalle += (f" — {textos['cautela']}: "
                        f"{segmento.motivo_de_cautela.get(idioma, '')}")
        lineas.append(f"- **{segmento.segmento_id}** ({marca}): {detalle}")
    lineas.append("")

    # --- faltantes ---------------------------------------------------------
    lineas.append(f"## {textos['faltantes']}")
    if perfil.politica_de_faltantes is None:
        lineas.append(textos["faltantes_no_declarada"])
    else:
        lineas.extend(_lineas_de_un_mapa_libre(perfil.politica_de_faltantes, idioma))
    lineas.append("")
    return "\n".join(lineas)


def _lineas_de_la_comparacion(perfil: "PerfilClinico", textos: dict[str, str],
                              idioma: str) -> list[str]:
    """La comparación con el baseline, su ausencia con motivo, o que no consta.

    Tres casos distintos y tres frases distintas: «no se pudo plantear» y «no
    consta» no son lo mismo, y confundirlos es decir de un perfil viejo que el
    estudio lo intentó. El veredicto se escribe con la frase de su categoría, y
    un intervalo que no se pudo calcular dice por qué en vez de quedar mudo.
    """
    comparacion = perfil.comparacion_con_el_baseline
    if comparacion is None:
        motivo_de_ausencia = perfil.sin_comparacion_con_el_baseline
        if motivo_de_ausencia is None:
            return [textos["comparacion_no_consta"]]
        return [textos[f"sin_comparacion_{motivo_de_ausencia}"]]
    lineas = [textos["comparacion_que"].format(metrica=comparacion.metric_id,
                                               baseline=perfil.baseline_comparado)]
    if comparacion.veredicto == "incomparable":
        lineas.append(str((comparacion.undefined_reason or {}).get(idioma, "")))
        return lineas
    lineas.append(f"- **{textos['comparacion_diferencia']}**: "
                  f"{_cifra(comparacion.diferencia_puntual)}")
    ic = comparacion.intervalo
    if ic is not None and ic.disponible:
        lineas.append(f"- **{textos['ic']} {ic.level * 100:.0f} %**: "
                      f"[{_cifra(ic.ci_low)}, {_cifra(ic.ci_high)}]")
    elif ic is not None:
        lineas.append(f"- **{textos['sin_ic']}**: "
                      f"{(ic.undefined_reason or {}).get(idioma, '')}")
    lineas.append(textos[f"comparacion_{comparacion.veredicto}"])
    return lineas


def _lineas_de_un_mapa_libre(mapa: Mapping[str, Any], idioma: str) -> list[str]:
    """Un mapa libre (`recalibracion`, `politica_de_faltantes`), clave a clave.

    Sin esquema propio, porque no lo tienen (ver `PerfilClinico.__post_init__`).
    Lo que no es texto se escribe en JSON y no con el `repr` de Python: el
    productor del Studio mete aquí listas de objetos, y `False`, `None` y las
    comillas simples son de Python, no de un documento —y el expediente del
    109-C3 ya escribe este mismo campo en JSON—. Un motivo bilingüe (`es`/`en`,
    la forma de este paquete) se escribe en el idioma de la ficha.
    """
    lineas = []
    for clave in sorted(mapa):
        valor = mapa[clave]
        if isinstance(valor, str):
            texto = valor
        elif isinstance(valor, Mapping) and set(valor) == set(IDIOMAS):
            texto = str(valor[idioma])
        else:
            texto = json.dumps(valor, ensure_ascii=False, sort_keys=True)
        lineas.append(f"- **{clave}**: {texto}")
    return lineas


def _tabla_de_la_curva(curva: CurvaDeDecision, textos: dict[str, str],
                       idioma: str) -> list[str]:
    if curva.undefined_reason is not None:
        return [f"{textos['dca_no_medida']}: "
                f"{curva.undefined_reason.get(idioma, '')}"]
    lineas = [f"| {textos['col_pt']} | {textos['col_modelo']} | {textos['col_todos']} "
              f"| {textos['col_nadie']} | {textos['col_supera']} |",
              "|---|---|---|---|---|"]
    for punto in curva.puntos:
        lineas.append(
            f"| {_cifra(punto.umbral_de_probabilidad, decimales=3)} "
            f"| {_cifra(punto.beneficio_neto)} "
            f"| {_cifra(punto.beneficio_neto_tratar_a_todos)} "
            f"| {_cifra(punto.beneficio_neto_no_tratar)} "
            f"| {textos['si'] if punto.supera_las_referencias else textos['no']} |")
    sin_ventaja = curva.umbrales_sin_ventaja
    if sin_ventaja and not curva.supera_en_algun_umbral:
        lineas.append("")
        lineas.append(textos["dca_nunca_supera"])
    elif sin_ventaja:
        lineas.append("")
        lineas.append(f"{textos['dca_sin_ventaja']}: "
                      + ", ".join(_cifra(u, decimales=3) for u in sin_ventaja))
    return lineas

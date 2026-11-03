# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C1 — el banco de pruebas de comportamiento («qué hace este modelo si…»). NÚCLEO: funciones
puras, sin disco, sin motores y sin Studio.

QUÉ HACE Y QUÉ NO. Recibe una FUNCIÓN DE PUNTUACIÓN que aporta el llamante (el patrón de
`segmentos.importancia_por_permutacion`: este módulo no sabe nada de ningún motor) y devuelve
documentos JSON serializables, versionados, con `a_json()` y `digest()`. Tres pruebas con el modelo
entregado, todas a demanda:

  A — «si falta una columna»: aquí solo se AGREGA (`caida_entre_pliegues`, `ordenar_por_caida`,
      `k_mayores`, `bloques_por_prefijo`). Quien reentrena por pliegue y puntúa la validación es el
      Studio (138-C3), con la métrica del estudio; este módulo no mide una métrica.
  B — sensibilidad a cambios pequeños (`sensibilidad_de_una_columna`): fracción de filas cuya
      decisión cambia y distribución de |Δsalida|/δ, con las filas de mayor pendiente.
  C — monotonía declarada (`monotonia_en_rejilla`): curvas ICE sobre una rejilla, retroceso, amplitud
      de la curva, fracción de filas que retroceden y veredicto de TRES palabras.

I2 — DESCRIBIR. Los resultados de A y B no llevan NINGUNA clave de veredicto; solo C la lleva, y solo
para la monotonía declarada. Una prueba recorre los dos documentos buscándola.

I7 — LA TOLERANCIA ES DEL CÁLCULO, no de la declaración. El resultado de C guarda por fila el
retroceso, el efecto y la amplitud de la curva, y los K peores contraejemplos, que NO dependen de ε ni
de q. `con_tolerancia(ε, q)` (y `con_epsilon(ε)`) rehace el veredicto con otra tolerancia sin volver a
predecir, y el documento dice qué ε y qué q usó.

LOS VALORES POR OMISIÓN SON MEDIDOS (149-C0-ter, `documentacion/149_C0_MEDICION.md` §4; regla en
`benchmarks/c149/REGLA_DE_ELECCION_TER.md`, guion `medir_c0ter.py`): `TOLERANCIA_POR_OMISION`, por
tarea. Regresión ε 5 % de la escala y q 5 %; binaria ε 20 % de la probabilidad y q 20 %. Multiclase no
tiene (decisión del 02-11: sin clase declarada no se comprueba) y exige darlos. Quien llama puede pasar
otros; no hay un 0,01 escrito en ningún otro sitio. El veredicto NO es el de una sola fila: el de este
módulo es el V'(ε, q) medido, y `tests/test_c149_c1bis_paridad_con_la_medicion.py` lo compara con el guion.

EL VEREDICTO (decidido por Roberto, 03-11, `149-veredicto`, opción A), en este orden:
  1. AMPLITUD. La de la columna es la MEDIANA, sobre las filas comprobadas, de (máximo − mínimo) de la
     curva de cada fila: sin signo y sin depender de los extremos (una U que acaba donde empezó se mueve
     mucho). Si no pasa de ε (`>` estricto): «no se pudo comprobar». Se mira PRIMERO.
  2. Si la fracción de filas con retroceso > ε es ≥ q: «con contraejemplos».
  3. Si no: «sin contraejemplos».

I8 — UN VALOR AUSENTE NO ES UN CERO. Sin predicción (columna sin rango, escala no válida, salida que no
es un número) el veredicto es «no se pudo comprobar» con su motivo bilingüe, jamás un retroceso 0 ni
«sin contraejemplos». Una fila sin predicción se EXCLUYE y se cuenta; no cuenta como una curva plana.

I5 — «SIN CONTRAEJEMPLOS» NUNCA SALE SOLO. Va con la amplitud de la curva típica. Si esa amplitud no
supera la tolerancia (una columna plana, o que se mueve menos de lo que tolera la prueba), la prueba no
distingue una dirección de la contraria —con amplitud ≈ 0 la dirección «contraria» tiene tantos
contraejemplos como la declarada— y el veredicto es «no se pudo comprobar» con ese motivo. (Hasta el
C1-bis la puerta era el efecto de extremo a extremo y CON SIGNO: dejaba «sin comprobar» a una columna
monótona declarada al revés y a las curvas en U; el efecto sigue en el documento, como dato.)

I1 — SOLO DESARROLLO. El resultado declara `medido_sobre`; la prueba reservada NO es un valor admitido
y se rechaza al construir, no al pintar.

I4 — LOS LÍMITES VAN PEGADOS AL VEREDICTO: el documento de cada columna lleva la frase fija dentro de
`veredicto`, no en un pie.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from matrixai.estudio.comportamiento_textos import (
    CLAVES_DE_VEREDICTO,
    CLAVE_DE_LOS_LIMITES,
    PorIdioma,
    cifra,
    motivo,
    porcentaje,
    porcentajes_de_la_lectura,
)
from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.validacion import digest_canonico, exigir_booleano, exigir_entero, exigir_real, exigir_texto
from matrixai.estudio.vocabulario import TAREAS, exigir_opcion

__all__ = [
    "DIRECCIONES_DE_MONOTONIA", "EscalaDeLaSalida", "K_PEORES_POR_OMISION", "MEDIDO_SOBRE",
    "MonotoniaDeUnaColumna", "Retroceso", "SensibilidadDeUnaColumna", "TOLERANCIA_POR_OMISION",
    "VERSION_DEL_COMPORTAMIENTO", "VEREDICTOS", "amplitud_de_una_curva", "bloques_por_prefijo",
    "caida_entre_pliegues", "elegir_filas", "escala_de_la_salida", "k_mayores", "monotonia_desde_curvas",
    "monotonia_en_rejilla", "ordenar_por_caida", "rejilla", "retroceso_de_una_curva",
    "sensibilidad_de_una_columna", "tolerancia_por_omision",
]

#: La versión del documento. Cambia con la FORMA (claves) o con la regla de una cifra.
#: 2.0.0 (149-C1-bis): el veredicto es V'(ε, q) por amplitud y fracción de filas, y el documento lleva q, la
#: amplitud y la fracción. La regla de «con contraejemplos» cambió (ya no basta una fila).
VERSION_DEL_COMPORTAMIENTO = "2.0.0"

DIRECCIONES_DE_MONOTONIA = ("sube", "baja")

#: Las tres palabras. Una prueba exige que sean EXACTAMENTE las claves del catálogo de textos.
VEREDICTOS = ("sin_contraejemplos", "con_contraejemplos", "no_se_pudo_comprobar")

#: Sobre qué filas se midió (I1). La prueba reservada NO está y no se puede añadir de pasada: pedirla
#: es un `EsquemaInvalido`. `fila_tipica` es el ICE sintético de «Probar una fila»: no hay datos.
MEDIDO_SOBRE = ("validacion_de_cada_pliegue", "muestra_de_desarrollo", "fila_tipica")

#: Cuántos contraejemplos (los de mayor retroceso, independientes de ε) se guardan por columna.
K_PEORES_POR_OMISION = 5

#: La tolerancia de partida POR TAREA: `(ε relativo a la escala, q = fracción mínima de filas)`. MEDIDA en
#: 149-C0-ter (regla `benchmarks/c149/REGLA_DE_ELECCION_TER.md`, medición `58f5e6d`, semillas 6-8): la regla
#: elige el menor ε con falsa alarma ≤ 10 % y potencia ≥ 80 %, y con él el menor q.
#:   · regresión: ε 5 %, q 5 % (margen amplio: cumple igual de ε 5 % a 15 %).
#:   · binaria: ε 20 %, q 20 % —ALTA a propósito—: es el borde de la rejilla medida, sin margen (la amplitud de
#:     una columna plana ronda el 10–15 % de la probabilidad con ~2.000 filas, por los escalones de los
#:     árboles), y por eso el documento lo dice con un aviso.
#: Multiclase no está: sin clase declarada no se comprueba (decisión del 02-11) y quien llame tiene que darlos.
TOLERANCIA_POR_OMISION = MappingProxyType({
    "regression": (0.05, 0.05),
    "binary_classification": (0.20, 0.20),
})


def tolerancia_por_omision(tarea: str) -> tuple[float, float] | None:
    """`(ε, q)` de partida de la tarea, o `None` si no tiene (multiclase)."""
    return TOLERANCIA_POR_OMISION.get(exigir_opcion(tarea, "tarea", TAREAS))


# ---------------------------------------------------------------------------
# La rejilla, el retroceso y la escala
# ---------------------------------------------------------------------------

def rejilla(minimo: float, maximo: float, puntos: int) -> tuple[float, ...]:
    """`puntos` valores equidistantes de `minimo` a `maximo`, AMBOS incluidos y en ORDEN CRECIENTE.
    El orden importa: el retroceso se mide recorriendo la rejilla de menor a mayor, y una rejilla al
    revés convertiría «sube» en «baja» sin que ningún número lo delatara. Los extremos se fijan a mano
    (`minimo + (maximo-minimo)*k/(n-1)` en el último punto puede no devolver `maximo` por redondeo)."""
    exigir_entero(puntos, "puntos", minimo=2)
    minimo = exigir_real(minimo, "minimo")
    maximo = exigir_real(maximo, "maximo")
    if not maximo > minimo:
        raise EsquemaInvalido("fuera_de_rango", campo="maximo", minimo=minimo, maximo="+inf", valor=repr(maximo))
    pasos = puntos - 1
    return tuple([minimo] + [minimo + (maximo - minimo) * k / pasos for k in range(1, pasos)] + [maximo])


@dataclass(frozen=True)
class Retroceso:
    """El mayor descenso de una curva respecto de su máximo corrido, EN LA DIRECCIÓN DECLARADA, y los
    dos puntos de la rejilla que lo delimitan: `a` (donde estaba el máximo corrido) y `b` (donde cae)."""
    valor: float
    a: int
    b: int


def _con_signo(direccion: str) -> float:
    return 1.0 if exigir_opcion(direccion, "direccion", DIRECCIONES_DE_MONOTONIA) == "sube" else -1.0


def retroceso_de_una_curva(curva: Sequence[float], direccion: str) -> Retroceso:
    """«Sube»: el mayor descenso respecto del máximo corrido. «Baja»: el mayor ascenso respecto del
    mínimo corrido (el mismo cálculo con la curva cambiada de signo). 0,0 si la curva nunca retrocede.
    Los empates se resuelven por el PRIMER punto, para que dos ejecuciones den el mismo contraejemplo."""
    signo = _con_signo(direccion)
    if len(curva) < 1:
        raise EsquemaInvalido("falta_campo", campo="curva")
    mejor, donde_el_mejor = signo * curva[0], 0
    peor, a, b = 0.0, 0, 0
    for j, valor in enumerate(curva):
        s = signo * valor
        if s > mejor:
            mejor, donde_el_mejor = s, j
        caida = mejor - s
        if caida > peor:
            peor, a, b = caida, donde_el_mejor, j
    return Retroceso(peor, a, b)


def amplitud_de_una_curva(curva: Sequence[float]) -> float:
    """(máximo − mínimo) de la curva de UNA fila: sin signo y sin depender de los extremos (una U que acaba
    donde empezó se mueve mucho; el efecto de extremo a extremo daría 0). Es la pieza de la puerta del veredicto."""
    if len(curva) < 1:
        raise EsquemaInvalido("falta_campo", campo="curva")
    return max(curva) - min(curva)


@dataclass(frozen=True)
class EscalaDeLaSalida:
    """La unidad en que se mide la tolerancia. `valor is None` = no hay escala válida y `motivo` dice
    por qué (I8): ninguna columna se comprueba con una escala inventada."""
    tarea: str
    valor: float | None
    clave_de_la_escala: str | None
    motivo: dict[str, str] | None
    clave_del_motivo: str | None = None

    def a_json(self) -> dict[str, Any]:
        return {"tarea": self.tarea, "valor": self.valor, "clave_de_la_escala": self.clave_de_la_escala,
                "descripcion": motivo(self.clave_de_la_escala) if self.clave_de_la_escala else None,
                "clave_del_motivo": self.clave_del_motivo, "motivo": self.motivo}


def escala_de_la_salida(tarea: str, *, amplitud_p90_p10: float | None = None,
                        pendiente_de_la_recalibracion: float | None = None) -> EscalaDeLaSalida:
    """Binaria: la PROBABILIDAD calibrada que ve la persona, escala 1 (puntos de probabilidad). La
    recalibración del estudio es `sigmoid(a + b·logit p)`: monótona solo si `b > 0`, así que con
    `b <= 0` (o no finita) la probabilidad ya no conserva el orden de la puntuación y NO hay escala.
    `None` en `pendiente_de_la_recalibracion` = el estudio no recalibró: vale la probabilidad del
    modelo. Regresión: la amplitud p90−p10 del objetivo de ENTRENAMIENTO (ya en el sobre del estudio:
    `reparto_del_objetivo_en_el_entrenamiento`), que ha de ser positiva. Multiclase: sin escala hasta
    que exista la clase declarada (D10)."""
    exigir_opcion(tarea, "tarea", TAREAS)
    if tarea == "binary_classification":
        b = pendiente_de_la_recalibracion
        if b is not None and not (isinstance(b, (int, float)) and not isinstance(b, bool)
                                  and math.isfinite(b) and b > 0):
            return EscalaDeLaSalida(tarea, None, None,
                                    motivo("recalibracion_no_conserva_el_orden", valor=cifra(b) if _es_finito(b) else "?"),
                                    "recalibracion_no_conserva_el_orden")
        return EscalaDeLaSalida(tarea, 1.0, "escala_de_probabilidad", None)
    if tarea == "regression":
        a = amplitud_p90_p10
        if not (_es_finito(a) and a > 0):
            return EscalaDeLaSalida(tarea, None, None,
                                    motivo("escala_de_regresion_no_valida", valor=cifra(a) if _es_finito(a) else "?"),
                                    "escala_de_regresion_no_valida")
        return EscalaDeLaSalida(tarea, float(a), "escala_de_regresion", None)
    return EscalaDeLaSalida(tarea, None, None, motivo("multiclase_sin_clase_declarada"),
                            "multiclase_sin_clase_declarada")


def _es_finito(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def elegir_filas(n_total: int, m: int, semilla: int) -> tuple[int, ...]:
    """Los índices de las `m` filas (o todas, si hay menos) de una muestra de `n_total`, elegidos con
    la semilla y devueltos en orden creciente: la misma semilla da las mismas filas en cualquier
    máquina (`random.Random` sobre una cadena es determinista) y todas las columnas comparten filas."""
    exigir_entero(n_total, "n_total", minimo=0)
    exigir_entero(m, "m", minimo=1)
    exigir_entero(semilla, "semilla")
    if n_total <= m:
        return tuple(range(n_total))
    return tuple(sorted(random.Random(f"{semilla}:banco_de_comportamiento").sample(range(n_total), m)))


def _medido_sobre(clave: str, semilla: int, filas_pedidas: int) -> dict[str, Any]:
    exigir_opcion(clave, "medido_sobre", MEDIDO_SOBRE)
    return {"clave": clave, "semilla": exigir_entero(semilla, "semilla"),
            "filas_pedidas": exigir_entero(filas_pedidas, "filas_pedidas", minimo=0)}


def _numero_o_none(valor: Any) -> float | None:
    """Una salida del modelo: un número finito, o `None`. Un booleano NO es un número aquí."""
    if valor is None:
        return None
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise _SalidaNoNumerica()
    return float(valor) if math.isfinite(valor) else None


class _SalidaNoNumerica(Exception):
    pass


# ---------------------------------------------------------------------------
# C — monotonía declarada
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonotoniaDeUnaColumna:
    """El resultado de la prueba C para UNA columna. Inmutable: `con_tolerancia` devuelve otro."""
    columna: str
    direccion: str
    rejilla: tuple[float, ...]
    escala: EscalaDeLaSalida
    epsilon_relativo: float | None             # None solo si la tarea no tiene tolerancia y no se dio (escala no válida)
    q_relativo: float | None                   # fracción mínima de filas que retroceden más que ε para «con contraejemplos»
    medido_sobre: Mapping[str, Any]
    ids: tuple[Any, ...]                       # una por fila pedida, en el orden de la muestra
    retrocesos: tuple[float | None, ...]       # None = esa fila no tuvo predicción (NO es un 0)
    efectos: tuple[float | None, ...]          # extremo a extremo, en la dirección declarada
    amplitudes: tuple[float | None, ...]       # (máx − mín) de la curva de cada fila, sin signo
    peores: tuple[Mapping[str, Any], ...]      # los K de mayor retroceso, independientes de ε
    k: int
    clave_de_no_comprobar: str | None = None   # por qué no hay curvas (antes de mirar ninguna fila)
    motivo_de_no_comprobar: Mapping[str, str] | None = None
    motor_de_arboles: bool | None = None       # ¿el modelo es de árboles? None = no se sabe. Solo decide qué avisa el documento

    # ---- lo calculado ------------------------------------------------------------------------
    @property
    def epsilon_absoluto(self) -> float | None:
        return None if self.escala.valor is None else self.epsilon_relativo * self.escala.valor

    @property
    def filas_comprobadas(self) -> int:
        return sum(1 for r in self.retrocesos if r is not None)

    @property
    def efecto_total(self) -> float | None:
        """Media, sobre las filas comprobadas, del cambio de la salida de un extremo de la rejilla al
        otro, medido EN LA DIRECCIÓN DECLARADA (positivo = se mueve como se declaró)."""
        validos = [e for e in self.efectos if e is not None]
        return statistics.fmean(validos) if validos else None

    @property
    def amplitud_tipica(self) -> float | None:
        """MEDIANA, sobre las filas comprobadas, de (máximo − mínimo) de la curva de cada fila. Sin signo y sin
        depender de los extremos. `None` si ninguna fila tuvo predicción (un valor ausente no es un cero)."""
        validas = [a for a in self.amplitudes if a is not None]
        return statistics.median(validas) if validas else None

    @property
    def filas_con_violacion(self) -> int | None:
        eps = self.epsilon_absoluto
        if eps is None or self.filas_comprobadas == 0:
            return None
        # `>` y no `>=`: un retroceso IGUAL a la tolerancia es lo que la tolerancia deja pasar. (Una
        # prueba con retroceso == ε exacto lo fija.)
        return sum(1 for r in self.retrocesos if r is not None and r > eps)

    @property
    def fraccion_con_violacion(self) -> float | None:
        """Las filas con retroceso > ε entre las comprobadas. `None` si no hay nada que comparar."""
        v = self.filas_con_violacion
        return None if v is None else v / self.filas_comprobadas

    def _clave_del_veredicto(self) -> tuple[str, str | None, dict[str, str] | None]:
        """`(veredicto, clave_del_motivo, motivo)`."""
        if self.clave_de_no_comprobar is not None:
            return "no_se_pudo_comprobar", self.clave_de_no_comprobar, dict(self.motivo_de_no_comprobar or {})
        eps = self.epsilon_absoluto
        if eps is None:                                  # escala no válida: el motivo ya viaja en `escala`
            return "no_se_pudo_comprobar", self.escala.clave_del_motivo, self.escala.motivo
        if self.filas_comprobadas == 0:
            clave = "sin_filas" if not self.ids else "sin_prediccion"
            return ("no_se_pudo_comprobar", clave,
                    motivo(clave) if clave == "sin_filas" else motivo(clave, filas=len(self.ids)))
        # 1. LA AMPLITUD, PRIMERO (I5): si la curva típica no se mueve más que la tolerancia, la prueba no distingue una
        #    dirección de la contraria. `>` estricto, como el guion medido (`veredicto_ter`). Va ANTES de contar filas:
        #    una columna plana con un par de filas ruidosas no es «con contraejemplos».
        amplitud = self.amplitud_tipica
        if amplitud is None or not amplitud > eps:
            # Qué PASÓ: cuántas filas sí se mueven más que la tolerancia (la mediana puede ser plana con un subgrupo que se mueve).
            mueven = sum(1 for a in self.amplitudes if a is not None and a > eps)
            huecos = dict(amplitud=cifra(amplitud if amplitud is not None else 0.0), tolerancia=cifra(eps),
                          filas=self.filas_comprobadas)
            return ("no_se_pudo_comprobar", "amplitud_no_distingue",
                    motivo("amplitud_no_distingue_con_filas_que_se_mueven", mueven=mueven, **huecos) if mueven
                    else motivo("amplitud_no_distingue", **huecos))
        # 2. LA FRACCIÓN: al menos el q de las filas comprobadas retroceden más que ε. Aritmética exacta (decimal contra
        #    racional): `v/n >= q` en coma flotante falla en el borde (0,07·100 = 7,000000000000001).
        if Fraction(self.filas_con_violacion, self.filas_comprobadas) >= Fraction(repr(self.q_relativo)):
            return "con_contraejemplos", None, None
        return "sin_contraejemplos", None, None

    @property
    def veredicto(self) -> str:
        return self._clave_del_veredicto()[0]

    @property
    def contraejemplos(self) -> tuple[Mapping[str, Any], ...]:
        """Los guardados que superan la tolerancia ACTUAL (hasta K) —y SOLO si el veredicto es «con contraejemplos»:
        listar contraejemplos bajo un «sin contraejemplos» (unas pocas filas por debajo de q) se leería como una
        contradicción. Los K peores siguen enteros en `para_recalcular`."""
        if self.veredicto != "con_contraejemplos":
            return ()
        return self._peores_sobre_la_tolerancia()

    def _peores_sobre_la_tolerancia(self) -> tuple[Mapping[str, Any], ...]:
        eps = self.epsilon_absoluto
        return () if eps is None else tuple(c for c in self.peores if c["retroceso"] > eps)

    @property
    def filas_que_retroceden(self) -> tuple[Mapping[str, Any], ...]:
        """Decisión del supervisor (remate del C1-bis): bajo «sin contraejemplos», las filas (hasta K) que retroceden más que
        la tolerancia pero no llegan al q del veredicto. Se enseñan APARTE y con otro rótulo —no son «contraejemplos» de un
        veredicto que no los tiene, pero esconderlas sería no decir lo que pasó—. Vacío con cualquier otro veredicto (con
        contraejemplos ya están en `contraejemplos`)."""
        return self._peores_sobre_la_tolerancia() if self.veredicto == "sin_contraejemplos" else ()

    def con_tolerancia(self, epsilon_relativo: float | None = None, q_relativo: float | None = None) -> "MonotoniaDeUnaColumna":
        """El mismo resultado con OTRA tolerancia (ε, q), sin volver a predecir (I7). `None` = conservar el actual."""
        eps = self.epsilon_relativo if epsilon_relativo is None else _epsilon(epsilon_relativo)
        q = self.q_relativo if q_relativo is None else _q(q_relativo)
        return MonotoniaDeUnaColumna(**{**self.__dict__, "epsilon_relativo": eps, "q_relativo": q})

    def con_epsilon(self, epsilon_relativo: float) -> "MonotoniaDeUnaColumna":
        """Atajo de `con_tolerancia` que cambia solo ε (q se conserva)."""
        return self.con_tolerancia(epsilon_relativo=epsilon_relativo)

    # ---- el documento -----------------------------------------------------------------------------
    def _veredicto_json(self) -> dict[str, Any]:
        clave, clave_motivo, texto_motivo = self._clave_del_veredicto()
        puntos, filas = len(self.rejilla), self.filas_comprobadas
        limites = motivo(CLAVE_DE_LOS_LIMITES, puntos=puntos, filas=filas, semilla=self.medido_sobre["semilla"])
        if clave == "no_se_pudo_comprobar":
            lectura = motivo("lectura_sin_comprobar")
        else:
            eps = self.epsilon_absoluto
            # La cifra escrita no puede contradecir al veredicto (hallazgo I1): nada de `round(fraccion, 3)`.
            fraccion, minimo = porcentajes_de_la_lectura(self.filas_con_violacion, filas, self.q_relativo)
            lectura = motivo(
                "lectura_con_contraejemplos" if clave == "con_contraejemplos" else "lectura_sin_contraejemplos",
                violadoras=self.filas_con_violacion, filas=filas,
                fraccion=fraccion, minimo=minimo,
                tolerancia=cifra(eps), retroceso=cifra(max(r for r in self.retrocesos if r is not None)),
                direccion=PorIdioma(motivo("direccion_" + self.direccion)), amplitud=cifra(self.amplitud_tipica))
        return {"clave": clave, "texto": motivo(CLAVES_DE_VEREDICTO[clave]), "lectura": lectura,
                "clave_del_motivo": clave_motivo, "motivo": texto_motivo,
                # I4: los límites van DENTRO del veredicto, no en un pie.
                "limites": limites}

    def a_json(self) -> dict[str, Any]:
        sin_prediccion = len(self.ids) - self.filas_comprobadas
        avisos = ([{"clave": "filas_sin_prediccion", "texto": motivo("filas_sin_prediccion", sin=sin_prediccion, total=len(self.ids))}]
                  if sin_prediccion and self.filas_comprobadas else [])
        # En binaria la tolerancia de partida es ALTA a propósito (149-C0-ter): lo dice el documento, no solo el contrato. Lo que
        # dice depende de DOS hechos (hallazgo I2): si lo usado ES lo de partida y si el motor es de árboles (lo medido).
        base = TOLERANCIA_POR_OMISION.get(self.escala.tarea)
        por_omision = (self.epsilon_relativo, self.q_relativo) == base
        if (self.escala.tarea == "binary_classification" and base is not None and self.escala.valor is not None
                and self.filas_comprobadas and self.epsilon_relativo is not None and self.epsilon_relativo >= base[0]):
            arboles = self.motor_de_arboles is True
            if por_omision:
                clave_del_aviso = "tolerancia_alta_en_binaria" if arboles else "tolerancia_alta_en_binaria_otro_motor"
            else:
                clave_del_aviso = ("tolerancia_alta_en_binaria_no_de_partida" if arboles
                                   else "tolerancia_alta_en_binaria_no_de_partida_otro_motor")
            avisos.append({"clave": "tolerancia_alta_en_binaria",
                           "texto": motivo(clave_del_aviso, tolerancia=porcentaje(self.epsilon_relativo))})
        return {
            "version": VERSION_DEL_COMPORTAMIENTO,
            "columna": self.columna, "direccion": self.direccion,
            "medido_sobre": dict(self.medido_sobre),
            "rejilla": {"puntos": len(self.rejilla), "valores": list(self.rejilla)},
            "tolerancia": {"relativa": self.epsilon_relativo, "absoluta": self.epsilon_absoluto,
                           "texto": None if self.epsilon_relativo is None else porcentaje(self.epsilon_relativo),
                           "fraccion_minima": {"relativa": self.q_relativo,
                                               "texto": None if self.q_relativo is None else porcentaje(self.q_relativo)},
                           # ¿es la de partida de la tarea (medida en C0-ter)? Lo dice el valor, no cómo se pasó.
                           "por_omision": por_omision,
                           "escala": self.escala.a_json()},
            "filas": {"pedidas": len(self.ids), "comprobadas": self.filas_comprobadas, "sin_prediccion": sin_prediccion,
                      "ids": list(self.ids)},
            "efecto_total": self.efecto_total,
            "amplitud_tipica": self.amplitud_tipica,
            "fraccion_con_violacion": self.fraccion_con_violacion,
            "retroceso_maximo": max((r for r in self.retrocesos if r is not None), default=None),
            "filas_con_violacion": self.filas_con_violacion,
            "contraejemplos": [dict(c) for c in self.contraejemplos],
            # Bajo «sin contraejemplos»: las filas que retroceden más que la tolerancia sin llegar al q, APARTE y con su rótulo.
            "filas_que_retroceden": {"rotulo": motivo("rotulo_filas_que_retroceden"),
                                     "filas": [dict(c) for c in self.filas_que_retroceden]},
            "motor_de_arboles": self.motor_de_arboles,
            "avisos": avisos,
            "veredicto": self._veredicto_json(),
            # Lo que NO depende de ε: con esto `con_epsilon` rehace el veredicto sin predecir (I7).
            "para_recalcular": {"k": self.k, "retrocesos": list(self.retrocesos), "efectos": list(self.efectos),
                                "amplitudes": list(self.amplitudes), "peores": [dict(c) for c in self.peores],
                                "clave_de_no_comprobar": self.clave_de_no_comprobar,
                                "motivo_de_no_comprobar": None if self.motivo_de_no_comprobar is None
                                else dict(self.motivo_de_no_comprobar)},
        }

    @classmethod
    def desde_json(cls, documento: Mapping[str, Any]) -> "MonotoniaDeUnaColumna":
        """El resultado de `a_json()` (leído de `comportamiento.json`) vuelto a un objeto, para re-evaluarlo con
        `con_tolerancia` sin predecir: quien lo pinte (C3a) NO reescribe la regla del veredicto. Solo se lee lo que no depende
        de la tolerancia (`para_recalcular`) y la tolerancia usada; el veredicto, la lectura y los avisos del documento se
        RECALCULAN, no se creen. Un documento de otra versión, o con listas desalineadas, es un `EsquemaInvalido`."""
        if not isinstance(documento, Mapping):
            raise EsquemaInvalido("falta_campo", campo="documento")
        if documento.get("version") != VERSION_DEL_COMPORTAMIENTO:
            raise EsquemaInvalido("version_no_legible", campo="version", valor=repr(documento.get("version")),
                                  opciones=VERSION_DEL_COMPORTAMIENTO)
        try:
            tol, esc, rec = documento["tolerancia"], documento["tolerancia"]["escala"], documento["para_recalcular"]
            ids = tuple(documento["filas"]["ids"])
            medido = documento["medido_sobre"]
            escala = EscalaDeLaSalida(esc["tarea"], esc["valor"], esc["clave_de_la_escala"], esc["motivo"],
                                      esc["clave_del_motivo"])
            largos = {len(rec[campo]) for campo in ("retrocesos", "efectos", "amplitudes")}
            if largos != {len(ids)}:
                raise EsquemaInvalido("filas_desalineadas", campo="para_recalcular", valor=sorted(largos), opciones=len(ids))
            malla = tuple(float(x) for x in documento["rejilla"]["valores"])
            _con_signo(documento["direccion"])
            return cls(
                exigir_texto(documento["columna"], "columna"), documento["direccion"], malla, escala,
                None if tol["relativa"] is None else _epsilon(tol["relativa"]),
                None if tol["fraccion_minima"]["relativa"] is None else _q(tol["fraccion_minima"]["relativa"]),
                _medido_sobre(medido["clave"], medido["semilla"], medido["filas_pedidas"]), ids,
                tuple(rec["retrocesos"]), tuple(rec["efectos"]), tuple(rec["amplitudes"]),
                tuple(dict(c) for c in rec["peores"]), exigir_entero(rec["k"], "k", minimo=1),
                rec["clave_de_no_comprobar"], None if rec["motivo_de_no_comprobar"] is None else dict(rec["motivo_de_no_comprobar"]),
                documento.get("motor_de_arboles"))
        except (KeyError, TypeError) as error:
            raise EsquemaInvalido("falta_campo", campo=f"documento ({error!r})") from None

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def _epsilon(valor: Any) -> float:
    return exigir_real(valor, "epsilon_relativo", minimo=0.0)


def _q(valor: Any) -> float:
    """La fracción mínima de filas: en (0, 1]. Con q = 0 cualquier columna tendría contraejemplos (0 ≥ 0)."""
    q = exigir_real(valor, "q_relativo", minimo=0.0, maximo=1.0)
    if q == 0.0:
        raise EsquemaInvalido("fuera_de_rango", campo="q_relativo", minimo="0 (excluido)", maximo=1.0, valor="0.0")
    return q


def _tolerancia(escala: EscalaDeLaSalida, epsilon_relativo: float | None,
                q_relativo: float | None) -> tuple[float | None, float | None]:
    """`(ε, q)` validados. Lo que no se da sale de `TOLERANCIA_POR_OMISION` de la tarea. Con una escala VÁLIDA y sin
    valor para la tarea, FALLA ruidoso: no se inventa uno (la escala no válida ya no comprueba nada, y ahí `None` es honesto)."""
    base = TOLERANCIA_POR_OMISION.get(escala.tarea)
    eps = epsilon_relativo if epsilon_relativo is not None else (base[0] if base else None)
    q = q_relativo if q_relativo is not None else (base[1] if base else None)
    if escala.valor is not None:
        for nombre, valor in (("epsilon_relativo", eps), ("q_relativo", q)):
            if valor is None:
                raise EsquemaInvalido("falta_campo", campo=nombre)
    return (None if eps is None else _epsilon(eps)), (None if q is None else _q(q))


def _peores(retrocesos: Sequence[float | None], curvas: Sequence[Sequence[float] | None], ids: Sequence[Any],
            direccion: str, malla: Sequence[float], k: int) -> tuple[dict[str, Any], ...]:
    candidatos = []
    for i, (r, curva) in enumerate(zip(retrocesos, curvas)):
        if r is None or curva is None:
            continue
        ret = retroceso_de_una_curva(curva, direccion)
        if ret.valor > 0:
            candidatos.append((ret.valor, i, ret))
    # Mayor retroceso primero; a igualdad, la fila que va antes en la muestra (determinista).
    candidatos.sort(key=lambda t: (-t[0], t[1]))
    return tuple({"fila": ids[i], "x_a": malla[ret.a], "x_b": malla[ret.b],
                  "y_a": curvas[i][ret.a], "y_b": curvas[i][ret.b], "retroceso": valor}
                 for valor, i, ret in candidatos[:k])


def _sin_curvas(columna: str, direccion: str, malla: Sequence[float], escala: EscalaDeLaSalida, epsilon_relativo: float | None,
                q_relativo: float | None, medido: dict[str, Any], clave: str, texto: dict[str, str],
                k: int, motor_de_arboles: bool | None = None) -> MonotoniaDeUnaColumna:
    return MonotoniaDeUnaColumna(columna, direccion, tuple(malla), escala, epsilon_relativo, q_relativo, medido, (), (), (), (),
                                 (), k, clave, texto, motor_de_arboles)


def monotonia_desde_curvas(columna: str, direccion: str, malla: Sequence[float],
                           curvas: Sequence[Sequence[float | None]], ids: Sequence[Any], *,
                           escala: EscalaDeLaSalida, medido_sobre: str, semilla: int,
                           epsilon_relativo: float | None = None, q_relativo: float | None = None,
                           k: int = K_PEORES_POR_OMISION, motor_de_arboles: bool | None = None) -> MonotoniaDeUnaColumna:
    """La prueba C sobre curvas YA calculadas: `curvas[i][j]` es la salida del modelo en la fila `i` con la
    columna puesta en `malla[j]` (`None` = sin predicción). Es lo que llama `monotonia_en_rejilla` y lo que
    usan las pruebas con curvas construidas a mano. `epsilon_relativo` y `q_relativo`: `None` = los de partida
    de la tarea (`TOLERANCIA_POR_OMISION`). `motor_de_arboles`: ¿el modelo es de árboles? (`None` = no se sabe): solo decide
    qué dice el aviso de la tolerancia alta en binaria, que cita una medición hecha con árboles."""
    exigir_texto(columna, "columna")
    _con_signo(direccion)
    eps_rel, q_rel = _tolerancia(escala, epsilon_relativo, q_relativo)
    exigir_entero(k, "k", minimo=1)
    if motor_de_arboles is not None:
        exigir_booleano(motor_de_arboles, "motor_de_arboles")
    if len(curvas) != len(ids):
        raise EsquemaInvalido("filas_desalineadas", campo="ids", valor=len(ids), opciones=len(curvas))
    malla = tuple(float(x) for x in malla)
    if len(malla) < 2 or any(b <= a for a, b in zip(malla, malla[1:])):
        raise EsquemaInvalido("falta_campo", campo="rejilla")
    medido = _medido_sobre(medido_sobre, semilla, len(curvas))
    retrocesos: list[float | None] = []
    efectos: list[float | None] = []
    amplitudes: list[float | None] = []
    limpias: list[list[float] | None] = []
    signo = _con_signo(direccion)
    for i, curva in enumerate(curvas):
        if curva is None:
            # Una fila SIN curva (el llamante no tiene nada que darle) es una fila sin predicción, como los huecos: se excluye
            # y se cuenta; no es una curva plana ni revienta.
            retrocesos.append(None); efectos.append(None); amplitudes.append(None); limpias.append(None)
            continue
        if len(curva) != len(malla):
            raise EsquemaInvalido("filas_desalineadas", campo=f"curvas[{i}]", valor=len(curva), opciones=len(malla))
        try:
            valores = [_numero_o_none(v) for v in curva]
        except _SalidaNoNumerica:
            return _sin_curvas(columna, direccion, malla, escala, eps_rel, q_rel, medido, "salida_no_numerica",
                               motivo("salida_no_numerica"), k, motor_de_arboles)
        if any(v is None for v in valores):
            # Una curva a medias NO se completa ni se toma por plana: la fila no cuenta.
            retrocesos.append(None); efectos.append(None); amplitudes.append(None); limpias.append(None)
            continue
        retrocesos.append(retroceso_de_una_curva(valores, direccion).valor)
        efectos.append(signo * (valores[-1] - valores[0]))
        amplitudes.append(amplitud_de_una_curva(valores))
        limpias.append(valores)
    return MonotoniaDeUnaColumna(
        columna, direccion, malla, escala, eps_rel, q_rel, medido, tuple(ids), tuple(retrocesos), tuple(efectos),
        tuple(amplitudes), _peores(retrocesos, limpias, ids, direccion, malla, k), k, motor_de_arboles=motor_de_arboles)


def monotonia_en_rejilla(columna: str, direccion: str, filas: Sequence[Mapping[str, Any]],
                         puntuar: Callable[[Sequence[Mapping[str, Any]]], Sequence[float | None]], *,
                         minimo: float | None, maximo: float | None, puntos: int, filas_m: int, semilla: int,
                         escala: EscalaDeLaSalida, medido_sobre: str,
                         epsilon_relativo: float | None = None, q_relativo: float | None = None,
                         ids: Sequence[Any] | None = None, k: int = K_PEORES_POR_OMISION,
                         motor_de_arboles: bool | None = None) -> MonotoniaDeUnaColumna:
    """La prueba C de una columna. Elige `filas_m` filas con la semilla, y para cada una recorre
    `puntos` valores de la rejilla entre `minimo` y `maximo` DE LO VISTO, con el resto de la fila fijo
    (curvas ICE). `puntuar` recibe TODAS las filas perturbadas de una vez (fila-mayor: fila 0 con los G
    valores, fila 1…) y devuelve una salida por fila, en el mismo orden; una `None` = sin predicción.

    Las filas que recibe `puntuar` son copias: la muestra del llamante no se toca. Sea cual sea el valor
    que traiga la columna (un hueco, `NA`, un texto), se sobrescribe: el ICE recorre la rejilla.

    `epsilon_relativo` y `q_relativo` (ε como fracción de la escala; q como fracción mínima de filas): `None` = los de
    partida de la tarea de la escala (`TOLERANCIA_POR_OMISION`); multiclase no los tiene y no comprueba nada.
    `motor_de_arboles`: ver `monotonia_desde_curvas`."""
    exigir_texto(columna, "columna")
    _con_signo(direccion)
    eps_rel, q_rel = _tolerancia(escala, epsilon_relativo, q_relativo)
    exigir_entero(puntos, "puntos", minimo=2)
    elegidas = elegir_filas(len(filas), exigir_entero(filas_m, "filas_m", minimo=1), semilla)
    if ids is not None and len(ids) != len(filas):
        raise EsquemaInvalido("filas_desalineadas", campo="ids", valor=len(ids), opciones=len(filas))
    ids_elegidos = tuple(ids[i] if ids is not None else i for i in elegidas)
    medido = _medido_sobre(medido_sobre, semilla, len(elegidas))

    def sin_comprobar(clave: str, texto: dict[str, str], malla: Sequence[float] = ()) -> MonotoniaDeUnaColumna:
        return _sin_curvas(columna, direccion, malla, escala, eps_rel, q_rel, medido, clave, texto, k, motor_de_arboles)

    if escala.valor is None:
        return sin_comprobar(escala.clave_del_motivo or "escala_no_valida", dict(escala.motivo or {}))
    if not (_es_finito(minimo) and _es_finito(maximo) and maximo > minimo):
        return sin_comprobar("rejilla_sin_rango", motivo("rejilla_sin_rango"))
    malla = rejilla(minimo, maximo, puntos)
    if not elegidas:
        return sin_comprobar("sin_filas", motivo("sin_filas"), malla)
    perturbadas = []
    for i in elegidas:
        for x in malla:
            fila = dict(filas[i])
            fila[columna] = x
            perturbadas.append(fila)
    salidas = list(puntuar(perturbadas))
    if len(salidas) != len(perturbadas):
        raise EsquemaInvalido("filas_desalineadas", campo="salidas", valor=len(salidas), opciones=len(perturbadas))
    curvas = [salidas[j * puntos:(j + 1) * puntos] for j in range(len(elegidas))]
    return monotonia_desde_curvas(columna, direccion, malla, curvas, ids_elegidos, epsilon_relativo=eps_rel,
                                  q_relativo=q_rel, escala=escala, medido_sobre=medido_sobre, semilla=semilla, k=k,
                                  motor_de_arboles=motor_de_arboles)


# ---------------------------------------------------------------------------
# B — sensibilidad a cambios pequeños (DESCRIBE: sin veredicto, I2)
# ---------------------------------------------------------------------------

def _percentil(ordenados: Sequence[float], q: float) -> float:
    if len(ordenados) == 1:
        return ordenados[0]
    posicion = q * (len(ordenados) - 1)
    bajo = math.floor(posicion)
    alto = min(bajo + 1, len(ordenados) - 1)
    return ordenados[bajo] + (ordenados[alto] - ordenados[bajo]) * (posicion - bajo)


def _distancia(a: Any, b: Any) -> float:
    """|Δ| de dos salidas: un número, o el mayor cambio de cualquier componente de un vector (la
    probabilidad de la clase que más se mueve)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(b) - float(a))
    return max(abs(float(y) - float(x)) for x, y in zip(a, b))


@dataclass(frozen=True)
class SensibilidadDeUnaColumna:
    columna: str
    medido_sobre: Mapping[str, Any]
    minimo: float | None
    maximo: float | None
    ids: tuple[Any, ...]
    pruebas: tuple[Mapping[str, Any], ...]     # una por (fracción, signo)
    clave_de_no_medir: str | None = None
    motivo_de_no_medir: Mapping[str, str] | None = None

    def a_json(self) -> dict[str, Any]:
        # I2: AQUÍ NO HAY VEREDICTO. Cifras, tasas y las filas de mayor pendiente.
        return {"version": VERSION_DEL_COMPORTAMIENTO, "columna": self.columna,
                "medido_sobre": dict(self.medido_sobre),
                "rango_visto": {"minimo": self.minimo, "maximo": self.maximo},
                "filas": {"pedidas": len(self.ids)},
                "disponible": self.clave_de_no_medir is None,
                "clave_del_motivo": self.clave_de_no_medir, "motivo": self.motivo_de_no_medir,
                "pruebas": [dict(p) for p in self.pruebas]}

    def digest(self) -> str:
        return digest_canonico(self.a_json())


def sensibilidad_de_una_columna(columna: str, filas: Sequence[Mapping[str, Any]],
                                puntuar: Callable[[Sequence[Mapping[str, Any]]], Sequence[Any]], *,
                                minimo: float | None, maximo: float | None, semilla: int, filas_n: int,
                                medido_sobre: str, fracciones: Sequence[float] = (0.01, 0.05),
                                decidir: Callable[[Any], Any] | None = None, ids: Sequence[Any] | None = None,
                                k: int = K_PEORES_POR_OMISION) -> SensibilidadDeUnaColumna:
    """Prueba B de una columna numérica: a `filas_n` filas con la semilla se les suma y resta
    δ = fracción × (`maximo` − `minimo`) en esa columna. Por cada (fracción, signo): cuántas filas
    cambian de DECISIÓN (`decidir(salida)`; sin ella —regresión— el campo es `None`, no un 0), la
    distribución de |Δsalida|/δ y las `k` filas de mayor pendiente. Un valor que no es un número en esa
    fila (un hueco, `NA`) se deja como estaba y la fila no cuenta: un hueco sigue siendo un hueco."""
    exigir_texto(columna, "columna")
    exigir_entero(k, "k", minimo=1)
    elegidas = elegir_filas(len(filas), exigir_entero(filas_n, "filas_n", minimo=1), semilla)
    ids_elegidos = tuple(ids[i] if ids is not None else i for i in elegidas)
    medido = _medido_sobre(medido_sobre, semilla, len(elegidas))
    fracciones = tuple(exigir_real(f, "fraccion", minimo=0.0) for f in fracciones)
    if any(f == 0.0 for f in fracciones):
        raise EsquemaInvalido("fuera_de_rango", campo="fraccion", minimo="0 (excluido)", maximo="+inf", valor="0.0")

    def no(clave: str) -> SensibilidadDeUnaColumna:
        return SensibilidadDeUnaColumna(columna, medido, minimo if _es_finito(minimo) else None,
                                        maximo if _es_finito(maximo) else None, ids_elegidos, (), clave, motivo(clave))

    if not (_es_finito(minimo) and _es_finito(maximo) and maximo > minimo):
        return no("sensibilidad_sin_rango")
    con_valor = [i for i in elegidas if _es_finito(filas[i].get(columna))]
    if not con_valor:
        return no("sensibilidad_sin_filas")
    base = [dict(filas[i]) for i in con_valor]
    salida_base = list(puntuar(base))
    if len(salida_base) != len(base):
        raise EsquemaInvalido("filas_desalineadas", campo="salidas", valor=len(salida_base), opciones=len(base))
    pos = {i: p for p, i in enumerate(elegidas)}
    pruebas = []
    for fraccion in fracciones:
        delta = fraccion * (maximo - minimo)
        for signo, rotulo in ((1.0, "mas"), (-1.0, "menos")):
            movidas = [{**f, columna: f[columna] + signo * delta} for f in base]
            salida = list(puntuar(movidas))
            if len(salida) != len(movidas):
                raise EsquemaInvalido("filas_desalineadas", campo="salidas", valor=len(salida), opciones=len(movidas))
            pares = [(j, a, b) for j, (a, b) in enumerate(zip(salida_base, salida)) if a is not None and b is not None]
            pendientes = sorted(((_distancia(a, b) / delta, j) for j, a, b in pares), key=lambda t: (-t[0], t[1]))
            ordenadas = sorted(p for p, _ in pendientes)
            cambian = None if decidir is None else sum(1 for _, a, b in pares if decidir(a) != decidir(b))
            pruebas.append({
                "fraccion": fraccion, "signo": rotulo, "delta": delta,
                "filas_comprobadas": len(pares), "filas_sin_prediccion": len(base) - len(pares),
                "filas_sin_valor_en_la_columna": len(elegidas) - len(con_valor),
                "cambian_de_decision": cambian,
                "fraccion_que_cambia_de_decision": (cambian / len(pares)) if cambian is not None and pares else None,
                "sin_decision": motivo("sensibilidad_sin_decision") if decidir is None else None,
                "pendiente": ({"mediana": _percentil(ordenadas, 0.5), "p90": _percentil(ordenadas, 0.9), "maximo": ordenadas[-1]}
                              if ordenadas else None),
                "mayores_pendientes": [
                    {"fila": ids_elegidos[pos[con_valor[j]]], "valor_antes": base[j][columna],
                     "valor_despues": movidas[j][columna], "salida_antes": salida_base[j], "salida_despues": salida[j],
                     "pendiente": p} for p, j in pendientes[:k]],
            })
    return SensibilidadDeUnaColumna(columna, medido, float(minimo), float(maximo), ids_elegidos, tuple(pruebas))


# ---------------------------------------------------------------------------
# A — «si falta una columna»: la agregación (DESCRIBE: sin veredicto, I2)
# ---------------------------------------------------------------------------

def caida_entre_pliegues(columna: str, caidas: Sequence[float | None], *, nunca_falto_al_entrenar: bool | None,
                         medido_sobre: str = "validacion_de_cada_pliegue") -> dict[str, Any]:
    """Caída media y dispersión ENTRE PLIEGUES de una columna. `caidas[p]` es la caída de la métrica del
    estudio en el pliegue `p` (signo unificado: positiva = peor sin la columna), o `None` si ese pliegue no
    la dio. La media se hace sobre los pliegues que sí la dieron y se dice cuántos fueron; ninguno =
    `caida_media: None` con su motivo, no un cero. `nunca_falto_al_entrenar` (`proporcion_faltante == 0`)
    añade el aviso columna a columna: entonces la caída mide extrapolación, no robustez."""
    exigir_texto(columna, "columna")
    if medido_sobre != "validacion_de_cada_pliegue":
        # El camino «en muestra» para A está PROHIBIDO por diseño (I1): mide 10–40 veces menos.
        raise EsquemaInvalido("no_es_del_vocabulario", campo="medido_sobre",
                              opciones=["validacion_de_cada_pliegue"], valor=repr(medido_sobre))
    validas = [float(c) for c in caidas if _es_finito(c)]
    media = statistics.fmean(validas) if validas else None
    return {
        "version": VERSION_DEL_COMPORTAMIENTO, "columna": columna, "medido_sobre": medido_sobre,
        "pliegues": len(caidas), "pliegues_con_caida": len(validas),
        "caidas": [float(c) if _es_finito(c) else None for c in caidas],
        "caida_media": media,
        "desviacion": (statistics.stdev(validas) if len(validas) > 1 else 0.0) if validas else None,
        "nunca_falto_al_entrenar": nunca_falto_al_entrenar,
        "aviso": motivo("falta_nunca_visto") if nunca_falto_al_entrenar else None,
        "motivo": None if validas else motivo("falta_sin_pliegues_con_valor"),
    }


def ordenar_por_caida(resultados: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """De mayor a menor caída media; las columnas sin cifra AL FINAL (no empatan con un 0); a igualdad,
    por nombre, para que el orden no dependa de cómo llegaron."""
    return sorted(resultados, key=lambda r: (r["caida_media"] is None, -(r["caida_media"] or 0.0), r["columna"]))


def k_mayores(ordenados: Sequence[Mapping[str, Any]], k: int) -> tuple[str, ...]:
    """Los nombres de las `k` primeras columnas CON cifra: la variante «las k más importantes juntas»
    (una columna cada vez no basta: las gemelas se compensan)."""
    exigir_entero(k, "k", minimo=1)
    return tuple(r["columna"] for r in ordenados if r["caida_media"] is not None)[:k]


def bloques_por_prefijo(columnas: Sequence[str], *, separador: str = "_", minimo: int = 2) -> dict[str, tuple[str, ...]]:
    """Los bloques de columnas que comparten prefijo hasta el primer `separador` (todas las `peor_*`), con
    al menos `minimo` miembros, en orden de aparición. La comparación es EXACTA —`Peor_x` y `peor_y` no
    son el mismo bloque—: agrupar de más diría que se quitó algo que no se quitó."""
    exigir_entero(minimo, "minimo", minimo=2)
    por_prefijo: dict[str, list[str]] = {}
    for c in columnas:
        if separador in c and not c.startswith(separador):
            por_prefijo.setdefault(c.split(separador, 1)[0], []).append(c)
    return {p: tuple(cs) for p, cs in por_prefijo.items() if len(cs) >= minimo}

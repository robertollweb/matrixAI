# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C3 — preparación dentro del entrenamiento: proponer y AJUSTAR, por
columna, una estrategia de faltantes y de categóricas — solo sobre TRAIN.
Test (o cualquier fila nueva) recibe `transformar_fila`, nunca vuelve a
ajustar nada: la política ya está fijada cuando llega ese momento.

TRES ESTADOS DE UNA CATEGÓRICA, NUNCA COLAPSADOS (texto literal del
criterio: «diferenciar categoría desconocida, faltante y categoría de
referencia»). `CATEGORIA_FALTANTE` es un valor ausente en la fila.
`CATEGORIA_DESCONOCIDA` es un valor PRESENTE que train nunca vio — dos
cosas de naturaleza distinta que un `None` compartido confundiría (una fila
sin dato no es lo mismo que una fila con un dato nuevo). `categoria_de_
referencia` (la más frecuente en train) es metadato declarado para quien
construya la codificación final (p. ej. qué columna dummy omitir en un
one-hot) — este módulo no hace el one-hot él mismo, eso es de quien
codifica (`categorical.py`, o el motor si tiene tratamiento nativo).

MEDIANA + INDICADOR, NO MEDIA. La mediana no la mueve un valor extremo
como sí la media; el indicador (`{columna}__faltante`, 0.0/1.0) declara
DÓNDE se imputó, para que un motor sin tratamiento nativo de faltantes no
pierda esa información al sustituir el valor.

POR MOTOR, SEGÚN CAPACIDAD — DOS BOOLEANOS, NO UN CATÁLOGO NUEVO.
`admite_categoricas`/`admite_faltantes` son los mismos dos campos de
`Capacidades` (102-C1, en `matrixai-engines`) pero recibidos aquí como
datos sueltos: este paquete es stdlib puro y no puede importar
`matrixai_engines` (que depende de numpy) — el llamante extrae esos dos
booleanos y los pasa, igual que 104-C3 recibía `necesita_red` sin importar
nada del motor real.

NO SE IMPUTA EL OBJETIVO. Una fila sin `objetivo` se EXCLUYE del train
efectivo, con su recuento — la única fila que este módulo nunca completa
con un valor inventado es precisamente la que se está intentando predecir.

NO BLOQUEAR POR N PEQUEÑO (texto literal del criterio). Menos de 100 filas
en el train efectivo no lanza excepción ni impide la política: añade un
`Limite` (103-C2) declarando el soporte reducido — la misma frontera
error/indefinición del resto del programa, aplicada aquí al tamaño de
muestra en vez de a una métrica.

FUERA DE ESTE CORTE, CON SU MOTIVO. Resampling y pesos («si se habilitan,
usan únicamente train y quedan registrados», texto literal) no se
implementan aquí: el criterio de terminado de este corte no exige una
política de resampling concreta, y construir una a medias sin un caso de
uso real que la fije sería peor que declararla pendiente — mismo
razonamiento que dejó fuera la corrección de multiplicidad en 105-C4/C5.
La codificación final a columnas (one-hot, ordinal, nativa) tampoco: este
módulo resuelve el VALOR (conocido/faltante/desconocido), no la
representación final en columnas.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from matrixai.training.fechas import (RecetaDeFecha, detectar_fechas, expandir_fila,
                                     variables_de_fecha)
from matrixai.training.diagnostico import Limite
from matrixai.training.preparacion_textos import motivo

__all__ = [
    "CATEGORIA_DESCONOCIDA", "CATEGORIA_FALTANTE", "MINIMO_FILAS_SIN_AVISO",
    "SUFIJO_FALTANTE", "UMBRAL_AVISO_FALTANTES", "PoliticaDePreparacion",
    "PropuestaDeColumna", "ajustar_preparacion", "nombre_de_indicador",
    "transformar_fila",
]

#: Sufijo del indicador «en esta celda se imputó», DECLARADO UNA VEZ.
#:
#: Estaba escrito a mano en los dos sitios que lo producen
#: (`transformar_fila` y `columnas_de_salida`) y ahora hace falta un tercero:
#: `dataset_project.py` cablea esta política y tiene que declarar el indicador
#: como una FEATURE del modelo. Ese tercero además no puede copiarlo tal cual
#: —el prompt tipado de GEN sanea los nombres de campo con `_sanitize_name`,
#: que colapsa `_+` a un solo `_`, así que un `__` no sobrevive— y por eso
#: importa más todavía que el sufijo salga de un solo sitio: lo que cada
#: camino adapta es la ESCRITURA del nombre, nunca la decisión de cuál es.
SUFIJO_FALTANTE = "__faltante"


def nombre_de_indicador(columna: str) -> str:
    """El nombre del indicador de `columna`, según esta política."""
    return f"{columna}{SUFIJO_FALTANTE}"


#: Presente en la fila, pero NUNCA visto en train — distinto de faltante.
CATEGORIA_DESCONOCIDA = "__desconocida__"

#: Ausente en la fila (falta el dato).
CATEGORIA_FALTANTE = "__faltante__"

#: «Proporción >50 % de faltantes puede activar aviso» (texto literal) —
#: medido con 40 % (no dispara) y 60 % (sí) en el criterio de terminado.
UMBRAL_AVISO_FALTANTES = 0.5

#: «No bloquear automáticamente toda recomendación con menos de 100 filas»
#: — 100 es el propio número que da el criterio de terminado, no una
#: elección libre de este módulo.
MINIMO_FILAS_SIN_AVISO = 100


def _es_faltante(valor: Any) -> bool:
    if valor is None:
        return True
    if isinstance(valor, str) and valor.strip() == "":
        return True
    if isinstance(valor, float) and valor != valor:  # NaN
        return True
    return False


def _es_numerica(valor: Any) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _mediana(valores: Sequence[float]) -> float:
    ordenados = sorted(valores)
    n = len(ordenados)
    medio = n // 2
    if n % 2 == 1:
        return ordenados[medio]
    return (ordenados[medio - 1] + ordenados[medio]) / 2.0


def _categoria_mas_frecuente(valores: Sequence[str]) -> str:
    conteo = Counter(valores)
    maximo = max(conteo.values())
    # empate: la primera en orden alfabético, para que sea determinista y no
    # dependa del orden de aparición en el CSV.
    return min(c for c, n in conteo.items() if n == maximo)


@dataclass(frozen=True)
class PropuestaDeColumna:
    """Lo que se decidió AJUSTAR para una columna, con lo que hace falta
    para aplicarlo después (`transformar_fila`) sin volver a mirar train."""

    columna: str
    tipo: str  # "numerica" | "categorica" | "fecha"
    admite_nativo: bool
    proporcion_faltante: float
    mediana: float | None = None
    #: LOS EXTREMOS QUE VIO TRAIN, medidos sobre las MISMAS filas y la misma
    #: lista de valores presentes de la que sale `mediana` -- nunca sobre
    #: test, nunca sobre una fila nueva. Los añade 2026-09-14 porque quien
    #: usa el modelo pedía un deslizador y no había con qué dibujarlo: esta
    #: propuesta publicaba un valor de partida (`mediana`) pero ningún
    #: extremo, y un deslizador sin extremos tendría que inventárselos.
    #:
    #: `None` en una CATEGÓRICA, las dos: ordenar cadenas para sacar un
    #: "mínimo" inventaría un orden que los datos no tienen. `None` también
    #: en una política vieja releída con `desde_json` -- ausente no es cero,
    #: y ese estudio de verdad no midió esto.
    #:
    #: `minimo == maximo` (columna constante en train) es una MEDIDA, no un
    #: fallo, y se publica tal cual: si con eso se puede dibujar algo o no
    #: es decisión de quien pinta, no de este modulo.
    minimo: float | None = None
    maximo: float | None = None
    categorias_conocidas: tuple[str, ...] = ()
    categoria_de_referencia: str | None = None
    #: 114-C5: el formato con que se lee una columna de tipo «fecha» (`fechas.py`).
    #: `None` en las demás. La columna original se queda en la política —es lo que
    #: pide el formulario de «Probar una fila», una fecha y no «fecha__anio»— y sus
    #: variables van en `PoliticaDePreparacion.derivadas_de_fecha`.
    formato_de_fecha: str | None = None

    def a_json(self) -> dict[str, Any]:
        return {
            "columna": self.columna, "tipo": self.tipo,
            "formato_de_fecha": self.formato_de_fecha,
            "admite_nativo": self.admite_nativo,
            "proporcion_faltante": self.proporcion_faltante,
            "mediana": self.mediana,
            "minimo": self.minimo, "maximo": self.maximo,
            "categorias_conocidas": list(self.categorias_conocidas),
            "categoria_de_referencia": self.categoria_de_referencia,
        }

    @classmethod
    def desde_json(cls, payload: dict[str, Any]) -> "PropuestaDeColumna":
        return cls(columna=payload["columna"], tipo=payload["tipo"],
                   admite_nativo=payload["admite_nativo"],
                   proporcion_faltante=payload["proporcion_faltante"],
                   mediana=payload.get("mediana"),
                   # `.get()` A PROPOSITO: los sobres de selección ya escritos
                   # (medidos el 2026-09-14: 7 claves, sin extremos) se releen
                   # con `minimo`/`maximo` en `None`. No se rellenan con el
                   # rango de nada -- ese estudio no lo midió.
                   minimo=payload.get("minimo"), maximo=payload.get("maximo"),
                   categorias_conocidas=tuple(payload.get("categorias_conocidas") or ()),
                   categoria_de_referencia=payload.get("categoria_de_referencia"),
                   formato_de_fecha=payload.get("formato_de_fecha"))


@dataclass(frozen=True)
class PoliticaDePreparacion:
    """El resultado de ajustar sobre train: una propuesta por columna, más
    los `Limite` (103-C2, reutilizados — no un tercer tipo de hallazgo)
    que esa misma preparación encontró motivo para declarar."""

    columnas: tuple[PropuestaDeColumna, ...]
    filas_excluidas_sin_objetivo: int
    filas_de_train_efectivas: int
    limites: tuple[Limite, ...] = field(default_factory=tuple)
    #: 114-C5: las variables de cada columna de fecha (`{col}__anio`, …), como
    #: propuestas NUMÉRICAS normales (mediana, extremos, indicador de faltante),
    #: ajustadas sobre las mismas filas de train. Aparte de `columnas` a propósito:
    #: quien pinta el formulario lee `columnas` y tiene que ver la fecha, no esto.
    derivadas_de_fecha: tuple[PropuestaDeColumna, ...] = field(default_factory=tuple)

    def derivadas_de(self, propuesta: PropuestaDeColumna) -> tuple[PropuestaDeColumna, ...]:
        """Las propuestas derivadas de una columna de fecha, en el orden de su receta."""
        receta = RecetaDeFecha(columna=propuesta.columna, formato=propuesta.formato_de_fecha)
        por_nombre = {d.columna: d for d in self.derivadas_de_fecha}
        return tuple(por_nombre[n] for n in receta.columnas_derivadas())

    def columna(self, nombre: str) -> PropuestaDeColumna | None:
        return next((c for c in self.columnas if c.columna == nombre), None)

    def columnas_de_salida(self) -> tuple[str, ...]:
        """Los nombres que `transformar_fila` de verdad emite, en su orden.

        Existe porque el llamante NECESITA declararlos: un motor que deriva
        sus columnas de `ProblemSpec.predictors` (102-C2, invariante «modelo,
        preparación, calibrador y decisión forman una unidad versionada»)
        descartaría los indicadores `{columna}__faltante` que esta política
        añade, porque no son predictores del problema — son predictores
        DERIVADOS de ellos, y solo esta clase sabe cuáles produce.

        Auditoría 2026-09-11: sin esto, cada llamante tendría que reconstruir
        el sufijo `__faltante` por su cuenta, que es el «dos sitios declarando
        lo mismo» que acaba divergiendo en cuanto esta función cambie. Se
        deriva del MISMO bucle que `transformar_fila`, no de una lista
        paralela.

        Se DEDUPLICA preservando el orden, porque `transformar_fila` escribe en
        un `dict` y un `dict` no repite claves. Un CSV que ya traiga una columna
        llamada `x__faltante` junto a una `x` numérica hace que el indicador
        derivado de `x` COLISIONE con ella: el dict se queda con tres claves y
        una lista ingenua declararía cuatro. Encontrado en la reauditoría del
        2026-09-11 —una lista con repetidos hacía que `ProblemSpec` rechazara la
        traducción y el estudio del producto respondiera un **HTTP 500 sin
        motivo**—, así que aquí se declara exactamente lo que sale, no lo que
        debería salir.

        La colisión en sí es anterior y sigue viva: la columna real `x__faltante`
        del CSV queda PISADA por el indicador derivado, en silencio. Eso no se
        arregla aquí (cambiar el nombre del indicador afecta a todo lo ya
        ajustado); queda declarado como defecto de `transformar_fila`.
        """
        salida: list[str] = []
        vistas: set[str] = set()
        # Una fecha no sale como tal: salen sus variables, cada una con su indicador.
        emitidas = [p for propuesta in self.columnas
                    for p in (self.derivadas_de(propuesta) if propuesta.tipo == "fecha" else (propuesta,))]
        for propuesta in emitidas:
            for nombre in (propuesta.columna,
                           *([nombre_de_indicador(propuesta.columna)]
                             if propuesta.tipo == "numerica" else ())):
                if nombre not in vistas:
                    vistas.add(nombre)
                    salida.append(nombre)
        return tuple(salida)

    def a_json(self) -> dict[str, Any]:
        return {
            "columnas": [c.a_json() for c in self.columnas],
            "filas_excluidas_sin_objetivo": self.filas_excluidas_sin_objetivo,
            "filas_de_train_efectivas": self.filas_de_train_efectivas,
            "limites": [l.a_json() for l in self.limites],
            "derivadas_de_fecha": [c.a_json() for c in self.derivadas_de_fecha],
        }

    @classmethod
    def desde_json(cls, payload: dict[str, Any]) -> "PoliticaDePreparacion":
        return cls(
            columnas=tuple(PropuestaDeColumna.desde_json(c) for c in payload["columnas"]),
            filas_excluidas_sin_objetivo=payload["filas_excluidas_sin_objetivo"],
            filas_de_train_efectivas=payload["filas_de_train_efectivas"],
            limites=tuple(Limite.desde_json(l) for l in payload.get("limites") or ()),
            # `.get()`: una política anterior al 114-C5 no trae fechas, y no se le inventan.
            derivadas_de_fecha=tuple(PropuestaDeColumna.desde_json(c)
                                     for c in payload.get("derivadas_de_fecha") or ()))


def tipar_columnas_numericas(filas: list[dict[str, Any]],
                             columnas: Sequence[str]) -> tuple[str, ...]:
    """Convierte EN SITIO a `float` las columnas cuyo texto parsea entero, y
    devuelve cuáles tocó.

    `ajustar_preparacion` exige valores YA TIPADOS: su detección de
    numérica/categórica es `isinstance(v, (int, float))` y nunca intenta
    parsear texto, **por diseño del núcleo** — que una columna «parezca»
    numérica no es lo mismo que serlo, y adivinar ahí sería justo lo que este
    proyecto prohíbe. Pero un `csv.DictReader` da TODO como texto, y un ARFF
    da como texto las columnas nominales aunque sus valores sean `"0"`/`"1"`.

    **Y los motores SÍ parsean texto.** Ahí está el choque, y ha costado dos
    fallos distintos:

      · Studio (2026-09): `x1`/`x2`, numéricas de verdad, salían categóricas
        con valores casi únicos por fila; casi todo pliegue de validación veía
        «categoría nunca vista en train», el core metía el centinela de texto
        `__desconocida__` y `float("__desconocida__")` reventaba dentro del
        subproceso aislado.
      · Fase 0, pasada exploratoria del 101-C3 (2026-09-07, diagnosticado el
        12): `Internet-Advertisements` tiene columnas nominales con valores
        `"0"`/`"1"`. Mismo choque, mismo centinela, **6 intentos perdidos** de
        `lightgbm` y `sklearn.lineal` — y con ellos el dataset entero para la
        regla de cierre, que cuenta un fallo como dataset perdido. El veredicto
        «lightgbm 9/12 = 0,750 < 0,800» se apoyaba en parte en eso.

    La reparación del Studio se escribió allí y el camino de benchmarks nunca
    la recibió. Se extrae aquí, al núcleo, para que no haya una tercera copia:
    «antes de copiar una decisión por segunda vez, extraerla».

    Se decide **sobre el conjunto COMPLETO de filas, no por pliegue**: el tipo
    de una columna no depende de qué filas le tocan a cada pliegue. Y una
    columna se convierte solo si TODOS sus valores no vacíos parsean; si no, es
    categórica de verdad y se deja tal cual.
    """
    tocadas: list[str] = []
    for columna in columnas:
        no_vacios = [f.get(columna) for f in filas]
        no_vacios = [v for v in no_vacios if v not in (None, "")]
        if not no_vacios:
            continue
        try:
            tipados = {v: float(v) for v in set(no_vacios)}
        except (TypeError, ValueError):
            continue  # columna categórica de verdad -- se deja tal cual
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in no_vacios):
            continue  # ya estaba tipada: no se toca ni se declara tocada
        for fila in filas:
            valor = fila.get(columna)
            if valor not in (None, ""):
                fila[columna] = tipados[valor]
        tocadas.append(columna)
    return tuple(tocadas)


def ajustar_preparacion(filas: Sequence[Mapping[str, Any]], *, objetivo: str,
                        columnas: Sequence[str], admite_categoricas: bool,
                        admite_faltantes: bool, con_fechas: bool = False) -> PoliticaDePreparacion:
    """Ajusta SOLO sobre `filas` — se espera que sea el train de un pliegue,
    nunca test (el llamante lo garantiza, igual que en 105-C3/105-C4: este
    módulo no tiene ningún parámetro para recibir una segunda partición).

    `con_fechas` (114-C5, medido: «mejora» en 4 de 5 conjuntos con fecha en un
    reparto temporal, `benchmarks/fechas_114c5/`): una columna cuyos valores de
    train se leen todos con un mismo formato de fecha (`fechas.detectar_fechas`,
    que no adivina las ambiguas) pasa a sus variables en vez de a categórica.
    Por omisión NO, y es a propósito: la Fase 0 mide con su protocolo sellado, y
    cambiar lo que su preparación hace sería medir otra cosa sin decirlo."""
    con_objetivo = [f for f in filas if not _es_faltante(f.get(objetivo))]
    excluidas = len(filas) - len(con_objetivo)
    n = len(con_objetivo)

    propuestas: list[PropuestaDeColumna] = []
    limites: list[Limite] = []
    derivadas: list[PropuestaDeColumna] = []
    recetas = {r.columna: r for r in detectar_fechas(con_objetivo, columnas)} if con_fechas else {}

    for columna in columnas:
        valores = [f.get(columna) for f in con_objetivo]
        if columna in recetas:
            receta = recetas[columna]
            presentes = [v for v in valores if not _es_faltante(v)]
            proporcion_faltante = (len(valores) - len(presentes)) / n if n else 0.0
            propuestas.append(PropuestaDeColumna(
                columna=columna, tipo="fecha", admite_nativo=admite_faltantes,
                proporcion_faltante=proporcion_faltante, formato_de_fecha=receta.formato))
            expandidas = [expandir_fila({columna: v}, (receta,))[0] for v in valores]
            for nombre in receta.columnas_derivadas():
                numericos = [float(e[nombre]) for e in expandidas if e[nombre] is not None]
                derivadas.append(PropuestaDeColumna(
                    columna=nombre, tipo="numerica", admite_nativo=admite_faltantes,
                    proporcion_faltante=(n - len(numericos)) / n if n else 0.0,
                    mediana=_mediana(numericos) if numericos else None,
                    minimo=min(numericos) if numericos else None,
                    maximo=max(numericos) if numericos else None))
            if proporcion_faltante > UMBRAL_AVISO_FALTANTES:
                limites.append(Limite(
                    clave="faltantes_por_encima_del_umbral", campo=columna,
                    motivo=motivo("faltantes_por_encima_del_umbral", campo=columna,
                                  valor=f"{proporcion_faltante:.0%}"),
                    medida={"proporcion_faltante": proporcion_faltante}))
            continue
        presentes = [v for v in valores if not _es_faltante(v)]
        proporcion_faltante = (len(valores) - len(presentes)) / n if n else 0.0
        es_numerica = bool(presentes) and all(_es_numerica(v) for v in presentes)

        if es_numerica:
            # UNA SOLA LISTA para los tres: mediana, mínimo y máximo salen de
            # los mismos valores presentes de las mismas filas de train. Que
            # sea la misma variable no es estilo -- es lo que impide que el
            # extremo y el valor de partida acaben describiendo conjuntos
            # distintos si mañana alguien toca uno de los dos filtros.
            numericos = [float(v) for v in presentes]
            propuestas.append(PropuestaDeColumna(
                columna=columna, tipo="numerica", admite_nativo=admite_faltantes,
                proporcion_faltante=proporcion_faltante,
                mediana=_mediana(numericos),
                minimo=min(numericos), maximo=max(numericos)))
        else:
            categorias_str = [str(v) for v in presentes]
            categorias_conocidas = tuple(sorted(set(categorias_str)))
            referencia = _categoria_mas_frecuente(categorias_str) if categorias_str else None
            propuestas.append(PropuestaDeColumna(
                columna=columna, tipo="categorica", admite_nativo=admite_categoricas,
                proporcion_faltante=proporcion_faltante,
                categorias_conocidas=categorias_conocidas,
                categoria_de_referencia=referencia))

        if proporcion_faltante > UMBRAL_AVISO_FALTANTES:
            limites.append(Limite(
                clave="faltantes_por_encima_del_umbral", campo=columna,
                motivo=motivo("faltantes_por_encima_del_umbral", campo=columna,
                              valor=f"{proporcion_faltante:.0%}"),
                medida={"proporcion_faltante": proporcion_faltante}))

    if n < MINIMO_FILAS_SIN_AVISO:
        limites.append(Limite(
            clave="pocas_filas_para_preparacion", campo=None,
            motivo=motivo("pocas_filas_para_preparacion", valor=n),
            medida={"filas": n}))

    return PoliticaDePreparacion(
        columnas=tuple(propuestas), filas_excluidas_sin_objetivo=excluidas,
        filas_de_train_efectivas=n, limites=tuple(limites),
        derivadas_de_fecha=tuple(derivadas))


def transformar_fila(fila: Mapping[str, Any], politica: PoliticaDePreparacion) -> dict[str, Any]:
    """Solo TRANSFORM — nunca reajusta ninguna estadística con esta fila,
    sea de test o de inferencia. Aplica `politica` tal como quedó ajustada."""
    resultado: dict[str, Any] = {}
    for propuesta in politica.columnas:
        valor = fila.get(propuesta.columna)
        if propuesta.tipo == "fecha":
            # Sus variables, cada una con la regla de una numérica. Una fecha que no
            # se lee con su formato sale FALTANTE en todas —nunca un cero ni una fecha
            # inventada—, y el indicador lo dice.
            variables = variables_de_fecha(valor, propuesta.formato_de_fecha)
            for i, derivada in enumerate(politica.derivadas_de(propuesta)):
                faltante = variables is None
                resultado[derivada.columna] = ((None if derivada.admite_nativo else derivada.mediana)
                                               if faltante else variables[i])
                resultado[nombre_de_indicador(derivada.columna)] = 1.0 if faltante else 0.0
            continue
        if propuesta.tipo == "numerica":
            faltante = _es_faltante(valor)
            if faltante:
                resultado[propuesta.columna] = None if propuesta.admite_nativo else propuesta.mediana
            else:
                resultado[propuesta.columna] = float(valor)
            resultado[nombre_de_indicador(propuesta.columna)] = 1.0 if faltante else 0.0
        else:
            if _es_faltante(valor):
                resultado[propuesta.columna] = CATEGORIA_FALTANTE
            elif str(valor) in propuesta.categorias_conocidas:
                resultado[propuesta.columna] = str(valor)
            else:
                resultado[propuesta.columna] = CATEGORIA_DESCONOCIDA
    return resultado

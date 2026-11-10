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
from matrixai.training.codigos_con_ceros import codigos_con_ceros_de, es_un_codigo_con_ceros
from matrixai.training.diagnostico import Limite
from matrixai.training.textos_de_faltante import (es_texto_de_faltante, recuento_por_texto,
                                                  sin_dato_en_un_objetivo_numerico, textos_leidos_como_faltante)
from matrixai.training.preparacion_textos import motivo


class ValorNoNumerico(ValueError):
    """Una celda de una columna NUMÉRICA que no se puede leer como número («sesenta», «1.234,5»), al
    transformar UNA fila de uso. Es un `ValueError` (quien ya lo recogía así sigue recogiéndolo) con lo que
    hace falta para decírselo a la persona: la columna, lo que escribió y el motivo en los dos idiomas.
    Antes subía como `ValueError: could not convert string to float` y el Studio contestaba un 500 que la
    pantalla enseñaba como «la conexión con el servidor falló» (visto por dos auditorías del 142, 29-10)."""

    def __init__(self, columna: str, valor: object):
        self.columna = columna
        self.valor = str(valor)
        self.motivo = motivo("valor_no_numerico", campo=columna, valor=self.valor)
        super().__init__(self.motivo["es"])

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


def es_faltante_en_una_numerica(valor: Any) -> bool:
    """¿Es FALTANTE este valor en una columna NUMÉRICA de la política? Lo vacío, `None`, el NaN y los textos de faltante
    («NA», «?», «#N/A»…, hallazgo 11). Es lo que lee `transformar_fila`, y quien replique sus valores (el Studio, 129-C4)
    importa ESTA función: con `_es_faltante` solo, el indicador salía 0 con «NA» (auditoría del hallazgo 11, M3). Y «lo
    visto» de una FECHA la usa también (`lo_visto._es_faltante_crudo`): allí la preparación lee los mismos textos."""
    return _es_faltante(valor) or es_texto_de_faltante(valor)


def _presentes_de_la_columna(valores: Sequence[Any]) -> tuple[list[Any], bool, frozenset[Any]]:
    """`(presentes, es_numerica, textos leídos como faltante)` de UNA columna. Los textos de `TEXTOS_DE_FALTANTE` cuentan
    como faltantes SOLO si, quitados, lo que queda es numérico y no vacío (`textos_leidos_como_faltante`, LA regla); si no,
    son valores (una categoría «NA»)."""
    presentes = [v for v in valores if not _es_faltante(v)]
    textos = textos_leidos_como_faltante(set(presentes), lambda resto: all(_es_numerica(v) for v in resto))
    if textos:
        return [v for v in presentes if v not in textos], True, textos
    return presentes, bool(presentes) and all(_es_numerica(v) for v in presentes), frozenset()


def limite_de_textos_leidos_como_faltante(columna: str, por_texto: Mapping[str, int], *,
                                         en_una_fecha: bool = False) -> Limite:
    """LA declaración de los textos leídos como faltante de UNA columna (`por_texto`: `{texto: celdas}`, de
    `recuento_por_texto`), con su frase en los dos idiomas. La usan `ajustar_preparacion` y el estudio del Studio, que tipa
    ANTES de ajustar (y entonces aquí ya llegan `None`): un solo sitio para la frase. `en_una_fecha`: la columna es una
    FECHA y la frase lo dice (re-auditoría del hallazgo 11, M1: decía «es numérica»); la clave es la misma."""
    celdas = sum(por_texto.values())
    textos = ", ".join(f"«{t}»" if len(por_texto) == 1 else f"«{t}» ({n})" for t, n in por_texto.items())
    frase = "textos_leidos_como_faltante_en_una_fecha" if en_una_fecha else "textos_leidos_como_faltante"
    return Limite(clave="textos_leidos_como_faltante", campo=columna,
                  motivo=motivo(frase, campo=columna, celdas=celdas, textos=textos),
                  medida={"textos": list(por_texto), "celdas": celdas, "por_texto": dict(por_texto)})


def limite_de_codigo_con_ceros(columna: str, ejemplos: Sequence[str], celdas: int) -> Limite:
    """LA declaración de una columna que el estudio dejó como TEXTO porque trae códigos con ceros a la izquierda («08001»,
    «007»; la regla es `codigos_con_ceros.es_un_codigo_con_ceros`, decisión A de Roberto, 09-11). `ejemplos`: hasta tres
    valores tal como están escritos; `celdas`: cuántas celdas llevan el cero delante. Misma forma que las demás del grupo
    (`a_json()`), para que Progreso y Resultados la pinten con la frase del núcleo sin tocar nada."""
    lista = ", ".join(f"«{e}»" for e in ejemplos)
    return Limite(clave="codigo_con_ceros_a_la_izquierda", campo=columna,
                  motivo=motivo("codigo_con_ceros_a_la_izquierda", campo=columna, celdas=celdas, ejemplos=lista),
                  medida={"celdas": celdas, "ejemplos": list(ejemplos)})


def limite_de_textos_del_objetivo_sin_dato(objetivo: str, valores: Sequence[Any]) -> Limite | None:
    """LA declaración de las filas que el OBJETIVO de una regresión trae sin dato ESCRITO («NA», «#N/A», «-», «.»…; la regla
    es `sin_dato_en_un_objetivo_numerico`): cuántas y qué texto dice cada una, con su frase en los dos idiomas (hallazgo 11, M3;
    decisión A de Roberto, 09-11). `valores`: las celdas CRUDAS del objetivo, antes de quitar nada. Las vacías no son un texto
    y no se nombran aquí (las cuenta `filas_sin_objetivo`, como siempre). `None` si no hay ninguna: no hay nada que decir."""
    textos = {v for v in valores if isinstance(v, str) and v.strip() != "" and sin_dato_en_un_objetivo_numerico(v)}
    por_texto = recuento_por_texto(valores, textos)
    if not por_texto:
        return None
    celdas = sum(por_texto.values())
    lista = ", ".join(f"«{t}»" if len(por_texto) == 1 else f"«{t}» ({n})" for t, n in por_texto.items())
    return Limite(clave="textos_del_objetivo_sin_dato", campo=objetivo,
                  motivo=motivo("textos_del_objetivo_sin_dato", campo=objetivo, celdas=celdas, textos=lista),
                  medida={"textos": list(por_texto), "celdas": celdas, "por_texto": dict(por_texto)})


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
                             columnas: Sequence[str], *,
                             leidos_como_faltante: dict[str, dict[str, Any]] | None = None,
                             conservar_ceros_a_la_izquierda: bool = False,
                             codigos_con_ceros: dict[str, dict[str, Any]] | None = None) -> tuple[str, ...]:
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

    **Los textos de faltante** (`TEXTOS_DE_FALTANTE`: «NA», «n/a», «?», «#N/A»…, hallazgo 11) no cuentan para
    decidir: si, quitados, todo lo demás parsea, la columna ES numérica y esas celdas pasan a `None` (faltante),
    no a categoría (la regla es `textos_de_faltante.textos_leidos_como_faltante`). Si se pasa `leidos_como_faltante`, se
    rellena con `{columna: {"textos": [...], "celdas": n, "por_texto": {texto: celdas}}}`, cada texto UNA vez (recortado),
    para que quien llama DIGA lo que se leyó así (`limite_de_textos_leidos_como_faltante`); lo vacío, los espacios y el NaN
    no se declaran: ya eran faltantes.

    **Los códigos con ceros a la izquierda** (decisión A de Roberto, 09-11; la regla es `codigos_con_ceros`): con
    `conservar_ceros_a_la_izquierda=True`, una columna que SE HABRÍA convertido pero en la que ALGÚN valor es «08001», «007»,
    «0012»… es un código (postal, de producto…), no un número: se deja como texto —categórica— y NO se declara tocada
    (`float("08001")` da 8001.0 y perdería el cero para siempre). Si se pasa `codigos_con_ceros`, se rellena con
    `{columna: {"celdas": n, "ejemplos": [hasta 3]}}` para que quien llama lo DIGA (`limite_de_codigo_con_ceros`).
    **Por omisión NO** (`False`), y es a propósito: la Fase 0 mide con su protocolo sellado (`benchmarks/fase0`) y las
    mediciones de fechas llaman sin ello; cambiar lo que tipan sería medir otra cosa sin decirlo. Lo pide el estudio del
    Studio, igual que el análisis ya trata esas columnas como no numéricas.
    """
    # 10-10 (medido en el arranque de APSFailure, 76.000 × 171: 19,5 → 12,9 s, salida IDÉNTICA): cada columna se lee UNA
    # vez (`valores`), los vacíos se quitan sobre el CONJUNTO de valores distintos y no sobre cada fila, y se escribe con
    # `zip`. Las mismas decisiones que antes, en el mismo orden: `test_tipar_columnas_igual_que_antes.py` lo compara con
    # una copia literal de la versión anterior en formas de datos reales.
    tocadas: list[str] = []
    for columna in columnas:
        valores = [f.get(columna) for f in filas]
        distintos = set(valores)
        distintos.discard(None)
        distintos.discard("")
        if not distintos:
            continue
        # Los textos de faltante («NA», «?», «#N/A»…) no cuentan: si quitados todo parsea, la columna ES numérica y
        # esas celdas pasan a `None` (faltante). Los que ya parseaban («nan») siguen igual que antes. Una columna SOLO de
        # textos de faltante no tiene números que digan que es numérica: no se toca (la regla lo devuelve vacío).
        tipados: dict[Any, float] = {}

        def _el_resto_parsea(resto: list[Any]) -> bool:
            try:
                for v in resto:
                    tipados[v] = float(v)
            except (TypeError, ValueError):
                return False
            return True

        textos = textos_leidos_como_faltante(distintos, _el_resto_parsea)
        if not textos:
            try:
                tipados = {v: float(v) for v in distintos}
            except (TypeError, ValueError):
                continue  # columna categórica de verdad -- se deja tal cual
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in distintos):
            continue  # ya estaba tipada: no se toca ni se declara tocada
        if conservar_ceros_a_la_izquierda:
            con_ceros = codigos_con_ceros_de(valores)  # en orden de aparición: sin depender del azar de los `set`
            if con_ceros:
                if codigos_con_ceros is not None:
                    codigos_con_ceros[columna] = {
                        "celdas": sum(1 for v in valores if es_un_codigo_con_ceros(v)),
                        "ejemplos": con_ceros[:3]}
                continue  # un código: se queda como texto (categoría), con sus ceros
        por_texto = recuento_por_texto(valores, {t for t in textos if t.strip() != ""})
        if por_texto and leidos_como_faltante is not None:
            leidos_como_faltante[columna] = {"textos": list(por_texto), "celdas": sum(por_texto.values()),
                                             "por_texto": por_texto}
        for fila, valor in zip(filas, valores):
            if valor in textos:
                fila[columna] = None
            elif valor is not None and valor != "":
                fila[columna] = tipados[valor]
        tocadas.append(columna)
    return tuple(tocadas)


def textos_leidos_como_faltante_en_las_fechas(filas: Sequence[Mapping[str, Any]],
                                              columnas: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Las columnas de `columnas` que la preparación leerá como FECHA con textos de faltante dentro («NA», «#N/A»…), con el
    MISMO formato que `leidos_como_faltante` de `tipar_columnas_numericas` (reparación del hallazgo 11, I4). No toca las filas:
    la fecha se expande al preparar. Es para quien tipa ANTES de ajustar (el estudio del Studio) y tiene que DECIR también
    lo que se leyó así en una fecha; la regla es la de `detectar_fechas(..., con_textos_de_faltante=True)`."""
    salida: dict[str, dict[str, Any]] = {}
    for receta in detectar_fechas(filas, columnas, con_textos_de_faltante=True):
        valores = [f.get(receta.columna) for f in filas]
        por_texto = recuento_por_texto(valores, {v for v in valores if es_texto_de_faltante(v) and v.strip() != ""})
        if por_texto:
            salida[receta.columna] = {"textos": list(por_texto), "celdas": sum(por_texto.values()), "por_texto": por_texto}
    return salida


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
    recetas = ({r.columna: r for r in detectar_fechas(con_objetivo, columnas, con_textos_de_faltante=True)}
               if con_fechas else {})

    for columna in columnas:
        valores = [f.get(columna) for f in con_objetivo]
        if columna in recetas:
            receta = recetas[columna]
            # Reparación del hallazgo 11 (I4): la fecha se detectó SIN sus textos de faltante (la regla), así que en ella
            # «NA», «?», «#N/A»… son faltantes: sus variables salen faltantes (`variables_de_fecha` no los lee) y se DECLARAN.
            presentes = [v for v in valores if not _es_faltante(v) and not es_texto_de_faltante(v)]
            proporcion_faltante = (len(valores) - len(presentes)) / n if n else 0.0
            por_texto = recuento_por_texto(valores, {v for v in valores if es_texto_de_faltante(v) and v.strip() != ""})
            if por_texto:
                limites.append(limite_de_textos_leidos_como_faltante(columna, por_texto, en_una_fecha=True))
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
        presentes, es_numerica, textos = _presentes_de_la_columna(valores)
        proporcion_faltante = (len(valores) - len(presentes)) / n if n else 0.0

        if es_numerica:
            # LOS TEXTOS LEÍDOS COMO FALTANTE SE DECLARAN (hallazgo 11): «NA», «?», «#N/A»… no son una categoría ni
            # un cero, y la persona tiene que poder ver que se leyeron así. Lo vacío y el NaN no: esos ya eran faltantes.
            por_texto = recuento_por_texto(valores, {t for t in textos if str(t).strip() != ""})
            if por_texto:
                limites.append(limite_de_textos_leidos_como_faltante(columna, por_texto))
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


def _numero_de_la_celda(columna: str, valor: object) -> float:
    """`float(valor)`, o `ValorNoNumerico` con la columna y lo escrito. La coma decimal NO se acepta aquí:
    esta función ve lo que manda el formulario, que ya la convierte; aceptarla también aquí sería una
    segunda regla de lectura de números distinta de la del entrenamiento."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        raise ValorNoNumerico(columna, valor) from None


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
            faltante = es_faltante_en_una_numerica(valor)   # «NA» en una numérica: faltante declarado
            if faltante:
                resultado[propuesta.columna] = None if propuesta.admite_nativo else propuesta.mediana
            else:
                resultado[propuesta.columna] = _numero_de_la_celda(propuesta.columna, valor)
            resultado[nombre_de_indicador(propuesta.columna)] = 1.0 if faltante else 0.0
        else:
            if _es_faltante(valor):
                resultado[propuesta.columna] = CATEGORIA_FALTANTE
            elif str(valor) in propuesta.categorias_conocidas:
                resultado[propuesta.columna] = str(valor)
            else:
                resultado[propuesta.columna] = CATEGORIA_DESCONOCIDA
    return resultado

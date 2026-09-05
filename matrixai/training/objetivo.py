# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El problema se CONFIRMA antes de entrenar — contrato 103, corte C1.

QUÉ ENTREGA. Que el core no empiece un estudio sin saber **qué se predice, para
qué unidad de observación, con qué tarea y —si es binaria— con qué clase
positiva**; y que, cuando aplique, diga también **desde cuándo** se predice y
**hasta dónde** mira el desenlace. Las dos rutas del producto pasan por aquí:

* **CSV** — `confirmar_desde_csv`. Hoy `analyze_dataset_csv` ya propone columnas
  objetivo con su motivo estructurado (`target_candidates[*].reason_codes`); lo
  que faltaba era el paso siguiente: que esa propuesta se CONFIRME y no se
  consuma como si fuera una decisión.
* **PROMPT** — `confirmar_desde_prompt`. Medido hoy contra el core: «analizar
  los datos de clientes de una empresa de telefonía» sale con `OUTPUT
  predicted_class` y `feature_1..4`, o sea con objetivo y entradas inventados y
  sin un solo aviso. Aquí una frase que no dice qué predecir no arranca ningún
  estudio: devuelve la pregunta.

EL RESULTADO ES UN DOCUMENTO, NO UN BOOLEANO. `Confirmacion` lleva la propuesta,
las PREGUNTAS que faltan por contestar, los BLOQUEOS que impiden el estudio y
las PISTAS que hay que mirar; y `problema` —un `ProblemSpec` del 104-C0— solo
cuando no queda ninguna pregunta ni ningún bloqueo. `ProblemSpec` es el sitio
donde vive el problema confirmado y aquí no se hace otro.

TRES REGLAS QUE SON EL CORTE ENTERO:

1. **Un objetivo no se inventa para poder continuar.** Ni el nombre («la última
   columna, seguro que es ésa»), ni la tarea («tres enteros distintos, será
   clasificación»), ni la clase positiva («la primera por orden alfabético»).
   Lo que no se sabe se PREGUNTA, y la pregunta viaja con su motivo en los dos
   idiomas.
2. **Los nombres parecidos son PISTAS, no veredictos.** El contrato 71 lo dejó
   medido con tres prompts: «clasificar el precio a partir del precio» es fuga
   real, y «predecir el salario a partir del salario del año pasado» es
   legítimo —el sueldo anterior predice bien el siguiente—. Un detector por
   parecido de nombres marcaría los dos. Aquí el parecido AVISA; lo que bloquea
   es la identidad CONFIRMADA: la misma columna declarada a la vez objetivo y
   entrada.
3. **Un valor ausente no es un cero y `None` es una respuesta.** Un objetivo sin
   confirmar es `problema=None` con sus preguntas, no un `ProblemSpec` a medias;
   una unidad de observación que nadie declaró no se rellena con «una fila».

LO QUE NO ES DE ESTE CORTE, y se dice para que no se dé por hecho: los
detectores de fuga con su semántica (correlación, proxies, IDs, disponibilidad
temporal) son el **C2**; la preparación ajustada dentro de train es el **C3**;
las particiones son el **C4**; y publicarlo en API y UI es el **C5**. Aquí solo
se confirma el problema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from matrixai.estudio import Horizonte, ProblemSpec, Restriccion
from matrixai.estudio.errores import ErrorDeEstudio
from matrixai.training.objetivo_textos import motivo

__all__ = [
    "Bloqueo", "Confirmacion", "ObjetivoSinConfirmar", "Pista", "Pregunta",
    "confirmar_desde_csv", "confirmar_desde_prompt", "corte_de_entrenamiento",
    "exigir_problema_confirmado", "no_entrenable_en_train", "pistas_por_nombre",
]


#: Tipos de columna que NO se pueden predecir. Un identificador no significa
#: nada fuera de su fila y una fecha no es un desenlace; `unknown` es una
#: columna vacía. Los nombres salen de `dataset_analysis._analyze_column`.
TIPOS_NO_PREDECIBLES = ("identifier", "date", "unknown")

#: Tipos cuya naturaleza YA dice que la tarea es de clasificación.
TIPOS_DE_CLASIFICACION = ("boolean", "categorical")

#: Tipos numéricos. Cuántos valores distintos tengan es lo que abre la pregunta
#: de tipo de tarea, y esa pregunta no se contesta sola.
TIPOS_NUMERICOS = ("number", "integer")

#: Tokens que no cuentan al comparar dos nombres: son enlaces, no significado.
#: Sin esta lista, `precio_del_metro` y `coste_del_barrio` compartirían `del` y
#: se acusarían mutuamente.
_TOKENS_VACIOS = frozenset({"de", "del", "la", "el", "los", "las", "un", "una",
                            "of", "the", "a", "an", "in", "on", "for", "por"})


class ObjetivoSinConfirmar(ErrorDeEstudio):
    """Se ha intentado arrancar un estudio con el problema sin confirmar.

    Hereda de `ErrorDeEstudio` (104-C0) para que quien capture las roturas del
    protocolo capture también ésta, pero compone su motivo con el catálogo de
    este corte: `ErrorDeEstudio.__init__` mira el catálogo del 104, que no
    conoce —ni tiene por qué— las preguntas del 103.
    """

    def __init__(self, clave: str, **campos: object) -> None:
        textos = motivo(clave, **campos)
        ValueError.__init__(self, textos["en"])
        self.clave = clave
        self.campos = dict(campos)
        self.es = textos["es"]
        self.en = textos["en"]


# ---------------------------------------------------------------------------
# Las tres cosas que puede decir una confirmación
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pregunta:
    """Algo que falta por decidir. Con `opciones` cuando las hay: preguntar sin
    enseñar las respuestas posibles obliga a adivinar."""

    clave: str
    motivo: dict[str, str]
    campo: str | None = None
    opciones: tuple[Any, ...] = ()

    def a_json(self) -> dict[str, Any]:
        return {"clave": self.clave, "campo": self.campo,
                "motivo": dict(self.motivo), "opciones": list(self.opciones)}


@dataclass(frozen=True)
class Bloqueo:
    """Un error ESTRUCTURAL del diseño (103, invariante 2): impide el estudio
    hasta corregirse, y no se levanta aceptándolo."""

    clave: str
    motivo: dict[str, str]
    campo: str | None = None

    def a_json(self) -> dict[str, Any]:
        return {"clave": self.clave, "campo": self.campo, "motivo": dict(self.motivo)}


@dataclass(frozen=True)
class Pista:
    """Una señal por NOMBRE. No decide nada y no bloquea nada (103-C1: «los
    nombres parecidos generan pistas, no veredictos»)."""

    clave: str
    objetivo: str
    entrada: str
    motivo: dict[str, str]

    def a_json(self) -> dict[str, Any]:
        return {"clave": self.clave, "objetivo": self.objetivo,
                "entrada": self.entrada, "motivo": dict(self.motivo)}


@dataclass(frozen=True)
class Confirmacion:
    """El problema, confirmado o con lo que le falta para estarlo."""

    problema: ProblemSpec | None = None
    propuesta: dict[str, Any] = field(default_factory=dict)
    preguntas: tuple[Pregunta, ...] = ()
    bloqueos: tuple[Bloqueo, ...] = ()
    pistas: tuple[Pista, ...] = ()

    @property
    def confirmado(self) -> bool:
        """Solo cuando hay `ProblemSpec` y no queda nada por decidir.

        Las PISTAS no cuentan: una pista es algo que mirar, no algo que
        impida. Convertirlas en bloqueo es justo lo que el 71 midió que
        acusaba a una columna legítima.
        """
        return (self.problema is not None and not self.preguntas
                and not self.bloqueos)

    def a_json(self) -> dict[str, Any]:
        return {
            "confirmado": self.confirmado,
            "problema": self.problema.a_json() if self.problema is not None else None,
            "propuesta": dict(self.propuesta),
            "preguntas": [p.a_json() for p in self.preguntas],
            "bloqueos": [b.a_json() for b in self.bloqueos],
            "pistas": [p.a_json() for p in self.pistas],
        }


def exigir_problema_confirmado(confirmacion: Confirmacion) -> ProblemSpec:
    """El `ProblemSpec`, o un rechazo con lo que falta. **Esto es la puerta.**

    Quien vaya a entrenar llama aquí: si el problema no está confirmado, no se
    entrena. El motivo enumera las claves que quedan pendientes en vez de decir
    «no válido», porque «no válido» obliga a leer el código para saber qué
    contestar.
    """
    if confirmacion.confirmado:
        return confirmacion.problema  # type: ignore[return-value]
    pendientes = [p.clave for p in confirmacion.preguntas]
    pendientes += [b.clave for b in confirmacion.bloqueos]
    raise ObjetivoSinConfirmar("estudio_sin_problema_confirmado",
                               valor=len(pendientes),
                               opciones=", ".join(pendientes) or "—")


# ---------------------------------------------------------------------------
# Pistas por nombre — señales, nunca veredictos
# ---------------------------------------------------------------------------

def _tokens(nombre: str) -> list[str]:
    from matrixai.training.dense_generator import _identifier  # noqa: PLC0415

    return [t for t in _identifier(nombre).split("_") if t]


def pistas_por_nombre(objetivo: str, entradas: Sequence[str]) -> tuple[Pista, ...]:
    """Qué entradas se PARECEN al objetivo por el nombre. Nada más que eso.

    Dos señales, y las dos son señales:

    * **Se llama igual.** Es la más fuerte, y aun así aquí no bloquea: dos
      columnas pueden compartir nombre y no ser la misma variable. Lo que
      bloquea es declararlas a la vez objetivo y entrada del MISMO estudio, y
      eso lo mira `confirmar_*`, no esta función.
    * **El nombre de la entrada contiene el del objetivo.** `last_year_salary`
      contra `salary`. El contrato 71 lo midió y lo etiquetó mal la primera vez:
      «el segundo caso lo etiqueté como fuga y no lo es. Ahí está el límite».

    LO QUE ESTA FUNCIÓN NO VE, dicho para que no se dé por hecho:

    * **Otro idioma.** El 71 registró que el LLM devolvía `price` para un prompt
      que pedía «el precio»: por nombre no hay parecido ninguno. Eso no se
      arregla con más nombres en una lista; se arregla confirmando el objetivo
      contra las columnas, que es lo que hace el resto de este módulo.
    * **El sentido contrario** (que el objetivo contenga a la entrada). Se deja
      fuera a propósito: un aviso falso cuesta más que un aviso ausente, y esa
      dirección marca `salario` como sospechosa de `salario_anual` en cuanto
      alguien nombra bien sus columnas.
    """
    obj_tokens = _tokens(objetivo)
    if not obj_tokens:
        return ()
    obj_norm = "_".join(obj_tokens)
    significativos = {t for t in obj_tokens if t not in _TOKENS_VACIOS}
    pistas: list[Pista] = []
    for entrada in entradas:
        ent_tokens = _tokens(entrada)
        if not ent_tokens:
            continue
        ent_norm = "_".join(ent_tokens)
        if ent_norm == obj_norm:
            pistas.append(Pista(
                clave="entrada_con_el_nombre_del_objetivo",
                objetivo=str(objetivo), entrada=str(entrada),
                motivo=motivo("entrada_con_el_nombre_del_objetivo",
                              campo=repr(str(entrada)), valor=repr(str(objetivo)))))
        elif significativos and significativos <= set(ent_tokens):
            pistas.append(Pista(
                clave="entrada_que_contiene_el_objetivo",
                objetivo=str(objetivo), entrada=str(entrada),
                motivo=motivo("entrada_que_contiene_el_objetivo",
                              campo=repr(str(entrada)), valor=repr(str(objetivo)))))
    return tuple(pistas)


# ---------------------------------------------------------------------------
# Entrenabilidad: qué ve de verdad el entrenamiento
# ---------------------------------------------------------------------------

def corte_de_entrenamiento(total: int, *, ratio: float | None = None) -> int:
    """Cuántas filas ve el ENTRENAMIENTO, con la regla del entrenador denso.

    La regla está copiada de `matrixai/training/dense_trainer.py` (el bloque que
    parte `examples` en `train_ex`/`val_ex`) y hay una prueba que la contrasta
    contra el entrenador REAL, porque dos sitios declarando lo mismo acaban
    divergiendo y aquí divergir significaría avisar de una partición que no es
    la que corre.

    Lo importante no es el 80 %: es que **no baraja**. El corte es por POSICIÓN,
    así que un CSV ordenado por la columna objetivo deja todas las filas de una
    clase fuera del entrenamiento sin que nada lo diga hoy.

    `ratio` es el declarado en `SPLIT ... mode=temporal`; sin él corre el 0.8
    fijo que el entrenador aplica ignorando lo que declare el contrato.
    """
    if total <= 1:
        return total
    if ratio is None:
        return max(1, int(total * 0.8))
    return max(1, min(total - 1, int(total * ratio)))


def no_entrenable_en_train(valores: Sequence[str], *, objetivo: str,
                           ratio: float | None = None,
                           filas_de_train: Sequence[int] | None = None) -> Bloqueo | None:
    """`Bloqueo` si el objetivo no varía en las filas de ENTRENAMIENTO.

    `valores` son los valores del objetivo **en el orden del fichero**, que es el
    orden en el que el entrenador los va a partir. Los nulos no cuentan como un
    valor: una fila sin objetivo se excluye del entrenamiento (103-C3), no
    aporta una clase.

    Por qué esto no lo cubría `constant_target_error`: aquélla mira el CSV
    ENTERO y acierta cuando la columna es constante de arriba abajo. Un CSV
    ordenado por la clase objetivo tiene dos valores en el fichero y uno solo en
    el 80 % inicial, que es lo que entrena — y ahí la pérdida llega a 0 sin que
    el modelo haya aprendido a distinguir nada. Las dos comprobaciones dicen lo
    mismo sobre el mismo CSV cuando la columna es constante; hay una prueba que
    lo fija para que no se separen.
    """
    from matrixai.training.dataset_analysis import _is_null  # noqa: PLC0415

    if filas_de_train is None:
        corte = corte_de_entrenamiento(len(valores), ratio=ratio)
        train = list(valores[:corte])
    else:
        train = [valores[i] for i in filas_de_train if 0 <= i < len(valores)]
    distintos: list[str] = []
    for value in train:
        if _is_null(value):
            continue
        limpio = str(value).strip()
        if limpio not in distintos:
            distintos.append(limpio)
        if len(distintos) > 1:
            return None
    if len(distintos) != 1:
        # Cero valores no nulos en train no es «una sola clase»: es que no hay
        # objetivo con el que entrenar, y ése es otro hecho (103-C3 lo cuenta
        # como filas excluidas). Decirlo aquí con el motivo de la clase única
        # mandaría a alguien a buscar variedad donde lo que falta son datos.
        return None
    return Bloqueo(
        clave="objetivo_con_una_sola_clase", campo=str(objetivo),
        motivo=motivo("objetivo_con_una_sola_clase", campo=repr(str(objetivo)),
                      valor=repr(distintos[0]), opciones=len(train)))


# ---------------------------------------------------------------------------
# Ruta CSV
# ---------------------------------------------------------------------------

def _onehot_max() -> int:
    """El umbral de «pocas categorías» del producto, pedido a quien lo fija.

    Es el mismo que decide one-hot frente a embedding en los generadores y el
    que usa `_rank_target_candidates` para llamar «pocas categorías» a una
    columna. Copiarlo aquí haría que la pregunta de tipo de tarea saltara con un
    número y el resto del producto usara otro.
    """
    from matrixai.training.dense_generator import _ONEHOT_MAX  # noqa: PLC0415

    return _ONEHOT_MAX


def _tarea_de_la_columna(tipo: str, cardinalidad: int) -> str | None:
    """La tarea que se DEDUCE del tipo, o `None` cuando hay que preguntar."""
    if tipo in TIPOS_DE_CLASIFICACION:
        return ("binary_classification" if cardinalidad == 2
                else "multiclass_classification")
    if tipo in TIPOS_NUMERICOS:
        if 2 <= cardinalidad <= _onehot_max():
            # Pocos valores numéricos distintos: pueden ser clases (1, 2, 3 =
            # niveles de riesgo) o una magnitud que se repite. Hoy el producto
            # decide por su cuenta y ADEMÁS decide distinto en cada sitio —
            # `_rank_target_candidates` lo propone como clasificación y
            # `generate_project_from_dataset` lo entrena como regresión—. Aquí
            # se pregunta.
            return None
        return "regression"
    return None


def _preguntas_de_momento(analisis: Mapping[str, Any], momento: str | None,
                          horizonte: Horizonte | None) -> list[Pregunta]:
    """Momento de predicción y horizonte, **cuando aplique** (103-C1).

    Aplica cuando los datos traen fecha: sin momento no se puede decir qué
    variable estaba disponible al predecir y cuál llegó después del desenlace
    (invariante 5 del 103). Sin fecha no se pregunta: exigir un momento a un
    dataset transversal sería pedir un dato que no existe.
    """
    temporales = list(analisis.get("temporal_columns") or [])
    preguntas: list[Pregunta] = []
    if temporales and momento is None:
        preguntas.append(Pregunta(
            clave="momento_de_prediccion", campo=None,
            opciones=tuple(temporales),
            motivo=motivo("momento_de_prediccion", opciones=", ".join(temporales))))
    if momento is not None and horizonte is None:
        preguntas.append(Pregunta(
            clave="horizonte_del_desenlace", campo=momento,
            motivo=motivo("horizonte_del_desenlace", campo=momento)))
    return preguntas


def confirmar_desde_csv(
    csv_text: str,
    *,
    objetivo: str | None = None,
    tarea: str | None = None,
    clases: Sequence[str] | None = None,
    clase_positiva: str | None = None,
    unidad_de_observacion: str | None = None,
    momento_de_prediccion: str | None = None,
    horizonte: Horizonte | None = None,
    uso_previsto: str | None = None,
    entradas: Sequence[str] | None = None,
    restricciones: Sequence[Restriccion] = (),
    filas_de_train: Sequence[int] | None = None,
    ratio_de_train: float | None = None,
    problem_id: str | None = None,
    analisis: Mapping[str, Any] | None = None,
    filas: Sequence[Mapping[str, Any]] | None = None,
) -> Confirmacion:
    """Propone el objetivo con su motivo y CONFIRMA el problema, o pregunta.

    `analisis` y `filas` son el resultado de `analyze_dataset_csv` y las filas ya
    leídas, para quien llama teniéndolos (el flujo desde datos calcula los dos
    una vez). Pasarlos evita repetir pasadas enteras sobre un CSV que puede
    tener cientos de miles de filas —medido en este producto: 0,24 s por pasada
    sobre 200.000 filas y 12,7 MB—, y no pasarlos no cambia nada del resultado.

    `entradas` son los predictores DECLARADOS. Sin ellas se proponen todas las
    columnas menos el objetivo — que es justo el criterio de cierre «la columna
    objetivo no entra en X», aquí por construcción. Si quien llama declara el
    objetivo TAMBIÉN como entrada, eso es identidad confirmada y bloquea.
    """
    from matrixai.training.dataset_analysis import analyze_dataset_csv  # noqa: PLC0415
    from matrixai.training.dataset_project import (  # noqa: PLC0415
        DatasetProjectError,
        _distinct_non_null,
        _normalize_labels,
        _read_rows,
    )

    if analisis is None:
        analisis = analyze_dataset_csv(csv_text)
    columnas: dict[str, Any] = dict(analisis.get("columns") or {})
    orden = list(analisis.get("column_order") or [])
    candidatas = list(analisis.get("target_candidates") or [])

    propuesta: dict[str, Any] = {
        "ruta": "csv",
        "columnas": orden,
        # LOS MOTIVOS YA EXISTEN Y SON ESTRUCTURADOS (`reason_codes`, contrato
        # 58 C3): se reenvían en vez de redactar aquí una segunda versión de la
        # misma explicación.
        #
        # Lo que NO se reenvía es `reasons`, la prosa **solo en castellano** que
        # `analyze_dataset_csv` conserva por retrocompatibilidad. Este documento
        # es bilingüe por construcción y va a viajar por la API a una aplicación
        # que puede estar en inglés: colar ahí «es la última columna del CSV»
        # dejaría media pantalla en español, que es el defecto que este producto
        # ya ha pagado varias veces. Los códigos los traduce quien pinta, que es
        # exactamente para lo que el contrato 58 C3 los creó.
        "candidatos": [{k: v for k, v in c.items() if k != "reasons"}
                       for c in candidatas],
        "columnas_temporales": list(analisis.get("temporal_columns") or []),
    }

    if objetivo is None:
        nombres = [str(c["column"]) for c in candidatas]
        return Confirmacion(
            propuesta=propuesta,
            preguntas=(Pregunta(
                clave="objetivo_no_elegido", campo=None, opciones=tuple(nombres),
                motivo=motivo("objetivo_no_elegido",
                              opciones=", ".join(nombres) or "—")),))

    objetivo = str(objetivo)
    if objetivo not in columnas:
        return Confirmacion(
            propuesta=propuesta,
            bloqueos=(Bloqueo(
                clave="objetivo_inexistente", campo=objetivo,
                motivo=motivo("objetivo_inexistente", campo=repr(objetivo),
                              opciones=", ".join(orden) or "—")),))

    info = dict(columnas[objetivo])
    tipo = str(info.get("type") or "unknown")
    if tipo in TIPOS_NO_PREDECIBLES:
        return Confirmacion(
            propuesta=propuesta,
            bloqueos=(Bloqueo(
                clave="objetivo_no_predecible", campo=objetivo,
                motivo=motivo("objetivo_no_predecible", campo=repr(objetivo),
                              valor=repr(tipo))),))

    preguntas: list[Pregunta] = []
    bloqueos: list[Bloqueo] = []

    # -- las entradas, y la identidad confirmada ---------------------------
    if entradas is None:
        predictores = tuple(c for c in orden if c != objetivo)
    else:
        predictores = tuple(str(e) for e in entradas)
        if objetivo in predictores:
            bloqueos.append(Bloqueo(
                clave="objetivo_entre_las_entradas", campo=objetivo,
                motivo=motivo("objetivo_entre_las_entradas", campo=repr(objetivo))))
    pistas = pistas_por_nombre(objetivo, predictores)

    # -- la tarea ----------------------------------------------------------
    cardinalidad = int(info.get("cardinality") or 0)
    if filas is None:
        filas = _read_rows(csv_text)
    valores_crudos = _distinct_non_null(list(filas), objetivo)
    if tarea is None:
        tarea = _tarea_de_la_columna(tipo, cardinalidad)
        if tarea is None:
            preguntas.append(Pregunta(
                clave="tipo_de_tarea", campo=objetivo,
                opciones=("binary_classification" if cardinalidad == 2
                          else "multiclass_classification", "regression"),
                motivo=motivo("tipo_de_tarea", campo=repr(objetivo),
                              valor=cardinalidad,
                              opciones=", ".join(valores_crudos[:10]))))
    propuesta["tipo_de_columna"] = tipo
    propuesta["cardinalidad"] = cardinalidad
    # Los valores solo cuando son pocos: enumerar los 200.000 precios distintos
    # de una regresión no informa de nada y engorda el documento que va a viajar
    # por HTTP. La cardinalidad, en cambio, va siempre.
    if len(valores_crudos) <= _onehot_max():
        propuesta["valores"] = list(valores_crudos)

    # -- las clases, EN SU ORDEN ------------------------------------------
    etiquetas: list[str] | None = None
    if tarea in ("binary_classification", "multiclass_classification"):
        crudas = [str(c) for c in clases] if clases is not None else list(valores_crudos)
        ausentes = [c for c in crudas if c not in valores_crudos]
        if ausentes:
            bloqueos.append(Bloqueo(
                clave="clases_declaradas_que_no_estan", campo=objetivo,
                motivo=motivo("clases_declaradas_que_no_estan", campo=repr(objetivo),
                              opciones=", ".join(repr(a) for a in ausentes))))
        else:
            try:
                # LAS ETIQUETAS QUE EL MODELO VA A EMITIR, con la MISMA
                # normalización y el MISMO orden con los que se escriben en el
                # `ProbabilityMap[...]` del `.mxai`. Recalcularlas aquí con otra
                # regla haría que el orden del manifiesto no fuera el del modelo.
                etiquetas, mapa = _normalize_labels(crudas, objetivo)
                propuesta["mapa_de_etiquetas"] = dict(mapa)
            except DatasetProjectError as exc:
                etiquetas = None
                bloqueos.append(Bloqueo(
                    clave="objetivo_no_predecible", campo=objetivo,
                    motivo=motivo("objetivo_no_predecible", campo=repr(objetivo),
                                  valor=repr(str(exc)))))
        if etiquetas is not None and len(etiquetas) < 2:
            bloqueos.append(Bloqueo(
                clave="objetivo_con_una_sola_clase", campo=objetivo,
                motivo=motivo("objetivo_con_una_sola_clase", campo=repr(objetivo),
                              valor=repr(etiquetas[0] if etiquetas else ""),
                              opciones=len(valores_crudos))))
            etiquetas = None
        propuesta["clases"] = list(etiquetas) if etiquetas else []

        if (etiquetas is not None and tarea == "binary_classification"
                and (clase_positiva is None or clase_positiva not in etiquetas)):
            preguntas.append(Pregunta(
                clave="clase_positiva", campo=objetivo, opciones=tuple(etiquetas),
                motivo=motivo("clase_positiva", campo=repr(objetivo),
                              opciones=", ".join(etiquetas))))
            clase_positiva = None

    # -- el objetivo, en las filas que de verdad entrenan ------------------
    bloqueo_train = no_entrenable_en_train(
        [str(row.get(objetivo) or "") for row in filas], objetivo=objetivo,
        ratio=ratio_de_train, filas_de_train=filas_de_train)
    if bloqueo_train is not None and not any(
            b.clave == "objetivo_con_una_sola_clase" for b in bloqueos):
        bloqueos.append(bloqueo_train)

    # -- unidad de observación, momento y horizonte ------------------------
    if unidad_de_observacion is None:
        preguntas.append(Pregunta(
            clave="unidad_de_observacion", campo=None,
            motivo=motivo("unidad_de_observacion")))
    preguntas.extend(_preguntas_de_momento(analisis, momento_de_prediccion, horizonte))

    return _confirmar(
        propuesta=propuesta, preguntas=preguntas, bloqueos=bloqueos, pistas=pistas,
        problem_id=problem_id or f"csv:{objetivo}", objetivo=objetivo, tarea=tarea,
        etiquetas=etiquetas, clase_positiva=clase_positiva,
        unidad_de_observacion=unidad_de_observacion, predictores=predictores,
        momento_de_prediccion=momento_de_prediccion, horizonte=horizonte,
        uso_previsto=uso_previsto, restricciones=restricciones)


# ---------------------------------------------------------------------------
# Ruta prompt
# ---------------------------------------------------------------------------

def confirmar_desde_prompt(
    prompt: str,
    *,
    objetivo: str | None = None,
    tarea: str | None = None,
    clases: Sequence[str] | None = None,
    clase_positiva: str | None = None,
    unidad_de_observacion: str | None = None,
    momento_de_prediccion: str | None = None,
    horizonte: Horizonte | None = None,
    uso_previsto: str | None = None,
    entradas: Sequence[str] = (),
    restricciones: Sequence[Restriccion] = (),
    problem_id: str | None = None,
) -> Confirmacion:
    """Lee el objetivo de la frase y confirma el problema, o pregunta.

    **Un prompt sin objetivo no arranca ningún estudio.** No se inventa
    `predicted_class` para poder seguir: se devuelve la pregunta, con la frase
    citada, para que quien la escribió diga qué hay que predecir o qué columna
    lo contiene.

    Las ENTRADAS se reciben, no se leen de la frase. Es deliberado: el generador
    determinista rellena con `feature_1..4` cuando no extrae ninguna (medido), y
    meter ese relleno en un `ProblemSpec` sería blanquear la invención. Quién
    son los predictores es del C2 y del C3; aquí se juzga a los que se declaren.
    """
    from matrixai.generation.prompt_objetivo import objetivo_declarado  # noqa: PLC0415
    from matrixai.training.dense_generator import (  # noqa: PLC0415
        DenseNetworkGenerator,
        _identifier,
    )

    texto = str(prompt or "")
    lectura = objetivo_declarado(texto) if objetivo is None else None
    propuesta: dict[str, Any] = {
        "ruta": "prompt",
        "lectura": lectura.a_json() if lectura is not None else None,
    }

    if objetivo is None and lectura is None:
        citado = " ".join(texto.split())
        return Confirmacion(
            propuesta=propuesta,
            preguntas=(Pregunta(
                clave="objetivo_no_declarado", campo=None,
                motivo=motivo("objetivo_no_declarado",
                              valor=repr(citado[:160] or ""))),))

    if objetivo is None and lectura is not None and lectura.nombre is None:
        # La frase dice QUÉ («si un cliente va a impagar») y no cómo se llama.
        return Confirmacion(
            propuesta=propuesta,
            preguntas=(Pregunta(
                clave="objetivo_sin_nombre", campo=None,
                motivo=motivo("objetivo_sin_nombre",
                              valor=repr(lectura.definicion))),))

    objetivo = str(objetivo) if objetivo is not None else str(lectura.nombre)

    preguntas: list[Pregunta] = []
    bloqueos: list[Bloqueo] = []

    predictores = tuple(str(e) for e in entradas)
    if any(_identifier(p) == _identifier(objetivo) for p in predictores):
        bloqueos.append(Bloqueo(
            clave="objetivo_entre_las_entradas", campo=objetivo,
            motivo=motivo("objetivo_entre_las_entradas", campo=repr(objetivo))))
    pistas = pistas_por_nombre(objetivo, predictores)

    dg = DenseNetworkGenerator()
    if tarea is None:
        # La tarea que el core deduce de la frase (contrato 70), traducida al
        # vocabulario del 104-C0. Es una PROPUESTA: sigue habiendo que
        # confirmarla, igual que en la ruta CSV.
        deducida = dg._detect_task(texto, None)
        tarea = {"regression": "regression",
                 "binary": "binary_classification",
                 "multiclass": "multiclass_classification"}.get(deducida, "regression")
    propuesta["tarea_leida"] = tarea

    etiquetas: list[str] | None = None
    if tarea in ("binary_classification", "multiclass_classification"):
        leidas = [str(c) for c in (clases if clases is not None
                                   else dg._extract_labels(texto) or [])]
        etiquetas = [e for e in (_identifier(c) for c in leidas) if e]
        propuesta["clases"] = list(etiquetas)
        if len(etiquetas) < 2:
            # NO se cae a `_default_labels`. Medido y documentado en el propio
            # core: «clasificar el nivel de riesgo del paciente» devolvía
            # `ProbabilityMap[class_a, class_b, class_c]` — marcadores de
            # posición cuyas salidas no significan nada. Un estudio no empieza
            # con clases inventadas.
            preguntas.append(Pregunta(
                clave="clases_no_declaradas", campo=objetivo,
                motivo=motivo("clases_no_declaradas", campo=repr(objetivo))))
            etiquetas = None
        elif tarea == "binary_classification" and len(etiquetas) != 2:
            tarea = "multiclass_classification"
        if (etiquetas is not None and tarea == "binary_classification"
                and (clase_positiva is None or clase_positiva not in etiquetas)):
            preguntas.append(Pregunta(
                clave="clase_positiva", campo=objetivo, opciones=tuple(etiquetas),
                motivo=motivo("clase_positiva", campo=repr(objetivo),
                              opciones=", ".join(etiquetas))))
            clase_positiva = None

    if unidad_de_observacion is None:
        preguntas.append(Pregunta(
            clave="unidad_de_observacion", campo=None,
            motivo=motivo("unidad_de_observacion")))
    if momento_de_prediccion is not None and horizonte is None:
        preguntas.append(Pregunta(
            clave="horizonte_del_desenlace", campo=momento_de_prediccion,
            motivo=motivo("horizonte_del_desenlace", campo=momento_de_prediccion)))

    return _confirmar(
        propuesta=propuesta, preguntas=preguntas, bloqueos=bloqueos, pistas=pistas,
        problem_id=problem_id or f"prompt:{objetivo}", objetivo=objetivo, tarea=tarea,
        etiquetas=etiquetas, clase_positiva=clase_positiva,
        unidad_de_observacion=unidad_de_observacion, predictores=predictores,
        momento_de_prediccion=momento_de_prediccion, horizonte=horizonte,
        uso_previsto=uso_previsto, restricciones=restricciones)


# ---------------------------------------------------------------------------
# El ensamblaje, común a las dos rutas
# ---------------------------------------------------------------------------

def _confirmar(*, propuesta: dict[str, Any], preguntas: list[Pregunta],
               bloqueos: list[Bloqueo], pistas: tuple[Pista, ...],
               problem_id: str, objetivo: str, tarea: str | None,
               etiquetas: list[str] | None, clase_positiva: str | None,
               unidad_de_observacion: str | None, predictores: tuple[str, ...],
               momento_de_prediccion: str | None, horizonte: Horizonte | None,
               uso_previsto: str | None,
               restricciones: Sequence[Restriccion]) -> Confirmacion:
    """El `ProblemSpec` solo si no falta nada. Un sitio, para las dos rutas."""
    if preguntas or bloqueos or tarea is None:
        return Confirmacion(problema=None, propuesta=propuesta,
                            preguntas=tuple(preguntas), bloqueos=tuple(bloqueos),
                            pistas=pistas)
    problema = ProblemSpec(
        problem_id=problem_id,
        target=objetivo,
        task=tarea,
        observation_unit=str(unidad_de_observacion),
        classes=tuple(etiquetas) if etiquetas else None,
        positive_label=clase_positiva,
        predictors=predictores,
        prediction_time=momento_de_prediccion,
        horizon=horizonte,
        intended_use=uso_previsto,
        constraints=tuple(restricciones),
    )
    return Confirmacion(problema=problema, propuesta=propuesta, pistas=pistas)

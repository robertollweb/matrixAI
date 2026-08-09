# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 C10 — DenseNetworkGenerator: genera NetworkSpec y textos .mxai/.mxtrain desde intención humana."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
from matrixai import limits as _limits
from matrixai.generation import parse_field_specs, strip_field_specs
from matrixai.training import architecture_policy as _architecture_policy
from matrixai.training.categorical import expand_categoricals

# GEN C2: a declared categorical with at most this many values becomes one-hot
# columns here (dense model); above it, it should be an embedding (composite path).
# Keeps one-hot column counts sane; aligns the old composite `vocab > 5`.
_ONEHOT_MAX = 12


class DenseNetworkGeneratorError(ValueError):
    """Fallo del generador. `details` lleva el payload estructurado cuando el
    motivo es un tope superado (`limits.limit_error`), para que quien lo capture
    pueda responder con "qué tope, cuánto te pasas y dónde se sube" en vez de
    solo un texto."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class DenseNetworkGenerationResult:
    prompt: str
    network_name: str
    input_name: str
    input_dim: int
    output_type: str
    output_activation: str
    loss_type: str
    hidden_layers: list[tuple[int, str]]
    output_units: int
    labels: list[str]
    mxai_text: str
    training_text: str
    dataset_template_text: str
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # GEN C2: categoricals declared in the prompt that were materialized as one-hot
    # ({campo: [valores humanos ordenados]}). The canonical human input stays the
    # original field; the .mxai/training_text carry the expanded columns. Empty when
    # no categorical was declared. Source of truth for the export's field_categories.
    field_categories: dict[str, list[str]] = field(default_factory=dict)
    # GEN C3: scalar ranges declared in the prompt ({campo: [min, max]}). NOT written
    # into the .mxai VECTOR type (training data is normalized to [0,1]; a raw range
    # there would make the training verifier reject every normalized row). This is
    # metadata only — the canonical source for the Studio/export's field_ranges.
    field_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # GEN C3: declared semantic types ({campo: "boolean"|"integer"}) for fields whose
    # .mxai type stays a bare Scalar (same reasoning as field_ranges). Canonical source
    # for the Studio/export's field_types.
    field_types: dict[str, str] = field(default_factory=dict)
    # CONTRATO 64 C4 — por qué esta red y no otra: entradas de la regla,
    # presupuesto, candidatas, límites aplicados y origen. Viaja en el RESULTADO
    # y no como estado del generador: una instancia compartida entre hilos haría
    # que dos generaciones simultáneas se pisaran la decisión.
    architecture_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def network_spec(self) -> NetworkSpec:
        all_layers = list(self.hidden_layers) + [(self.output_units, self.output_activation)]
        specs: list[DenseLayerSpec] = []
        dim = self.input_dim
        for i, (units, activation) in enumerate(all_layers, start=1):
            specs.append(DenseLayerSpec(
                index=i,
                units=units,
                activation=activation,
                input_shape=[dim],
                output_shape=[units],
            ))
            dim = units
        output_name = _output_name(self.output_activation)
        return NetworkSpec(
            name=self.network_name,
            input=self.input_name,
            layers=specs,
            output=output_name,
            output_type_str=self.output_type,
        )


class DenseNetworkGenerator:
    _REGRESSION_KEYWORDS = [
        "precio", "price", "predecir", "estim", "regres",
        "temperatura", "temperature", "consumo", "duracion",
        "valor", "value", "cantidad", "amount",
    ]
    _BINARY_KEYWORDS = [
        "spam", "fraude", "fraud", "binario", "binary",
        "dos clases", "two classes", "positivo o negativo",
        "detec", "detect",
    ]
    _MULTICLASS_KEYWORDS = [
        "clasifica", "classify", "categoriza", "categor",
        "multiclase", "multiclass", "clases", "categorias",
    ]
    # Strong, unambiguous classification intent. These outrank regression keywords:
    # a clinical feature named "temperatura" must not flip an explicit classifier
    # ("clasificación multiclase") into a regressor.
    _STRONG_CLASSIFICATION_KEYWORDS = [
        "clasifica", "classify", "classification", "multiclase", "multiclass",
        "categoriza", "categorize",
    ]

    _FIELD_RE = re.compile(
        r"(?:campos|fields|variables|features|entradas|inputs)\s*(?::|=|son|are)?\s*(?P<fields>[^.;\n]+)",
        re.IGNORECASE,
    )
    _LABEL_RE = re.compile(
        r"(?:labels?|etiquetas|clases|categorias|categories|niveles?|levels?)\s*(?::|=|son|are)?\s*(?P<labels>[^.;\n]+)",
        re.IGNORECASE,
    )
    # Truncates the captured label region at descriptive connectors so trailing
    # prose (architecture/feature descriptions) is not parsed as class names.
    _LABEL_STOP_RE = re.compile(
        r"\s+(?:con|usando|mediante|para|seg[uú]n|a\s+partir\s+de|y\s+una|"
        r"with|using|from|based\s+on|"
        r"features?|caracter[ií]sticas?|variables?|columnas?|atributos?)\b.*$",
        re.IGNORECASE | re.DOTALL,
    )
    _NAME_RE = re.compile(
        r"\b(?:network|red|modelo|model)\s*(?:llamad[ao]|named|called)?\s*(?P<name>[A-Za-z_][\w]*)",
        re.IGNORECASE,
    )
    _ENTITY_RE = re.compile(
        r"\b(?:entidad|entity|entrada|input)\s*(?:llamad[ao]|named|called)?\s*(?P<name>[A-Za-z_][\w]*)",
        re.IGNORECASE,
    )
    # "12 capas", "12 capas Dense ocultas", "12 hidden layers", "12 layers". La palabra
    # "Dense"/"densas"/"ocultas" entre medias NO debe romper la detección (antes exigía
    # "capas ocultas" juntas → "capas Dense ocultas" no casaba y caía al default).
    _DEPTH_RE = re.compile(
        r"(\d+)\s*(?:capas|hidden\s+layers?|layers?)\b",
        re.IGNORECASE,
    )
    # M12: el tope de profundidad ahora es configurable en runtime (limits.cap(..,"max_depth"));
    # este valor es solo el default histórico (perfil equilibrado), conservado como referencia.
    _MAX_EXPLICIT_DEPTH = 12
    # M12 — ancho de capa desde el prompt ("2048 unidades", "units=2048"). Sin esto el
    # ancho lo fija el tapering (máx 256) y no se pueden pedir redes grandes (la GPU no
    # se carga). El tope es de cordura (evita typos tipo units=999999), no de capacidad:
    # la máquina del usuario manda (ver M12 en MEJORAS_FUTURAS).
    _WIDTH_RE = re.compile(
        r"units?\s*[:=]\s*(\d+)|(\d+)\s*(?:unidades|neuronas|units|de\s+ancho)",
        re.IGNORECASE,
    )
    _MAX_EXPLICIT_WIDTH = 16384
    _EPOCHS_RE = re.compile(
        r"(?:\bepochs?\b|\bepocas?\b)\s*[:=]?\s*(\d+)|(\d+)\s*(?:\bepochs?\b|\bepocas?\b)",
        re.IGNORECASE,
    )
    _EARLY_STOP_RE = re.compile(
        r"early[_\s-]?stop\w*\s+patience\s*[:=]?\s*(\d+)(?:\s+metric\s*[:=]?\s*([A-Za-z_][\w.]*))?",
        re.IGNORECASE,
    )
    _MAX_EPOCHS = 1000
    _DEFAULT_EPOCHS = 50

    def generate(
        self,
        prompt: str,
        *,
        input_fields: list[str] | None = None,
        labels: list[str] | None = None,
        network_name: str | None = None,
        input_name: str | None = None,
        hidden_layers: list[tuple[int, str]] | None = None,
        # CONTRATO 64 C2 — filas del dataset con el que se va a entrenar, si se
        # conocen. NO agrandan la red: solo ponen techo al presupuesto de
        # parámetros (diez ejemplos por parámetro). 0 = flujo por prompt, donde
        # todavía no hay dataset y solo manda el tope del perfil.
        rows: int = 0,
        # CONTRATO 64 C3/C4 — quién propuso `hidden_layers`, cuando lo pone el
        # llamante: 'llm' o 'user_override'. El generador no puede distinguirlos
        # por sí mismo y la decisión registrada tiene que decir la verdad sobre
        # su origen.
        hidden_layers_source: str = "",
        # En que MAQUINA se va a entrenar esto, si quien llama lo sabe.
        #
        # La estimacion de recursos iba con `device="cpu"` FIJO, y el
        # comentario lo justificaba diciendo que «el core no detecta
        # hardware». Es cierto que no lo detecta AQUI —y sigue sin
        # hacerlo: se lo dicen—, pero el numero que se adjuntaba
        # describia una CPU con lote 8 aunque hubiera una GPU delante.
        # Medido: el mismo modelo estima 0,000008 GiB con lote 8 (cpu) y
        # 0,003742 GiB con lote 16384 (cuda), x468.
        #
        # Por defecto "cpu", que es lo que habia: quien no lo sepa no
        # cambia de comportamiento.
        device: str = "cpu",
    ) -> DenseNetworkGenerationResult:
        clean = " ".join(prompt.strip().split())
        if not clean:
            raise DenseNetworkGeneratorError("DenseNetworkGenerator requires a non-empty prompt")

        # GEN C4: task/label resolution shared with the composite generator (invariant
        # 5) — an explicit ProbabilityMap[...]/Label[...] bracket wins over caller
        # labels and task keywords; 2 declared labels mean 2-class softmax, never
        # the 1-unit sigmoid (see resolve_task_and_labels).
        task, resolved_labels, label_warnings = resolve_task_and_labels(self, clean, labels)
        # GEN C1/C2/C3: honor explicit field-type declarations from the prompt (shared
        # with the composite generator so both use the SAME policy — invariant 5).
        resolved_fields, specs_by_name, field_ranges, field_types, spec_warnings = \
            resolve_prompt_fields(self, prompt, input_fields)
        input_dim = len(resolved_fields)
        resolved_name = network_name or self._extract_name(clean) or _default_network_name(task)
        resolved_entity = input_name or self._extract_entity(clean) or "Input"

        output_activation, output_type, output_units, loss_type = _output_config(task, resolved_labels)

        # CONTRATO 64 C2 — las categóricas declaradas se resuelven ANTES de decidir
        # la arquitectura. La expansión one-hot ocurre más abajo, pero la política
        # necesita la dimensión EFECTIVA de entrada: un campo categórico de 8
        # valores no entra con un peso por neurona sino con ocho, y dimensionar
        # por número de campos subestimaría la red justo donde más importa.
        categoricals = {
            name: list(specs_by_name[name].values or [])
            for name in resolved_fields
            if name in specs_by_name and specs_by_name[name].kind == "categorical"
            and 2 <= len(specs_by_name[name].values or []) <= _ONEHOT_MAX
        }
        effective_dim = _architecture_policy.effective_input_dim(
            input_dim, one_hot_widths={k: len(v) for k, v in categoricals.items()})

        policy_decision = None
        if hidden_layers:
            resolved_hidden = hidden_layers
        else:
            resolved_hidden, policy_decision = self._extract_hidden_layers(
                clean, input_dim, effective_dim=effective_dim,
                output_units=output_units, task=task, rows=rows)
        # M8-A1: sanitize whatever architecture we got (default / prompt / LLM)
        # so no source can emit a dying-ReLU bottleneck before the output.
        resolved_hidden, sanitizer_notes = sanitize_hidden_layers(resolved_hidden)

        # CONTRATO 64 C1/C3 — el presupuesto de parámetros es un límite DURO y se
        # aplica DESPUÉS de resolver la arquitectura, venga de donde venga: del
        # prompt, del LLM, del Modo experto o de la política (invariante 2). Se
        # ESTRECHA en vez de fallar —mismo criterio que `_limits.cap` con la
        # profundidad— pero nunca en silencio: quien pidió una red que no cabe se
        # entera de cuál es el tope y de dónde se sube.
        resolved_hidden, budget_notes, arch_decision = _apply_param_budget(
            resolved_hidden,
            input_dim=effective_dim, output_units=output_units,
            requested_source=(hidden_layers_source or "caller") if hidden_layers
                             else ("prompt_override" if policy_decision is None
                                   else "policy"),
            policy_decision=policy_decision,
            task=task, rows=rows,
        )
        sanitizer_notes = list(sanitizer_notes) + budget_notes
        epochs = self._extract_epochs(clean)
        early_stop = self._extract_early_stop(clean)

        mxai_text = _build_mxai_text(
            resolved_name, resolved_entity, resolved_fields,
            resolved_hidden, output_units, output_activation, output_type,
        )
        out_name = _output_name(output_activation)
        ds_target_type = _dataset_target_type(task, resolved_labels if task == "multiclass" else None)
        training_text = _build_training_text(
            resolved_name, resolved_entity, resolved_fields,
            out_name, ds_target_type, loss_type,
            epochs=epochs, early_stop=early_stop,
        )

        # GEN C2: materialize declared low-cardinality categoricals as one-hot. Reuse
        # expand_categoricals, which rewrites the .mxai VECTOR and the training_text
        # FROM COLUMNS together, so training/inference use the expanded columns while
        # the human canonical input stays the original field. High-cardinality
        # categoricals (> _ONEHOT_MAX) are left for the embedding/composite path.
        # field_ranges/field_types already resolved by resolve_prompt_fields (C3).
        field_categories: dict[str, list[str]] = {}
        # GEN C5: a declared categorical beyond one-hot territory needs the embedding
        # (composite) path — the playground dispatch routes it there. A DIRECT dense
        # call leaves it scalar; say so loudly instead of dropping the declaration
        # in silence (spec_warnings feeds result.warnings below).
        for name in resolved_fields:
            spec = specs_by_name.get(name)
            if (spec is not None and spec.kind == "categorical" and spec.values
                    and len(spec.values) > _ONEHOT_MAX):
                spec_warnings.append(
                    f"'{name}': Categorical de {len(spec.values)} valores "
                    f"(> {_ONEHOT_MAX}) requiere el path composite (embedding); "
                    "el generador denso la deja como escalar."
                )
        template_fields = resolved_fields
        if categoricals:
            expansion = expand_categoricals(mxai_text, training_text, categoricals)
            mxai_text = expansion.mxai_text
            training_text = expansion.training_text
            field_categories = {c: list(v) for c, v in categoricals.items()}
            template_fields = _expanded_field_order(resolved_fields, expansion.groups)
            input_dim = len(template_fields)

        header = template_fields + [out_name]
        # Binary target type is Probability (numeric) — dummy must be float, not a label string.
        # Multiclass target type is Label — dummy is the first label string.
        dummy_target = resolved_labels[0] if task == "multiclass" else "0.0"
        dummy_values = ["0.0"] * len(template_fields) + [dummy_target]
        dataset_template_text = ",".join(header) + "\n" + ",".join(dummy_values) + "\n"

        depth_note = f"depth from prompt ({len(resolved_hidden)} layers)" if self._DEPTH_RE.search(_norm(clean)) else "default depth"
        assumptions = [
            f"DenseNetworkGenerator inferred task={task} ({self._porque_esa_tarea(clean, resolved_labels or None, task)})",
            f"input_dim={input_dim} from {len(resolved_fields)} fields",
            f"hidden architecture: {resolved_hidden} ({depth_note})",
            f"loss={loss_type}, output_activation={output_activation}",
            "Architecture is a heuristic — tune for production",
        ]
        warnings: list[str] = list(sanitizer_notes)
        warnings.extend(spec_warnings)  # GEN C1/C3: rango inválido, categórica <2, etc.
        warnings.extend(label_warnings)  # GEN C4: labels= ignorados / bracket recortado
        # El aviso viejo decia «using defaults» SIN usar ningun valor por
        # defecto, y el modelo salia con un softmax de una unidad que el
        # verificador del core rechaza. Ahora lo arregla —y lo explica—
        # `_multiclase_sin_clases`, en el resolutor que comparten los tres
        # generadores, asi que aqui no queda nada que avisar.

        return DenseNetworkGenerationResult(
            prompt=clean,
            network_name=resolved_name,
            input_name=resolved_entity,
            input_dim=input_dim,
            output_type=output_type,
            output_activation=output_activation,
            loss_type=loss_type,
            hidden_layers=resolved_hidden,
            output_units=output_units,
            labels=resolved_labels,
            mxai_text=mxai_text,
            training_text=training_text,
            dataset_template_text=dataset_template_text,
            assumptions=assumptions,
            warnings=warnings,
            field_categories=field_categories,
            field_ranges=field_ranges,
            field_types=field_types,
            architecture_decision=_con_estimacion_de_recursos(
                arch_decision, mxai_text, training_text, rows=rows, device=device),
        )

    def _extract_hidden_layers(
        self, prompt: str, input_dim: int, *,
        effective_dim: int | None = None,
        output_units: int = 1,
        task: str = "",
        rows: int = 0,
    ) -> tuple[list[tuple[int, str]], Any]:
        """La arquitectura, por orden de autoridad (CONTRATO 64 C3).

        Lo que el prompt pide EXPLÍCITAMENTE (ancho, profundidad) sigue mandando
        sobre la política: es la voz de la persona. Lo que cambia con el contrato
        64 es el último escalón —antes `_default_hidden_layers`, tres tramos por
        dimensión que saturaban en `128-64-32` para cualquier entrada de más de
        diez columnas— y que ahora nada de lo anterior puede saltarse el
        presupuesto duro de parámetros (eso se aplica en `generate`).
        """
        norm = _norm(prompt)
        width = self._extract_width(norm)
        m = self._DEPTH_RE.search(norm)
        if m:
            n = _limits.cap(int(m.group(1)), "max_depth")
            # M12: ancho del prompt → capas uniformes de ese ancho; si no, tapering.
            if width is not None:
                return [(width, "relu")] * n, None
            return _hidden_layers_for_depth(n, input_dim), None
        if width is not None:
            # Ancho explícito sin profundidad → profundidad por defecto con ese ancho.
            n = len(_default_hidden_layers(input_dim))
            return [(width, "relu")] * n, None
        # CONTRATO 64 C2 — política determinista. Sustituye al tapering fijo como
        # ORIGEN del tamaño: ahora hay una regla que se puede explicar y auditar.
        decision = _architecture_policy.propose(
            input_dim=effective_dim if effective_dim is not None else input_dim,
            output_units=output_units, task=task, rows=rows)
        return list(decision.hidden_layers), decision

    def _extract_width(self, norm_prompt: str) -> int | None:
        m = self._WIDTH_RE.search(norm_prompt)
        if not m:
            return None
        raw = int(m.group(1) or m.group(2))
        if raw <= 0:
            return None
        return min(raw, self._MAX_EXPLICIT_WIDTH)

    def _extract_epochs(self, prompt: str) -> int:
        return extract_epochs_from_prompt(prompt)

    def _extract_early_stop(self, prompt: str) -> tuple[int, str] | None:
        return extract_early_stop_from_prompt(prompt)

    # CONTRATO 70 C1 — conectores que separan el OBJETIVO de las COLUMNAS.
    # "predecir X a partir de Y": lo que se predice es X; Y son features.
    # AUDITORIA C70 1a pasada: faltaban la forma CON ARTICULO («a partir
    # del precio») y el «usando» español —estaba solo el «using» ingles—,
    # y sin conector la frase entera cuenta como objetivo, asi que una
    # columna volvia a decidir la tarea. Medido: «detectar averias usando
    # temperatura» salia regresion.
    _CONECTORES_DE_ENTRADA = [
        " a partir de ", " a partir del ", " a partir de la ",
        " a partir de los ", " a partir de las ",
        " segun ", " segun el ", " segun la ",
        " en funcion de ", " en funcion del ", " basandose en ", " mediante ",
        " usando ", " utilizando ", " con base en ",
        " from ", " from the ", " based on ", " using ", " given ",
    ]

    # CONTRATO 70 C2 — formas que describen un SI/NO, sea cual sea el verbo.
    # "predecir si llovera" es una pregunta de si o no, no un numero.
    _PREGUNTA_SI_NO = [
        "predecir si ", "prever si ", "saber si ", "decir si ", "determinar si ",
        "predict if ", "predict whether ", "know if ", "tell if ", "determine whether ",
    ]

    # AUDITORIA C70 1a pasada: la lista de arriba exige que el verbo y el
    # «si» esten PEGADOS, y «quiero un modelo que me diga si un cliente va
    # a impagar» tiene cuatro palabras en medio — caia al valor por
    # defecto, que es regresion. Ir añadiendo conjugaciones («diga»,
    # «digan», «indique»…) seria perseguir el idioma sin alcanzarlo.
    #
    # Asi que se generaliza: un VERBO de prediccion en cualquier parte, y
    # un marcador de si/no DESPUES de el.
    _VERBOS_DE_PREDICCION = [
        "predec", "predic", "prever", "prev", "saber", "sepa", "diga", "dig",
        "determin", "detect", "estim", "anticip", "adivin",
        "predict", "know", "tell", "guess", "forecast",
    ]
    _MARCADORES_SI_NO = [" si ", " whether ", " if "]

    def _es_pregunta_de_si_o_no(self, text: str) -> bool:
        """CONTRATO 70 C2 — ¿lo que se pide es un SI o un NO?

        Dos caminos. El primero es la lista literal de formas pegadas
        («predecir si», «predict whether»), que es la de mas confianza.

        El segundo lo añadio la 1a pasada de auditoria: un verbo de
        prediccion en cualquier parte y un marcador de si/no DESPUES.
        «Quiero un modelo que me diga si un cliente va a impagar» tiene
        cuatro palabras entre el verbo y el «si», y caia al valor por
        defecto —regresion— por no estar pegados.

        Se exige el ORDEN (verbo antes que marcador) a proposito: en «si
        tengo datos, predecir el consumo» el «si» es un condicional y no
        introduce lo que se predice. Sin esa condicion, ese prompt saldria
        binario.
        """
        if _any(text, self._PREGUNTA_SI_NO):
            return True
        for verbo in self._VERBOS_DE_PREDICCION:
            i = text.find(verbo)
            if i < 0:
                continue
            resto = text[i + len(verbo):]
            if _any(resto, self._MARCADORES_SI_NO):
                return True
        return False

    def _objetivo_del_prompt(self, text: str) -> str:
        """CONTRATO 70 C1 — la parte del prompt que dice QUE se predice.

        `_REGRESSION_KEYWORDS` contiene `temperatura`, `consumo`, `valor`,
        `precio`… que son justo los nombres que la gente le pone a las
        COLUMNAS DE ENTRADA. Medido: «detectar una averia rara a partir de
        vibracion, temperatura y velocidad del viento» salia REGRESION —
        tiene "detec" (binaria) en el verbo y "temperatura" en una feature,
        y ganaba la de regresion porque se miraba antes.

        El propio comentario de este metodo ya lo admitia: «which may
        match feature names». Ahora se corta por el conector y las
        palabras de regresion solo cuentan en el objetivo.

        Sin conector, se devuelve el prompt entero: cortar por donde no
        hay corte seria inventarse una separacion.

        AUDITORIA C70 2a pasada: se cortaba en el primer conector de LA
        LISTA, no en el primero del TEXTO. Con dos conectores en la misma
        frase —«detectar averias USANDO temperatura A PARTIR DEL sensor»—
        se cortaba en el segundo y la temperatura se colaba en el
        objetivo: salia regresion. El corte va donde EMPIEZAN las
        columnas, que es el conector mas temprano.
        """
        corte = min(
            (i for i in (text.find(c) for c in self._CONECTORES_DE_ENTRADA) if i > 0),
            default=-1,
        )
        return text[:corte] if corte > 0 else text

    def _porque_esa_tarea(self, prompt: str, labels: list[str] | None, task: str) -> str:
        """CONTRATO 70 C3 — POR QUE se eligio esa tarea, y como cambiarla.

        `assumptions` decia `inferred task=regression`: un resultado sin
        razon. Quien lee eso no sabe si el core entendio su frase o si
        cayo en el valor por defecto — y son cosas muy distintas cuando
        la tarea equivocada devuelve un numero en vez de un si/no.
        """
        text = _norm(prompt).lower()
        if _any(text, self._PREGUNTA_SI_NO) or _any(text, ["clasificar si ", "classify if "]):
            return "la frase pregunta SI o NO"
        if _any(text, self._STRONG_CLASSIFICATION_KEYWORDS):
            return "la frase dice clasificar"
        if _any(self._objetivo_del_prompt(text), self._REGRESSION_KEYWORDS):
            return "lo que se predice es una magnitud"
        if labels is not None:
            return f"vienen {len(labels)} etiquetas declaradas"
        if _any(text, self._BINARY_KEYWORDS) or _any(text, self._MULTICLASS_KEYWORDS):
            return "por el vocabulario de la frase"
        # El unico caso en que NO se ha entendido nada: hay que decirlo,
        # y decir como se arregla.
        return (
            "POR DEFECTO: la frase no dice si se predice un numero o una clase. "
            "Escribe «clasificar si …» para un si/no, «clasificar … en A, B, C» "
            "para varias clases, o nombra la magnitud para una regresion"
        )

    def _detect_task(self, prompt: str, labels: list[str] | None) -> str:
        # Regression keywords in the prompt take priority over LLM-supplied labels,
        # preventing an over-eager LLM from turning "predict price" into a classifier.
        # Exception: explicit classification vocabulary ("clasificación multiclase",
        # "classify") outranks regression keywords, which may match feature names.
        #
        # CONTRATO 70: esa prioridad creo el sesgo CONTRARIO. Se conserva
        # —el caso «predict price» es lo que protege el contrato 59— pero
        # ahora (C1) solo mira el OBJETIVO, no las columnas, y (C2) una
        # pregunta de si/no gana sobre ella.
        text = _norm(prompt).lower()
        if _any(text, self._STRONG_CLASSIFICATION_KEYWORDS):
            # UNA etiqueta no es una multiclase: es que no se sabe cuantas
            # clases hay. Antes esto era `if labels is not None:` y el
            # `else` se tragaba el caso de una sola, devolviendo
            # `multiclass` SIN llegar a leer la frase — que dos lineas mas
            # abajo habria contestado «binary» para «clasificar SI …».
            #
            # Medido el 2026-08-09: 1 de cada 6 generaciones de «clasificar
            # si un cliente cancela…» salia asi, y acababa en un `softmax`
            # de una unidad que el verificador del propio core rechaza.
            # Contar etiquetas solo decide cuando hay etiquetas que contar.
            if labels is not None and len(labels) >= 2:
                return "binary" if len(labels) == 2 else "multiclass"
            # CONTRATO 70 C2 — «clasificar SI …» tambien es un si/no. El
            # contrato dice «sea cual sea el verbo», y sin esto caia en
            # `multiclass` con etiquetas inventadas por defecto: tres
            # clases para una pregunta de dos.
            if _any(text, ["clasificar si ", "categorizar si ", "classify if ", "classify whether "]):
                return "binary"
            # GEN C4 fix: "binari" (stem) catches both "binario" and the
            # feminine-agreement "binaria" ("clasificación binaria"), which the
            # exact word "binario" was silently missing — that miss used to fall
            # through to "multiclass" with 3 fake default labels for the
            # contract's own retrocompat example ("clasificación binaria" a secas).
            if _any(text, ["binari", "binary", "dos clases", "two classes"]):
                return "binary"
            return "multiclass"
        # C2 — «predecir SI …» es un si/no, sea cual sea el verbo. Va
        # ANTES que las palabras de regresion: «predecir» esta entre
        # ellas, asi que sin esto la pregunta no se llega a leer nunca.
        if self._es_pregunta_de_si_o_no(text):
            if labels is not None and len(labels) > 2:
                return "multiclass"
            return "binary"
        # C1 — solo en el OBJETIVO, no en las columnas de entrada.
        if _any(self._objetivo_del_prompt(text), self._REGRESSION_KEYWORDS):
            return "regression"
        if labels is not None:
            if len(labels) == 2:
                return "binary"
            if len(labels) > 2:
                return "multiclass"
        if _any(text, self._BINARY_KEYWORDS):
            return "binary"
        if _any(text, self._MULTICLASS_KEYWORDS):
            return "multiclass"
        return "regression"

    def _extract_bracket_labels(self, prompt: str) -> list[str]:
        """Labels declared EXPLICITLY via `ProbabilityMap[...]`/`Label[...]` in the
        prompt — the most reliable source (avoids capturing prose like "(6 clases)").
        GEN C4: such a bracket is a declared OUTPUT TYPE, so it forces the softmax
        classification path and wins over caller labels (resolve_task_and_labels).
        Returns the FULL declared list, uncapped: the max_labels limit is the
        caller's job, because truncating an explicitly declared output must warn
        (resolve_task_and_labels), never happen silently."""
        mb = re.search(r"(?:ProbabilityMap|Label)\s*\[\s*(?P<labels>[^\]]+)\]", prompt, re.IGNORECASE)
        if not mb:
            return []
        parts = [p for p in re.split(r",|;", mb.group("labels")) if p.strip()]
        bracket = [_identifier(p) for p in parts if _identifier(p)]
        return bracket if len(bracket) >= 2 else []

    def _extract_labels(self, prompt: str) -> list[str]:
        bracket = self._extract_bracket_labels(prompt)
        if bracket:
            m_labels = _limits.get_limit("max_labels")
            return bracket if m_labels is None else bracket[:m_labels]
        m = self._LABEL_RE.search(prompt)
        if not m:
            return self._clases_en_prosa(prompt)
        raw = m.group("labels")
        # Drop trailing descriptive prose so it is not swallowed as labels, e.g.
        # "BAJO MEDIO ALTO con una red profunda…" → "BAJO MEDIO ALTO". The connectors
        # introduce architecture/feature descriptions, not class names.
        raw = self._LABEL_STOP_RE.sub("", raw).strip()
        parts = [p for p in re.split(r",|;|\s+y\s+|\s+and\s+|\s+o\s+|\s+or\s+", raw,
                                     flags=re.IGNORECASE) if p.strip()]
        # Space-separated short labels ("BAJO MEDIO ALTO") when no explicit separator
        # produced a list. Multi-word labels stay intact when comma/connector-separated.
        if len(parts) < 2:
            ws = raw.split()
            if len(ws) >= 2:
                parts = ws
        result = [_identifier(p) for p in parts if _identifier(p)]
        if len(result) < 2:
            return []
        m_labels = _limits.get_limit("max_labels")
        return result if m_labels is None else result[:m_labels]

    # «clasificar la incidencia EN critica, media o baja» — la forma normal
    # de escribirlo, que el extractor NO leia: exige la palabra literal
    # «clases/categorias/niveles/etiquetas» o un bracket. Medido en la 3a
    # pasada: ese prompt salia `ProbabilityMap[class_a, class_b, class_c]`.
    _CLASES_EN_PROSA_RE = re.compile(
        r"\b(?:en|como|entre|into|as|among)\s+(?P<labels>[^.;\n]+)",
        re.IGNORECASE,
    )

    def _clases_en_prosa(self, prompt: str) -> list[str]:
        """Las clases dichas en prosa, SOLO con separador explicito.

        La guarda importa mas que el patron. «clasificar los pedidos en
        funcion del peso» no nombra ninguna clase, y sin exigir una coma o
        un «o» acabaria devolviendo ['funcion', 'del', 'peso'] por el
        reparto por espacios que usa el camino de arriba — tres clases
        inventadas de una frase que no habla de clases.

        Por eso aqui: se parte SOLO por separadores explicitos, nunca por
        espacios, y hacen falta al menos dos partes que sobrevivan.
        """
        m = self._CLASES_EN_PROSA_RE.search(prompt)
        if not m:
            return []
        raw = self._LABEL_STOP_RE.sub("", m.group("labels")).strip()
        partes = [p for p in re.split(r",|;|\s+y\s+|\s+and\s+|\s+o\s+|\s+or\s+", raw,
                                      flags=re.IGNORECASE) if p.strip()]
        if len(partes) < 2:
            return []
        # Una «clase» de varias palabras casi siempre es prosa colada
        # («funcion del peso»). Las clases se nombran con una palabra o dos.
        if any(len(p.split()) > 2 for p in partes):
            return []
        resultado = [_identifier(p) for p in partes if _identifier(p)]
        if len(resultado) < 2:
            return []
        m_labels = _limits.get_limit("max_labels")
        return resultado if m_labels is None else resultado[:m_labels]

    def _extract_fields(self, prompt: str, *, min_count: int = 2) -> list[str]:
        m = self._FIELD_RE.search(prompt)
        if not m:
            return []
        raw = m.group("fields")
        # Si tras la palabra clave queda una CABECERA con dos puntos antes de la lista
        # ("NUMÉRICAS (24), normalizables a su rango físico: vibracion_axial, ..."), la
        # lista real empieza tras el PRIMER ':' — lo anterior es prosa, no son campos.
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        # Quita rangos/anotaciones entre corchetes o paréntesis de cada campo
        # ("vibracion_axial [0-50]" → "vibracion_axial", "(24)" → "").
        raw = re.sub(r"[\[(][^\])]*[\])]", " ", raw)
        parts = re.split(r",|;|\s+y\s+|\s+and\s+", raw, flags=re.IGNORECASE)
        result = [_identifier(p) for p in parts if _identifier(p)]
        return result if len(result) >= min_count else []

    def _extract_name(self, prompt: str) -> str:
        m = self._NAME_RE.search(prompt)
        return _titlecase(m.group("name")) if m else ""

    def _extract_entity(self, prompt: str) -> str:
        m = self._ENTITY_RE.search(prompt)
        return _titlecase(m.group("name")) if m else ""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _detect_task_from_labels(labels: list[str] | None) -> str | None:
    if labels is None:
        return None
    return "binary" if len(labels) == 2 else "multiclass"


def _output_config(task: str, labels: list[str]) -> tuple[str, str, int, str]:
    if task == "regression":
        return "linear", "Scalar", 1, "mse"
    if task == "binary":
        return "sigmoid", "Probability", 1, "binary_cross_entropy"
    label_str = ", ".join(labels)
    return "softmax", f"ProbabilityMap[{label_str}]", len(labels), "cross_entropy"


def _dataset_target_type(task: str, labels: list[str] | None = None) -> str:
    """Return the TARGET type expected in the .mxtrain DATASET (CSV column type, not model output)."""
    if task == "regression":
        return "Scalar"
    if task == "binary":
        return "Probability"
    if labels:
        return f"Label[{', '.join(labels)}]"
    return "Label"


# M8-A1: minimum width for a ReLU hidden layer. A narrow ReLU layer (especially
# the one feeding the softmax, e.g. Dense(n_classes, relu) → Dense(n_classes,
# softmax)) is a dying-ReLU trap: its few units can all die during training,
# collapsing the model to a constant predictor. This floor matches the width the
# deterministic generator already uses by default, and is enforced on EVERY
# source (default / prompt / LLM) so no path can emit the bottleneck.
_MIN_RELU_WIDTH = 16


def sanitize_hidden_layers(
    hidden_layers: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], list[str]]:
    """M8-A1 — widen narrow ReLU hidden layers to avoid dying-ReLU bottlenecks.

    Returns (sanitized_layers, notes) where notes describes any change made, for
    auditability (surfaced in the pipeline). Non-ReLU layers are left untouched.
    """
    out: list[tuple[int, str]] = []
    notes: list[str] = []
    for i, (units, activation) in enumerate(hidden_layers):
        if activation == "relu" and units < _MIN_RELU_WIDTH:
            notes.append(
                f"capa oculta {i + 1}: ancho {units}→{_MIN_RELU_WIDTH} "
                f"(evita un cuello ReLU que colapsaría el modelo)"
            )
            units = _MIN_RELU_WIDTH
        out.append((units, activation))
    return out, notes


# Auditoría C5 [MEDIA]: `sanitize_hidden_layers` de arriba SOLO ensancha
# ReLU demasiado estrechas — no valida tipos ni acota profundidad/ancho, así
# que no basta como defensa para `architecture_hints`, un payload que
# `/api/analyze` acepta TAL CUAL del cliente (`playground.py`, canal C5).
# Reproducido exactamente: `architecture_hints="bad"` → `AttributeError`
# sin capturar (HTTP 500, no un error controlado); `hidden_layers=[("bad",
# "relu")]` → `TypeError` al comparar unidades; `hidden_layers=[(64,
# "relu")]*100` → aceptado sin más, 101 capas Dense generadas, muy por
# encima del tope de 12 capas / 16384 unidades que exige el propio C5. Las
# únicas activaciones que este generador produce para capas OCULTAS en
# cualquier camino (prompt/LLM/default) son "relu" — ver `_parse_layers`
# en `intent_llm.py` y `_default_hidden_layers`/`_hidden_layers_for_depth`
# aquí mismo — así que es el único valor aceptado desde la entrada pública.
_ALLOWED_HIDDEN_ACTIVATIONS = {"relu"}


def validate_architecture_hints(hints: Any) -> tuple[dict[str, Any], str | None]:
    """Valida `architecture_hints` (canal público C5) ANTES de que
    `analyze_playground_request` lo toque — a diferencia de
    `sanitize_hidden_layers` (que asume una forma ya correcta, propuesta
    por el propio core), esta función es la primera línea de defensa contra
    un payload arbitrario de `/api/analyze`.

    Devuelve `({}, None)` si `hints` está ausente/vacío, `(hints_limpios,
    None)` si es válido (unidades coeridas a `int` normal, nunca `bool`), o
    `({}, mensaje)` si no lo es — el caller debe tratar el segundo caso como
    un error controlado (`{"ok": False, "error": mensaje}`), nunca dejar
    que la excepción de más abajo llegue sin capturar al handler HTTP.

    Reauditoría C5 [BAJA]: `not hints` trataba CUALQUIER valor falsy como
    "ausente" — `""`, `[]`, `0`, `False` pasaban con `ok=true` en vez de
    rechazarse por tipo, contradiciendo la validación estricta de más abajo
    (que si exige `isinstance(hints, dict)`). Solo `None` y `{}` son
    "vacío" de verdad; cualquier otro valor falsy pero de tipo incorrecto
    debe rechazarse igual que un valor truthy del tipo incorrecto (p.ej.
    `"bad"`)."""
    if hints is None or hints == {}:
        return {}, None
    if not isinstance(hints, dict):
        return {}, "architecture_hints debe ser un objeto (diccionario)."
    unknown = set(hints) - {"hidden_layers"}
    if unknown:
        return {}, f"architecture_hints: claves no reconocidas {sorted(unknown)!r} (solo se admite 'hidden_layers')."
    if "hidden_layers" not in hints:
        return {}, None
    layers = hints["hidden_layers"]
    if not isinstance(layers, list) or not layers:
        return {}, "architecture_hints.hidden_layers debe ser una lista no vacía de [unidades, activación]."
    if len(layers) > DenseNetworkGenerator._MAX_EXPLICIT_DEPTH:
        return {}, (
            "architecture_hints.hidden_layers supera la profundidad máxima "
            f"({DenseNetworkGenerator._MAX_EXPLICIT_DEPTH} capas)."
        )
    cleaned: list[tuple[int, str]] = []
    for i, item in enumerate(layers):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return {}, f"architecture_hints.hidden_layers[{i}] debe ser un par [unidades, activación]."
        units, activation = item
        if isinstance(units, bool) or not isinstance(units, int):
            return {}, f"architecture_hints.hidden_layers[{i}]: unidades debe ser un entero."
        if not (1 <= units <= DenseNetworkGenerator._MAX_EXPLICIT_WIDTH):
            return {}, (
                f"architecture_hints.hidden_layers[{i}]: unidades fuera de rango "
                f"(1..{DenseNetworkGenerator._MAX_EXPLICIT_WIDTH})."
            )
        if activation not in _ALLOWED_HIDDEN_ACTIVATIONS:
            return {}, (
                f"architecture_hints.hidden_layers[{i}]: activación no permitida "
                f"{activation!r} (solo {sorted(_ALLOWED_HIDDEN_ACTIVATIONS)!r})."
            )
        cleaned.append((int(units), activation))
    return {"hidden_layers": cleaned}, None


def _con_estimacion_de_recursos(
    decision: Any, mxai_text: str, training_text: str, rows: int = 0,
    device: str = "cpu",
) -> dict[str, Any] | None:
    """Adjunta la estimación de recursos a la decisión (CONTRATO 64 C1/C4).

    Es INFORMATIVA, no una puerta: `estimate_model_resources` está definido como
    orientativo y sin umbral, y tratarlo como límite duro sería inventarse un
    tope que nadie ha fijado. El límite duro es `max_params`; esto responde
    "¿cuánta memoria va a necesitar esto?" sin tener que entrenar.

    Nunca hace fallar la generación: si el `.mxai` recién construido no se puede
    parsear aquí, el modelo sigue siendo válido y la decisión se entrega sin la
    estimación en vez de perderse entera.
    """
    if decision is None:
        return None
    datos = decision.to_dict()
    try:
        from matrixai.parser import parse_text  # noqa: PLC0415
        from matrixai.resources import estimate_model_resources  # noqa: PLC0415
        est = estimate_model_resources(
            parse_text(mxai_text), rows=rows, training_text=training_text,
            device=device)
        datos["resource_estimate"] = {
            # INTRÍNSECOS: dependen solo de la arquitectura, así que valen para
            # siempre y para cualquier máquina.
            "param_count": est.param_count,
            "weights_gib": round(est.weights_gib, 6),
            # DEPENDIENTES DEL CONTEXTO: la VRAM cambia con el dispositivo y con
            # el batch efectivo, que a su vez depende de las filas. Aquí se
            # calcula con supuestos EXPLÍCITOS —el core no detecta hardware— y se
            # declaran junto al número: sin ellos, "0,000035 GiB" parecía una
            # verdad sobre la GPU cuando describía una CPU con batch 8.
            "vram_train_gib": round(est.vram_train_gib, 6),
            "effective_batch": est.effective_batch,
            # El supuesto declarado JUNTO al numero, que es lo que hace
            # que el numero se pueda juzgar (contrato 64). Ahora dice la
            # maquina de verdad en vez de decir siempre «cpu».
            "assumptions": {"device": device, "rows": int(rows or 0),
                            "batch": est.effective_batch},
            "orientative": True,
        }
    except Exception:  # noqa: BLE001
        pass
    return datos


def _apply_param_budget(
    hidden_layers: list[tuple[int, str]],
    *,
    input_dim: int,
    output_units: int,
    requested_source: str,
    policy_decision: Any,
    task: str,
    rows: int,
) -> tuple[list[tuple[int, str]], list[str], Any]:
    """Aplica el tope DURO de parámetros y construye la decisión auditable.

    CONTRATO 64 C1/C3/C4. Si la arquitectura pedida no cabe se estrecha por
    mitades —la profundidad no se toca: viene de la complejidad de la tarea y
    quitarla cambia lo que la red puede representar, no solo cuánto— y se emite
    un aviso que nombra el tope y dónde se sube.
    """
    tope = _limits.get_limit("max_params")
    params = _architecture_policy.param_count(input_dim, hidden_layers, output_units)
    notas: list[str] = []
    candidatas: list[dict[str, Any]] = []
    limites: list[str] = []

    if policy_decision is not None:
        candidatas = list(policy_decision.candidates)
        limites = list(policy_decision.limits_applied)
        if policy_decision.limit_error:
            # La política ya determinó que ni la red mínima cabe en el tope duro.
            raise DenseNetworkGeneratorError(
                policy_decision.limit_error["error"],
                details=policy_decision.limit_error,
            )

    original = list(hidden_layers)
    # REAUDITORÍA [MEDIA-BAJA]: la arquitectura SOLICITADA se registra siempre,
    # también cuando viene de fuera (prompt, LLM, Modo experto). Antes, una
    # propuesta externa aceptada dejaba `candidates=[]` y una estrechada empezaba
    # el registro por el primer recorte, omitiendo lo que se había pedido — justo
    # el dato que hace falta para entender la decisión.
    if policy_decision is None:
        candidatas.append({
            "hidden_layers": [[u, a] for u, a in original],
            "params": params,
            "accepted": tope is None or params <= tope,
            "reason": f"solicitada ({requested_source})",
        })
    while tope is not None and params > tope and max(u for u, _ in hidden_layers) > _MIN_RELU_WIDTH:
        hidden_layers = [(max(_MIN_RELU_WIDTH, u // 2), a) for u, a in hidden_layers]
        params = _architecture_policy.param_count(input_dim, hidden_layers, output_units)
        candidatas.append({"hidden_layers": [[u, a] for u, a in hidden_layers],
                           "params": params, "accepted": params <= tope,
                           "reason": "estrechada por el tope de parámetros"})

    # REAUDITORÍA [ALTO] — comprobación FINAL. El bucle de arriba termina al
    # llegar al suelo de ancho, y hasta ahora nadie volvía a mirar si la red
    # había quedado dentro del tope: con `max_params=100` salía una red de
    # cientos de parámetros, sin error y sin nada anotado. `max_params` y
    # `_MIN_RELU_WIDTH` no se pueden cumplir a la vez; el límite duro manda.
    if tope is not None and params > tope:
        raise DenseNetworkGeneratorError(
            _limits.limit_error("max_params", params)["error"],
            details=_limits.limit_error("max_params", params),
        )

    if hidden_layers != original:
        limites.append("max_params")
        pedidos = _architecture_policy.param_count(input_dim, original, output_units)
        _m = _architecture_policy.miles
        notas.append(
            f"La arquitectura solicitada ({'-'.join(str(u) for u, _ in original)}, "
            f"{_m(pedidos)} parámetros) supera el tope del perfil "
            f"({_m(tope)} parámetros); se ha estrechado a "
            f"{'-'.join(str(u) for u, _ in hidden_layers)} ({_m(params)} "
            "parámetros). Puedes subirlo en Ajustes → Límites."
        )

    if policy_decision is not None and hidden_layers == list(policy_decision.hidden_layers):
        decision = _architecture_policy.ArchitectureDecision(
            hidden_layers=list(hidden_layers), params=params,
            source=policy_decision.source, inputs=dict(policy_decision.inputs),
            budget=dict(policy_decision.budget), candidates=candidatas,
            limits_applied=limites, rationale=policy_decision.rationale,
            warnings=list(policy_decision.warnings),
        )
    else:
        # La arquitectura NO viene de la política (prompt explícito, LLM, Modo
        # experto). Se registra igual —"la decisión se registra o no existe"— con
        # su origen real y el presupuesto contra el que se comprobó.
        decision = _architecture_policy.ArchitectureDecision(
            hidden_layers=list(hidden_layers), params=params,
            source=requested_source,
            inputs={"input_dim": input_dim, "output_units": output_units,
                    "task": task, "rows": int(rows or 0)},
            budget=_architecture_policy.budget_for(rows),
            candidates=candidatas, limits_applied=limites,
            rationale=(f"arquitectura de origen '{requested_source}': "
                       + "-".join(str(u) for u, _ in hidden_layers)
                       + f", {_architecture_policy.miles(params)} parámetros"),
            warnings=list(policy_decision.warnings) if policy_decision else [],
        )
    # Los avisos de la POLÍTICA (hoy: dataset pequeño para su dimensión de
    # entrada) viajan con los del presupuesto. Sin esto se quedaban dentro de la
    # decisión, que no la ve nadie salvo en Modo experto — y el aviso más útil
    # del contrato es precisamente para quien no está en Modo experto.
    if policy_decision is not None:
        notas = list(policy_decision.warnings) + notas
    return hidden_layers, notas, decision


def _default_hidden_layers(input_dim: int) -> list[tuple[int, str]]:
    if input_dim <= 4:
        return [(32, "relu"), (16, "relu")]
    if input_dim <= 10:
        return [(64, "relu"), (32, "relu"), (16, "relu")]
    return [(128, "relu"), (64, "relu"), (32, "relu")]


def _hidden_layers_for_depth(n: int, input_dim: int) -> list[tuple[int, str]]:
    """Generate exactly n hidden layers with a tapering unit schedule."""
    base = 64 if input_dim <= 4 else (128 if input_dim <= 10 else 256)
    return [(max(16, base >> (i // 2)), "relu") for i in range(n)]


def _default_labels(task: str) -> list[str]:
    if task == "binary":
        return ["negative", "positive"]
    if task == "multiclass":
        return ["class_a", "class_b", "class_c"]
    return []


def _default_fields() -> list[str]:
    return ["feature_1", "feature_2", "feature_3", "feature_4"]


def _expanded_field_order(fields: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Field order after one-hot expansion: each categorical field replaced, in
    place, by its ordered one-hot columns (from ExpansionResult.groups)."""
    out: list[str] = []
    for f in fields:
        out.extend(groups[f] if f in groups else [f])
    return out


def resolve_task_and_labels(dg, prompt, labels):
    """Task + label resolution shared by the dense AND composite generators
    (invariant 5: same policy in both paths). Returns (task, labels, warnings).

    GEN C4 (+audit): an explicit `ProbabilityMap[...]`/`Label[...]` bracket in the
    prompt with >=2 labels is a DECLARED OUTPUT TYPE and always wins (invariant 1):
    - over caller/LLM ``labels`` (they are ignored, with a warning if they differ);
    - over task keyword detection ("predecir precio ... ProbabilityMap[A,B,C]" is
      a classifier, not a regressor — before the audit the extracted labels were
      silently dropped and the output stayed linear/mse);
    - with EXACTLY 2 labels this means 2-class softmax + cross_entropy +
      Label[...] target, never the 1-unit sigmoid. Reusing "multiclass" for n=2
      is deliberate: `_output_config`/`_dataset_target_type` already handle any
      label count generically, so no separate "labeled binary" task is needed.
    The max_labels limit still applies to a declared bracket, but with an explicit
    warning — never a silent truncation of an output the user spelled out.

    Without such a bracket, resolution is unchanged (retrocompat): caller labels,
    then prose labels, then task keywords; a bare "clasificación binaria" prompt
    still gets the 1-unit sigmoid.
    """
    warnings: list[str] = []
    bracket = dg._extract_bracket_labels(prompt)
    if len(bracket) >= 2:
        m_labels = _limits.get_limit("max_labels")
        if m_labels is not None and len(bracket) > m_labels:
            warnings.append(
                f"El prompt declara {len(bracket)} labels explícitos pero el límite "
                f"max_labels={m_labels} los recorta a {bracket[:m_labels]} "
                f"(sube MATRIXAI_MAX_LABELS o el perfil de límites para conservarlos)."
            )
            bracket = bracket[:m_labels]
        if len(bracket) >= 2:
            if labels is not None and [_identifier(str(l)) for l in labels] != bracket:
                warnings.append(
                    f"labels={list(labels)} ignorados: el prompt declara explícitamente "
                    f"ProbabilityMap/Label{bracket} y el prompt gana (invariante 1)."
                )
            return "multiclass", bracket, warnings
    task = dg._detect_task(prompt, labels)
    # De DONDE salen las clases, que no es lo mismo que cuales son.
    de_ejemplo = False
    resolved_labels = list(labels or dg._extract_labels(prompt) or [])
    if not resolved_labels:
        resolved_labels = _default_labels(task)
        de_ejemplo = task == "multiclass" and bool(resolved_labels)
    # Inventarse las clases EN SILENCIO es lo peor de las dos opciones.
    #
    # 3a pasada de auditoria, medido: «clasificar el nivel de riesgo del
    # paciente a partir de la edad y la tension» devolvia
    # `ProbabilityMap[class_a, class_b, class_c]` con CERO avisos. Nadie
    # habia nombrado esas clases; son marcadores de posicion, y quien lo
    # lea se lleva un modelo cuyas salidas no significan nada.
    #
    # El aviso viejo no cubria esto: solo saltaba con MENOS de dos
    # etiquetas, y los valores por defecto son tres. El exportador si lo
    # advierte («A downloadable model must name its classes») — pero eso
    # llega despues de entrenar, que es tarde.
    if de_ejemplo:
        warnings.append(
            f"No se han podido leer las clases del prompt, asi que se usan las de ejemplo "
            f"{resolved_labels}: NO son tus clases, son marcadores de posicion. "
            f"Nombralas en el prompt —«clasificar en alto, medio o bajo», o "
            f"ProbabilityMap[alto, medio, bajo]— antes de entrenar."
        )
    # Que la propuesta del LLM se descarte NO puede pasar en silencio: es
    # quien la lea el que tiene que poder decidir si la frase estaba mal
    # escrita o si el modelo esta mal construido.
    if labels is not None and 0 < len(labels) < 2:
        warnings.append(
            f"El LLM propuso {len(labels)} clase(s) {list(labels)}, y con menos de dos "
            f"no hay un conjunto de clases: decide la frase, y aqui sale "
            f"task={task}. Para varias clases, nombralas en el prompt como "
            f"ProbabilityMap[una, otra, otra_mas]."
        )
    task, resolved_labels, aviso = _multiclase_sin_clases(dg, prompt, task, resolved_labels)
    if aviso is not None:
        warnings.append(aviso)
    return task, resolved_labels, warnings


def _multiclase_sin_clases(dg, prompt, task, resolved_labels):
    """Una multiclase con MENOS DE DOS clases no es una multiclase.

    Encontrado el 2026-08-09 conduciendo la interfaz por fases, y medido
    despues por el API: **1 de cada 6** generaciones de «clasificar si un
    cliente cancela su suscripcion…» devolvia un modelo que el propio
    verificador del core RECHAZA:

        Verifier Agent  error  softmax output requires units >= 2, got units=1
        Type Check      error  (el mismo)

    El LLM proponia `multiclass` con UNA etiqueta, y nadie lo corregia:
    `labels or _extract_labels(...) or _default_labels(task)` solo cae a
    los valores por defecto cuando la lista viene VACIA, y una lista de un
    elemento es verdadera. `_output_config` remataba con
    `units=len(labels)` — un `softmax` de una sola salida, que devuelve
    siempre 1 y no es una clasificacion de nada.

    Y el aviso que se emitia MENTIA: decia «using defaults» sin usar
    ningun valor por defecto. Media verdad tranquilizadora.

    Que se hace en su lugar, por orden:

    1. **Si la frase pregunta un SI o un NO** —la deteccion del contrato
       70, que es la que sabe de esto— se construye una **binaria**. Es la
       respuesta correcta a la pregunta que se hizo, y sale un modelo
       valido: un `sigmoid` de una unidad SI significa algo.
    2. **Si no**, se usan de verdad las clases por defecto, y el aviso lo
       dice nombrandolas. Inventarse clases es peor que no tenerlas, asi
       que se declara que son inventadas y que hay que escribirlas.

    Lo que NO se hace es emitir un `softmax` de menos de dos unidades.
    Ese modelo no es corregible aguas abajo: no hay dato que lo salve.
    """
    if task != "multiclass" or len(resolved_labels) >= 2:
        return task, resolved_labels, None

    tenia = list(resolved_labels)
    if dg._es_pregunta_de_si_o_no(_norm(prompt)):
        return "binary", _default_labels("binary"), (
            f"El LLM propuso una clasificacion multiclase con {len(tenia)} clase(s) "
            f"{tenia}, y con menos de dos no hay multiclase. La frase pregunta un si "
            f"o un no, asi que se construye una BINARIA (sigmoid). Para varias clases, "
            f"nombralas en el prompt: ProbabilityMap[una, otra, otra_mas]."
        )

    porDefecto = _default_labels("multiclass")
    return task, porDefecto, (
        f"El LLM propuso una clasificacion multiclase con {len(tenia)} clase(s) "
        f"{tenia}, y con menos de dos no hay multiclase. Se usan clases de ejemplo "
        f"{porDefecto} para que el modelo valide: NO son tus clases. Nombralas en el "
        f"prompt como ProbabilityMap[una, otra, otra_mas]."
    )


def resolve_prompt_fields(dg, prompt, input_fields):
    """Field resolution + C3 metadata shared by the dense AND composite generators.

    Honors the prompt's explicit type declarations (invariants 1 & 5): the typed
    fields always survive (even when a caller passes input_fields), clean names come
    from parse_field_specs (not the mangling _extract_fields), and bare untyped
    fields are added from the prompt with the typed declarations stripped.

    GEN C5: caller/LLM ``input_fields`` are sanitized here — a name that is not a
    valid identifier ("customer age") would be written verbatim into the .mxai
    VECTOR and crash the parser downstream, so it is normalized with `_identifier`
    (or dropped if nothing survives), with a warning. Valid names pass verbatim.

    Returns ``(resolved_fields, specs_by_name, field_ranges, field_types, warnings)``.
    field_ranges/field_types are METADATA only (never written into the .mxai VECTOR;
    training data is [0,1]-normalized — see GENERACION_TIPOS_PROMPT_CONTRACT.md C3).
    """
    parsed = parse_field_specs(prompt)
    specs_by_name = parsed.by_name()
    warnings = list(parsed.warnings)
    typed_names = [f.name for f in parsed.fields]
    # SECUENCIAS_PRODUCTO C2 (auditoría [MEDIA]): collapsing ALL whitespace
    # (incl. newlines) destroyed the `\n` boundary _FIELD_RE relies on to stop
    # capturing ("variables: edad, ingreso\nOUTPUT clase: ProbabilityMap[...]"
    # swallowed the OUTPUT line into the field list and lost "edad, ingreso"
    # entirely). Normalize horizontal whitespace PER LINE, keep line breaks.
    bare_clean = "\n".join(" ".join(line.split()) for line in strip_field_specs(prompt).split("\n"))
    bare_names = [n for n in (dg._extract_fields(bare_clean) or []) if n not in specs_by_name]
    caller_fields: list[str] = []
    for raw in (input_fields or []):
        name = str(raw)
        if not _VALID_FIELD_NAME_RE.fullmatch(name):
            fixed = _identifier(name)
            if not fixed:
                warnings.append(
                    f"input_fields: nombre inválido {name!r} descartado "
                    "(no queda un identificador utilizable)."
                )
                continue
            warnings.append(
                f"input_fields: nombre inválido {name!r} normalizado a '{fixed}' "
                "(el nombre crudo rompería el .mxai)."
            )
            name = fixed
        caller_fields.append(name)
    resolved_fields: list[str] = []
    for n in caller_fields + typed_names + bare_names:
        if n not in resolved_fields:
            resolved_fields.append(n)
    if not resolved_fields:
        resolved_fields = list(dg._extract_fields(" ".join(prompt.split())) or _default_fields())

    field_ranges: dict[str, tuple[float, float]] = {
        name: specs_by_name[name].range
        for name in resolved_fields
        if name in specs_by_name and specs_by_name[name].kind == "scalar"
        and specs_by_name[name].range is not None
    }
    field_types: dict[str, str] = {}
    for name in resolved_fields:
        spec = specs_by_name.get(name)
        if spec is None:
            continue
        if spec.kind == "boolean":
            field_types[name] = "boolean"
        elif spec.kind == "scalar" and spec.integer:
            field_types[name] = "integer"
    return resolved_fields, specs_by_name, field_ranges, field_types, warnings


def _default_network_name(task: str) -> str:
    return {"regression": "Regressor", "binary": "BinaryClassifier", "multiclass": "Classifier"}[task]


def _output_name(activation: str) -> str:
    return {"linear": "predicted_value", "sigmoid": "predicted_prob", "softmax": "predicted_class"}.get(activation, "output")


def _build_mxai_text(
    network_name: str,
    input_name: str,
    fields: list[str],
    hidden_layers: list[tuple[int, str]],
    output_units: int,
    output_activation: str,
    output_type: str,
) -> str:
    field_lines = "\n".join(f"  {f}: Scalar" for f in fields)
    layer_lines = "\n".join(f"  LAYER Dense units={u} activation={a}" for u, a in hidden_layers)
    layer_lines += f"\n  LAYER Dense units={output_units} activation={output_activation}"
    out_name = _output_name(output_activation)
    return (
        f"PROJECT {network_name}Project\n\n"
        f"VECTOR {input_name}[{len(fields)}]\n{field_lines}\nEND\n\n"
        f"NETWORK {network_name}\n"
        f"  INPUT {input_name}\n"
        f"{layer_lines}\n"
        f"  OUTPUT {out_name}: {output_type}\n"
        f"END\n\n"
        f"GRAPH\n"
        f"  {input_name} -> {network_name}\n"
        f"END\n\n"
        f"AUDIT\n"
        f"  EXPLAIN {input_name} -> {network_name}\n"
        f"END\n"
    )


def extract_epochs_from_prompt(prompt: str) -> int:
    """EPOCHS from the prompt (`epochs=300`, `300 epocas`), capped at the sanity
    ceiling; default when absent. Shared by the dense and composite generators."""
    m = DenseNetworkGenerator._EPOCHS_RE.search(_norm(prompt))
    if m:
        n = int(m.group(1) or m.group(2))
        return max(1, min(n, DenseNetworkGenerator._MAX_EPOCHS))
    return DenseNetworkGenerator._DEFAULT_EPOCHS


def extract_early_stop_from_prompt(prompt: str) -> tuple[int, str] | None:
    """(patience, metric) from `early_stop patience=20 metric=validation_loss`."""
    m = DenseNetworkGenerator._EARLY_STOP_RE.search(_norm(prompt))
    if m:
        return (max(1, int(m.group(1))), m.group(2) or "validation_loss")
    return None


def _build_training_text(
    network_name: str,
    input_name: str,
    fields: list[str],
    output_name: str,
    dataset_target_type: str,
    loss_type: str,
    epochs: int = 50,
    early_stop: tuple[int, str] | None = None,
) -> str:
    field_list = "[" + ", ".join(fields) + "]"
    loss_name = f"{network_name}Loss"
    optimizer_name = f"{network_name}Optimizer"
    lines = [
        f"MODEL {network_name}Project.mxai",
        "",
        f"DATASET {network_name}TrainingSet",
        f'  SOURCE csv("{network_name.lower()}.train.csv")',
        f"  INPUT {input_name} FROM COLUMNS {field_list}",
        f"  TARGET {output_name}: {dataset_target_type}",
        "  SPLIT train=0.8 validation=0.2 seed=42",
        "  BATCH size=8",
        "END",
        "",
        f"LOSS {loss_name}",
        f"  TYPE {loss_type}",
        f"  PREDICTION {network_name}",
        f"  TARGET {output_name}",
        "END",
        "",
        f"OPTIMIZER {optimizer_name}",
        "  TYPE sgd",
        "  LEARNING_RATE 0.01",
        f"  UPDATE {network_name}.*",
        "END",
        "",
        "RUN",
        f"  EPOCHS {epochs}",
    ]
    if early_stop is not None:
        patience, metric = early_stop
        lines.append(f"  EARLY_STOP patience={patience} metric={metric}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _norm(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


# GEN C5: what the .mxai parser accepts as a VECTOR field name. Anything else
# written verbatim into the VECTOR block raises MatrixAIParseError downstream.
_VALID_FIELD_NAME_RE = re.compile(r"[A-Za-z_]\w*")


def _identifier(value: str) -> str:
    text = _norm(value).strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text if text and not text[0].isdigit() else ""


def _titlecase(value: str) -> str:
    text = _norm(value).strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:1].upper() + text[1:] if text else ""


def _any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)

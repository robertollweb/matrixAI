# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""SECUENCIAS_PRODUCTO C3 — generación sintética de texto (decisión 5 del
contrato, 3 orígenes de etiqueta):

- **random**: texto de palabras neutras SIN relación con la etiqueta
  (asignada también al azar) — sin señal, declarado como tal
  (`label_origin="synthetic_random"`, honestidad M8-C1).
- **template**: plantillas deterministas con un TOKEN-SEÑAL por clase — el
  "suelo sin LLM" (decisión 5): siempre produce una tarea de clasificación
  real y aprendible, sin depender de ningún proveedor externo
  (`label_origin="synthetic_template"`).
- **LLM (M8v2)**: el LLM redacta N ejemplos de texto por clase en una única
  llamada; se validan (cobertura mínima por clase) y, si fallan o no hay
  proveedor, se cae al determinista de plantillas — nunca al azar (la
  plantilla ya es señal real; degradar a random tirarla seria peor que no
  usar LLM en absoluto). `label_origin="synthetic_llm_examples"`.

Nunca importa `matrixai.text`/tokenizador SALVO para el chequeo de
colisiones post-tokenización (`_tokenized_collisions`): produce pares
(texto, etiqueta) crudos — la tokenización "real" ocurre en el boundary de
train (`CSVTextDataAdapter`), no aquí (invariante 2 del contrato). El
chequeo de colisiones existe porque una señal que sobrevive como TEXTO
puede, aun así, ser indistinguible de otra clase una vez truncada a
`Text[L]` — auditoría C3 [MEDIA].
"""
from __future__ import annotations

import random
import re

_RANDOM_WORDS = [
    "lorem", "ipsum", "dato", "elemento", "valor", "muestra", "registro",
    "articulo", "nota", "entrada", "bloque", "fragmento", "seccion", "unidad",
    "informe", "detalle", "resumen", "consulta",
]

_TEMPLATE_FILLERS = [
    "el producto", "el servicio", "la experiencia", "el pedido", "la entrega",
    "la atencion", "el resultado", "el proceso", "la respuesta", "el equipo",
]

_LLM_TEXT_SYSTEM = (
    "You write short example texts for a text classification dataset. For "
    "EACH class given, write exactly N distinct example sentences a human "
    "would recognize as belonging to that class. Output STRICTLY as lines "
    "in the form 'CLASS: example text' — one per line, nothing else (no "
    "headers, no numbering, no explanations). Keep each example under 200 "
    "characters. Never repeat an example."
)

_LLM_LINE_RE = re.compile(r"^\s*(?P<label>[^:\n]{1,64}):\s*(?P<text>.+?)\s*$")

# Auditoría C3 [MEDIA]: el system prompt YA pide <200 caracteres, pero nada
# lo hacía cumplir — un LLM que ignore la instrucción colaba ejemplos
# larguísimos. Tope real, aplicado en el parseo.
_LLM_MAX_CHARS = 200

# Auditoría C3 [MEDIA]: alfabeto compacto (1 byte/carácter ASCII) para la
# señal de plantilla — hasta 36 clases quedan distinguibles por su PRIMER
# byte incluso si Text[L] trunca todo lo demás (L tan pequeño como 1).
_SIGNAL_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def generate_random_examples(n: int, labels: list[str], seed: int) -> list[tuple[str, str]]:
    """`n` ejemplos con texto de palabras aleatorias sin relación con la
    etiqueta, también asignada al azar."""
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for _ in range(n):
        length = rng.randint(4, 12)
        text = " ".join(rng.choice(_RANDOM_WORDS) for _ in range(length))
        examples.append((text, rng.choice(labels)))
    return examples


def _index_to_signal(index: int) -> str:
    """Base36 corta y determinista: índice 0 -> "0", 1 -> "1", ..., 35 ->
    "z", 36 -> "10", ... Cada índice < 36 produce un ÚNICO carácter (1
    byte ASCII) — el máximo de clases distinguibles cuando Text[L] trunca a
    un solo byte."""
    if index < 0:
        raise ValueError("signal index must be >= 0")
    base = len(_SIGNAL_ALPHABET)
    digits: list[str] = []
    n = index
    while True:
        digits.append(_SIGNAL_ALPHABET[n % base])
        n //= base
        if n == 0:
            break
    return "".join(reversed(digits))


def _signal_token(label: str, labels: list[str]) -> str:
    """Token-señal determinista y ÚNICO POR POSICIÓN en `labels` — mismo
    label produce SIEMPRE el mismo token, clases distintas SIEMPRE reciben
    tokens distintos.

    Auditoría C3 [MEDIA]: la versión anterior derivaba el token de
    `_identifier(label)` (normalización textual) — dos labels que colapsan
    al mismo identificador ("A-B" y "A B" -> ambos "a_b") recibían la MISMA
    señal, así que el modelo no podía distinguir esas clases pese a que la
    generación las declaraba "aprendibles". El índice en `labels` es único
    por construcción, sin depender de cómo se escriba el nombre."""
    return _index_to_signal(labels.index(label))


def _tokenized_collisions(
    rows: list[tuple[str, str]], seq_length: int,
) -> list[tuple[str, str, str]]:
    """Auditoría C3 [MEDIA]: una señal que sobrevive como TEXTO puede, aun
    así, quedar indistinguible de otra clase una vez tokenizada y truncada a
    `Text[L]` (`ByteTokenizer` trunca por el final). Devuelve
    `(texto, label_primera_clase, label_segunda_clase)` por cada colisión
    encontrada entre DOS clases distintas — lista vacía si no hay ninguna."""
    from matrixai.text.tokenizer import ByteTokenizer

    tokenizer = ByteTokenizer(seq_length)
    seen: dict[tuple[int, ...], str] = {}
    collisions: list[tuple[str, str, str]] = []
    for text, label in rows:
        key = tuple(tokenizer.encode(text))
        prior = seen.get(key)
        if prior is not None and prior != label:
            collisions.append((text, prior, label))
        else:
            seen[key] = label
    return collisions


def generate_template_examples(
    n: int, labels: list[str], seed: int, seq_length: int,
) -> list[tuple[str, str]]:
    """Plantillas deterministas: token-señal de la clase (único por índice,
    ver `_signal_token`) SIEMPRE al principio del texto — `ByteTokenizer`
    trunca por el FINAL, así que es la única posición que garantiza que la
    señal sobrevive cuando `Text[L]` es pequeño — seguido de relleno neutro
    hasta agotar el presupuesto de bytes de `seq_length` (nunca se añade
    relleno que se saldría de L). Cobertura uniforme entre clases
    (round-robin), orden barajado con el mismo seed.

    Auditoría C3 [MEDIA]: la versión anterior ni conocía `seq_length` ni
    protegía la posición de la señal — con L pequeño (o incluso moderado
    frente al relleno elegido) la señal quedaba truncada al azar, colapsando
    filas de clases distintas a la MISMA representación tokenizada
    ("siempre aprendible" era falso). Se valida el resultado real tokenizado
    — si `seq_length` es demasiado pequeño para el número de clases
    (físicamente imposible: hacen falta >= len(labels) primeros-bytes
    distintos), se rechaza con un error accionable en vez de devolver un
    dataset que parece aprendible y no lo es."""
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for i in range(n):
        label = labels[i % len(labels)]
        token = _signal_token(label, labels)
        budget = seq_length - len(token.encode("utf-8"))
        text = token
        if budget > 1:
            fillers: list[str] = []
            remaining = budget - 1  # 1 byte para el separador tras la señal
            for _ in range(rng.randint(2, 5)):
                word = rng.choice(_TEMPLATE_FILLERS)
                cost = len(word.encode("utf-8")) + 1  # +1 separador
                if cost > remaining:
                    break
                fillers.append(word)
                remaining -= cost
            if fillers:
                # La señal va SIEMPRE primera (nunca en posición aleatoria):
                # `ByteTokenizer` trunca por el final, así que es la única
                # posición que garantiza supervivencia con L pequeño.
                text = " ".join([token, *fillers])
        examples.append((text, label))

    collisions = _tokenized_collisions(examples, seq_length)
    if collisions:
        text, label_a, label_b = collisions[0]
        raise ValueError(
            f"synthetic_template: Text[{seq_length}] es demasiado pequeño "
            f"para distinguir {len(labels)} clases de forma aprendible — "
            f"{label_a!r} y {label_b!r} colisionan tras tokenizar/truncar "
            f"(p.ej. {text!r}). Sube la longitud del campo Text o reduce el "
            "número de clases."
        )
    rng.shuffle(examples)
    return examples


def parse_llm_text_examples(
    raw: str, labels: list[str], max_chars: int = _LLM_MAX_CHARS,
) -> dict[str, list[str]]:
    """Parsea 'CLASE: texto' por línea. Solo sobreviven líneas cuya CLASE
    (normalizada, sin distinguir mayúsculas/espacios) coincide EXACTAMENTE
    con una de las `labels` declaradas — cabeceras/numeración/explicaciones
    que el LLM haya colado se descartan en vez de intentar interpretarlas.

    Auditoría C3 [MEDIA]: la deduplicación era solo POR CLASE — una
    respuesta con el MISMO texto para dos clases distintas (p.ej. "genial"
    en NEG y en POS) se aceptaba entera, produciendo un ejemplo
    contradictorio y declarado igualmente `synthetic_llm_examples`. La
    deduplicación ahora es GLOBAL entre todas las clases: el primer texto
    (por orden de aparición) se queda con él, cualquier repetición
    posterior — misma clase u otra — se descarta. El límite de longitud del
    system prompt (200 caracteres) también se aplica de verdad aquí, no
    solo se pide."""
    by_label: dict[str, list[str]] = {label: [] for label in labels}
    label_lookup = {label.strip().lower(): label for label in labels}
    seen_texts: set[str] = set()
    for line in raw.splitlines():
        m = _LLM_LINE_RE.match(line)
        if not m:
            continue
        canonical = label_lookup.get(m.group("label").strip().lower())
        if canonical is None:
            continue
        text = m.group("text").strip()
        if not text or len(text) > max_chars:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        by_label[canonical].append(text)
    return by_label


def validate_llm_text_examples(
    by_label: dict[str, list[str]],
    labels: list[str],
    seq_length: int,
    min_per_class: int = 2,
) -> list[str]:
    """Lista de problemas (vacía = usable) — mismo patrón que
    `DomainRules.validate`. Auditoría C3 [MEDIA]: además de la cobertura
    mínima por clase, valida que los ejemplos no colisionen entre sí una vez
    tokenizados/truncados a `Text[seq_length]` (mismo chequeo que
    `generate_template_examples`) — el dedup global de
    `parse_llm_text_examples` evita texto IDÉNTICO entre clases, pero dos
    textos DISTINTOS pueden seguir colapsando a la misma representación tras
    truncar."""
    missing = [l for l in labels if len(by_label.get(l, [])) < min_per_class]
    if missing:
        return [
            f"insufficient examples for classes: {', '.join(missing)} "
            f"(need >= {min_per_class} each)"
        ]
    rows = [(text, label) for label, texts in by_label.items() for text in texts]
    collisions = _tokenized_collisions(rows, seq_length)
    if collisions:
        text, label_a, label_b = collisions[0]
        return [
            f"LLM examples collide after truncating to Text[{seq_length}]: "
            f"{label_a!r} and {label_b!r} both reduce to the same ids "
            f"(e.g. {text!r}) — indistinguishable for the model"
        ]
    return []


def llm_text_examples(prompt: str, labels: list[str], n_per_class: int) -> dict[str, list[str]] | None:
    """Pide al LLM N ejemplos de texto por clase en una única llamada;
    devuelve `None` ante CUALQUIER fallo (sin proveedor, error de red,
    respuesta vacía) — el caller valida y cae al determinista de plantillas
    (mismo patrón que `_llm_domain_rules`, M8v2)."""
    from matrixai.agents.llm_proposal import ChatCompletionsLLMProposalProvider
    try:
        provider = ChatCompletionsLLMProposalProvider.from_env()
        user = (
            f"Problem: {prompt}\n"
            f"Classes: {', '.join(labels)}\n"
            f"Examples per class: {n_per_class}"
        )
        text = provider.complete(_LLM_TEXT_SYSTEM, user)
        return parse_llm_text_examples(text, labels)
    except Exception:  # noqa: BLE001
        return None


def _rows_from_by_label(
    by_label: dict[str, list[str]], n: int, labels: list[str], seed: int,
) -> list[tuple[str, str]]:
    """`n` filas cubriendo las clases round-robin, reciclando el pool de
    ejemplos por clase si `n` pide más filas de las que el LLM redactó."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    counters = {label: 0 for label in labels}
    for i in range(n):
        label = labels[i % len(labels)]
        pool = by_label[label]
        rows.append((pool[counters[label] % len(pool)], label))
        counters[label] += 1
    rng.shuffle(rows)
    return rows


def generate_text_examples(
    mode: str,
    prompt: str,
    n: int,
    labels: list[str],
    seed: int,
    seq_length: int,
    use_llm: bool = False,
) -> tuple[list[tuple[str, str]], str]:
    """Punto de entrada único de los 3 orígenes (decisión 5 del contrato).
    `mode == "random"` → sin señal; cualquier otro `mode` (típicamente
    `"coherent"`, igual que el generador tabular) → plantillas deterministas,
    o LLM si `use_llm=True` y la respuesta valida, con fallback a plantillas
    — NUNCA a random (la plantilla ya es señal real). `seq_length` es la `L`
    de `Text[L]` del modelo — necesaria para garantizar que la señal
    generada (plantilla o LLM) sobrevive a la tokenización/truncado real
    (auditoría C3 [MEDIA]).

    Devuelve `(filas, label_origin)`."""
    if not labels:
        raise ValueError("generate_text_examples requires at least one label")
    if mode == "random":
        return generate_random_examples(n, labels, seed), "synthetic_random"
    if use_llm:
        n_per_class = max(1, -(-n // len(labels)))  # ceil division
        by_label = llm_text_examples(prompt, labels, n_per_class)
        if by_label is not None and not validate_llm_text_examples(by_label, labels, seq_length):
            return _rows_from_by_label(by_label, n, labels, seed), "synthetic_llm_examples"
    return generate_template_examples(n, labels, seed, seq_length), "synthetic_template"

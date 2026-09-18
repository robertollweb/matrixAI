# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""CONTRATO 82-C4 — la plantilla del Space, que es DEL USUARIO.

*«El Space no es de MatrixAI, es de cada usuario: su modelo, sus datos,
su Space»* (Roberto, 2026-08-19). MatrixAI aporta **la plantilla**, como
ya aporta el README — y nada más: esto es TEXTO que viaja dentro del
paquete, no código que cree repositorios en la cuenta de nadie.

Publicar el Space es una casilla al publicar, nunca un efecto
automático. Crear recursos en la cuenta de alguien porque sí es
exactamente lo que este proyecto evita en todo lo demás.

REVISIÓN 2026-09-16 — Space ESTÁTICO con ONNX Runtime Web, no Gradio.

Decisión de Roberto, por lo medido el 2026-08-21 con una cuenta real
(ver `matrixai_studio.publish_hf`, la tabla del 402): un Space de
**Gradio** devuelve 402 Payment Required a cualquier cuenta sin HF PRO
—público o privado, da igual—, así que ningún usuario gratuito podía
llegar nunca a ver la demo. Un Space **estático** se crea en los dos
casos con la misma cuenta gratuita, medido igual ese día.

La plantilla ya no ejecuta nada en un servidor: es HTML + JS que corre
el modelo con ONNX Runtime Web (WASM) **en el navegador de quien
visita el Space**. Eso trae una consecuencia que hay que DECIR, no
callar: un Space estático no tiene Python ni proceso, así que **no
puede ejecutar `matrixai verify`**. Lo que antes era un botón dentro
del Space ahora es una frase con el comando para correrlo en local —que
es donde siempre pudo demostrar algo, un botón "Verify" dentro del
propio paquete solo puede comprobar lo que el propio paquete decida
enseñar.

El HISTORIAL de la plantilla de Gradio (por qué llevaba `gradio` y
`matrixai-core` en su propio `requirements.txt`, por qué el Space
ejecutaba `matrixai verify` dentro) queda en el historial de git de
este fichero — no se repite aquí porque ya no describe lo que se
genera, y un comentario que describe un diseño que ya no existe es
peor que no tenerlo.
"""

from __future__ import annotations

import html as _html

from matrixai.export.wasm_exporter import ORT_WEB_MIN_VERSION

__all__ = ["SPACE_DIR", "space_index_html", "space_readme_md"]

#: Dónde viaja dentro del paquete.
SPACE_DIR = "space"


def space_readme_md(model_name: str) -> str:
    """El `README.md` del Space, con el front-matter que HF entiende.

    Sin front-matter, Hugging Face no reconoce el repositorio como Space
    y lo trata como ficheros sueltos: la plantilla no serviría de nada.

    `sdk: static` + `app_file: index.html` — y SIN `sdk_version`: ese
    campo solo tiene sentido para Gradio/Streamlit, que arrancan un
    runtime con una versión que fijar (`SpaceCardData.sdk_version`, en
    `huggingface_hub.repocard_data`, lo documenta así: "if Gradio/
    Streamlit"). Un Space estático no arranca ningún runtime, HF solo
    sirve los ficheros tal cual — fijar una versión que no se usa sería
    inventar precisión donde no la hay. Confirmado localmente contra el
    `huggingface_hub` instalado: `constants.SPACES_SDK_TYPES` incluye
    `"static"`, y `create_repo(space_sdk="static", ...)` lo acepta.
    """
    return f"""---
title: {model_name}
emoji: 🧮
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# {model_name}

A MatrixAI model you can try **in your browser** — no server, no
account, and nothing you type here ever leaves your machine. It runs
with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
(WASM): the browser downloads the model once, and every prediction
after that runs locally.

## What this page does — and does not — do

- It runs the model **in your browser**. Your inputs are never sent
  anywhere.
- It does **not** verify this package. Checking that these files match
  what the package declares needs Python and runs on your own machine,
  not in a static page:

  ```bash
  matrixai verify .            # integrity + rebuild the dataset
  matrixai verify . --retrain  # …and train again
  ```

Everything the browser needs travels in this repository: the ONNX
model, its inference spec and this page.
"""


def space_index_html(model_name: str) -> str:
    """La página del Space: HTML + JS que corre el modelo con ONNX
    Runtime Web, y nada más.

    NO reimplementa la inferencia: carga `predict.js` —el que genera
    `matrixai.export.wasm_exporter._build_predict_js`, el MISMO que ya
    usa el bundle WASM— y esta página solo añade lo que a `predict.js`
    le falta para ser una demo: un formulario legible por humanos,
    construido en el navegador a partir de `inference_spec.json`, y la
    traducción de sus valores brutos al vector que `predict()` espera.
    Dos generadores de la parte que habla con ONNX Runtime acabarían
    divergiendo — por eso esa parte se REUSA en vez de escribirse dos
    veces; la codificación de campo->vector es la mitad que le falta a
    `predict.js` (que solo sabe de un vector ya construido) y que
    `predict.py` ya resuelve en Python para quien predice en local —
    aquí es su equivalente en JS, sobre el mismo contrato
    (`inference_spec.json`), no una segunda implementación de la
    inferencia en sí.

    `inference_spec.json`, `example_input.json` y `model.onnx` se piden
    con rutas RELATIVAS (`./…`) porque al publicar,
    `matrixai_studio.publish_hf._upload_space` aplana el paquete entero
    (menos `space/`) y ENCIMA el contenido de `space/`: los ficheros
    acaban todos en la raíz del Space, unos junto a otros — así es como
    `model.onnx` llega junto a esta página sin viajar DUPLICADO dentro
    de `space/` en el ZIP descargable (el paquete ya lo lleva una vez,
    en su raíz).

    Dentro del ZIP sin publicar esos ficheros NO están juntos
    (`model.onnx` vive un nivel por encima de `space/`): abrir este
    fichero suelto antes de publicar no los encuentra, y la página lo
    DICE en vez de quedarse en blanco — la misma regla que ya seguía la
    plantilla de Gradio con sus artefactos.

    Solo soporta el caso TABULAR (`inference_spec.json` con `fields`,
    contrato EXPORT C1/C2): una entrada SEQUENCE (texto/tokens) no
    declara `fields`, y la página lo detecta y lo dice en vez de fingir
    un formulario que no puede construir — igual que la plantilla de
    Gradio hacía cuando `inference_spec.json` faltaba o no era legible.
    """
    nombre_seguro = _html.escape(model_name, quote=True)
    return (
        _INDEX_HTML_TEMPLATE
        .replace("__MODEL_NAME__", nombre_seguro)
        .replace("__ORT_WEB_MIN_VERSION__", ORT_WEB_MIN_VERSION)
    )


# Plantilla PLANA (no f-string): el cuerpo es HTML+JS lleno de sus
# propias llaves ({} de JS de punta a punta), así que interpolar con un
# f-string exigiría escapar cada una como {{ }} — más ruido que
# claridad, y un sitio más donde un olvido rompe el HTML generado en
# silencio. Los DOS valores que varían por modelo se sustituyen con
# `.replace()` sobre marcadores que no pueden aparecer por accidente.
_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__MODEL_NAME__ — MatrixAI</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 40rem;
    margin: 2rem auto;
    padding: 0 1rem 3rem;
    line-height: 1.5;
  }
  .does, .does-not {
    border-left: 4px solid;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
  }
  .does { border-color: #2a8c5e; }
  .does-not { border-color: #c9752f; }
  pre {
    background: rgba(127, 127, 127, 0.12);
    padding: 0.75rem;
    overflow-x: auto;
    border-radius: 4px;
  }
  .campo { margin: 0.85rem 0; display: flex; flex-direction: column; gap: 0.25rem; }
  .campo label { font-weight: 600; }
  .pista { font-size: 0.85em; opacity: 0.75; }
  input, select, button { font: inherit; padding: 0.4rem; }
  button {
    cursor: pointer;
    margin-top: 1rem;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
  }
  button:disabled { cursor: wait; opacity: 0.6; }
  #resultado { white-space: pre-wrap; }
</style>
</head>
<body>
<h1>__MODEL_NAME__</h1>

<div class="does">
  <strong>What this page does:</strong> it runs this model
  <strong>in your browser</strong>, with
  <a href="https://onnxruntime.ai/docs/tutorials/web/">ONNX Runtime Web</a>
  (WASM). The model file downloads once; every prediction after that
  runs locally. Nothing you type here is sent anywhere.
</div>

<div class="does-not">
  <strong>What this page does NOT do:</strong> it does not verify this
  package. Checking that these files match what the package declares
  needs Python and runs on your own machine, not in a static page:
  <pre><code>matrixai verify .            # integrity + rebuild the dataset
matrixai verify . --retrain  # …and train again</code></pre>
</div>

<p id="estado" role="status">Loading model…</p>
<form id="formulario" hidden></form>
<button id="predecir" type="button" hidden>Predict</button>
<pre id="resultado" hidden></pre>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@__ORT_WEB_MIN_VERSION__/dist/ort.min.js"></script>
<script src="./predict.js"></script>
<script>
"use strict";

// Raw record -> flat vector. Mirrors predict.py's MatrixAIModel._encode
// exactly (scalar / scalar01 / one_hot / embedding_index) so the same
// inference_spec.json describes both, in Python and in the browser.
// This never touches ONNX Runtime itself — that part is predict.js,
// loaded above and reused as-is.
const TRUE_WORDS = ["true", "1", "yes", "si", "sí", "y", "t"];
const FALSE_WORDS = ["false", "0", "no", "n", "f"];

function parseBoolField(field, value) {
  if (typeof value === "boolean") return value ? 1 : 0;
  const text = String(value).trim().toLowerCase();
  if (TRUE_WORDS.includes(text)) return 1;
  if (FALSE_WORDS.includes(text)) return 0;
  throw new Error('Field "' + field + '": expected a boolean (true/false, yes/no).');
}

function encodeRecord(spec, record) {
  const order = spec.input_order;
  const vector = new Array(order.length).fill(0);
  const indexOf = {};
  order.forEach(function (name, i) { indexOf[name] = i; });

  for (const [field, entry] of Object.entries(spec.fields)) {
    const enc = entry.encoding;
    if (enc === "scalar" || enc === "scalar01") {
      const raw = record[field];
      if (raw === undefined || raw === "")
        throw new Error('Missing required field "' + field + '".');
      let number;
      if (entry.type === "boolean") {
        number = parseBoolField(field, raw);
      } else {
        number = Number(raw);
        if (!Number.isFinite(number))
          throw new Error('Field "' + field + '": expected a number, got "' + raw + '".');
        if (entry.type === "integer" && !Number.isInteger(number))
          throw new Error('Field "' + field + '": expected an integer, got "' + raw + '".');
      }
      let normalized = number;
      if (enc === "scalar") {
        const lo = entry.range[0];
        const hi = entry.range[1];
        const span = (hi - lo) || 1;
        normalized = (number - lo) / span;
      }
      vector[indexOf[field]] = Math.min(1, Math.max(0, normalized));
    } else if (enc === "one_hot") {
      const raw = String(record[field] !== undefined ? record[field] : "");
      const match = entry.values.find(function (v) { return String(v.raw) === raw; });
      if (!match)
        throw new Error('Field "' + field + '": unknown category "' + raw + '".');
      vector[indexOf[match.column]] = 1;
    } else if (enc === "embedding_index") {
      const column = entry.column || field;
      let idx;
      if (entry.vocab) {
        idx = entry.vocab.indexOf(String(record[field]));
        if (idx < 0)
          throw new Error('Field "' + field + '": unknown category "' + record[field] + '".');
      } else {
        idx = parseInt(record[field], 10);
        if (!Number.isInteger(idx) || idx < 0 || idx >= entry.vocab_size)
          throw new Error('Field "' + field + '": index out of range [0, ' + (entry.vocab_size - 1) + '].');
      }
      vector[indexOf[column]] = idx;
    } else {
      throw new Error('Field "' + field + '": encoding "' + enc + '" is not supported by this browser demo.');
    }
  }
  return vector;
}

function decodeOutput(spec, raw) {
  const values = Array.from(raw);
  const out = spec.output || {};
  const labels = out.labels || [];
  if (out.kind === "classification") {
    const result = {};
    labels.forEach(function (label, i) { result[label] = values[i]; });
    return result;
  }
  if (out.kind === "binary_classification") {
    const p = values[0];
    const result = {};
    result[labels[0]] = 1 - p;
    result[labels[1]] = p;
    return result;
  }
  if (out.kind === "regression") {
    let value = values[0];
    if (out.range) {
      const lo = out.range[0];
      const hi = out.range[1];
      value = value * (hi - lo) + lo;
    }
    return value;
  }
  return { values: values };
}

// ---------------------------------------------------------------------
// form, built from inference_spec.json — nothing here is hardcoded per
// model, so this same page works for any tabular MatrixAI export.
// ---------------------------------------------------------------------
function buildField(field, entry) {
  const wrap = document.createElement("div");
  wrap.className = "campo";
  const label = document.createElement("label");
  label.textContent = field;
  label.htmlFor = "campo-" + field;
  wrap.appendChild(label);

  let input;
  if (entry.encoding === "one_hot" || (entry.encoding === "embedding_index" && entry.vocab)) {
    input = document.createElement("select");
    const opciones = entry.encoding === "one_hot"
      ? entry.values.map(function (v) { return v.raw; })
      : entry.vocab;
    for (const opcion of opciones) {
      const option = document.createElement("option");
      option.value = String(opcion);
      option.textContent = String(opcion);
      input.appendChild(option);
    }
  } else if (entry.type === "boolean") {
    input = document.createElement("select");
    for (const valor of ["true", "false"]) {
      const option = document.createElement("option");
      option.value = valor;
      option.textContent = valor;
      input.appendChild(option);
    }
  } else {
    input = document.createElement("input");
    input.type = "number";
    input.step = entry.type === "integer" ? "1" : "any";
    if (entry.encoding === "scalar" && entry.range) {
      input.min = String(entry.range[0]);
      input.max = String(entry.range[1]);
    }
  }
  input.id = "campo-" + field;
  input.name = field;
  wrap.appendChild(input);

  if (entry.encoding === "scalar" && entry.range) {
    const hint = document.createElement("span");
    hint.className = "pista";
    hint.textContent = "range: " + entry.range[0] + " – " + entry.range[1];
    wrap.appendChild(hint);
  }
  return wrap;
}

async function arrancar() {
  const estado = document.getElementById("estado");
  const formulario = document.getElementById("formulario");
  const botonPredecir = document.getElementById("predecir");
  const resultado = document.getElementById("resultado");

  let spec;
  try {
    const r = await fetch("./inference_spec.json");
    if (!r.ok) throw new Error("HTTP " + r.status);
    spec = await r.json();
  } catch (err) {
    estado.textContent =
      "Could not load inference_spec.json (" + err.message + "). This page " +
      "only has what it needs once published as a Hugging Face Space (or " +
      "served together with the rest of the package): it expects " +
      "model.onnx and inference_spec.json to sit right next to it.";
    return;
  }

  if (!spec.fields || Object.keys(spec.fields).length === 0) {
    estado.textContent =
      "This model's input is not a plain table of fields (for example, a " +
      "text/sequence input), so this browser demo cannot build a form for " +
      "it yet. Download the package and run predict.py on your own machine.";
    return;
  }

  // Best-effort: the package's own worked example, if it shipped one
  // (a large model streamed from .mxw may not have one). Missing is not
  // a failure — the form just starts empty.
  let ejemplo = {};
  try {
    const r = await fetch("./example_input.json");
    if (r.ok) ejemplo = await r.json();
  } catch (err) {
    // no example_input.json reachable from here: nothing to prefill.
  }

  for (const [field, entry] of Object.entries(spec.fields)) {
    formulario.appendChild(buildField(field, entry));
    if (Object.prototype.hasOwnProperty.call(ejemplo, field)) {
      document.getElementById("campo-" + field).value = String(ejemplo[field]);
    }
  }

  estado.textContent = "Ready.";
  formulario.hidden = false;
  botonPredecir.hidden = false;

  botonPredecir.addEventListener("click", async function () {
    resultado.hidden = false;
    resultado.textContent = "Predicting…";
    botonPredecir.disabled = true;
    try {
      const record = {};
      for (const campo of formulario.elements) {
        if (campo.name) record[campo.name] = campo.value;
      }
      const vector = encodeRecord(spec, record);
      const raw = await predict(vector);
      resultado.textContent = JSON.stringify(decodeOutput(spec, raw), null, 2);
    } catch (err) {
      resultado.textContent = "Error: " + err.message;
    } finally {
      botonPredecir.disabled = false;
    }
  });
}

arrancar().catch(function (err) {
  document.getElementById("estado").textContent = "Error: " + err.message;
});
</script>
</body>
</html>
"""

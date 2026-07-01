# Referencia de la CLI de MatrixAI

> **Nota de desarrollo:** Este documento fue elaborado como entregable anticipado del corte PR5-C6
> (API REST completa y Studio desacoplado), durante la sesión de cierre de PR4 (2026-05-28).

MatrixAI se invoca como `python -m matrixai <comando> [opciones]` o `matrixai <comando> [opciones]` tras la instalación.

## Opciones comunes

La mayoría de comandos aceptan estas opciones:

| Opción | Descripción |
|--------|-------------|
| `--json` | Imprimir la salida en JSON legible por máquina en lugar de texto legible por humanos |
| `--output, -o <ruta>` | Escribir la salida en un fichero en lugar de stdout |

Los comandos de registro usan por defecto el directorio `matrixai_registry/` del directorio actual. Se puede sobreescribir con `--registry-path`.

---

## Grupos de comandos

- [Proyecto](#proyecto)
- [Desarrollo de modelos](#desarrollo-de-modelos)
- [Análisis y diagnóstico](#análisis-y-diagnóstico)
- [Compilación y backends](#compilación-y-backends)
- [Parámetros](#parámetros)
- [Entrenamiento](#entrenamiento)
- [Servicio](#servicio)
- [Exportación y empaquetado](#exportación-y-empaquetado)
- [Acciones](#acciones)
- [Registro](#registro)
- [Gestión de claves](#gestión-de-claves)
- [Aprendizaje continuo](#aprendizaje-continuo)
- [Refinamiento](#refinamiento)

---

## Proyecto

### matrixai init

Crear un nuevo proyecto MatrixAI a partir de una plantilla.

```
matrixai init <nombre_proyecto> [--template <nombre>] [--output-dir <dir>] [--list-templates]
```

| Argumento / Opción | Por defecto | Descripción |
|--------------------|-------------|-------------|
| `nombre_proyecto` | — | Nombre del proyecto a crear |
| `--template` | `classification` | Plantilla de inicio a usar |
| `--output-dir` | `.` | Directorio donde se crea la carpeta del proyecto |
| `--list-templates` | — | Mostrar plantillas disponibles y salir |

Crea una carpeta de proyecto con un `.mxai` inicial, `.mxtrain`, CSV de ejemplo e instrucciones de inicio rápido.

---

## Desarrollo de modelos

### matrixai prompt

Generar un programa `.mxai` a partir de un prompt en lenguaje natural.

```
matrixai prompt <prompt...> [-o <fichero>] [--semantic] [--json]
```

| Argumento / Opción | Por defecto | Descripción |
|--------------------|-------------|-------------|
| `prompt` | — | Descripción del modelo. Usar `-` para leer desde stdin |
| `-o, --output` | stdout | Escribir el `.mxai` generado en este fichero |
| `--semantic` | — | Imprimir también la especificación semántica intermedia |
| `--json` | — | Imprimir el resultado completo de síntesis como JSON |

---

### matrixai propose

Generar propuestas semánticas candidatas mediante LLM y supervisarlas de forma interactiva.

```
matrixai propose <prompt...> [--max-candidates <n>] [--provider <nombre>] [--json] [--include-compiled]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--max-candidates` | — | Número máximo de candidatos a presentar para supervisión |
| `--provider` | `deterministic` | Backend de propuestas: `deterministic` u `chat-completions-compatible` |
| `--json` | — | Imprimir la decisión como JSON |
| `--include-compiled` | — | Incluir el código Python compilado en la salida JSON |

---

### matrixai supervise-prompt

Supervisar un artefacto semántico generado por prompt o propuesto.

```
matrixai supervise-prompt <prompt...> [--proposal <fichero>] [--json] [--include-compiled]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--proposal` | — | Fichero `.semantic` a supervisar en lugar de sintetizar desde el prompt |
| `--json` | — | Imprimir el informe como JSON |
| `--include-compiled` | — | Incluir el código Python compilado en la salida JSON |

---

### matrixai architect

Generar un fichero `.mxai` a partir de un fichero de especificación semántica.

```
matrixai architect <fichero> [-o <salida>] [--json]
```

---

### matrixai validate

Validar la estructura y semántica de un fichero `.mxai`.

```
matrixai validate <fichero>
```

Sale con código 0 si es correcto, distinto de cero si hay errores.

---

### matrixai validate-plan

Validar una especificación semántica antes de generar el `.mxai`.

```
matrixai validate-plan <fichero>
```

---

### matrixai lint

Analizar ficheros `.semantic` o `.mxai` e informar de diagnósticos.

```
matrixai lint <fichero> [--json] [--strict]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--json` | — | Imprimir diagnósticos como JSON |
| `--strict` | — | Salir con código distinto de cero si hay advertencias |

---

### matrixai format

Imprimir o escribir el formato canónico de ficheros `.semantic` o `.mxai`.

```
matrixai format <fichero> [--check | --write]
```

| Opción | Descripción |
|--------|-------------|
| `--check` | Salir con código distinto de cero si el formato cambiaría el fichero (útil en CI) |
| `--write` | Formatear el fichero en su lugar |

Sin opciones, imprime la salida formateada en stdout.

---

### matrixai typecheck

Inferir y validar tipos MatrixAI para ficheros `.mx` o `.mxai`.

```
matrixai typecheck <fichero> [--json] [--registry-path <dir>] [--allow-mutable-imports]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--registry-path` | `matrixai_registry` | Ruta del registro para resolver declaraciones `IMPORT` |
| `--allow-mutable-imports` | — | Permitir `@latest` y otras etiquetas mutables en `IMPORT` |
| `--json` | — | Imprimir el informe de tipos como JSON |

---

### matrixai parse

Analizar un fichero `.mxai` e imprimir su representación intermedia JSON.

```
matrixai parse <fichero>
```

---

## Análisis y diagnóstico

### matrixai graph

Renderizar el grafo de computación de un fichero `.semantic` o `.mxai`.

```
matrixai graph <fichero> [--format <fmt>] [-o <fichero>]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--format` | `mermaid` | Formato de salida: `mermaid`, `dot` o `json` |
| `-o, --output` | stdout | Escribir el grafo en fichero |

---

### matrixai diagnose

Comparar la salida del intérprete en tiempo de ejecución y el Python compilado para la misma entrada.

```
matrixai diagnose <fichero> --input <fichero_json> [--json] [--tolerance <float>]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--input` | requerido | Fichero de entrada JSON |
| `--tolerance` | `1e-9` | Tolerancia de comparación numérica |
| `--json` | — | Imprimir el informe de diagnóstico como JSON |

---

### matrixai optimize

Analizar un fichero `.mxai` y sugerir optimizaciones.

```
matrixai optimize <fichero> [--json]
```

---

### matrixai permissions

Revisar los permisos de sandbox requeridos por las acciones de un fichero `.mxai`.

```
matrixai permissions <fichero> [--json]
```

---

### matrixai mathematize

Traducir reglas if/else discretas a expresiones continuas de MatrixAI.

```
matrixai mathematize <fichero> [--json]
```

`<fichero>` es un fichero de texto con una regla por línea, o `-` para leer desde stdin.

---

## Compilación y backends

### matrixai compile

Compilar un fichero `.mxai` a un backend ejecutable.

```
matrixai compile <fichero> [--target <backend>] [-o <fichero>]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--target` | `python` | Destino de compilación: `python` o `differentiable-python` |
| `-o, --output` | stdout | Escribir la salida compilada en fichero |

---

### matrixai eval

Evaluar un fichero de expresiones matemáticas `.mx`.

```
matrixai eval <fichero> [--input <json>] [--call <fn>] [--json] [--trace] [--graph]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--input` | — | Datos de entrada JSON: ruta de fichero o cadena JSON directa |
| `--call` | todos | Llamar solo a esta función específica |
| `--trace` | — | Incluir la traza de evaluación en la salida |
| `--graph` | — | Imprimir el grafo de computación |
| `--json` | — | Imprimir la salida como JSON |

---

### matrixai backend-report

Informar sobre la portabilidad de un fichero `.mxai` a un backend diferenciable.

```
matrixai backend-report <fichero> [--target <backend>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--target` | `differentiable_python` | Backend: `differentiable_python` o `torch` |
| `--json` | — | Imprimir el informe como JSON |

---

### matrixai backend-run

Ejecutar un fichero `.mxai` a través de un backend diferenciable.

```
matrixai backend-run <fichero> --input <fichero_json> [--target <backend>] [--parameters <json>] [--device <dev>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--input` | requerido | Fichero de entrada JSON |
| `--target` | `differentiable-python` | Backend: `differentiable-python` o `torch` |
| `--parameters` | — | Fichero JSON de ParameterSet, o `initial` para usar valores iniciales generados |
| `--device` | — | Dispositivo de cómputo: `cpu`, `cuda`, `mps` (cuda/mps requieren `--target torch`) |
| `--json` | — | Imprimir el resultado como JSON |

---

### matrixai backend-parameters

Inspeccionar o validar los parámetros del backend diferenciable para un fichero `.mxai`.

```
matrixai backend-parameters <fichero> [--target <backend>] [--validate <json>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--target` | `differentiable-python` | Backend destino |
| `--validate` | — | Fichero JSON de ParameterSet a validar contra el modelo |
| `--json` | — | Imprimir el informe de parámetros como JSON |

---

## Parámetros

### matrixai init-parameters

Crear un ParameterSet versionado con valores por defecto a partir de un fichero `.mxai`.

```
matrixai init-parameters <fichero> [-o <fichero>] [--parameter-set-id <id>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `-o, --output` | stdout | Escribir el ParameterSet JSON en fichero |
| `--parameter-set-id` | auto | Identificador para el ParameterSet generado |
| `--json` | — | Imprimir el ParameterSet como JSON |

---

### matrixai validate-parameters

Validar un fichero JSON de ParameterSet contra un programa `.mxai`.

```
matrixai validate-parameters <fichero> --params <fichero_json> [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--params` | requerido | Fichero JSON de ParameterSet a validar |
| `--json` | — | Imprimir el informe de validación como JSON |

---

## Entrenamiento

### matrixai validate-training

Validar una especificación de entrenamiento supervisado `.mxtrain`.

```
matrixai validate-training <fichero> [--json]
```

---

### matrixai generate-training

Generar una especificación `.mxtrain` y una plantilla de dataset CSV a partir de un fichero `.mxai` y un prompt.

```
matrixai generate-training <fichero_mxai> <prompt...> [-o <fichero>] [--dataset-output <fichero>] [--epochs <n>] [--learning-rate <lr>] [--batch-size <n>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `-o, --output` | stdout | Escribir `.mxtrain` en fichero |
| `--dataset-output` | — | Escribir la plantilla CSV en fichero |
| `--dataset-source` | — | Ruta del dataset a incrustar en la especificación |
| `--dataset-name` | — | Sobreescribir el nombre del bloque `DATASET` |
| `--target-name` | — | Sobreescribir el nombre de la columna `TARGET` |
| `--labels` | — | Lista de etiquetas separada por comas |
| `--epochs` | — | Sobreescribir `RUN EPOCHS` |
| `--learning-rate` | — | Sobreescribir la tasa de aprendizaje SGD |
| `--batch-size` | — | Sobreescribir el tamaño del lote |
| `--json` | — | Imprimir el resultado de generación como JSON |

---

### matrixai generate-supervised

Generar `.mxai`, `.mxtrain` y una plantilla CSV a partir de un único prompt.

```
matrixai generate-supervised <prompt...> -o <directorio_salida> [--stem <nombre>] [--epochs <n>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `-o, --output-dir` | requerido | Directorio para los artefactos generados |
| `--stem` | auto | Sobreescribir el nombre base de los ficheros generados |
| `--dataset-name` | — | Sobreescribir el nombre del bloque `DATASET` |
| `--target-name` | — | Sobreescribir el nombre de la columna `TARGET` |
| `--labels` | — | Lista de etiquetas separada por comas |
| `--epochs` | — | Sobreescribir `RUN EPOCHS` |
| `--json` | — | Imprimir el resultado de generación como JSON |

---

### matrixai generate-dataset

Generar un dataset sintético reproducible a partir de un fichero `.mxai` + `.mxtrain`.

```
matrixai generate-dataset <fichero_mxai> --training <fichero_mxtrain> [-o <dir_salida>] [--rows <n>] [--seed <n>] [--mode <modo>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--training` | requerido | Fichero de especificación `.mxtrain` |
| `--rows` | `200` | Total de filas (rango: 2–50.000) |
| `--seed` | `42` | Semilla aleatoria para reproducibilidad |
| `--mode` | `random` | `random` o `coherent` (coherente es consistente con la semántica del modelo) |
| `-o, --output-dir` | `.` | Directorio de salida para los CSV y el manifiesto |
| `--stem` | auto | Nombre base de los ficheros |
| `--json` | — | Imprimir el resultado de generación como JSON |

---

### matrixai train

Entrenar un modelo supervisado y escribir los artefactos de entrenamiento MatrixAI.

```
matrixai train <fichero_mxai> --training <fichero_mxtrain> -o <directorio_salida> [--backend <nombre>] [--device <dev>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--training` | requerido | Fichero de especificación `.mxtrain` |
| `-o, --output` | requerido | Directorio de salida de la ejecución |
| `--backend` | de la especificación | `stdlib` o `torch` |
| `--device` | de la especificación | `cpu`, `cuda` o `mps` |
| `--json` | — | Imprimir el resultado de entrenamiento como JSON |

---

### matrixai train-supervised

Pipeline completo: generar `.mxai` + `.mxtrain` desde un prompt, entrenar y evaluar.

```
matrixai train-supervised <prompt...> -o <directorio_salida> [--train-data <csv>] [--eval-data <csv>] [--dataset-manifest <json>] [--epochs <n>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `-o, --output-dir` | requerido | Directorio de salida |
| `--train-data` | — | Fichero CSV de entrenamiento |
| `--eval-data` | — | Fichero CSV de evaluación |
| `--dataset-manifest` | — | JSON de manifiesto de dataset versionado |
| `--dataset-split` | — | Partición/fold nombrada en el manifiesto |
| `--stem` | auto | Nombre base de los artefactos |
| `--run-name` | `run` | Nombre del directorio de artefactos de ejecución |
| `--epochs` | — | Sobreescribir `RUN EPOCHS` |
| `--json` | — | Imprimir el resultado completo como JSON |

---

### matrixai evaluate

Evaluar un ParameterSet entrenado en un dataset CSV supervisado.

```
matrixai evaluate <fichero_mxai> --params <fichero_json> --training <fichero_mxtrain> [-o <informe>] [--data <csv>] [--backend <nombre>] [--device <dev>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--params` | requerido | ParameterSet JSON (pesos entrenados) |
| `--training` | requerido | Fichero de especificación `.mxtrain` |
| `--data` | — | Fichero CSV de dataset opcional |
| `-o, --output` | — | Escribir `evaluation_report.json` en esta ruta |
| `--backend` | — | `stdlib` o `torch` |
| `--device` | — | `cpu`, `cuda` o `mps` |
| `--json` | — | Imprimir el resultado de evaluación como JSON |

---

## Servicio

### matrixai run

Ejecutar un fichero `.mxai` una vez con entrada JSON e imprimir el resultado.

```
matrixai run <fichero> --input <fichero_json> [--params <fichero_json>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--input` | requerido | Fichero de entrada JSON |
| `--params` | — | Fichero JSON de ParameterSet |
| `--json` | — | Imprimir el resultado JSON sin procesar |

---

### matrixai serve

Servir un modelo entrenado y/o un registry de modelos por HTTP (servidor de producción).

```
matrixai serve [<fichero_mxai>] [--params <json>] [--port <n>] [--host <dir>]
               [--api-key <clave>] [--api-key-read <clave>] [--registry <ruta>]
               [--contract <mxact>] [--allow-real-actions]
               [--continual-policy <politica>] [--reference-accuracy <float>]
               [--rate-limit <n>] [--cors-origin <origen>] [--backend <nombre>]
```

`<fichero_mxai>` es opcional cuando se proporciona `--registry`.

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--params` | — | Fichero JSON de ParameterSet (pesos entrenados) |
| `--port` | `8000` | Puerto HTTP |
| `--host` | `127.0.0.1` | Dirección de enlace |
| `--backend` | `stdlib` | Backend de ejecución: `stdlib` o `torch` |
| `--api-key` | auto-generada | Clave de escritura — acceso completo (env: `MATRIXAI_API_KEY`) |
| `--api-key-read` | — | Clave de solo lectura — solo GET y predict (env: `MATRIXAI_API_KEY_READ`) |
| `--registry` | — | Ruta a un directorio de registry — habilita los endpoints `/api/v1/registry/*` |
| `--contract` | — | Fichero `.mxact` para habilitar `POST /api/v1/execute-action` |
| `--allow-real-actions` | desactivado | Habilitar ejecución real de acciones (requerido junto con `--contract`) |
| `--signing-key` | — | Clave HMAC hex para firma de ActionTrace (env: `MATRIXAI_ACTION_SIGNING_KEY`) |
| `--continual-policy` | — | Fichero de política `.mxcontinual` para habilitar monitorización de drift y `POST /api/v1/feedback` |
| `--reference-accuracy` | de params | Referencia de precisión base para detección de drift |
| `--rate-limit` | `60` | Máximo de peticiones/minuto por IP. `0` desactiva. (env: `MATRIXAI_RATE_LIMIT`) |
| `--cors-origin` | `*` | Origen CORS permitido. Repetible. (env: `MATRIXAI_CORS_ORIGINS`) |

---

### matrixai playground / matrixai studio

Iniciar el entorno local de desarrollo de modelos en el navegador.

```
matrixai playground [--host <dir>] [--port <n>] [--open]
matrixai studio    [--host <dir>] [--port <n>] [--open]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--host` | `127.0.0.1` | Dirección de enlace |
| `--port` | `8765` | Puerto HTTP |
| `--open` | — | Abrir la URL en el navegador predeterminado automáticamente |

`studio` y `playground` son alias del mismo comando.

---

### matrixai pack

Empaquetar un modelo `.mxai` como bundle de artefactos desplegable, opcionalmente con soporte Docker.

```
matrixai pack <fichero_mxai> [--params <json>] [--contract <mxact>] [--outdir <dir>] [--docker]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--params` | — | Fichero JSON de ParameterSet |
| `--contract` | — | Fichero de contrato `.mxact` (incluido en la imagen Docker) |
| `--outdir` | `dist` | Directorio de salida |
| `--docker` | — | Generar `Dockerfile`, `docker-compose.yml` y `.env.example` |

Todos los ficheros generados usan codificación UTF-8.

---

## Exportación y empaquetado

### matrixai export-onnx

Exportar un modelo entrenado a formato ONNX.

```
matrixai export-onnx <fichero_mxai> --params <json> -o <salida.onnx> [--validate] [--manifest <ruta>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--params` | requerido | ParameterSet JSON (pesos entrenados) |
| `-o, --output` | requerido | Ruta del fichero `.onnx` de salida |
| `--validate` | — | Ejecutar comprobación de equivalencia contra `onnxruntime` |
| `--manifest` | — | Escribir `export_manifest.json` aquí (requiere `--validate`) |
| `--json` | — | Imprimir el resultado de exportación como JSON |

---

### matrixai export-bundle

Crear un bundle de edge autocontenido: `model.onnx` + manifiestos + README.

```
matrixai export-bundle <fichero_mxai> --params <json> --outdir <dir> \
  [--inference-metadata <json>] [--no-validate] [--force] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--params` | requerido | ParameterSet JSON |
| `--outdir` | requerido | Directorio de salida |
| `--inference-metadata` | — | Sidecar JSON con la metadata de normalización; hace el bundle auto-usable (añade `predict.py`, `inference_spec.json`, `requirements.txt`, `example_input.json`, `expected_output.json`) |
| `--no-validate` | — | Omitir la comprobación de equivalencia (no recomendado en producción) |
| `--force` | — | Sobreescribir el directorio del bundle si existe |
| `--json` | — | Imprimir el resultado del bundle como JSON |

**Bundle auto-usable.** Con `--inference-metadata` el bundle incluye un `predict.py`
standalone (solo `numpy` + `onnxruntime`) que recibe valores **crudos** y devuelve una
predicción etiquetada — la normalización y la codificación de categorías se aplican por
ti. Los labels también fluyen del `ProbabilityMap[...]` del modelo. El comando imprime
`Self-usable: yes/no` (e `inference_spec_skipped_reason` con `--json`).

El sidecar se valida de forma estricta — una clave malformada aborta con `Bundle error`
en vez de producir en silencio un bundle que normaliza mal:

```json
{
  "field_ranges":     {"edad": [0, 120], "imc": [10, 70]},
  "field_categories": {"color": ["red", "green", "blue"]},
  "field_types":      {"edad": "integer", "activo": "boolean"},
  "labels":           ["BAJO", "ALTO"],
  "example_input":    {"edad": 60, "imc": 40, "color": "red"}
}
```

Los valores de `field_ranges` deben ser `[min, max]` numéricos finitos con `min < max`;
los de `field_categories`, listas de strings no vacías; `field_types` solo admite
`number` / `integer` / `boolean`.

**Usar tu modelo descargado:**

```bash
cd mi_bundle
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python predict.py --input example_input.json   # debe reproducir expected_output.json
```

---

### matrixai export-wasm

Crear un bundle de despliegue ONNX Runtime Web (WebAssembly).

```
matrixai export-wasm <fichero_mxai> --params <json> --outdir <dir> [--no-validate] [--force] [--json]
```

Mismas opciones que `export-bundle`. La salida incluye `model.onnx`, `predict.js` y manifiestos WASM para despliegue en navegador.

---

## Acciones

Las acciones requieren un fichero de contrato `.mxact` y la variable de entorno `MATRIXAI_ALLOW_REAL_ACTIONS=true` (o la opción `--allow-real-actions`) para la ejecución real.

### matrixai validate-actions

Validar un contrato `.mxact` contra un programa `.mxai`.

```
matrixai validate-actions <contrato.mxact> <programa.mxai> [--json]
```

---

### matrixai dry-run-action

Simular una acción de contrato sin efectos secundarios e imprimir el `DryRunReport`.

```
matrixai dry-run-action <contrato.mxact> <programa.mxai> --contract-name <nombre> [--input <json>] [--model-hash <hash>] [--param-set <id>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--contract-name` | requerido | Nombre del `ACTION_CONTRACT` a simular |
| `--input` | — | Objeto JSON con `input_data`. Usar `-` para stdin |
| `--model-hash` | `cli` | Identificador de hash del modelo |
| `--param-set` | `default` | Identificador del conjunto de parámetros |
| `--json` | — | Imprimir el informe como JSON |

---

### matrixai execute-action

Ejecutar una acción de contrato con efectos secundarios completos.

```
matrixai execute-action <contrato.mxact> <programa.mxai> --contract-name <nombre> --allow-real-actions [--input <json>] [--signing-key <hex>] [--json]
```

`--allow-real-actions` es obligatorio. Sin él el comando termina con error.

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--contract-name` | requerido | Nombre del `ACTION_CONTRACT` a ejecutar |
| `--allow-real-actions` | desactivado | Habilita la ejecución real |
| `--input` | — | Objeto JSON con `input_data`. Usar `-` para stdin |
| `--model-hash` | — | Identificador de hash del modelo |
| `--param-set` | `default` | Identificador del conjunto de parámetros |
| `--signing-key` | — | Clave HMAC hex (env: `MATRIXAI_ACTION_SIGNING_KEY`) |
| `--json` | — | Imprimir el resultado como JSON |

---

### matrixai audit-action

Verificar la firma HMAC de un fichero JSON `ActionTrace`.

```
matrixai audit-action <fichero_traza> [--signing-key <hex>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--signing-key` | — | Clave HMAC hex (env: `MATRIXAI_ACTION_SIGNING_KEY`) |
| `--json` | — | Imprimir el resultado como JSON |

---

## Registro

Las entradas del registro se referencian como `nombre@version` (p.ej. `credit-scoring@v1.0`).

### matrixai registry push

Registrar una ejecución de entrenamiento en el registro.

```
matrixai registry push <directorio_ejecucion> --name <nombre> --version <version> [--registry-path <dir>]
```

| Argumento / Opción | Por defecto | Descripción |
|--------------------|-------------|-------------|
| `directorio_ejecucion` | — | Ruta al directorio de ejecución de entrenamiento |
| `--name` | requerido | Nombre del modelo (minúsculas) |
| `--version` | requerido | Etiqueta de versión (p.ej. `v1.0`) |
| `--registry-path` | `matrixai_registry` | Directorio del registro |

---

### matrixai registry pull

Copiar una entrada del registro de un registro a otro.

```
matrixai registry pull <nombre@version> --from <dir_origen> --to <dir_destino>
```

---

### matrixai registry list

Listar todas las entradas del registro.

```
matrixai registry list [--name <nombre>] [--registry-path <dir>] [--json]
```

---

### matrixai registry show

Mostrar el manifiesto de una entrada específica del registro.

```
matrixai registry show <nombre@version> [--registry-path <dir>] [--json]
```

---

### matrixai registry tag

Crear o mover un alias de etiqueta (p.ej. `latest`, `prod`) a una versión específica.

```
matrixai registry tag <nombre@version> <nombre_etiqueta> [--registry-path <dir>]
```

---

### matrixai registry verify

Verificar la integridad de una entrada del registro (sumas de comprobación y completitud).

```
matrixai registry verify <nombre@version> [--registry-path <dir>]
```

Sale con código 0 si la entrada está intacta, distinto de cero en caso contrario.

---

### matrixai registry diff

Comparar los manifiestos de dos versiones de un modelo.

```
matrixai registry diff <nombre@version_a> <nombre@version_b> [--registry-path <dir>]
```

---

## Gestión de claves

Las claves de firma se almacenan por defecto en `<registry-path>/.matrixai_key_history.json`.

### matrixai keys rotate

Retirar la clave de firma actual y registrarla en el historial de claves.

```
matrixai keys rotate --purpose <proposito> [--key <valor>] [--history-path <fichero>] [--registry-path <dir>]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--purpose` | requerido | Qué clave rotar: `action` o `registry` |
| `--key` | de env | Valor de la clave a retirar |
| `--history-path` | `<registro>/.matrixai_key_history.json` | Fichero de historial de claves |
| `--registry-path` | `matrixai_registry` | Directorio del registro |

---

### matrixai keys list

Listar todas las claves de firma registradas (activas y retiradas) con estado y huellas digitales.

```
matrixai keys list [--history-path <fichero>] [--registry-path <dir>] [--json]
```

---

## Aprendizaje continuo

Los comandos continual operan sobre un fichero de política `.mxcontinual`. Gestionan la detección de drift, la promoción de versiones y el rollback en producción.

### matrixai continual init

Validar y mostrar un resumen de una política `.mxcontinual`.

```
matrixai continual init <politica.mxcontinual> [--json]
```

---

### matrixai continual ingest

Registrar una etiqueta de verdad real (ground truth) para un `ActionTrace` de producción (usado por el monitor de drift).

```
matrixai continual ingest <politica.mxcontinual> --trace-id <id> --label <valor> [--trace-file <json>] [--signing-key <hex>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--trace-id` | requerido | `report_id` del `ActionTrace` a anotar |
| `--label` | requerido | Etiqueta o valor de verdad real |
| `--trace-file` | — | Ruta al fichero JSON del `ActionTrace` |
| `--signing-key` | — | Clave HMAC hex (env: `MATRIXAI_CONTINUAL_SIGNING_KEY`) |

---

### matrixai continual status

Mostrar la versión actual del registro, métricas e historial de rollbacks para una política.

```
matrixai continual status <politica.mxcontinual> [--registry-dir <dir>] [--json]
```

---

### matrixai continual promote

Promover un ParameterSet candidato a producción mediante el `ContinualVersioner`.

```
matrixai continual promote <politica.mxcontinual> --approval-report <json> --candidate-params <json> [--registry-dir <dir>] [--update-id <id>] [--human-approved] [--approved-by <identidad>] [--signing-key <hex>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--approval-report` | requerido | Fichero JSON `ApprovalGateReport` |
| `--candidate-params` | requerido | Fichero JSON del ParameterSet candidato |
| `--registry-dir` | `matrixai_registry` | Directorio del registro |
| `--update-id` | auto | Identificador de actualización continual |
| `--human-approved` | — | Registrar una decisión de aprobación humana |
| `--approved-by` | `$USER` | Identidad del aprobador humano |
| `--signing-key` | — | Clave HMAC hex para verificar el token `PendingApproval` |

---

### matrixai continual rollback

Revertir a la versión anterior del registro.

```
matrixai continual rollback <politica.mxcontinual> [--registry-dir <dir>] [--dry-run] [--signing-key <hex>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--dry-run` | — | Mostrar qué se revertiría sin ejecutarlo |
| `--signing-key` | — | Clave HMAC hex (env: `MATRIXAI_CONTINUAL_SIGNING_KEY`) |

---

### matrixai continual audit

Mostrar la configuración de auditoría y opcionalmente generar una sugerencia de refinamiento basada en drift.

```
matrixai continual audit <politica.mxcontinual> [--drift-report <json>] [--prompt <texto>] [--drift-persistence-days <n>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--drift-report` | — | Fichero JSON `DriftReport` para la sugerencia de refinamiento |
| `--prompt` | — | Prompt original (requerido con `--drift-report`) |
| `--drift-persistence-days` | — | Días que el drift se ha observado continuamente (activa `REFINEMENT_DRIFT_PERSISTENCE_DAYS`) |

---

## Refinamiento

### matrixai refine

Proponer un prompt refinado basado en un resultado de auditoría o evaluación.

```
matrixai refine <prompt...> [--audit <json>] [--evaluation <json>] [--mxai <fichero>] [--hint <texto>] [--iteration <n>] [--chain <json>] [--parent-hash <sha256>] [-o <fichero>] [--mxai-output <fichero>] [--chain-output <fichero>] [--accept] [--max-iterations <n>] [--json]
```

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `prompt` | — | Texto del prompt original. Usar `-` para stdin |
| `--audit` | — | Fichero JSON de auditoría (activa el modo `audit_driven`) |
| `--evaluation` | — | `evaluation_report.json` (activa el modo `metric_driven`) |
| `--mxai` | — | Fichero `.mxai` actual (añade contexto del modelo al refinamiento) |
| `--hint` | — | Sugerencia adicional del usuario. Repetible |
| `--iteration` | `1` | Número de iteración actual |
| `--chain` | — | Fichero JSON con la `refinement_chain` previa (lista de IDs) |
| `--parent-hash` | — | SHA-256 del prompt raíz original |
| `-o, --output` | — | Escribir el prompt propuesto en fichero (requiere `--accept`) |
| `--mxai-output` | — | Escribir el `.mxai` generado en fichero (requiere `--accept`) |
| `--chain-output` | — | Escribir el JSON actualizado de `refinement_chain` en fichero |
| `--accept` | — | Aceptar explícitamente la propuesta para habilitar la escritura de ficheros de salida |
| `--max-iterations` | por defecto | Límite máximo de iteraciones. Código de salida `2` si se supera |
| `--json` | — | Imprimir el `RefinementProposal` completo como JSON |

---

## Variables de entorno

| Variable | Usada por | Descripción |
|----------|-----------|-------------|
| `MATRIXAI_API_KEY` | `serve` | Clave API para autenticación HTTP |
| `MATRIXAI_ACTION_SIGNING_KEY` | `serve`, `execute-action`, `audit-action` | Clave HMAC hex para firma de trazas de acción |
| `MATRIXAI_CONTINUAL_SIGNING_KEY` | `continual ingest`, `continual rollback` | Clave HMAC hex para eventos de aprendizaje continuo |
| `MATRIXAI_ALLOW_REAL_ACTIONS` | `serve` | Establecer a `true` para habilitar `/execute-action` |
| `MATRIXAI_RATE_LIMIT` | `serve` | Máximo de peticiones/minuto por IP (entero) |
| `MATRIXAI_CORS_ORIGINS` | `serve` | Orígenes CORS permitidos separados por comas |

---

## Códigos de salida

| Código | Significado |
|--------|-------------|
| `0` | Éxito |
| `1` | Error (análisis, validación, tiempo de ejecución) |
| `2` | Límite de iteraciones superado (`refine`) |

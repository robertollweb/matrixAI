# MatrixAI Quickstart — 5 minutos

> **English:** [docs/en/QUICKSTART.md](../en/QUICKSTART.md)

Vamos a entrenar un modelo de clasificación binaria con tres señales numéricas y hacer una predicción. Sin configuración, sin credenciales externas, sin complicaciones.

## Requisitos

- **Python 3.10+** — descarga en [python.org/downloads](https://www.python.org/downloads/)
- Terminal / línea de comandos
- 5 minutos

> **Windows:** usa `python` en lugar de `python3` en todos los comandos. En Windows, Python se instala solo como `python`.  
> Si el comando `matrixai` no se encuentra tras instalar, usa `python -m matrixai` en su lugar.

## 1. Instalar

**Desde el código fuente (actual — publicación en PyPI próximamente):**

```bash
git clone https://github.com/robertollweb/matrixAI.git
cd matrixAI
pip install -e .
```

**Con soporte de exportación ONNX / WASM:**

```bash
pip install -e ".[export]"
```

**Con soporte de entrenamiento GPU (requiere PyTorch):**

```bash
pip install -e ".[torch]"
```

## 2. Crear un proyecto nuevo

```bash
python3 -m matrixai init my-first-classifier --template classification
```

**Verás:**
```
✓ Project 'my-first-classifier' created at /home/YOUR_USER/matrixAI/my-first-classifier

Next steps:
  1. python3 -m matrixai train /home/YOUR_USER/matrixAI/my-first-classifier/my-first-classifier.mxai --training /home/YOUR_USER/matrixAI/my-first-classifier/my-first-classifier.mxtrain --output /home/YOUR_USER/matrixAI/my-first-classifier/runs/v1
  2. python3 -m matrixai run /home/YOUR_USER/matrixAI/my-first-classifier/my-first-classifier.mxai --params /home/YOUR_USER/matrixAI/my-first-classifier/runs/v1/params.best.json --input /home/YOUR_USER/matrixAI/my-first-classifier/input/sample.json --json
```

## 3. Entrenar el modelo

```bash
python3 -m matrixai train my-first-classifier/my-first-classifier.mxai --training my-first-classifier/my-first-classifier.mxtrain --output my-first-classifier/runs/v1
```

**Verás:**
```
Training OK: v1
Best epoch: 30
Best validation loss: 0.083059
Accuracy: 1.000000
Artifacts: my-first-classifier/runs/v1
```

## 4. Hacer una predicción

```bash
python3 -m matrixai run my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/v1/params.best.json --input my-first-classifier/input/sample.json --json
```

**Verás (fragmento):**
```json
{
  "state": {
    "R": 0.91,
    "Classification": {
      "type": "Normal",
      "mean": 0.91,
      "sigma": 0.05
    }
  }
}
```

Interpretación: `R` es la probabilidad aprendida para la clase positiva. Con el ejemplo incluido (`input/sample.json`), el modelo responde con una probabilidad alta.

## 5. Opcional: servir el mismo modelo por HTTP

Si quieres pasar del CLI a HTTP con el modelo que acabas de entrenar, usa los artefactos de `runs/v1`.

Terminal 1:

```bash
python3 -m matrixai serve my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/v1/params.best.json --api-key dev-secret
```
Navegador:

```text
http://127.0.0.1:8000/docs
```

Terminal 2:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.85}'
```

## ¿Qué acabas de hacer?

1. **Creaste un proyecto** con estructura, datos y configuración listos para entrenar
2. **Entrenaste un modelo** usando tus datos: el archivo `dataset/train.csv` tiene ejemplos numéricos positivos y negativos
3. **Hiciste una predicción** sobre un ejemplo nuevo sin editar código
4. **Opcionalmente, serviste el modelo por HTTP** usando los mismos artefactos de `runs/v1`

El modelo aprendió a clasificar ejemplos binarios en ~30 segundos. Eso es MatrixAI: **velocidad + auditabilidad + verificabilidad**.

## ¿Qué sigue?

Felicidades, tienes un modelo. Tres opciones:

> ¿No entiendes lo que ves en la interfaz web? → [Guía de la interfaz HTTP](HTTP_INTERFACE.md)

1. **Tutorial de 30 minutos** — [TUTORIAL.md](TUTORIAL.md)
   - Entiende qué es un `.mxai` y un `.mxtrain`
   - Cambia hiperparámetros y ve cómo mejora la precisión
   - Sirve el modelo con un servidor HTTP
   - Explora el modelo visualmente en Studio

2. **Documentación de referencia**
   - [Referencia CLI](api/CLI_REFERENCE.md) — todos los comandos y opciones
   - [Referencia API REST](api/REST_API.md) — endpoints HTTP para servicio e integración
   - [Especificación del lenguaje](api/LANGUAGE_SPEC.md) — sintaxis y semántica de `.mxai`

3. **Casos de uso reales por industria** — [docs/es/CASOS_DE_USO.md](CASOS_DE_USO.md)
   - Financiero, salud, routing SaaS, agente automatizado
   - Cada uno ejecutable con un comando, datos incluidos

## ¿Problemas?

- **"No module named matrixai"** → Ejecuta los comandos desde la raíz del repo clonado (`matrixAI/`)
- **`python3` no se reconoce (Windows)** → Usa `python` en lugar de `python3`
- **`PermissionError: [WinError 10013]` al usar `serve`** → El puerto 8000 está bloqueado por el firewall de Windows o en uso. Prueba con otro puerto: añade `--port 8080` al comando serve y accede a `http://127.0.0.1:8080/docs`
- **El proyecto ya existe al hacer `init`** → El repo incluye `my-first-classifier` como ejemplo. Usa otro nombre: `python3 -m matrixai init my-proyecto --template classification`
- **Otros errores** → Abre un issue: [github.com/robertollweb/matrixAI/issues](https://github.com/robertollweb/matrixAI/issues)

---

**Tiempo total: ~5 minutos. Tu modelo ya entrena y responde. Siguiente: [Tutorial de 30 minutos →](TUTORIAL.md)**

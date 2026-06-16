# MatrixAI Quickstart — 5 minutes

Train a binary classification model with three numeric signals and make a prediction. No configuration, no external credentials, no complexity.

> **Español:** [docs/es/QUICKSTART.md](../es/QUICKSTART.md)

## Requirements

- **Python 3.10+** — download at [python.org/downloads](https://www.python.org/downloads/)
- Terminal / command line
- 5 minutes

> **Windows:** use `python` instead of `python3` in all commands. On Windows, Python is installed as `python` only.  
> If the `matrixai` command is not found after install, use `python -m matrixai` instead.

## 1. Install

**From source (current — PyPI release coming soon):**

```bash
git clone https://github.com/robertollweb/matrixAI.git
cd matrixAI
pip install -e .
```

**With ONNX / WASM export support:**

```bash
pip install -e ".[export]"
```

**With GPU training support (requires PyTorch):**

```bash
pip install -e ".[torch]"
```

## 2. Create a new project

```bash
python3 -m matrixai init my-first-classifier --template classification
```

**You'll see:**
```
✓ Project 'my-first-classifier' created at /home/YOUR_USER/matrixAI/my-first-classifier

Next steps:
  1. python3 -m matrixai train ...
  2. python3 -m matrixai run ...
```

## 3. Train the model

```bash
python3 -m matrixai train my-first-classifier/my-first-classifier.mxai --training my-first-classifier/my-first-classifier.mxtrain --output my-first-classifier/runs/v1
```

**You'll see:**
```
Training OK: v1
Best epoch: 30
Best validation loss: 0.083059
Accuracy: 1.000000
Artifacts: my-first-classifier/runs/v1
```

## 4. Make a prediction

```bash
python3 -m matrixai run my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/v1/params.best.json --input my-first-classifier/input/sample.json --json
```

**You'll see (excerpt):**
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

`R` is the learned probability for the positive class. With the included sample (`input/sample.json`), the model responds with a high probability.

## 5. Optional: serve the model over HTTP

**Terminal 1:**

```bash
python3 -m matrixai serve my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/v1/params.best.json --api-key dev-secret
```

> **Windows:** if port 8000 is blocked, add `--port 8080` and open `http://127.0.0.1:8080/docs`.

**Browser:**

```text
http://127.0.0.1:8000/docs
```

**Terminal 2:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.85}'
```

> Don't know what you're seeing in the web interface? → [HTTP Interface Guide](HTTP_INTERFACE.md)

## What did you just do?

1. **Created a project** with ready-to-train structure, data and configuration
2. **Trained a model** on your data: `dataset/train.csv` contains positive and negative numeric examples
3. **Made a prediction** on a new example without editing any code
4. **Optionally, served the model over HTTP** using the same `runs/v1` artifacts

The model learned to classify binary examples in ~30 seconds. That's MatrixAI: **speed + auditability + verifiability**.

## What's next?

Congratulations, you have a model. Three options:

> Don't understand what you see in the web interface? → [HTTP Interface Guide](HTTP_INTERFACE.md)

1. **30-minute tutorial** — [TUTORIAL.md](TUTORIAL.md)
   - Understand what `.mxai` and `.mxtrain` are
   - Change hyperparameters and see how accuracy improves
   - Serve the model with an HTTP server
   - Explore the model visually in Studio

2. **Reference documentation**
   - [CLI Reference](api/CLI_REFERENCE.md) — every command and flag
   - [REST API Reference](api/REST_API.md) — HTTP endpoints for serving and integration
   - [Language Spec](api/LANGUAGE_SPEC.md) — `.mxai` syntax and semantics

3. **Real use cases by industry** — [docs/en/USE_CASES.md](USE_CASES.md)
   - Financial, healthcare, SaaS routing, automated agent
   - Each runnable with one command, data included

## Troubleshooting

- **"No module named matrixai"** → Run commands from the root of the cloned repo (`matrixAI/`)
- **`python3` not recognized (Windows)** → Use `python` instead of `python3`
- **`PermissionError: [WinError 10013]` with `serve`** → Port 8000 is blocked by Windows Firewall or already in use. Try another port: add `--port 8080` to the serve command and open `http://127.0.0.1:8080/docs`
- **Project already exists on `init`** → The repo includes `my-first-classifier` as an example. Use another name: `python3 -m matrixai init my-project --template classification`
- **Other errors** → Open an issue: [github.com/robertollweb/matrixAI/issues](https://github.com/robertollweb/matrixAI/issues)

---

**Total time: ~5 minutes. Your model trains and responds. Next: [30-minute Tutorial →](TUTORIAL.md)**

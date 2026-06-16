# HTTP Interface Guide — `/docs`

When you start the server with `matrixai serve`, MatrixAI exposes an interactive web interface at `http://127.0.0.1:8000/docs`. This guide explains what you see, what each section does, and how to make your first prediction from the browser.

> **Español:** [docs/es/HTTP_INTERFACE.md](../es/HTTP_INTERFACE.md)

---

## Start the server

```bash
python3 -m matrixai serve my-project/my-project.mxai --params my-project/runs/v1/params.best.json --api-key dev-secret
```

Open the browser at `http://127.0.0.1:8000/docs`.

> **Windows:** if port 8000 is blocked, add `--port 8080` and open `http://127.0.0.1:8080/docs`.

---

## What you see when you open `/docs`

The page is called **Swagger UI**. It's a standard interface that describes and lets you test a REST API directly from the browser, with nothing extra to install.

```
MatrixAI Prediction API  1.0.0  OAS 3.0
/openapi.json
Auto-generated OpenAPI specification based on the deployed .mxai program.

                                          [ Authorize 🔒 ]

default ∧
  POST  /predict          Run prediction on the model        🔒
  POST  /execute-action   Execute a real action              🔒
  GET   /docs             Swagger UI
  GET   /health           Health Check

Schemas ∧
  FeaturesInput { ... }
  PredictionInput { ... }
  ...
```

**OAS 3.0** means the schema follows the OpenAPI 3.0 standard — the same format used by thousands of production APIs.

The 🔒 icon on an endpoint means it requires authentication.

---

## Step 1 — Authenticate

Click **Authorize** (top right corner).

A dialog appears. In the **Value** field, type the API key you used when starting the server:

```
dev-secret
```

Click **Authorize** then **Close**. The lock icon is now closed — your requests will include the token automatically.

> Without this step, `/predict` and `/execute-action` return **401 Unauthorized**.

---

## Step 2 — Make a prediction (`POST /predict`)

This is the main endpoint. It receives your model's input data and returns the result.

### How to use it

1. Click the green **POST /predict** bar to expand it
2. Click **Try it out** (right side)
3. The **Request body** field becomes editable and shows a JSON example with your model's fields:

```json
{
  "feature_1": 0.5,
  "feature_2": 0.5,
  "feature_3": 0.5
}
```

4. Change the values if you want (they must be between 0.0 and 1.0 for this model)
5. Click **Execute**

### What it returns

The response appears below. Real example:

```json
{
  "state": {
    "feature_1": 0.9,
    "feature_2": 0.8,
    "feature_3": 0.85,
    "ClassifierModel": 0.9156,
    "R": 0.9156,
    "Classification": {
      "type": "Normal",
      "mean": 0.9156,
      "sigma": 0.05
    }
  },
  "trace": [
    { "step": 1, "status": "ok", "node": "Features", "value": [0.9, 0.8, 0.85] },
    { "step": 2, "status": "ok", "node": "ClassifierModel", "value": 0.9156 },
    { "step": 3, "status": "ok", "node": "Classification", "value": { "type": "Normal", "mean": 0.9156, "sigma": 0.05 } }
  ],
  "actions": []
}
```

### What each field means

| Field | What it is |
|---|---|
| `state.R` | The model's numeric result. For binary classification: probability of the positive class. `0.91` = 91% probability. |
| `state.Classification` | The result interpreted as a distribution. `mean` is the same value as `R`. `sigma` is the uncertainty. |
| `trace` | List of graph execution steps. Each node in the `.mxai` appears here with its value and `status: ok`. If a step failed, it would show `status: error`. |
| `actions` | List of real actions triggered (email sending, external API call, etc.). Empty if the model has no configured actions or none were activated. |

**`R` is the number that matters.** The rest is audit trail — you can ignore it until you need to debug something.

### Interpreting `R`

`R` is a probability between 0.0 and 1.0. It indicates how confident the model is that the example belongs to the **positive class** — the class with `label = 1` in your training data.

- **`R` close to 1.0** → model predicts positive class with high confidence
- **`R` close to 0.0** → model predicts negative class with high confidence
- **`R` ≈ 0.5** → model is uncertain; the example is on the decision boundary

**Is `R = 1` good or bad?** It depends entirely on what you defined as label=1 in training:

| Use case | label=1 means | R=1 is... |
|---|---|---|
| Spam detector | Is spam | Bad (it's spam) |
| Credit approval | Approved | Good (approved) |
| Disease detection | Positive test | Context-dependent |
| Quality control | Defective part | Bad (defect detected) |

The quickstart model uses generic example data — label=1 has no business meaning yet. When you build your own model, you define what label=1 means in your training data, and that determines how to interpret `R`.

The typical decision threshold is 0.5: above → positive class, below → negative class. You can adjust it based on how much you want to prioritize false positives vs. false negatives.

---

## `POST /execute-action`

This endpoint is only relevant if your model has an action contract (`.mxact`) loaded. It allows triggering a real action (send an email, call an external system, log a decision) in an audited way.

If you only have a basic classification model, **you can ignore this endpoint** — it doesn't affect `/predict`.

---

## `GET /health`

Returns the server status. Useful to know if the server is alive and how many requests it has processed.

```json
{
  "status": "ok",
  "service": "MatrixAI Server",
  "metrics": {
    "requests_total": 5,
    "requests_successful": 5,
    "requests_failed": 0,
    "items_processed": 5,
    "last_request_ms": 0.12,
    "uptime_seconds": 342
  }
}
```

Doesn't require authentication. You can open it directly in the browser: `http://127.0.0.1:8000/health`.

---

## "Schemas" section (at the bottom of the page)

Shows the data structure the model expects and returns. Auto-generated from your `.mxai`.

- **FeaturesInput** — the fields you must send and their type (`number`, range `0.0–1.0`)
- **PredictionInput** — the full body format (accepts `FeaturesInput` directly or wrapped in an object)
- **PredictionResult** — the response format

You don't need to read this to use the endpoint. It's useful if you want to integrate the model into another application and need to know exactly what structure to expect.

---

## Making the same prediction from the terminal

If you prefer not to use the web interface, the exact equivalent with `curl`:

**Linux/Mac:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.9, "feature_2": 0.8, "feature_3": 0.85}'
```

**Windows (PowerShell):**
```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Authorization: Bearer dev-secret" -H "Content-Type: application/json" -d "{\"feature_1\": 0.9, \"feature_2\": 0.8, \"feature_3\": 0.85}"
```

---

## Summary

| I want to... | I use... |
|---|---|
| See my model's result | `POST /predict` → look for `state.R` |
| Check if the server is alive | `GET /health` |
| Trigger a real action | `POST /execute-action` (requires `.mxact`) |
| See the API schema | `/openapi.json` or Schemas section in `/docs` |

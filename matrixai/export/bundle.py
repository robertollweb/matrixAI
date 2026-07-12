# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Edge bundle: packages .mxai + params + model.onnx + manifests into a deployable directory."""
from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrixai.ir import MatrixAIProgram
from matrixai.export.onnx_exporter import validate_export_parameter_set
from matrixai.parameters.store import ParameterSet, program_hash
from matrixai.export.onnx_exporter import OnnxExporter, OnnxExportResult, OnnxExportError
from matrixai.export.inference_spec import (
    build_inference_spec,
    build_example_input,
    InferenceSpecError,
)

# The standalone predict.py shipped inside every usable bundle (copied verbatim).
_PREDICT_TEMPLATE = str(Path(__file__).resolve().parent / "predict_template.py")
_REQUIREMENTS = "numpy>=1.24\nonnxruntime>=1.16\n"
from matrixai.export.equivalence import (
    OnnxEquivalenceResult,
    OnnxEquivalenceValidator,
    write_export_manifest,
    ort_available,
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    _DEFAULT_N_SAMPLES,
)


class EdgeBundleError(ValueError):
    pass


@dataclass(frozen=True)
class EdgeBundleResult:
    bundle_dir: str
    files: list[str]
    model_hash: str
    parameter_set_id: str
    export_result: OnnxExportResult
    equivalence_result: OnnxEquivalenceResult | None = None
    # Why the bundle has no inference_spec.json (SEQUENCE/multi-input, unlabelled
    # classification, ...). None means the spec was produced. The bundle stays valid
    # either way; this makes the omission observable instead of silent.
    inference_spec_skipped_reason: str | None = None
    # PESOS_GRANDES C7b: por qué la equivalencia ONNX==referencia NO se
    # verificó (modelo grande vía `state_dict` — verificarla exigiría
    # materializar tensores a listas Python, el `.tolist()` que este export
    # evita). `None` significa que SÍ se verificó (comportamiento de siempre).
    equivalence_skipped_reason: str | None = None
    # PESOS_GRANDES C7 auditoría: en el modo STREAMING (mxw_path), el bundle NO
    # escribe `model.onnx.data` — el caller (Studio) lo streamea desde el `.mxw`
    # directo al zip final. Aquí van los tensores del `.mxw` en el orden EXACTO
    # en que deben concatenarse en el `.data` (offsets del grafo). `None` en el
    # modo normal/state_dict (el `.onnx.data` ya está en `bundle_dir`).
    external_data_layout: list[dict] | None = None

    @property
    def equivalence_passed(self) -> bool:
        return self.equivalence_result is not None and self.equivalence_result.passed

    @property
    def has_inference_spec(self) -> bool:
        return self.inference_spec_skipped_reason is None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bundle_dir": self.bundle_dir,
            "files": self.files,
            "model_hash": self.model_hash,
            "parameter_set_id": self.parameter_set_id,
            "equivalence_passed": self.equivalence_passed,
            "export": self.export_result.to_dict(),
            "inference_spec_skipped_reason": self.inference_spec_skipped_reason,
            "equivalence_skipped_reason": self.equivalence_skipped_reason,
        }
        if self.equivalence_result is not None:
            d["equivalence_check"] = self.equivalence_result.to_dict()
        return d


class EdgeBundler:
    """Creates a self-contained edge bundle directory from a .mxai + ParameterSet."""

    def bundle(
        self,
        program: MatrixAIProgram,
        parameter_set: ParameterSet | None,
        mxai_path: str | Path,
        params_path: str | Path | None,
        outdir: str | Path,
        *,
        state_dict: dict[str, Any] | None = None,
        mxw_path: str | Path | None = None,
        mxw_header: dict[str, Any] | None = None,
        model_hash: str | None = None,
        parameter_schema_hash: str | None = None,
        validate: bool = True,
        atol: float = _DEFAULT_ATOL,
        rtol: float = _DEFAULT_RTOL,
        n_samples: int = _DEFAULT_N_SAMPLES,
        force: bool = False,
        field_ranges: dict[str, tuple[float, float]] | None = None,
        field_categories: dict[str, list[str]] | None = None,
        field_types: dict[str, str] | None = None,
        labels: list[str] | None = None,
        example_input: dict[str, Any] | None = None,
    ) -> EdgeBundleResult:
        """PESOS_GRANDES C7b: `state_dict` (tensores torch crudos de un modelo
        grande guardado en `.mxw`) es la alternativa a un `parameter_set` con
        valores. Cuando se da, `parameter_set` sigue siendo obligatorio para
        `_build_model_manifest`/`build_inference_spec` (metadata: hash/shape/
        schema, nunca `.values`) — puede ser una PLANTILLA sin valores
        (`values=None`, el mismo objeto que ya construye
        `build_parameter_template_for_state`). `params_path=None` omite
        `params.best.json` del zip (escribirlo exigiría el JSON completo, el
        mismo `.tolist()` que este camino evita); la validación de
        equivalencia (que sí necesita una REFERENCIA con valores reales) se
        salta con motivo registrado.
        """
        using_state_dict = state_dict is not None
        # PESOS_GRANDES C7 auditoría: modo STREAMING — los pesos NUNCA se traen
        # a RAM; el grafo ONNX se construye desde la cabecera del `.mxw` y el
        # `model.onnx.data` lo streamea el caller (Studio) directo al zip.
        using_mxw_streaming = mxw_path is not None
        external_data_layout: list[dict] | None = None
        outdir = Path(outdir)
        if outdir.exists() and not force:
            raise EdgeBundleError(
                f"Bundle directory {outdir} already exists. "
                "Pass force=True to overwrite."
            )

        # Validate ParameterSet shapes/schema before touching disk. Se salta
        # con state_dict/streaming: el validador genérico (`validate_parameter_
        # set`) exige VALORES reales (no solo shapes) incluso para una
        # plantilla — `OnnxExporter.export`/`export_dense_onnx_graph_external`
        # ya validan `model_hash` y la presencia de cada tensor esperado por su
        # cuenta, así que esto no es un hueco, es la MISMA validación en el
        # sitio correcto.
        if not using_state_dict and not using_mxw_streaming:
            val = validate_export_parameter_set(program, parameter_set)
            if not val.ok:
                raise EdgeBundleError(
                    f"ParameterSet validation failed: {'; '.join(val.errors)}"
                )

        # Build in a temp dir; rename to outdir only when everything succeeds.
        tmp_parent = outdir.parent
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent, prefix=".bundle_tmp_") as _tmp:
            work = Path(_tmp)

            # 1. Copy .mxai and params (params.best.json se omite para un
            # grande vía state_dict — ver docstring).
            shutil.copy2(str(mxai_path), str(work / "model.mxai"))
            if params_path is not None:
                shutil.copy2(str(params_path), str(work / "params.best.json"))

            # 2. Export ONNX
            onnx_dest = work / "model.onnx"
            try:
                if using_mxw_streaming:
                    # PESOS_GRANDES C7 auditoría: solo el GRAFO se escribe aquí
                    # (`model.onnx`, initializers EXTERNAL); el `model.onnx.data`
                    # lo streamea el caller desde el `.mxw`. `ordered_metas` da el
                    # orden de concatenación que casa con los offsets del grafo.
                    from matrixai.export.onnx_exporter import export_onnx_graph_external
                    export_result, external_data_layout = export_onnx_graph_external(
                        program, mxw_header, onnx_dest,
                        model_hash=model_hash, parameter_schema_hash=parameter_schema_hash,
                    )
                else:
                    export_result = OnnxExporter().export(
                        program, parameter_set, onnx_dest,
                        state_dict=state_dict, model_hash=model_hash,
                        parameter_schema_hash=parameter_schema_hash,
                    )
            except OnnxExportError as exc:
                raise EdgeBundleError(f"ONNX export failed: {exc}") from exc

            # 3. Equivalence validation
            eq_result: OnnxEquivalenceResult | None = None
            eq_skipped_reason: str | None = None
            if validate and (using_state_dict or using_mxw_streaming):
                # PESOS_GRANDES C7b: verificar equivalencia exige correr un
                # forward de REFERENCIA con valores reales (`parameter_set`
                # con `.values`) — para un state_dict grande eso es
                # exactamente el `.tolist()` que este camino evita. Se salta
                # con el motivo registrado (mismo patrón que
                # `inference_spec_skipped_reason`), nunca en silencio.
                eq_skipped_reason = (
                    "Equivalencia no verificada: el modelo es grande (pesos en "
                    ".mxw binario) y verificarla exigiría materializar los "
                    "tensores a listas Python — el mismo .tolist() que este "
                    "export evita. El ONNX se generó directamente desde los "
                    "mismos tensores entrenados; validado por separado en el "
                    "cierre duro del contrato."
                )
            elif validate:
                if not ort_available():
                    raise EdgeBundleError(
                        "Equivalence validation requires 'onnxruntime'. "
                        "Install with: pip install matrixai-core[export]"
                    )
                eq_result = OnnxEquivalenceValidator().validate(
                    program, parameter_set, onnx_dest,
                    atol=atol, rtol=rtol, n_samples=n_samples,
                )
                if not eq_result.passed:
                    raise EdgeBundleError(
                        f"Equivalence check FAILED: max_abs_diff={eq_result.max_abs_diff:.2e} "
                        f"exceeds tolerance atol={atol:.0e} + rtol={rtol:.0e}. "
                        "Bundle not created."
                    )

            # 4. model_manifest.json
            (work / "model_manifest.json").write_text(
                json.dumps(_build_model_manifest(program, parameter_set), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

            # 4b. inference_spec.json — the "tokenizer": how a raw record becomes the
            # normalized float32 vector the ONNX graph expects. Without it the bundle
            # is not self-usable (see EXPORT_MODELO_DESCARGABLE_CONTRACT C1).
            # Best-effort: models that cannot produce a usable spec (SEQUENCE,
            # multi-input, unlabelled classification) keep producing the rest of the
            # bundle as before; they just don't get an inference_spec (nor, later,
            # predict.py). The reason is surfaced on the result and as a warning so the
            # omission is observable instead of silent (the Studio turns it into a
            # user-facing message in C4).
            spec_skipped_reason: str | None = None
            try:
                inference_spec = build_inference_spec(
                    program, parameter_set, export_result,
                    field_ranges=field_ranges,
                    field_categories=field_categories,
                    field_types=field_types,
                    labels=labels,
                    example_input=example_input,
                )
                (work / "inference_spec.json").write_text(
                    json.dumps(inference_spec, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except InferenceSpecError as exc:
                inference_spec = None
                spec_skipped_reason = str(exc)
                warnings.warn(
                    f"inference_spec.json omitted from bundle: {spec_skipped_reason} "
                    "The bundle is still valid but is not self-usable for prediction.",
                    stacklevel=2,
                )

            # 4c. Self-usable prediction artifacts (EXPORT C2): the standalone
            # predict.py wrapper, its requirements, a safe raw example and the
            # expected output of running it. Only when a usable spec exists.
            example_record: dict[str, Any] | None = None
            smoke_test_skipped = False
            if inference_spec is not None:
                shutil.copy2(_PREDICT_TEMPLATE, str(work / "predict.py"))
                (work / "requirements.txt").write_text(_REQUIREMENTS, encoding="utf-8")
                example_record = example_input or build_example_input(inference_spec)
                (work / "example_input.json").write_text(
                    json.dumps(example_record, indent=2, ensure_ascii=False), encoding="utf-8")
                if using_mxw_streaming:
                    # PESOS_GRANDES C7 auditoría: NO se ejecuta el smoke-test
                    # (`_run_prediction`) para un modelo grande — el
                    # `model.onnx.data` ni siquiera está presente aquí (lo
                    # streamea el caller directo al zip), y cargar un ONNX de
                    # GiBs en onnxruntime + un forward CPU dentro del POST de
                    # export es exactamente el coste que este corte evita. El
                    # bundle es self-usable igual (predict.py + inference_spec +
                    # example_input); solo omite el `expected_output.json` de
                    # referencia (que el usuario puede regenerar con `python
                    # predict.py example_input.json`).
                    smoke_test_skipped = True
                else:
                    # A usable bundle is only shipped fully formed: predict.py is smoke-tested
                    # at packaging time and expected_output.json is its baseline. If we cannot
                    # run it (no onnxruntime), we refuse to stamp the bundle as usable rather
                    # than emit it half-built with a README that references a missing file.
                    if not ort_available():
                        raise EdgeBundleError(
                            "Model yields a usable inference_spec, but onnxruntime is not "
                            "installed, so predict.py cannot be smoke-tested and "
                            "expected_output.json cannot be generated. Install it "
                            "(pip install 'matrixai-core[export]') to produce a self-usable bundle."
                        )
                    expected = _run_prediction(work, example_record)
                    (work / "expected_output.json").write_text(
                        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")

            # 5. export_manifest.json
            if eq_result is not None:
                write_export_manifest(export_result, eq_result, work / "export_manifest.json")
            else:
                _write_export_manifest_no_eq(export_result, work / "export_manifest.json")

            # 6. README.md — refleja los ficheros REALES del bundle (BAJA C7
            # auditoría): con external-data lista `model.onnx.data`; sin
            # `params.best.json` cuando se omitió (grande); sin
            # `expected_output.json` si el smoke-test se saltó.
            (work / "README.md").write_text(
                _build_readme(program, export_result, eq_result,
                              inference_spec=inference_spec,
                              example_input=example_record,
                              has_params_json=params_path is not None,
                              external_data=export_result.external_data,
                              smoke_test_skipped=smoke_test_skipped),
                encoding="utf-8",
            )

            # PESOS_GRANDES C7 auditoría: en streaming el `.onnx.data` NO está
            # en `work` (lo escribe el caller directo al zip) — se excluye de
            # `files` para que el listado sea honesto (el caller lo añade).
            # Atomic promotion: remove stale outdir then rename temp into place
            if outdir.exists():
                shutil.rmtree(str(outdir))
            shutil.copytree(str(work), str(outdir))

        files = sorted(str(p.relative_to(outdir)) for p in outdir.iterdir() if p.is_file())
        if using_mxw_streaming:
            files = sorted(set(files) | {"model.onnx.data"})

        return EdgeBundleResult(
            bundle_dir=str(outdir),
            files=files,
            # `export_result` (no `parameter_set`) es la fuente de verdad: para
            # el camino state_dict, `parameter_set_id` es "torch_state" ahí,
            # mientras que la plantilla pasada como `parameter_set` podría
            # llevar otro id de plantilla — evita la discrepancia.
            model_hash=export_result.model_hash,
            parameter_set_id=export_result.parameter_set_id,
            export_result=export_result,
            equivalence_result=eq_result,
            inference_spec_skipped_reason=spec_skipped_reason,
            equivalence_skipped_reason=eq_skipped_reason,
            external_data_layout=external_data_layout,
        )


def create_edge_bundle(
    program: MatrixAIProgram,
    parameter_set: ParameterSet | None,
    mxai_path: str | Path,
    params_path: str | Path | None,
    outdir: str | Path,
    *,
    state_dict: dict[str, Any] | None = None,
    mxw_path: str | Path | None = None,
    mxw_header: dict[str, Any] | None = None,
    model_hash: str | None = None,
    parameter_schema_hash: str | None = None,
    validate: bool = True,
    force: bool = False,
    field_ranges: dict[str, tuple[float, float]] | None = None,
    field_categories: dict[str, list[str]] | None = None,
    field_types: dict[str, str] | None = None,
    labels: list[str] | None = None,
    example_input: dict[str, Any] | None = None,
) -> EdgeBundleResult:
    return EdgeBundler().bundle(
        program, parameter_set, mxai_path, params_path, outdir,
        state_dict=state_dict, mxw_path=mxw_path, mxw_header=mxw_header,
        model_hash=model_hash, parameter_schema_hash=parameter_schema_hash,
        validate=validate, force=force,
        field_ranges=field_ranges,
        field_categories=field_categories,
        field_types=field_types,
        labels=labels,
        example_input=example_input,
    )


# ---------------------------------------------------------------------------
# Prediction artifacts (C2)
# ---------------------------------------------------------------------------

def _run_prediction(bundle_work: Path, record: dict[str, Any]) -> Any | None:
    """Run the bundled predict.py on the example to produce expected_output.json.

    Best-effort smoke test executed at packaging time: it exercises the very code
    the consumer will run. Returns None if onnxruntime is unavailable.
    """
    if not ort_available():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_matrixai_bundled_predict", str(bundle_work / "predict.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MatrixAIModel(str(bundle_work / "inference_spec.json"))
    return model.predict(record)


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

def _build_model_manifest(program: MatrixAIProgram, parameter_set: ParameterSet) -> dict:
    from matrixai.compiler import BackendContractAnalyzer
    report = BackendContractAnalyzer().analyze(program)
    inputs: list[dict] = []
    for v in program.vectors:
        inputs.append({"kind": "vector", "name": v.name, "size": v.size,
                       "dtype": "float32", "fields": list(v.fields)})
    for s in program.sequences:
        inputs.append({"kind": "sequence", "name": s.name, "length": s.length,
                       "vocab_size": s.vocab_size, "dtype": "int64"})
    return {
        "project": program.project,
        "model_hash": parameter_set.model_hash,
        "parameter_schema_hash": parameter_set.parameter_schema_hash,
        "parameter_set_id": parameter_set.parameter_set_id,
        "inputs": inputs,
        "vectors": [
            {"name": v.name, "size": v.size, "fields": list(v.fields)}
            for v in program.vectors
        ],
        "sequences": [
            {"name": s.name, "length": s.length, "vocab_size": s.vocab_size}
            for s in program.sequences
        ],
        "functions": [
            {"name": f.name, "kind": f.semantic.kind}
            for f in program.functions
        ],
        "backend_contract": {
            "target": report.target,
            "ok": report.ok,
            # BackendNodeReport is a dataclass-like domain object, not JSON
            # serializable.  Transformer programs deliberately keep one
            # unsupported program-level forward node even though export is
            # supported, so C5 is the first bundle path that reliably reaches
            # this non-empty branch.
            "unsupported_nodes": [node.to_dict() for node in report.unsupported_nodes],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_export_manifest_no_eq(export_result: OnnxExportResult, path: Path) -> None:
    data = {
        "model_hash": export_result.model_hash,
        "parameter_schema_hash": export_result.parameter_schema_hash,
        "parameter_set_id": export_result.parameter_set_id,
        "format": "onnx",
        "format_version": export_result.opset_version,
        "input_name": export_result.input_name,
        "input_shape": export_result.input_shape,
        "output_name": export_result.output_name,
        "output_shape": export_result.output_shape,
        "exported_function": export_result.exported_functions[0] if export_result.exported_functions else None,
        "tolerance": None,
        "equivalence_check": None,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _readme_quickstart(inference_spec: dict[str, Any], example_input: dict[str, Any] | None,
                       smoke_test_skipped: bool = False) -> str:
    out = inference_spec.get("output", {})
    kind = out.get("kind", "")
    labels = out.get("labels") or []
    if kind == "classification" or kind == "binary_classification":
        out_desc = f"a probability per class ({', '.join(labels)})"
    elif kind == "regression":
        out_desc = "a single numeric value"
    else:
        out_desc = "a raw output vector"
    example_json = json.dumps(example_input or {}, ensure_ascii=False)
    # PESOS_GRANDES C7 auditoría: para un modelo grande no se generó
    # `expected_output.json` (el smoke-test se saltó) — el README no promete
    # reproducirlo; en su lugar explica cómo generarlo.
    reproduce = (
        "That produces the prediction; there is no bundled `expected_output.json` for "
        "this (large) model — running the command above once writes your own baseline."
        if smoke_test_skipped else
        "That should reproduce `expected_output.json`."
    )
    return f"""
## Quick start

This model is self-usable: feed **raw, human-readable values** and get back {out_desc}.
Normalization and category encoding are handled for you by `predict.py`.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python predict.py --input example_input.json
```

{reproduce} From your own code:

```python
from predict import MatrixAIModel

model = MatrixAIModel()                 # loads inference_spec.json next to predict.py
print(model.predict({example_json}))
```
"""


def _build_readme(
    program: MatrixAIProgram,
    export_result: OnnxExportResult,
    eq_result: OnnxEquivalenceResult | None,
    *,
    inference_spec: dict[str, Any] | None = None,
    example_input: dict[str, Any] | None = None,
    has_params_json: bool = True,
    external_data: bool = False,
    smoke_test_skipped: bool = False,
) -> str:
    project = program.project
    out_name = export_result.output_name
    out_shape = export_result.output_shape

    is_sequence = bool(program.sequences)
    if is_sequence:
        seq = program.sequences[0]
        input_name = seq.name
        mask_name = f"{seq.name}_mask"
        input_desc = (
            f"`{seq.name}` shape `{export_result.input_shape}` (int64 token IDs) + "
            f"`{mask_name}` with the same shape (float32; 1 real / 0 padding)"
        )
        inference_snippet = (
            f"sess = ort.InferenceSession(\"model.onnx\")\n"
            f"ids = np.array([[...]], dtype=np.int64)  # shape [batch, {seq.length}]\n"
            f"mask = np.ones_like(ids, dtype=np.float32)\n"
            f"result = sess.run(None, {{\"{seq.name}\": ids, \"{mask_name}\": mask}})[0]  # {out_shape}"
        )
    else:
        vec = program.vectors[0] if program.vectors else None
        input_name = vec.name if vec else "input"
        vec_size = vec.size if vec else "?"
        input_desc = f"`{input_name}` shape `{export_result.input_shape}` (float32)"
        inference_snippet = (
            f"sess = ort.InferenceSession(\"model.onnx\")\n"
            f"x = np.array([[...]], dtype=np.float32)  # shape [batch, {vec_size}]\n"
            f"result = sess.run(None, {{\"{input_name}\": x}})[0]  # {out_shape}"
        )

    eq_line = ""
    if eq_result is not None:
        status = "PASS" if eq_result.passed else "FAIL"
        eq_line = (
            f"\nEquivalence check: {status} "
            f"(max_abs_diff={eq_result.max_abs_diff:.2e}, "
            f"atol={eq_result.atol:.0e}, n={eq_result.n_samples})\n"
        )

    skipped = ""
    if export_result.skipped_functions:
        skipped = (
            f"\nNote: the following functions were not exported (unsupported kind): "
            f"{', '.join(export_result.skipped_functions)}\n"
        )

    quickstart = ""
    usable_files = ""
    if inference_spec is not None:
        quickstart = _readme_quickstart(inference_spec, example_input, smoke_test_skipped)
        usable_files = (
            "| `inference_spec.json` | How a raw record maps to the model input (the \"tokenizer\") |\n"
            "| `predict.py` | Standalone wrapper: raw values in, labelled prediction out |\n"
            "| `requirements.txt` | Minimal deps to run predict.py (numpy + onnxruntime) |\n"
            "| `example_input.json` | A ready-to-run raw example |\n"
        )
        # PESOS_GRANDES C7 auditoría (BAJA): el README lista los ficheros
        # REALES — `expected_output.json` solo si el smoke-test corrió (para un
        # grande se salta, ver docstring del bundle).
        if not smoke_test_skipped:
            usable_files += (
                "| `expected_output.json` | The output predict.py should produce for that example |\n"
            )

    # BAJA C7: `params.best.json` solo se lista si de verdad va en el zip
    # (se omite para un modelo grande — el ONNX lleva los pesos); con
    # external-data, además hay un `model.onnx.data` que hay que listar.
    params_row = "| `params.best.json` | Trained parameter weights |\n" if has_params_json else ""
    onnx_data_row = (
        "| `model.onnx.data` | ONNX external weights (loaded automatically next to model.onnx) |\n"
        if external_data else ""
    )

    return f"""# {project} Edge Bundle

MatrixAI model exported for edge/production inference.
Actions remain `simulate_only`. This bundle only provides predictions.
{quickstart}
## Files

| File | Description |
|------|-------------|
| `model.mxai` | MatrixAI model definition (source of truth) |
{params_row}| `model.onnx` | ONNX model, opset {export_result.opset_version} |
{onnx_data_row}| `model_manifest.json` | Model metadata, hashes and backend contract |
| `export_manifest.json` | Export metadata, tolerance and equivalence check |
{usable_files}| `README.md` | This file |

## Model info

- Project: `{project}`
- Model hash: `{export_result.model_hash}`
- Parameter set: `{export_result.parameter_set_id}`
- Input: {input_desc}
- Output: `{out_name}` shape `{out_shape}`
{eq_line}{skipped}
## Advanced: raw onnxruntime access

For most uses prefer `predict.py` above (it handles normalization and labels).
The raw ONNX graph expects an already-normalized float32 vector:

```python
import onnxruntime as ort
import numpy as np

{inference_snippet}
```

## Verifying integrity

```python
import json
with open("model_manifest.json") as f:
    manifest = json.load(f)
assert manifest["model_hash"] == "{export_result.model_hash}"
assert manifest["parameter_schema_hash"] == "{export_result.parameter_schema_hash}"
```
"""

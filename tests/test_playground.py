from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from matrixai.agents import ChatCompletionsLLMProposalProvider
from matrixai.playground import (
    DEFAULT_INPUT,
    DEFAULT_PROMPT,
    _INDEX_HTML,
    _diag,
    _example_payload,
    analyze_playground_request,
)


ROOT = Path(__file__).resolve().parents[1]


class MatrixAIPlaygroundTest(unittest.TestCase):
    def test_prompt_analysis_returns_full_pipeline_artifacts(self) -> None:
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            result = analyze_playground_request(
                {
                    "mode": "prompt",
                    "prompt": DEFAULT_PROMPT,
                    "input_json": json.dumps(DEFAULT_INPUT),
                }
            )

        check_names = [check["name"] for check in result["checks"]]

        self.assertTrue(result["ok"])
        self.assertTrue(result["accepted"])
        self.assertIn("architect_plan", check_names)
        self.assertIn("planner_verifier", check_names)
        self.assertIn("runtime_compiler_diagnose", check_names)
        self.assertIn("PROJECT", result["semantic_text"])
        self.assertIn("PROJECT", result["mxai"])
        self.assertIn("flowchart LR", result["graph_mermaid"])
        self.assertTrue(result["diagnose"]["ok"])
        self.assertEqual(result["visual_model"]["project"], "EmailAgent")
        self.assertEqual(result["workflow"][0]["label"], "Modelo")
        self.assertTrue(result["visual_model"]["actions"][0]["simulated"])

    def test_mxai_analysis_supports_typed_domain_example(self) -> None:
        mxai_text = (ROOT / "examples" / "email-agent.typed.mxai").read_text(encoding="utf-8")
        input_text = (ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8")

        result = analyze_playground_request(
            {"mode": "mxai", "mxai_text": mxai_text, "input_json": input_text}
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["program"]["project"], "EmailAgentTyped")
        self.assertTrue(result["typecheck"]["ok"])
        self.assertTrue(result["run_result"]["actions"])
        self.assertTrue(result["diagnose"]["ok"])
        visual = result["visual_model"]
        self.assertEqual(visual["overview"]["inputs"], 8)
        self.assertEqual(visual["inputs"][0]["type"], "Score")
        self.assertEqual(visual["inputs"][0]["range"], "[0.0, 1.0]")
        self.assertEqual(visual["security"]["policy"], "mvp_simulate_only")
        self.assertIn("Bearer", visual["serving"]["auth"])
        self.assertIn("/openapi.json", visual["serving"]["openapi_url"])
        self.assertIn("curl -X POST", visual["serving"]["curl"])
        workflow = {step["id"]: step for step in result["workflow"]}
        self.assertEqual(workflow["training"]["status"], "warning")
        self.assertIn("sin manifiesto", workflow["training"]["summary"])
        self.assertTrue(result["training_artifacts"]["warnings"])

    def test_mxai_analysis_connects_real_training_artifacts(self) -> None:
        example = _example_payload("fall-risk")

        result = analyze_playground_request(example)

        self.assertTrue(result["ok"])
        training = result["training_artifacts"]
        self.assertTrue(training["available"])
        self.assertTrue(training["ok"])
        self.assertEqual(training["spec"]["dataset"]["target"]["name"], "risk_label")
        self.assertEqual(training["dataset"]["rows"], 8)
        self.assertEqual(training["verification"]["trainable_parameters"][0]["name"], "W1")
        self.assertTrue(training["manifest"]["ok"])
        self.assertEqual(result["visual_model"]["training"]["dataset"]["rows"], 8)
        self.assertTrue(result["evaluation_artifacts"]["available"])
        self.assertEqual(result["evaluation_artifacts"]["report"]["accuracy"], 1.0)
        workflow = {step["id"]: step["status"] for step in result["workflow"]}
        self.assertEqual(workflow["training"], "valid")
        self.assertEqual(workflow["evaluation"], "valid")

    def test_mxai_analysis_summarizes_evaluation_report(self) -> None:
        example = _example_payload("fall-risk")
        example["evaluation_report_text"] = json.dumps(
            {
                "model": "examples/fall-risk.typed.mxai",
                "model_hash": "model-hash",
                "parameter_schema_hash": "schema-hash",
                "parameter_set_id": "fall-risk-best",
                "dataset": "examples/fall-risk.test.csv",
                "dataset_fingerprint": "dataset-fingerprint",
                "dataset_schema": {},
                "rows": 4,
                "loss": 0.25,
                "accuracy": 0.75,
                "labels": ["low", "high"],
                "confusion_matrix": {
                    "low": {"low": 2, "high": 0},
                    "high": {"low": 1, "high": 1},
                },
                "per_label": {
                    "low": {"precision": 0.66, "recall": 1.0, "f1": 0.8},
                    "high": {"precision": 1.0, "recall": 0.5, "f1": 0.66},
                },
                "macro_precision": 0.83,
                "macro_recall": 0.75,
                "macro_f1": 0.73,
                "backend": {"target": "stdlib"},
            }
        )

        result = analyze_playground_request(example)

        self.assertTrue(result["ok"])
        evaluation = result["evaluation_artifacts"]
        self.assertTrue(evaluation["available"])
        self.assertTrue(evaluation["ok"])
        self.assertEqual(evaluation["report"]["accuracy"], 0.75)
        self.assertEqual(result["visual_model"]["evaluation"]["backend"]["target"], "stdlib")
        workflow = {step["id"]: step["status"] for step in result["workflow"]}
        self.assertEqual(workflow["evaluation"], "valid")

    def test_mxai_analysis_exposes_backend_contract(self) -> None:
        example = _example_payload("fall-risk")

        result = analyze_playground_request(example)

        self.assertIn("backend_contract", result)
        bc = result["backend_contract"]
        self.assertIn("target", bc)
        self.assertIn("parameter_manifest", bc)
        self.assertGreater(len(bc["parameter_manifest"]), 0)
        # visual_model should also carry it
        vm_bc = result["visual_model"]["backend_contract"]
        self.assertEqual(vm_bc["target"], bc["target"])
        self.assertEqual(len(vm_bc["parameter_manifest"]), len(bc["parameter_manifest"]))

    def test_invalid_manifest_blocks_analysis_result(self) -> None:
        example = _example_payload("fall-risk")
        manifest = json.loads(example["manifest_text"])
        manifest["datasets"][0]["source"] = "missing.csv"
        example["manifest_text"] = json.dumps(manifest)

        result = analyze_playground_request(example)

        self.assertFalse(result["ok"])
        self.assertFalse(result["training_artifacts"]["ok"])
        workflow = {step["id"]: step for step in result["workflow"]}
        self.assertEqual(workflow["training"]["status"], "blocked")
        self.assertIn("bloqueado", workflow["training"]["summary"])

    def test_missing_training_csv_blocks_analysis_result(self) -> None:
        example = _example_payload("email-agent")
        example["training_text"] = example["training_text"].replace(
            "email-agent.train.csv",
            "email-agent.traind.csv",
        )

        result = analyze_playground_request(example)

        self.assertFalse(result["ok"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["training_artifacts"]["ok"])
        self.assertIn("DATASET source not found", result["training_artifacts"]["verification"]["errors"][0])
        workflow = {step["id"]: step for step in result["workflow"]}
        self.assertEqual(workflow["training"]["status"], "blocked")

    def test_invalid_mode_is_reported_without_exception(self) -> None:
        result = analyze_playground_request({"mode": "unknown"})

        self.assertFalse(result["ok"])
        self.assertIn("mode must be", result["error"])

    def test_playground_html_contains_guided_demo_controls(self) -> None:
        self.assertIn('id="demoGuide"', _INDEX_HTML)
        self.assertIn('data-demo-example="email-agent"', _INDEX_HTML)
        self.assertIn('data-demo-example="fall-risk"', _INDEX_HTML)
        self.assertIn('data-demo-tab=', _INDEX_HTML)
        self.assertIn('id="analysisNote"', _INDEX_HTML)
        self.assertIn('id="statusDetail"', _INDEX_HTML)
        self.assertIn('id="diagnosticsSummary"', _INDEX_HTML)
        self.assertIn("Bloqueos detectados", _INDEX_HTML)
        self.assertIn("focusFirstBlockedView", _INDEX_HTML)
        self.assertIn("training: 'dataView'", _INDEX_HTML)
        self.assertIn("Accepted with warnings", _INDEX_HTML)
        self.assertIn("OK con aviso", _INDEX_HTML)
        self.assertIn("No hay manifiesto cargado", _INDEX_HTML)
        self.assertIn("no hay cambios visibles", _INDEX_HTML)
        self.assertIn("byId('run').disabled = true", _INDEX_HTML)
        self.assertIn("tab: 'evaluationView'", _INDEX_HTML)
        self.assertIn("dataset_template_text", _INDEX_HTML)
        self.assertNotIn("byId('runOut').innerHTML = ''", _INDEX_HTML)
        self.assertNotIn("join('\n')", _INDEX_HTML)

    def test_diag_stdout_requires_debug_one_but_always_logs_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "matrixai_diag.log"
            with unittest.mock.patch.dict(
                os.environ,
                {"MATRIXAI_DIAG_LOG": str(log_path)},
                clear=False,
            ):
                os.environ.pop("MATRIXAI_DEBUG", None)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    _diag("silent")
                self.assertEqual(stdout.getvalue(), "")

                os.environ["MATRIXAI_DEBUG"] = "0"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    _diag("debug-zero")
                self.assertEqual(stdout.getvalue(), "")

                os.environ["MATRIXAI_DEBUG"] = "1"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    _diag("debug-one")
                self.assertIn("[matrixai] debug-one", stdout.getvalue())

                text = log_path.read_text(encoding="utf-8")
                self.assertIn("[matrixai] silent", text)
                self.assertIn("[matrixai] debug-zero", text)
                self.assertIn("[matrixai] debug-one", text)


class P8PlaygroundPipelineTest(unittest.TestCase):
    """Tests for P8: pipeline_stages, artifacts and origin labels."""

    def _prompt_result(self) -> dict:
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            return analyze_playground_request({"mode": "prompt", "prompt": DEFAULT_PROMPT})

    def test_prompt_result_has_pipeline_stages(self) -> None:
        result = self._prompt_result()
        self.assertIn("pipeline_stages", result)
        stages = result["pipeline_stages"]
        self.assertEqual(len(stages), 7)
        names = [s["name"] for s in stages]
        self.assertIn("architect_plan", names)
        self.assertIn("python_compiler", names)

    def test_prompt_pipeline_stages_all_ok(self) -> None:
        result = self._prompt_result()
        for stage in result["pipeline_stages"]:
            self.assertIn(stage["status"], ("ok", "warning"), msg=f"{stage['name']} has unexpected status")

    def test_prompt_result_has_artifacts_with_source(self) -> None:
        result = self._prompt_result()
        self.assertIn("artifacts", result)
        arts = result["artifacts"]
        self.assertIn("semantic", arts)
        self.assertIn("mxai", arts)
        self.assertEqual(arts["semantic"]["source"], "PromptAgent")
        self.assertEqual(arts["mxai"]["source"], "derived_from_prompt")
        self.assertIn("PROJECT", arts["mxai"]["text"])

    def test_mxai_mode_has_pipeline_stages_and_artifacts(self) -> None:
        mxai_text = (ROOT / "examples" / "email-agent.typed.mxai").read_text(encoding="utf-8")
        result = analyze_playground_request({"mode": "mxai", "mxai_text": mxai_text})
        self.assertIn("pipeline_stages", result)
        names = [s["name"] for s in result["pipeline_stages"]]
        self.assertIn("parser", names)
        self.assertIn("python_compiler", names)
        arts = result["artifacts"]
        self.assertEqual(arts["mxai"]["source"], "playground:mxai")

    def test_pipeline_fail_stage_marked_correctly(self) -> None:
        bad_semantic = "PROJECT BadModel\nINTENT x\nMODE classification\nENTITY X\n"
        with unittest.mock.patch.object(
            ChatCompletionsLLMProposalProvider,
            "from_env",
            side_effect=ValueError("no_api_key"),
        ):
            result = analyze_playground_request({"mode": "semantic", "semantic_text": bad_semantic})
        stages = result["pipeline_stages"]
        statuses = {s["name"]: s["status"] for s in stages}
        # At least one stage should be fail or skipped
        self.assertTrue(
            any(v in ("fail", "skipped") for v in statuses.values()),
            msg=f"Expected fail/skipped stage in: {statuses}",
        )

    def test_html_contains_pipeline_view_elements(self) -> None:
        self.assertIn("pipelineView", _INDEX_HTML)
        self.assertIn("pipeline_stages", _INDEX_HTML)
        self.assertIn("renderPipelineView", _INDEX_HTML)
        self.assertIn("Reanalizar como .semantic", _INDEX_HTML)
        self.assertIn("reanalizeSemanticBtn", _INDEX_HTML)
        self.assertIn("semanticSourceBadge", _INDEX_HTML)
        self.assertIn("mxaiSourceBadge", _INDEX_HTML)


class P9TrainingLoopTest(unittest.TestCase):
    """Tests for P9: generate-training, validate-csv, train, run-with-params."""

    MXAI = (ROOT / "examples" / "fall-risk.typed.mxai").read_text(encoding="utf-8")
    CSV = (ROOT / "examples" / "fall-risk.train.csv").read_text(encoding="utf-8")

    def _get_training_text(self) -> str:
        from matrixai.playground import _generate_training_from_mxai
        r = _generate_training_from_mxai(self.MXAI)
        self.assertTrue(r["ok"], r.get("error"))
        return r["training_text"]

    def test_generate_training_returns_mxtrain_and_template(self) -> None:
        from matrixai.playground import _generate_training_from_mxai
        r = _generate_training_from_mxai(self.MXAI)
        self.assertTrue(r["ok"])
        self.assertIn("MODEL", r["training_text"])
        self.assertIn("DATASET", r["training_text"])
        self.assertGreater(len(r["dataset_template_text"]), 0)

    def test_generate_training_empty_mxai_returns_error(self) -> None:
        from matrixai.playground import _generate_training_from_mxai
        r = _generate_training_from_mxai("")
        self.assertFalse(r["ok"])
        self.assertIn("error", r)

    def test_validate_csv_ok_with_real_data(self) -> None:
        from matrixai.playground import _validate_training_csv
        training_text = self._get_training_text()
        r = _validate_training_csv(self.MXAI, training_text, self.CSV)
        self.assertTrue(r["ok"], r)
        self.assertGreater(r["rows"], 0)

    def test_validate_csv_rejects_oversized(self) -> None:
        from matrixai.playground import _validate_training_csv, _P9_MAX_CSV_BYTES
        training_text = self._get_training_text()
        big_csv = "x" * (_P9_MAX_CSV_BYTES + 1)
        r = _validate_training_csv(self.MXAI, training_text, big_csv)
        self.assertFalse(r["ok"])
        self.assertIn("límite", r["error"])

    def test_validate_csv_rejects_invalid_training_text(self) -> None:
        from matrixai.playground import _validate_training_csv
        r = _validate_training_csv(self.MXAI, "NOT A VALID MXTRAIN", self.CSV)
        self.assertFalse(r["ok"])

    def test_run_playground_training_returns_epoch_trace(self) -> None:
        from matrixai.playground import _run_playground_training
        training_text = self._get_training_text()
        r = _run_playground_training(self.MXAI, training_text, self.CSV, epochs_override=2)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual(len(r["epochs"]), 2)
        self.assertIn("train_loss", r["epochs"][0])
        self.assertIn("validation_loss", r["epochs"][0])
        self.assertGreater(r["best_epoch"], 0)

    def test_run_playground_training_returns_params_best(self) -> None:
        from matrixai.playground import _run_playground_training
        training_text = self._get_training_text()
        r = _run_playground_training(self.MXAI, training_text, self.CSV, epochs_override=2)
        self.assertTrue(r["ok"])
        self.assertIsNotNone(r["params_best"])
        self.assertIn("parameter_set_id", r["params_best"])
        self.assertIn("W1", r["params_best"]["parameters"])

    def test_run_playground_training_caps_epochs(self) -> None:
        # The sanity ceiling clamps absurd values, but an explicit in-range request
        # is honoured verbatim (the user controls their machine). Tested on the pure
        # cap function to avoid actually training up to the ceiling.
        from matrixai.playground import _apply_epoch_cap, _run_playground_training, _P9_MAX_EPOCHS
        from matrixai.training.parser import parse_training_text
        training = parse_training_text(self._get_training_text())
        self.assertEqual(_apply_epoch_cap(training, _P9_MAX_EPOCHS + 50), _P9_MAX_EPOCHS)
        self.assertEqual(_apply_epoch_cap(training, 7), 7)  # explicit value honoured
        # e2e: a small explicit override actually limits the run
        r = _run_playground_training(self.MXAI, self._get_training_text(), self.CSV, epochs_override=5)
        self.assertTrue(r["ok"])
        self.assertLessEqual(len(r["epochs"]), 5)

    def test_playground_run_with_params_uses_trained_weights(self) -> None:
        from matrixai.playground import _run_playground_training, _playground_run_with_params
        import json as _json
        training_text = self._get_training_text()
        train_r = _run_playground_training(self.MXAI, training_text, self.CSV, epochs_override=2)
        self.assertTrue(train_r["ok"])
        params_json = _json.dumps(train_r["params_best"])
        input_json = _json.dumps({
            "FallRisk": {
                "age": 0.8, "balance_score": 0.3, "medication_count": 0.7,
                "previous_falls": 1.0, "mobility_score": 0.2,
                "vision_score": 0.4, "cognitive_score": 0.5, "living_alone": 1.0,
            }
        })
        r = _playground_run_with_params(self.MXAI, params_json, input_json)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn("result", r)

    def test_html_contains_p9_train_elements(self) -> None:
        self.assertIn("generateTrainingBtn", _INDEX_HTML)
        self.assertIn("generateTraining()", _INDEX_HTML)
        self.assertIn("trainBtn", _INDEX_HTML)
        self.assertIn("runTraining()", _INDEX_HTML)
        self.assertIn("validateCsv()", _INDEX_HTML)
        self.assertIn("csvPaste", _INDEX_HTML)
        self.assertIn("epochProgress", _INDEX_HTML)
        self.assertIn("artifactsPanel", _INDEX_HTML)
        self.assertIn("predBtn", _INDEX_HTML)
        self.assertIn("runPrediction()", _INDEX_HTML)
        self.assertIn("/api/generate-training", _INDEX_HTML)
        self.assertIn("/api/validate-csv", _INDEX_HTML)
        self.assertIn("/api/train", _INDEX_HTML)
        self.assertIn("/api/run-with-params", _INDEX_HTML)
        # async endpoints and stop button
        self.assertIn("stopTrainBtn", _INDEX_HTML)
        self.assertIn("cancelTraining()", _INDEX_HTML)
        self.assertIn("/api/train-start", _INDEX_HTML)
        self.assertIn("/api/train-cancel", _INDEX_HTML)
        # P8: edit-as-mxai button
        self.assertIn("editAsMxai()", _INDEX_HTML)
        self.assertIn("editAsMxaiBtn", _INDEX_HTML)

    def test_run_playground_training_returns_evaluation_report(self) -> None:
        from matrixai.playground import _run_playground_training
        training_text = self._get_training_text()
        r = _run_playground_training(self.MXAI, training_text, self.CSV, epochs_override=2)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn("evaluation_report", r)
        self.assertIsNotNone(r["evaluation_report"])

    def test_run_playground_training_returns_backend_field(self) -> None:
        from matrixai.playground import _run_playground_training
        training_text = self._get_training_text()
        r = _run_playground_training(self.MXAI, training_text, self.CSV, epochs_override=2)
        self.assertTrue(r["ok"])
        self.assertIn("backend", r)

    def test_submit_training_job_returns_job_id(self) -> None:
        from matrixai.playground import _submit_training_job
        training_text = self._get_training_text()
        r = _submit_training_job(self.MXAI, training_text, self.CSV, epochs_override=2)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn("job_id", r)
        self.assertIsInstance(r["job_id"], str)

    def test_get_job_status_returns_status_and_epochs(self) -> None:
        import time
        from matrixai.playground import _get_job_status, _submit_training_job, _cancel_job
        training_text = self._get_training_text()
        r = _submit_training_job(self.MXAI, training_text, self.CSV, epochs_override=2)
        self.assertTrue(r["ok"])
        job_id = r["job_id"]
        # Poll until terminal (max 35s)
        deadline = time.time() + 35
        status: dict = {}
        while time.time() < deadline:
            status = _get_job_status(job_id)
            if status["status"] in ("done", "error", "cancelled", "timeout"):
                break
            time.sleep(0.2)
        _cancel_job(job_id)  # cleanup so concurrency test can run
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertGreater(len(status["epochs"]), 0)
        self.assertIn("params_best", status)
        self.assertIn("evaluation_report", status)

    def test_cancel_job_sets_cancelled_status(self) -> None:
        from matrixai.playground import _cancel_job, _get_job_status, _submit_training_job
        training_text = self._get_training_text()
        r = _submit_training_job(self.MXAI, training_text, self.CSV, epochs_override=50)
        self.assertTrue(r["ok"])
        job_id = r["job_id"]
        cancel_r = _cancel_job(job_id)
        self.assertTrue(cancel_r["ok"])
        # Status is either cancelled (if worker still running) or done (already finished) — both are valid
        self.assertIn(cancel_r["status"], ("cancelled", "done"))
        status = _get_job_status(job_id)
        self.assertIn(status["status"], ("cancelled", "done"))

    def test_submit_blocks_when_job_already_running(self) -> None:
        import time
        from matrixai.playground import _cancel_job, _submit_training_job
        training_text = self._get_training_text()
        r1 = _submit_training_job(self.MXAI, training_text, self.CSV, epochs_override=50)
        self.assertTrue(r1["ok"])
        try:
            r2 = _submit_training_job(self.MXAI, training_text, self.CSV, epochs_override=2)
            self.assertFalse(r2["ok"])
            self.assertIn("error", r2)
        finally:
            _cancel_job(r1["job_id"])
            time.sleep(0.1)  # let worker notice cancel

    def test_get_job_status_unknown_job(self) -> None:
        from matrixai.playground import _get_job_status
        r = _get_job_status("nonexistent_id")
        self.assertFalse(r["ok"])
        self.assertIn("error", r)


class C9DenseBinaryE2ETest(unittest.TestCase):
    """E2E test for C9: NETWORK dense binary flow (Probability target).

    Exercises the full path: synthetic dataset generation → CSV validation → training
    without calling the LLM. Uses inline mxai + training text as if produced by DNG.
    """

    MXAI = """\
PROJECT ReadmissionRisk

VECTOR Patient[4]
  age: Score
  los_days: Score
  prior_count: Score
  comorbidity_index: Score
END

NETWORK Predictor
  INPUT Patient
  LAYER Dense units=16 activation=relu
  LAYER Dense units=1 activation=sigmoid
  OUTPUT readmit_prob: Probability
END

GRAPH
  Patient -> Predictor
END
"""

    TRAINING = """\
MODEL readmission_risk.mxai

DATASET ReadmissionDataset
  SOURCE csv("test.csv")
  INPUT Patient FROM COLUMNS [age, los_days, prior_count, comorbidity_index]
  TARGET label: Probability
END

LOSS BCE
  TYPE binary_cross_entropy
  PREDICTION readmit_prob
  TARGET label
END

OPTIMIZER SGD
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE W1, b1, W2, b2
END

RUN
  EPOCHS 5
END
"""

    def test_get_prediction_kind_returns_network_call(self) -> None:
        from matrixai.playground import _get_prediction_kind
        kind = _get_prediction_kind(self.MXAI, self.TRAINING)
        self.assertEqual(kind, "network_call")

    def test_generate_synthetic_dataset_probability_target(self) -> None:
        from matrixai.playground import _generate_synthetic_dataset
        r = _generate_synthetic_dataset(self.MXAI, self.TRAINING, rows=20, seed=42, mode="random")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn("csv_text", r)
        # target column must be 0.0 or 1.0
        lines = r["csv_text"].strip().splitlines()
        self.assertGreater(len(lines), 1)
        header = [col.strip() for col in lines[0].split(",")]
        label_idx = header.index("label")
        for line in lines[1:]:
            val = float(line.split(",")[label_idx].strip())
            self.assertIn(val, (0.0, 1.0))

    def test_validate_csv_with_synthetic_probability_data(self) -> None:
        from matrixai.playground import _generate_synthetic_dataset, _validate_training_csv
        gen = _generate_synthetic_dataset(self.MXAI, self.TRAINING, rows=20, seed=7, mode="random")
        self.assertTrue(gen["ok"], gen.get("error"))
        r = _validate_training_csv(self.MXAI, self.TRAINING, gen["csv_text"])
        self.assertTrue(r["ok"], r.get("error"))
        self.assertGreater(r["rows"], 0)

    def test_run_playground_training_dense_binary_produces_epochs(self) -> None:
        from matrixai.playground import _generate_synthetic_dataset, _run_playground_training
        gen = _generate_synthetic_dataset(self.MXAI, self.TRAINING, rows=30, seed=1, mode="random")
        self.assertTrue(gen["ok"], gen.get("error"))
        r = _run_playground_training(self.MXAI, self.TRAINING, gen["csv_text"], epochs_override=3)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual(len(r["epochs"]), 3)
        self.assertIn("train_loss", r["epochs"][0])
        self.assertIn("validation_loss", r["epochs"][0])

    def test_run_playground_training_dense_binary_best_epoch_in_range(self) -> None:
        from matrixai.playground import _generate_synthetic_dataset, _run_playground_training
        gen = _generate_synthetic_dataset(self.MXAI, self.TRAINING, rows=30, seed=2, mode="random")
        self.assertTrue(gen["ok"], gen.get("error"))
        r = _run_playground_training(self.MXAI, self.TRAINING, gen["csv_text"], epochs_override=3)
        self.assertTrue(r["ok"], r.get("error"))
        self.assertGreaterEqual(r["best_epoch"], 1)
        self.assertLessEqual(r["best_epoch"], 3)

    def test_submit_training_job_dense_binary_completes(self) -> None:
        import time
        from matrixai.playground import (
            _cancel_job,
            _generate_synthetic_dataset,
            _get_job_status,
            _submit_training_job,
        )
        gen = _generate_synthetic_dataset(self.MXAI, self.TRAINING, rows=30, seed=3, mode="random")
        self.assertTrue(gen["ok"], gen.get("error"))
        r = _submit_training_job(self.MXAI, self.TRAINING, gen["csv_text"], epochs_override=3)
        self.assertTrue(r["ok"], r.get("error"))
        job_id = r["job_id"]
        # Poll until terminal (max 35s)
        deadline = time.time() + 35
        status: dict = {}
        while time.time() < deadline:
            status = _get_job_status(job_id)
            if status["status"] in ("done", "error", "cancelled", "timeout"):
                break
            time.sleep(0.2)
        _cancel_job(job_id)
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertGreater(len(status["epochs"]), 0)
        self.assertIn("epoch", status["epochs"][0])


if __name__ == "__main__":
    unittest.main()

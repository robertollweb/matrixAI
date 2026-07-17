# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""TRANSFORMER C4 (auditoría [ALTA-2]) — ruta CLI .mxtrain para BLOCK TRANSFORMER.

El contrato exige poder entrenar por CLI con CSVs de token-ids pre-tokenizados
(formato P11: columnas t0..t{L-1}). Antes de este módulo, el CLI seleccionaba
trainers que exigen un VECTOR y una SEQUENCE moría antes de llegar al trainer
de C4. Este trainer:

  - adapta el CSV (columnas del DATASET INPUT) a ejemplos {sequence: [L] ids},
  - resuelve el type_result CON sequence_map,
  - entrena vía train_composite_network_torch (torch es EL backend del bloque
    — invariante 6: el stdlib es solo referencia de test) honrando el
    OPTIMIZER TYPE declarado (adam|sgd),
  - guarda parameter_set.json + training_trace.json y devuelve el mismo
    TrainingRunResult que los demás trainers CLI.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable

from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.parameters.store import program_hash, write_parameter_set
from matrixai.parser import parse_file
from matrixai.text.tokenizer import ByteTokenizer
from matrixai.training.data import CSVDataAdapter, CSVTextDataAdapter
from matrixai.training.dense_trainer import _examples_to_xy, _labels_from_spec, _resolve_path
from matrixai.training.spec import TrainingRunResult, TrainingSpec
from matrixai.training.verifier import TrainingVerifier


def _resolve_transformer_dataset(
    training: TrainingSpec,
    base: Path,
    *,
    model_path: Path | None = None,
    data_path: str | None = None,
) -> dict[str, Any]:
    """Carga compartida train/evaluate (auditoría C6 [ALTA-1]): programa +
    typecheck CON sequence_map + ejemplos ``{sequence: [L] ids}`` desde el CSV.

    Antes de esto el evaluador CLI no existía y `mx evaluate` caía en el
    evaluador denso, que no sabe leer una SEQUENCE y producía un
    evaluation_report VACÍO (rows=0, accuracy=0.0) en verde — justo el
    fichero que `mx registry push` exige, así que un transformer podía
    registrarse en P21 con métricas mentirosas.
    """
    from matrixai.types import check_composite_network_types

    if model_path is None:
        model_path = _resolve_path(training.model, base)
        if model_path is None:
            raise FileNotFoundError(f"Model file not found: {training.model}")
    program = parse_file(model_path)
    if not program.networks:
        raise ValueError(f"No NETWORK blocks found in {training.model}")
    net = program.networks[0]
    if not getattr(net, "transformer_blocks", []):
        raise ValueError(
            f"NETWORK {net.name} has no BLOCK TRANSFORMER — use the dense/"
            f"stdlib trainers"
        )

    vector_map = {v.name: v for v in program.vectors}
    sequence_map = {s.name: s for s in program.sequences}
    type_result = check_composite_network_types(net, vector_map, sequence_map)
    if not type_result.ok:
        raise ValueError("; ".join(type_result.errors))
    seq = sequence_map.get(net.input)
    if seq is None:
        raise ValueError(f"INPUT {net.input!r} is not a declared SEQUENCE")

    # SECUENCIAS_PRODUCTO C3: el DATASET INPUT admite dos formas — UNA
    # columna de texto CRUDO (canónica del producto, tokenizada aquí mismo
    # con el mismo ByteTokenizer que declara la SEQUENCE — invariante 1:
    # nunca se piden ids al usuario) o una columna por posición con ids
    # pre-tokenizados (formato legacy del contrato A, conservado para no
    # romper .mxtrain/CSVs ya generados por CLI/tests).
    columns = list(training.dataset.input.columns)
    loss_fn = training.loss.type if training.loss else "cross_entropy"
    labels = _labels_from_spec(training)

    source = data_path if data_path else training.dataset.source
    resolved_data = _resolve_path(source, base)
    if resolved_data is None:
        raise FileNotFoundError(f"Dataset not found: {source}")

    # Auditoría C3 [MEDIA] + residual: con seq.length == 1 AMBAS formas
    # tendrían 1 columna. Un heurístico basado en el NOMBRE de columna
    # ("t0" = legacy) fue intentado y descartado: "t0" es también un
    # nombre de campo Text perfectamente válido declarado en el prompt
    # (`t0: Text[1]`), así que el heurístico rompía la forma canónica del
    # propio producto para ese nombre. Ningún fixture/test real de este
    # repo usa el formato legacy pre-tokenizado degenerado a L=1 (grep
    # confirmado) — UNA columna es SIEMPRE texto crudo, sin excepción; el
    # formato legacy de una columna por posición solo aplica cuando hay
    # más de una posición (seq.length > 1), donde no hay ambigüedad
    # posible por conteo.
    if len(columns) == 1:
        tokenizer = ByteTokenizer(seq.length)
        adapter = CSVTextDataAdapter(
            resolved_data, seq.name, columns[0], training.dataset.target.name,
            tokenizer, labels if labels else None,
        )
        # Auditoría C3 [ALTA]: PAD=256 debe quedar fuera de atención/pooling
        # — el pad_id viaja hasta el trainer/evaluador torch (que ya sabe
        # construir la máscara, __mask__ = ids != pad_id) SOLO para esta
        # rama de texto crudo real.
        pad_id: int | None = ByteTokenizer.PAD
    elif len(columns) == seq.length:
        adapter = CSVDataAdapter(
            resolved_data, seq.name, columns,
            training.dataset.target.name, labels if labels else None,
        )
        # Legacy pre-tokenizado: filas ya fijas a L, sin relleno conocido —
        # ninguna posición se enmascara (comportamiento previo intacto).
        pad_id = None
    else:
        raise ValueError(
            f"DATASET INPUT declares {len(columns)} columns but SEQUENCE "
            f"{seq.name!r} has length {seq.length} — must be either ONE "
            f"raw-text column, or one per position (t0..t{seq.length - 1}) "
            "of pre-tokenized ids"
        )
    xy = _examples_to_xy(adapter.examples(), loss_fn, labels)
    examples = [({seq.name: [int(v) for v in x]}, y) for x, y in xy]
    if not examples:
        raise ValueError(f"Dataset produced no valid examples: {resolved_data}")

    return {
        "program": program,
        "net": net,
        "type_result": type_result,
        "seq": seq,
        "examples": examples,
        "labels": labels,
        "loss_fn": loss_fn,
        "resolved_data": resolved_data,
        "pad_id": pad_id,
    }


class TransformerSupervisedTrainer:
    """Train a composite network with BLOCK TRANSFORMER from a .mxtrain spec."""

    def train(
        self,
        training: TrainingSpec,
        output_dir: str | None = None,
        base_path: Path | None = None,
        training_path: Path | None = None,
        epoch_callback: Callable[[dict[str, Any]], None] | None = None,
        seed: int = 42,
    ) -> TrainingRunResult:
        from matrixai.parameters.tensor_bridge import torch_available
        from matrixai.training.composite_torch_trainer import (
            evaluate_composite_network_torch,
            train_composite_network_torch,
        )

        if not torch_available():
            raise ValueError(
                "BLOCK TRANSFORMER training requires torch (the block's product "
                "backend — invariante 6); install torch or use the reference "
                "forward for mini test shapes"
            )

        base = base_path or Path(".")
        out = Path(output_dir or "output")
        out.mkdir(parents=True, exist_ok=True)

        report = TrainingVerifier().verify(training, base_path=base)
        if not report.ok:
            raise ValueError("; ".join(report.errors))

        loaded = _resolve_transformer_dataset(
            training, base, model_path=Path(report.model_path),
        )
        program = loaded["program"]
        net = loaded["net"]
        type_result = loaded["type_result"]
        seq = loaded["seq"]
        examples = loaded["examples"]
        labels = loaded["labels"]
        loss_fn = loaded["loss_fn"]
        pad_id = loaded["pad_id"]

        lr = training.optimizer.learning_rate if training.optimizer else 0.01
        opt_type = training.optimizer.type if training.optimizer else "adam"
        epochs = training.run.epochs if training.run else 50
        patience = training.run.early_stop_patience if training.run else None
        batch_size = (
            training.dataset.batch.size
            if (training.dataset and training.dataset.batch) else None
        )
        # Auditoría C4 ronda 2 [ALTA-1]: propagar el device RESUELTO (el CLI ya
        # mezcló --device con el BACKEND del .mxtrain) — antes el trainer
        # inferior usaba su default "cpu" aunque el usuario pidiera cuda.
        device = training.backend.device if training.backend else "cpu"

        # Auditoría C4 ronda 2 [MEDIA-2]: honrar el DATASET SPLIT declarado
        # (ratio + seed de barajado — misma semántica que _split_examples del
        # trainer stdlib) y pasar las particiones EXPLÍCITAS al trainer.
        import random as _random
        split_spec = training.dataset.split if training.dataset else None
        indices = list(range(len(examples)))
        # BIBLIOTECA_PROYECTOS_INTELIGENTES C3: mode=temporal nunca baraja,
        # aunque haya seed (mismo criterio que `_split_examples` del
        # trainer stdlib — ver su comentario). mode ausente/"random" es el
        # comportamiento de siempre, byte-idéntico.
        if split_spec and split_spec.seed is not None and split_spec.mode != "temporal":
            _random.Random(split_spec.seed).shuffle(indices)
        train_ratio = split_spec.train if split_spec else 0.8
        n_train = max(1, min(len(examples) - 1, int(len(examples) * train_ratio))) \
            if len(examples) > 1 else len(examples)
        train_idx = set(indices[:n_train])
        train_examples = [ex for i, ex in enumerate(examples) if i in train_idx]
        val_examples = [ex for i, ex in enumerate(examples) if i not in train_idx] \
            or train_examples[:1]

        mhash = program_hash(program)
        # with_values=False: init nativo torch (M15(f)) — el camino del bloque
        ps = build_composite_network_parameter_set(
            net, type_result, mhash, seed=seed, with_values=False,
        )

        epoch_trace: list[dict[str, Any]] = []

        def _cb(entry: dict[str, Any]) -> None:
            row = {
                "epoch": entry["epoch"],
                "train_loss": round(entry["loss"], 6),
                "validation_loss": round(entry["val_loss"], 6),
                "accuracy": None,
            }
            epoch_trace.append(row)
            if epoch_callback is not None:
                epoch_callback(row)

        tr = train_composite_network_torch(
            net, ps, train_examples, loss_fn,
            lr=lr, epochs=epochs,
            early_stop=(patience, "validation_loss") if patience else None,
            seed=seed, batch_size=batch_size,
            epoch_callback=_cb,
            type_result=type_result,
            optimizer=opt_type,
            device=device,
            validation_examples=val_examples,
            pad_id=pad_id,
        )
        best_ps = tr["best_params"]
        best_state = tr["best_state_dict"]

        accuracy = 0.0
        if val_examples:
            eval_result = evaluate_composite_network_torch(
                net, best_ps, val_examples, loss_fn,
                labels=labels or None, type_result=type_result,
                device=device, state_dict=best_state, pad_id=pad_id,
            )
            accuracy = (
                max(0.0, min(1.0, eval_result.r2))
                if eval_result.is_regression() else eval_result.accuracy
            )

        run_id = str(uuid.uuid4())[:8]
        artifacts: dict[str, str] = {}
        # Auditoría C6 ronda 2 [ALTA-4]: el run dir lleva el .mxai EXACTO con
        # el que se entrenó — sin él, push_run_dir no puede cruzar
        # pesos↔modelo y el registro P21 rechaza la entrada (el cross-check
        # ya no se salta en silencio).
        import shutil as _shutil
        mxai_copy = out / "model.mxai"
        _shutil.copyfile(report.model_path, mxai_copy)
        artifacts["model_mxai"] = str(mxai_copy)
        if best_ps is not None:
            ps_path = out / "parameter_set.json"
            write_parameter_set(ps_path, best_ps)
            artifacts["parameter_set"] = str(ps_path)
        elif best_state is not None:
            # Auditoría C4 ronda 2 [ALTA-2]: un entrenamiento por encima del
            # umbral PESOS_GRANDES devolvía best_params=None y el adapter no
            # persistía NADA — los pesos entrenados se perdían. Se guardan en
            # el formato binario PESOS_GRANDES (.mxw, atómico, con hashes).
            from matrixai.parameters.binary_store import write_mxw
            mxw_path = out / "weights.mxw"
            write_mxw(
                mxw_path, best_state,
                model_hash=mhash,
                parameter_schema_hash=ps.parameter_schema_hash,
            )
            artifacts["weights_mxw"] = str(mxw_path)

        trace_path = out / "training_trace.json"
        trace_path.write_text(
            json.dumps({
                "run_id": run_id,
                "epochs": len(epoch_trace),
                "best_epoch": tr["best_epoch"],
                "best_val_loss": tr["best_val_loss"],
                "network": net.name,
                "optimizer": tr["optimizer"],
                "param_count": tr["param_count"],
                "backend": tr["backend"],
                "device": tr["device"],
                "materialized": best_ps is not None,
                "n_train": tr["n_train"],
                "n_val": tr["n_val"],
            }, indent=2),
            encoding="utf-8",
        )
        artifacts["training_trace"] = str(trace_path)

        return TrainingRunResult(
            run_id=run_id,
            output_dir=str(out),
            best_epoch=tr["best_epoch"],
            best_validation_loss=tr["best_val_loss"],
            final_train_loss=tr["train_loss"],
            final_validation_loss=tr["best_val_loss"],
            accuracy=accuracy,
            artifacts=artifacts,
        )


class TransformerSupervisedEvaluator:
    """Evaluate a trained transformer ParameterSet on a supervised CSV dataset.

    Auditoría C6 [ALTA-1/2]: `mx evaluate` no tenía rama transformer — el
    default (stdlib) caía en el evaluador denso, que no lee SEQUENCEs y
    escribía un evaluation_report VACÍO en verde; `--backend torch` moría en
    el validador genérico de ParameterSet. Este evaluador cierra el ciclo
    CLI del registro P21 (train → evaluate → registry push) con las MISMAS
    métricas que el resto (reusa `result_from_predictions` vía
    `evaluate_composite_network_torch`) y el mismo formato de report.
    """

    def evaluate(
        self,
        training: TrainingSpec,
        parameter_set: Any,
        data_path: str | None = None,
        base_path: Path | None = None,
        state_dict: dict[str, Any] | None = None,
        weights_header: dict[str, Any] | None = None,
    ) -> Any:
        from matrixai.parameters.tensor_bridge import torch_available
        from matrixai.training.composite_torch_trainer import (
            evaluate_composite_network_torch,
        )
        from matrixai.training.data import dataset_fingerprint
        from matrixai.training.spec import EvaluationResult

        if not torch_available():
            raise ValueError(
                "BLOCK TRANSFORMER evaluation requires torch (the block's "
                "product backend — invariante 6)"
            )
        if parameter_set is None and state_dict is None:
            raise ValueError("evaluate requires a parameter_set or a state_dict")

        base = base_path or Path(".")
        loaded = _resolve_transformer_dataset(training, base, data_path=data_path)
        net = loaded["net"]
        type_result = loaded["type_result"]
        examples = loaded["examples"]
        labels = loaded["labels"]
        loss_fn = loaded["loss_fn"]
        resolved_data = loaded["resolved_data"]
        pad_id = loaded["pad_id"]
        device = training.backend.device if training.backend else "cpu"

        program_digest = program_hash(loaded["program"])
        if state_dict is not None:
            # Auditoría C6 ronda 2 [ALTA-1]: camino weights.mxw — el header
            # declara a qué modelo pertenecen los pesos; evaluarlos contra
            # otro .mxai es un error, no una coincidencia de shapes.
            header = weights_header or {}
            header_model = str(header.get("model_hash", ""))
            if header_model and header_model != program_digest:
                raise ValueError(
                    f"weights.mxw belongs to a different model (header "
                    f"model_hash {header_model!r}, program hashes to "
                    f"{program_digest!r})"
                )

        result = evaluate_composite_network_torch(
            net, parameter_set, examples, loss_fn,
            labels=labels or None, type_result=type_result, device=device,
            state_dict=state_dict, pad_id=pad_id,
        )

        per_label: dict[str, dict[str, float]] = {}
        if labels and result.precision:
            for lbl in labels:
                per_label[lbl] = {
                    "precision": result.precision.get(lbl, 0.0),
                    "recall": result.recall.get(lbl, 0.0),
                    "f1": result.f1.get(lbl, 0.0),
                }
        macro_p = (
            sum(result.precision.values()) / len(result.precision)
            if result.precision else 0.0
        )
        macro_r = (
            sum(result.recall.values()) / len(result.recall)
            if result.recall else 0.0
        )

        if parameter_set is not None:
            schema_hash = parameter_set.parameter_schema_hash
            ps_id = parameter_set.parameter_set_id
        else:
            schema_hash = str((weights_header or {}).get("parameter_schema_hash", ""))
            ps_id = "torch_state"
        return EvaluationResult(
            model=training.model,
            model_hash=program_digest,
            parameter_schema_hash=schema_hash,
            parameter_set_id=ps_id,
            dataset=str(resolved_data),
            dataset_fingerprint=dataset_fingerprint(resolved_data),
            dataset_schema={},
            rows=result.rows,
            loss=result.loss,
            accuracy=result.accuracy,
            labels=list(labels or []),
            confusion_matrix=result.confusion_matrix,
            per_label=per_label,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=result.macro_f1,
            backend={"target": "torch", "device": device},
            mae=result.mae,
            rmse=result.rmse,
            r2=result.r2,
        )

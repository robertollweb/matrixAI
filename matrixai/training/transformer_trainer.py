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
from matrixai.training.data import CSVDataAdapter
from matrixai.training.dense_trainer import _examples_to_xy, _labels_from_spec, _resolve_path
from matrixai.training.spec import TrainingRunResult, TrainingSpec
from matrixai.training.verifier import TrainingVerifier


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
        from matrixai.types import check_composite_network_types

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

        program = parse_file(Path(report.model_path))
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

        # Columnas del DATASET INPUT = una por posición de la SEQUENCE
        columns = list(training.dataset.input.columns)
        if len(columns) != seq.length:
            raise ValueError(
                f"DATASET INPUT declares {len(columns)} columns but SEQUENCE "
                f"{seq.name!r} has length {seq.length} — one column per position "
                f"(t0..t{seq.length - 1})"
            )

        loss_fn = training.loss.type if training.loss else "cross_entropy"
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
        labels = _labels_from_spec(training)

        data_path = _resolve_path(training.dataset.source, base)
        if data_path is None:
            raise FileNotFoundError(f"Dataset not found: {training.dataset.source}")
        adapter = CSVDataAdapter(
            data_path, seq.name, columns,
            training.dataset.target.name, labels if labels else None,
        )
        xy = _examples_to_xy(adapter.examples(), loss_fn, labels)
        examples = [({seq.name: [int(v) for v in x]}, y) for x, y in xy]
        if not examples:
            raise ValueError("Dataset produced no valid examples")

        # Auditoría C4 ronda 2 [MEDIA-2]: honrar el DATASET SPLIT declarado
        # (ratio + seed de barajado — misma semántica que _split_examples del
        # trainer stdlib) y pasar las particiones EXPLÍCITAS al trainer.
        import random as _random
        split_spec = training.dataset.split if training.dataset else None
        indices = list(range(len(examples)))
        if split_spec and split_spec.seed is not None:
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
        )
        best_ps = tr["best_params"]
        best_state = tr["best_state_dict"]

        accuracy = 0.0
        if val_examples:
            eval_result = evaluate_composite_network_torch(
                net, best_ps, val_examples, loss_fn,
                labels=labels or None, type_result=type_result,
                device=device, state_dict=best_state,
            )
            accuracy = (
                max(0.0, min(1.0, eval_result.r2))
                if eval_result.is_regression() else eval_result.accuracy
            )

        run_id = str(uuid.uuid4())[:8]
        artifacts: dict[str, str] = {}
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

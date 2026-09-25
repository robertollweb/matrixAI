# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 audit fix — DenseSupervisedTrainer and DenseSupervisedEvaluator for CLI integration."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from matrixai.parameters import (
    ParameterSet,
    build_network_parameter_set,
    write_parameter_set,
)
from matrixai.parameters.store import program_hash
from matrixai.parser import parse_file
from matrixai.training.data import CSVDataAdapter, SupervisedExample, dataset_fingerprint
from matrixai.training.dense_backprop import dense_train_step, compute_loss
from matrixai.training.particion import particion_para, reparte
from matrixai.training.dense_evaluator import (
    DenseEvaluationResult,
    effective_labels,
    evaluate_dense_network,
)
from matrixai.training.spec import (
    EvaluationResult,
    TrainingRunResult,
    TrainingSpec,
    esfuerzo_de_entrenamiento,
)


def _resolve_path(value: str, base: Path) -> Path | None:
    direct = Path(value)
    for candidate in ([direct] if direct.is_absolute() else [direct, base / value]):
        if candidate.exists():
            return candidate
    return None


def _labels_from_spec(training: TrainingSpec) -> list[str]:
    """Extract class labels from training spec target type, if any."""
    try:
        target_type = training.dataset.target.type
        args = target_type.parameters.get("args", [])
        return [str(a) for a in args] if args else []
    except AttributeError:
        return []


def _examples_to_xy(
    examples: list[SupervisedExample],
    loss_fn: str,
    labels: list[str],
) -> list[tuple[list[float], list[float]]]:
    """Convert SupervisedExample list to (input, target) float pairs."""
    result: list[tuple[list[float], list[float]]] = []
    for ex in examples:
        x = ex.vector
        if loss_fn == "mse":
            y_val = ex.target_value if ex.target_value is not None else float(ex.label)
            result.append((x, [y_val]))
        elif loss_fn == "binary_cross_entropy":
            if ex.target_value is not None:
                result.append((x, [ex.target_value]))
            else:
                pos_label = labels[1] if len(labels) >= 2 else "positive"
                result.append((x, [1.0 if ex.label == pos_label else 0.0]))
        elif loss_fn == "cross_entropy":
            if labels:
                one_hot = [1.0 if ex.label == lbl else 0.0 for lbl in labels]
            else:
                try:
                    y_val = float(ex.label)
                    one_hot = [y_val]
                except (ValueError, TypeError):
                    one_hot = [0.0]
            result.append((x, one_hot))
        else:
            y_val = ex.target_value if ex.target_value is not None else 0.0
            result.append((x, [y_val]))
    return result


class DenseSupervisedTrainer:
    """Train a dense neural network defined as a NETWORK block in .mxai."""

    def train(
        self,
        training: TrainingSpec,
        output_dir: str | None = None,
        base_path: Path | None = None,
        training_path: Path | None = None,
        epoch_callback: Any | None = None,
        seed: int = 42,
    ) -> TrainingRunResult:
        base_path = base_path or Path(".")
        output_dir = output_dir or "output"
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_path = _resolve_path(training.model, base_path)
        if model_path is None:
            raise FileNotFoundError(f"Model file not found: {training.model}")

        program = parse_file(model_path)
        if not program.networks:
            raise ValueError(f"No NETWORK blocks found in {training.model}")

        net = program.networks[0]
        loss_fn = training.loss.type if training.loss else "mse"
        lr = training.optimizer.learning_rate if training.optimizer else 0.01
        # Auditoría C4 [ALTA-1]: este trainer stdlib SOLO implementa sgd — un
        # TYPE distinto falla cerrado, nunca se sustituye en silencio.
        opt_type = training.optimizer.type if training.optimizer else "sgd"
        if opt_type != "sgd":
            raise ValueError(
                f"OPTIMIZER TYPE {opt_type!r} is not implemented by the stdlib "
                f"dense trainer (sgd only) — use --backend torch"
            )
        # CONTRATO 118-C3b: WEIGHT_DECAY/SCHEDULE son solo-torch. Este
        # trainer (`dense_train_step`/`dense_backprop.py`) solo sabe SGD
        # plano sin regularización ni programa de tasa — declararlos y
        # entrenar de todos modos sería ignorarlos en silencio (la receta
        # declarada no coincidiría con la que se ejecutó).
        _extras_no_admitidos = (
            training.optimizer.ajustes_no_admitidos_por_stdlib() if training.optimizer else []
        )
        if _extras_no_admitidos:
            raise ValueError(
                f"OPTIMIZER {', '.join(_extras_no_admitidos)} is not implemented "
                f"by the stdlib dense trainer — use --backend torch"
            )
        epochs = training.run.epochs if training.run else 50
        # Early stopping: the only metric this trainer computes is validation_loss,
        # so any declared metric maps to it. best_ps already implements save_best.
        patience = training.run.early_stop_patience if training.run else None

        labels = _labels_from_spec(training)
        mhash = program_hash(program)

        # Run type-checking to resolve layer shapes (input_shape/output_shape)
        from matrixai.types import check_network_types
        vector_map = {v.name: v for v in program.vectors}
        type_result = check_network_types(net, vector_map)
        resolved_layers = type_result.resolved_layers if type_result.resolved_layers else net.layers

        dataset_source = training.dataset.source if training.dataset else ""
        data_path = _resolve_path(dataset_source, base_path) if dataset_source else None
        examples = self._load_examples(program, net, training, data_path)

        # CÓMO SE PARTE, EN UN SOLO SITIO (contrato 101-C0). Antes esto estaba
        # aquí dentro y cada entrenador tenía su copia; ahora lo decide
        # `training.particion`, que conserva EXACTAMENTE los dos caminos de
        # antes —el 0,8 fijo secuencial y el temporal del 57-C3— y añade el
        # tercero, que solo se activa si el `.mxtrain` declara `protocol=2`.
        #
        # Sin esa declaración no cambia ni un byte: es lo que permite que
        # `matrixai verify --retrain` siga reproduciendo un proyecto antiguo.
        particion = particion_para(len(examples), training.dataset.split)
        train_ex, val_ex, test_ex = reparte(examples, particion)
        # EL TRAMO DE PRUEBA NO SE TOCA AQUÍ. No entrena, no elige la época y no
        # aporta un rango ni una categoría: el entrenador solo lo aparta y dice
        # cuál es. Medirlo es un acto posterior y de otro (104-C0 lo hace
        # cumplir con su registro de accesos); hacerlo aquí sería elegir sobre
        # los datos con los que luego se afirma.

        ps = build_network_parameter_set(net, resolved_layers, mhash, seed=seed)

        best_ps = ps
        best_val_loss = float("inf")
        best_epoch = 1
        train_loss = 0.0

        from matrixai.forward.dense_forward import dense_forward

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for x, y in train_ex:
                ps, loss = dense_train_step(net, ps, x, y, loss_fn, learning_rate=lr)
                epoch_loss += loss
            train_loss = epoch_loss / len(train_ex) if train_ex else 0.0

            val_loss = 0.0
            for x, y in val_ex:
                pred = dense_forward(net, ps, x)
                val_loss += compute_loss(loss_fn, pred, y)
            val_loss = val_loss / len(val_ex) if val_ex else train_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ps = ps
                best_epoch = epoch

            if epoch_callback is not None:
                epoch_callback({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "validation_loss": round(val_loss, 6),
                    "accuracy": None,
                })

            if patience is not None and (epoch - best_epoch) >= patience:
                break

        accuracy = 0.0
        validation_metrics: dict[str, Any] | None = None
        if val_ex:
            eval_result = evaluate_dense_network(net, best_ps, val_ex, loss_fn, labels=labels or None)
            if eval_result.is_regression():
                accuracy = max(0.0, min(1.0, eval_result.r2))
            else:
                accuracy = eval_result.accuracy
                # Y LA MATRIZ DE ESA MISMA EVALUACIÓN, que hasta ahora se
                # tiraba.
                #
                # Se conservaba sólo `accuracy` y quien pintaba la pantalla
                # tenía que sacar la matriz de otra evaluación —la que
                # puntúa el dataset entero— así que los dos números no
                # cuadraban: sumar la matriz daba un porcentaje distinto
                # del que decía la exactitud, y encima más alto, porque
                # incluía las filas con las que el modelo había entrenado.
                # Aquí van las dos juntas, medidas sobre lo mismo.
                validation_metrics = {
                    "rows": eval_result.rows,
                    "accuracy": eval_result.accuracy,
                    "confusion_matrix": eval_result.confusion_matrix,
                    "macro_f1": eval_result.macro_f1,
                    "labels": list(effective_labels(labels, eval_result)),
                    "precision": dict(eval_result.precision or {}),
                    "recall": dict(eval_result.recall or {}),
                    "f1": dict(eval_result.f1 or {}),
                }

        run_id = str(uuid.uuid4())[:8]
        ps_path = out / "parameter_set.json"
        write_parameter_set(ps_path, best_ps)
        # Y CON EL NOMBRE QUE TODO EL MUNDO CONOCE (2026-08-25).
        #
        # Medido: el entrenador de los modelos FUNCTION escribe
        # `params.best.json` —y lo declara en su manifiesto como
        # `selected_parameter_set`—, y éste escribía solo `parameter_set.json`.
        # Dos nombres para la misma cosa según por dónde entres: quien entrena
        # una RED y sigue el QUICKSTART se lleva un `No such file or
        # directory`, porque la documentación (correctamente) enseña el del
        # otro camino.
        #
        # Se escriben los DOS y no se retira el suyo: hay guardado y export que
        # ya buscan `parameter_set.json`, y renombrar rompería lo que funciona
        # para arreglar un nombre.
        write_parameter_set(out / "params.best.json", best_ps)

        trace_path = out / "training_trace.json"
        trace_path.write_text(
            json.dumps({
                "run_id": run_id,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "network": net.name,
                # BIBLIOTECA C2 (auditoría): sin esta clave,
                # `_collect_training_result` (playground.py) caía SIEMPRE al
                # default "classification" — cualquier NETWORK de regresión
                # entrenado por el camino stdlib (DenseSupervisedTrainer, el
                # fallback cuando use_torch=False) reportaba task_kind
                # incorrecto, aunque el entrenamiento en sí fuera correcto.
                # Mismo criterio que ya usa el camino torch (is_reg = loss_fn
                # == "mse").
                "task_kind": "regression" if loss_fn == "mse" else "classification",
                # LA EXACTITUD, ESCRITA Y NO SOLO IMPRESA (2026-08-25).
                #
                # El CLI la enseñaba por pantalla (`Accuracy: 0.968750`) y no
                # constaba en ningún fichero, así que **no podía viajar al
                # paquete ni contrastarse en R3**: un número que no está en
                # disco no se puede comparar después. Y estaba calculada aquí
                # mismo, a cuatro líneas del sitio donde se escribe la traza.
                #
                # Va con las métricas de clasificación que este mismo camino ya
                # calcula, por su nombre: `validation_metrics` es lo que la
                # pantalla y el export saben leer.
                "accuracy": accuracy,
                "validation_metrics": validation_metrics or None,
            }, indent=2),
            encoding="utf-8",
        )

        return TrainingRunResult(
            run_id=run_id,
            output_dir=str(out),
            best_epoch=best_epoch,
            best_validation_loss=best_val_loss,
            final_train_loss=train_loss,
            final_validation_loss=best_val_loss,
            accuracy=accuracy,
            validation_metrics=validation_metrics,
            artifacts={
                "parameter_set": str(ps_path),
                "training_trace": str(trace_path),
            },
            # El lote REAL de este camino es 1: el bucle actualiza los
            # pesos ejemplo a ejemplo e IGNORA el `BATCH size=` del spec.
            # Medido, no leído: 153 actualizaciones para 51 filas × 3
            # épocas con el spec pidiendo lote 8.
            #
            # Se declara en vez de repetir aquí lo que dice el contrato,
            # porque lo que hay que poder comparar entre dos máquinas es
            # lo que PASÓ, no lo que se pidió.
            effort=esfuerzo_de_entrenamiento(len(train_ex), 1, epochs),
        )

    def _load_examples(
        self,
        program: Any,
        net: Any,
        training: TrainingSpec,
        data_path: Path | None,
    ) -> list[tuple[list[float], list[float]]]:
        ejemplos, aplicados = _cargar_ejemplos(program, net, training, data_path)
        self.rangos_aplicados = aplicados
        return ejemplos


def _cargar_ejemplos(
    program: Any, net: Any, training: TrainingSpec, data_path: Path | None,
) -> tuple[list[tuple[list[float], list[float]]], dict[str, tuple[float, float]]]:
    """Los ejemplos del CSV, **normalizados por los rangos que el modelo declara**.

    VIVE UNA SOLA VEZ A PROPÓSITO. Este cuerpo estaba duplicado literalmente en
    el entrenador y en el evaluador, y al añadir la normalización (hallazgo 13,
    decisión de Roberto del 2026-08-25) esa duplicación pasaba de fea a
    peligrosa: normalizar solo en uno haría que el modelo se entrenara con
    valores escalados y se puntuara con los crudos, y el número que saliera no
    describiría nada.

    Devuelve también **los rangos aplicados**, porque quien exporta tiene que
    meterlos en el `inference_spec`: normalizar al entrenar y no al predecir
    produce un paquete que parece bueno y predice mal.
    """
    if data_path is None or not data_path.exists():
        return [], {}
    vector_map = {v.name: v for v in program.vectors}
    vector = vector_map.get(net.input)
    if vector is None:
        return [], {}
    loss_fn = training.loss.type if training.loss else "mse"
    labels = _labels_from_spec(training)
    target_col = training.dataset.target.name
    adapter = CSVDataAdapter(
        data_path, vector.name, list(vector.fields), target_col,
        labels if labels else None,
    )
    xs_ys = _examples_to_xy(adapter.examples(), loss_fn, labels)

    from matrixai.training.normalizacion import (  # noqa: PLC0415
        normalizar_filas, rangos_declarados_del_vector)

    xs = [x for x, _ in xs_ys]
    xs, aplicados = normalizar_filas(
        xs, list(vector.fields), rangos_declarados_del_vector(vector))
    return [(x, y) for x, (_, y) in zip(xs, xs_ys)], aplicados


class DenseSupervisedEvaluator:
    """Evaluate a dense neural network with a loaded ParameterSet."""

    def evaluate(
        self,
        training: TrainingSpec,
        parameter_set: ParameterSet,
        data_path: str | None = None,
        base_path: Path | None = None,
        target_range: tuple[float, float] | None = None,
    ) -> EvaluationResult:
        # CONTRATO 59 C1: `target_range` (rango de dominio del target
        # normalizado) reescala MAE/RMSE a la unidad real — ver
        # `result_from_predictions` en dense_evaluator.py. `None`
        # (retrocompatible) deja el comportamiento previo intacto.
        target_scale = (
            (target_range[1] - target_range[0]) if target_range is not None else None
        )
        base_path = base_path or Path(".")

        model_path = _resolve_path(training.model, base_path)
        if model_path is None:
            raise FileNotFoundError(f"Model file not found: {training.model}")

        program = parse_file(model_path)
        if not program.networks:
            raise ValueError(f"No NETWORK blocks found in {training.model}")

        net = program.networks[0]
        loss_fn = training.loss.type if training.loss else "mse"
        labels = _labels_from_spec(training)

        dataset_source = training.dataset.source if training.dataset else ""
        if data_path:
            resolved_data = _resolve_path(data_path, base_path)
        else:
            resolved_data = _resolve_path(dataset_source, base_path) if dataset_source else None

        examples = self._load_examples(program, net, training, resolved_data)

        if not examples:
            dense_result = DenseEvaluationResult(rows=0, loss=0.0, loss_fn=loss_fn)
        else:
            dense_result = evaluate_dense_network(
                net, parameter_set, examples, loss_fn, labels=labels or None,
                target_scale=target_scale,
            )

        mhash = program_hash(program)
        data_fp = dataset_fingerprint(resolved_data) if resolved_data and resolved_data.exists() else ""

        # Las clases REALES del cálculo, no solo las declaradas: una binaria
        # sin bloque LABELS las lleva inventadas dentro del resultado
        # (`effective_labels` explica por qué y con qué orden). Sin esto,
        # `labels` salía `[]` y `per_label` `{}` al lado de una matriz de
        # confusión con dos clases dentro.
        report_labels = effective_labels(labels, dense_result)

        per_label: dict[str, dict[str, float]] = {}
        if report_labels and dense_result.precision:
            for lbl in report_labels:
                per_label[lbl] = {
                    "precision": dense_result.precision.get(lbl, 0.0),
                    "recall": dense_result.recall.get(lbl, 0.0),
                    "f1": dense_result.f1.get(lbl, 0.0),
                }

        macro_p = sum(dense_result.precision.values()) / len(dense_result.precision) if dense_result.precision else 0.0
        macro_r = sum(dense_result.recall.values()) / len(dense_result.recall) if dense_result.recall else 0.0

        return EvaluationResult(
            model=training.model,
            model_hash=mhash,
            parameter_schema_hash=parameter_set.parameter_schema_hash,
            parameter_set_id=parameter_set.parameter_set_id,
            dataset=str(resolved_data or ""),
            dataset_fingerprint=data_fp,
            dataset_schema={},
            rows=dense_result.rows,
            loss=dense_result.loss,
            accuracy=dense_result.accuracy,
            labels=list(report_labels),
            confusion_matrix=dense_result.confusion_matrix,
            per_label=per_label,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=dense_result.macro_f1,
            mae=dense_result.mae,
            rmse=dense_result.rmse,
            r2=dense_result.r2,
            loss_fn=dense_result.loss_fn,
        )

    def _load_examples(
        self,
        program: Any,
        net: Any,
        training: TrainingSpec,
        data_path: Path | None,
    ) -> list[tuple[list[float], list[float]]]:
        ejemplos, aplicados = _cargar_ejemplos(program, net, training, data_path)
        self.rangos_aplicados = aplicados
        return ejemplos

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from matrixai.training.parser import MatrixAITrainingParseError, parse_training_file, parse_training_text
from matrixai.training.spec import (
    BackendSpec,
    DatasetBatchSpec,
    DatasetInputSpec,
    DatasetSpec,
    DatasetSplitSpec,
    DatasetTargetSpec,
    EvaluationResult,
    LossSpec,
    MetricSpec,
    OptimizerSpec,
    RunSpec,
    TrainingRunResult,
    TrainingSpec,
)
from matrixai.training.data import (
    CSVDataAdapter,
    DataAdapter,
    DatasetSchema,
    InMemoryDataAdapter,
    MatrixAIBatch,
    dataset_fingerprint,
)
from matrixai.training.dataset_manifest import (
    DATASET_MANIFEST_VERSION,
    SYNTHETIC_GENERATOR_VERSION,
    DatasetManifest,
    DatasetManifestEntry,
    DatasetManifestSplit,
    DatasetManifestSplitPartition,
    DatasetManifestVerificationResult,
    GeneratorSpec,
    build_synthetic_manifest,
    load_dataset_manifest,
    verify_dataset_manifest,
)
from matrixai.training.differentiability import DifferentiabilityVerificationResult, DifferentiabilityVerifier
from matrixai.training.generator import TrainingGenerationResult, TrainingPromptGenerator
from matrixai.training.supervised_prompt import (
    SupervisedPromptGenerationResult,
    SupervisedPromptGenerator,
    SupervisedPromptRunResult,
    SupervisedPromptRunner,
)
from matrixai.training.torch_trainer import TorchSupervisedTrainer
from matrixai.training.trainer import (
    GenericSupervisedEvaluator,
    GenericSupervisedTrainer,
    SupervisedEvaluator,
    SupervisedTrainer,
)
from matrixai.training.verifier import TrainingVerificationResult, TrainingVerifier
from matrixai.training.dense_evaluator import (
    DenseEvaluationResult,
    compute_accuracy,
    compute_mae,
    compute_r2,
    compute_rmse,
    evaluate_dense_network,
)
from matrixai.training.dense_generator import (
    DenseNetworkGenerationResult,
    DenseNetworkGenerator,
    DenseNetworkGeneratorError,
)
from matrixai.training.dense_backprop import (
    DenseBackpropError,
    binary_cross_entropy_loss,
    compute_loss,
    cross_entropy_loss,
    dense_compute_gradients,
    dense_train_step,
    mse_loss,
)
from matrixai.training.dense_trainer import DenseSupervisedEvaluator, DenseSupervisedTrainer
from matrixai.training.composite_evaluator import (
    composite_examples_from_csv,
    evaluate_composite_network,
)
from matrixai.training.composite_generator import (
    CompositeNetworkGenerationResult,
    CompositeNetworkGenerator,
    CompositeNetworkGeneratorError,
)

__all__ = [
    "BackendSpec",
    "CSVDataAdapter",
    "DataAdapter",
    "DATASET_MANIFEST_VERSION",
    "SYNTHETIC_GENERATOR_VERSION",
    "GeneratorSpec",
    "build_synthetic_manifest",
    "DatasetBatchSpec",
    "DatasetInputSpec",
    "DatasetManifest",
    "DatasetManifestEntry",
    "DatasetManifestSplit",
    "DatasetManifestSplitPartition",
    "DatasetManifestVerificationResult",
    "DatasetSchema",
    "DatasetSpec",
    "DatasetSplitSpec",
    "DatasetTargetSpec",
    "DifferentiabilityVerificationResult",
    "DifferentiabilityVerifier",
    "EvaluationResult",
    "GenericSupervisedEvaluator",
    "GenericSupervisedTrainer",
    "InMemoryDataAdapter",
    "LossSpec",
    "MatrixAIBatch",
    "MatrixAITrainingParseError",
    "MetricSpec",
    "OptimizerSpec",
    "RunSpec",
    "SupervisedEvaluator",
    "SupervisedPromptGenerationResult",
    "SupervisedPromptGenerator",
    "SupervisedPromptRunResult",
    "SupervisedPromptRunner",
    "SupervisedTrainer",
    "TorchSupervisedTrainer",
    "TrainingRunResult",
    "TrainingGenerationResult",
    "TrainingPromptGenerator",
    "TrainingSpec",
    "TrainingVerificationResult",
    "TrainingVerifier",
    "parse_training_file",
    "parse_training_text",
    "dataset_fingerprint",
    "load_dataset_manifest",
    "verify_dataset_manifest",
    "DenseNetworkGenerationResult",
    "DenseNetworkGenerator",
    "DenseNetworkGeneratorError",
    "DenseEvaluationResult",
    "compute_accuracy",
    "compute_mae",
    "compute_r2",
    "compute_rmse",
    "evaluate_dense_network",
    "DenseBackpropError",
    "binary_cross_entropy_loss",
    "compute_loss",
    "cross_entropy_loss",
    "dense_compute_gradients",
    "dense_train_step",
    "mse_loss",
    "DenseSupervisedTrainer",
    "DenseSupervisedEvaluator",
    "composite_examples_from_csv",
    "evaluate_composite_network",
    "CompositeNetworkGenerationResult",
    "CompositeNetworkGenerator",
    "CompositeNetworkGeneratorError",
]

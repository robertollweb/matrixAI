# my-first-classifier — Binary Classification Model

A binary classification model built with MatrixAI.

## Quick Start

Train the model:
```bash
python3 -m matrixai train my-first-classifier.mxai --training my-first-classifier.mxtrain --output runs/v1
```

Make a prediction:
```bash
python3 -m matrixai run my-first-classifier.mxai --params runs/v1/params.best.json --input input/sample.json --json
```

## Files

- `my-first-classifier.mxai` — Model architecture (mathematical function)
- `my-first-classifier.mxtrain` — Training configuration (dataset, hyperparameters)
- `dataset/train.csv` — Training examples (features, label)
- `dataset/test.csv` — Test examples for evaluation
- `input/sample.json` — Example prediction input
- `models/best/params.json` — Best model parameters (updated after training)

## Data Format

CSV with four columns:
- `feature_1`, `feature_2`, `feature_3`: Input numeric features (0.0 to 1.0)
- `label`: Ground truth label (0 or 1)

Example:
```
feature_1,feature_2,feature_3,label
0.9,0.8,0.85,1
0.1,0.15,0.12,0
```

## Next Steps

1. Replace `dataset/train.csv` and `dataset/test.csv` with your own data
2. Adjust hyperparameters in `my-first-classifier.mxtrain` (learning_rate, epochs, batch_size)
3. Check results in `runs/v1/params.best.json` and `runs/v1/metrics.json` and `runs/v1/validation_report.json`

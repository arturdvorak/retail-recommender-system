# MLflow Usage Guide

## Viewing Experiments

To view your experiments in the MLflow UI, run:

```bash
cd '/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system'
mlflow ui --backend-store-uri mlruns
```

Then open your browser to: `http://localhost:5000`

## What Gets Tracked

**Training (`model_training.py`):**
- Hyperparameters: factors, regularization, iterations, date window
- Training data stats: number of users, items, interactions
- Model artifact: saved model file

**Evaluation (`evaluate_model.py`):**
- Hyperparameters: same as training + eval_set, top_k
- Metrics: precision@K for validation or test set
- Evaluation data stats: number of users, items, interactions

## Experiment Organization

- **Experiment Name**: `ALS_Recommendation_System`
- **Run Names**: 
  - Training: `training`
  - Evaluation: `evaluation_{validation|test}_{experiment_name}`

## Benefits Over JSON

- Web UI for easy comparison
- Filter and search experiments
- Visualize metrics over time
- Compare multiple runs side-by-side
- Download/restore models from UI


---
name: Hyperparameter Tuning ALS BPR
overview: Create a hyperparameter tuning script using Bayesian optimization (Optuna) to find the best hyperparameters for ALS and BPR models, optimizing for Precision@K on the validation set.
todos: []
---

# Hyperparameter Tuning Plan for ALS and BPR Models

## Overview

Create a hyperparameter tuning script that uses Bayesian optimization (Optuna) to systematically search for the best hyperparameters for both ALS and BPR models. The script will train multiple model variants, evaluate them on the validation set, and track results in MLflow.

## Implementation Steps

### 1. Create Hyperparameter Tuning Script (`src/hyperparameter_tuning.py`)

**Purpose**: Automate the search for optimal hyperparameters using Optuna's Bayesian optimization.

**Key Components**:

- Define search spaces for ALS and BPR hyperparameters
- Use Optuna to intelligently explore the hyperparameter space
- Train models with different hyperparameter combinations
- Evaluate each combination on validation set using Precision@K
- Track all trials in MLflow
- Save best hyperparameters

**ALS Hyperparameter Search Space**:

- `factors`: [32, 64, 128, 256] - latent dimensions
- `regularization`: [0.01, 0.1, 1.0, 10.0] - controls overfitting
- `iterations`: [10, 20, 50, 100] - training steps

**BPR Hyperparameter Search Space**:

- `factors`: [32, 64, 128, 256] - latent dimensions
- `learning_rate`: [0.001, 0.01, 0.1] - learning speed
- `regularization`: [0.001, 0.01, 0.1] - controls overfitting
- `iterations`: [50, 100, 200] - training steps

**Workflow**:

1. Load train_val and validation matrices (from tune dataset)
2. Create Optuna study for the selected model type
3. For each trial:

   - Sample hyperparameters from search space
   - Train model on train_val
   - Evaluate on validation set (Precision@K)
   - Log trial to MLflow
   - Return Precision@K as objective (to maximize)

4. After all trials, identify best hyperparameters
5. Save best hyperparameters to a config file

### 2. Update `model_training.py` to Accept Hyperparameters

**Changes**:

- Add command-line argument `--hyperparameters` to load best hyperparameters from saved file
- If `--hyperparameters` is provided, load from `models/best_hyperparameters.json`
- If not provided, use default hyperparameters (current hardcoded values)
- This allows using best hyperparameters found during tuning for final model training
- Format: `--hyperparameters` flag loads best hyperparameters for the specified model type

### 3. Integration with Existing Code

**Files to Modify**:

- `src/model_training.py`: Add hyperparameter arguments (optional, defaults to current values)
- Create `src/hyperparameter_tuning.py`: New tuning script

**MLflow Integration**:

- Each trial logged as separate run in MLflow
- Best trial clearly marked
- Hyperparameters and metrics tracked for comparison

### 4. Output

**Best Hyperparameters File**:

- Save best hyperparameters to `models/best_hyperparameters.json` (or similar)
- Format: `{"als": {"factors": X, "regularization": Y, ...}, "bpr": {...}}`

**Usage**:

```bash
# Tune ALS hyperparameters
python src/hyperparameter_tuning.py --model als --n-trials 50

# Tune BPR hyperparameters  
python src/hyperparameter_tuning.py --model bpr --n-trials 50

# Train final model using best hyperparameters
python src/model_training.py --model als --mode final --hyperparameters
python src/model_training.py --model bpr --mode final --hyperparameters
```

## Technical Details

**Optuna Configuration**:

- Study direction: maximize Precision@K
- Pruning: Enable early stopping for unpromising trials
- Number of trials: Configurable (default: 50)

**Evaluation**:

- Use validation set from tune dataset
- Metric: Precision@K (as specified)
- Same evaluation logic as `evaluate_model.py`

**Dependencies**:

- Add `optuna` to `requirements.txt`

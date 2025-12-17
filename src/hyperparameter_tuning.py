"""Hyperparameter tuning for ALS, BPR, and LightFM models using Bayesian optimization (Optuna).

This script uses Optuna to intelligently search for the best hyperparameters
by learning from previous trials. It trains multiple model variants and
evaluates them on the validation set to find optimal settings.

Bayesian optimization: Instead of testing random combinations, Optuna learns
which hyperparameter ranges are promising and focuses search there.
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from lightfm import LightFM
from lightfm.data import Dataset

# Import configuration and utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_prep import TRAIN_VAL_START, TRAIN_VAL_END
from src.evaluate_utils import evaluate_precision_and_recall_at_k, evaluate_lightfm_precision_and_recall_at_k

# Paths to prepared data and model artifacts
# Warm matrices for ALS/BPR
MATRICES_TUNE_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune_warm.pkl"
)
# Full matrices for LightFM (includes cold-start users)
MATRICES_TUNE_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune_full.pkl"
)
ENCODERS_TUNE_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_tune_full.pkl"
)
MODELS_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)
BEST_HYPERPARAMETERS_PATH = MODELS_DIR / "best_hyperparameters.json"

# Evaluation configuration
TOP_K = 10  # Number of recommendations to evaluate per user


def train_and_evaluate_als(trial, train_matrix, eval_matrix):
    """Train ALS model with trial hyperparameters and evaluate on validation set.
    
    Args:
        trial: Optuna trial object that suggests hyperparameter values
        train_matrix: Training interaction matrix (train_val)
        eval_matrix: Validation interaction matrix (validation_warm)
    
    Returns:
        Precision@K score (what we want to maximize)
    """
    # Sample hyperparameters from search space
    # Optuna will intelligently choose values based on previous trials
    factors = trial.suggest_categorical("factors", [32, 64, 128, 256])
    regularization = trial.suggest_categorical("regularization", [0.01, 0.1, 1.0, 10.0])
    iterations = trial.suggest_categorical("iterations", [10, 20, 50, 100])
    
    # Train ALS model with these hyperparameters
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        dtype=train_matrix.dtype,
    )
    model.fit(train_matrix)
    
    # Evaluate on validation set
    metrics = evaluate_precision_and_recall_at_k(
        model=model,
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        k=TOP_K,
        recalculate_user=False,
        show_progress=False,  # Disable progress bar for cleaner output during tuning
    )
    
    precision_at_k = metrics["precision_at_k"]
    recall_at_k = metrics["recall_at_k"]
    
    # Report metrics to Optuna (for tracking and pruning)
    trial.set_user_attr("recall_at_k", recall_at_k)
    
    return precision_at_k  # This is what we want to maximize


def train_and_evaluate_bpr(trial, train_matrix, eval_matrix):
    """Train BPR model with trial hyperparameters and evaluate on validation set.
    
    Args:
        trial: Optuna trial object that suggests hyperparameter values
        train_matrix: Training interaction matrix (train_val)
        eval_matrix: Validation interaction matrix (validation_warm)
    
    Returns:
        Precision@K score (what we want to maximize)
    """
    # Sample hyperparameters from search space
    factors = trial.suggest_categorical("factors", [32, 64, 128, 256])
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.01, 0.1])
    regularization = trial.suggest_categorical("regularization", [0.001, 0.01, 0.1])
    iterations = trial.suggest_categorical("iterations", [50, 100, 200])
    
    # Train BPR model with these hyperparameters
    model = BayesianPersonalizedRanking(
        factors=factors,
        learning_rate=learning_rate,
        regularization=regularization,
        iterations=iterations,
        random_state=42,  # For reproducibility
    )
    model.fit(train_matrix)
    
    # Evaluate on validation set
    metrics = evaluate_precision_and_recall_at_k(
        model=model,
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        k=TOP_K,
        recalculate_user=False,
        show_progress=False,  # Disable progress bar for cleaner output during tuning
    )
    
    precision_at_k = metrics["precision_at_k"]
    recall_at_k = metrics["recall_at_k"]
    
    # Report metrics to Optuna (for tracking and pruning)
    trial.set_user_attr("recall_at_k", recall_at_k)
    
    return precision_at_k  # This is what we want to maximize


def train_and_evaluate_lightfm(trial, train_matrix, eval_matrix, dataset, encoders):
    """Train LightFM model with trial hyperparameters and evaluate on validation set.
    
    Args:
        trial: Optuna trial object that suggests hyperparameter values
        train_matrix: Training interaction matrix (train_val) - sparse matrix
        eval_matrix: Validation interaction matrix (validation_full) - sparse matrix
        dataset: LightFM Dataset object (already built)
        encoders: Encoder mappings for user/item IDs
    
    Returns:
        Precision@K score (what we want to maximize)
    """
    # Sample hyperparameters from search space
    no_components = trial.suggest_categorical("no_components", [32, 64, 128, 256])
    learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1])
    loss = trial.suggest_categorical("loss", ['logistic', 'bpr', 'warp'])
    epochs = trial.suggest_categorical("epochs", [10, 20, 50])
    
    # Convert sparse matrix to interactions format
    coo_matrix = train_matrix.tocoo()
    interactions = list(zip(coo_matrix.row, coo_matrix.col, coo_matrix.data))
    
    # Build interaction matrix in LightFM format
    (interactions_matrix, weights_matrix) = dataset.build_interactions(interactions)
    
    # Train LightFM model with these hyperparameters
    model = LightFM(
        no_components=no_components,
        learning_rate=learning_rate,
        loss=loss,
        random_state=42,  # For reproducibility
    )
    model.fit(
        interactions_matrix,
        sample_weight=weights_matrix,
        epochs=epochs,
        verbose=False,  # Disable verbose output during tuning
    )
    
    # Evaluate on validation set using LightFM-specific evaluation
    metrics = evaluate_lightfm_precision_and_recall_at_k(
        model=model,
        dataset=dataset,
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        k=TOP_K,
        show_progress=False,  # Disable progress bar for cleaner output during tuning
    )
    
    precision_at_k = metrics["precision_at_k"]
    recall_at_k = metrics["recall_at_k"]
    
    # Report metrics to Optuna (for tracking and pruning)
    trial.set_user_attr("recall_at_k", recall_at_k)
    
    return precision_at_k  # This is what we want to maximize


def objective_als(trial):
    """Objective function for ALS hyperparameter tuning.
    
    This function is called by Optuna for each trial. It trains a model,
    evaluates it, and returns the metric we want to optimize.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Precision@K score to maximize
    """
    # Load matrices once (they're the same for all trials)
    if not hasattr(objective_als, 'matrices'):
        objective_als.matrices = joblib.load(MATRICES_TUNE_PATH)
        objective_als.train_matrix = objective_als.matrices["train_val"].tocsr().astype("float32")
        objective_als.eval_matrix = objective_als.matrices["validation_warm"].tocsr().astype("float32")
    
    # Train and evaluate model with trial hyperparameters
    precision = train_and_evaluate_als(
        trial,
        objective_als.train_matrix,
        objective_als.eval_matrix
    )
    
    return precision


def objective_bpr(trial):
    """Objective function for BPR hyperparameter tuning.
    
    This function is called by Optuna for each trial. It trains a model,
    evaluates it, and returns the metric we want to optimize.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Precision@K score to maximize
    """
    # Load matrices once (they're the same for all trials)
    if not hasattr(objective_bpr, 'matrices'):
        objective_bpr.matrices = joblib.load(MATRICES_TUNE_PATH)
        objective_bpr.train_matrix = objective_bpr.matrices["train_val"].tocsr().astype("float32")
        objective_bpr.eval_matrix = objective_bpr.matrices["validation_warm"].tocsr().astype("float32")
    
    # Train and evaluate model with trial hyperparameters
    precision = train_and_evaluate_bpr(
        trial,
        objective_bpr.train_matrix,
        objective_bpr.eval_matrix
    )
    
    return precision


def objective_lightfm(trial):
    """Objective function for LightFM hyperparameter tuning.
    
    This function is called by Optuna for each trial. It trains a model,
    evaluates it, and returns the metric we want to optimize.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Precision@K score to maximize
    """
    # Load matrices and encoders once (they're the same for all trials)
    if not hasattr(objective_lightfm, 'matrices'):
        objective_lightfm.matrices = joblib.load(MATRICES_TUNE_FULL_PATH)
        objective_lightfm.train_matrix = objective_lightfm.matrices["train_val"].tocsr()
        objective_lightfm.eval_matrix = objective_lightfm.matrices["validation_full"].tocsr()
        objective_lightfm.encoders = joblib.load(ENCODERS_TUNE_FULL_PATH)
        
        # Build LightFM Dataset object once
        n_users, n_items = objective_lightfm.train_matrix.shape
        objective_lightfm.dataset = Dataset()
        objective_lightfm.dataset.fit(
            users=range(n_users),
            items=range(n_items)
        )
    
    # Train and evaluate model with trial hyperparameters
    precision = train_and_evaluate_lightfm(
        trial,
        objective_lightfm.train_matrix,
        objective_lightfm.eval_matrix,
        objective_lightfm.dataset,
        objective_lightfm.encoders
    )
    
    return precision


def main() -> None:
    """Run hyperparameter tuning using Optuna Bayesian optimization.
    
    Trains multiple model variants, evaluates them, and finds the best
    hyperparameters. Results are tracked in MLflow and saved to JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for ALS, BPR, or LightFM model using Bayesian optimization."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["als", "bpr", "lightfm"],
        default="als",
        help="Model type: 'als' for Alternating Least Squares, 'bpr' for Bayesian Personalized Ranking, 'lightfm' for LightFM (default: als)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of hyperparameter combinations to try (default: 50)",
    )
    args = parser.parse_args()
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set model-specific configuration
    if args.model == "als":
        experiment_name = "ALS_Recommendation_System"
        objective_func = objective_als
    elif args.model == "bpr":
        experiment_name = "BPR_Recommendation_System"
        objective_func = objective_bpr
    else:  # lightfm
        experiment_name = "LightFM_Recommendation_System"
        objective_func = objective_lightfm
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(experiment_name)
    
    # Create Optuna study (Bayesian optimization)
    # direction='maximize' means we want to find the highest Precision@K
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.model}_hyperparameter_tuning",
    )
    
    print(f"Starting hyperparameter tuning for {args.model.upper()} model...")
    print(f"Number of trials: {args.n_trials}")
    print(f"Optimizing: Precision@{TOP_K}")
    print("-" * 60)
    
    # Run optimization trials
    # Each trial tests a different hyperparameter combination
    for trial_num in range(args.n_trials):
        # Create MLflow run for this trial
        with mlflow.start_run(run_name=f"trial_{trial_num + 1}", nested=True):
            # Run one trial (Optuna suggests hyperparameters, we train and evaluate)
            trial = study.ask()
            
            # Train and evaluate model with suggested hyperparameters
            precision_at_k = objective_func(trial)
            
            # Report result back to Optuna
            study.tell(trial, precision_at_k)
            
            # Log to MLflow
            params = trial.params.copy()
            params["model_type"] = args.model.upper()
            params["trial_number"] = trial_num + 1
            mlflow.log_params(params)
            
            recall_at_k = trial.user_attrs.get("recall_at_k", 0.0)
            mlflow.log_metrics({
                "validation_precision_at_k": precision_at_k,
                "validation_recall_at_k": recall_at_k,
            })
            
            print(f"Trial {trial_num + 1}/{args.n_trials}: Precision@{TOP_K} = {precision_at_k:.4f}, "
                  f"Recall@{TOP_K} = {recall_at_k:.4f}, "
                  f"Params: {params}")
    
    # Get best trial (highest Precision@K)
    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    
    print("-" * 60)
    print(f"Best trial: Precision@{TOP_K} = {best_trial.value:.4f}")
    print(f"Best hyperparameters: {best_params}")
    
    # Save best hyperparameters to JSON file
    # Load existing file if it exists, otherwise create new dict
    if BEST_HYPERPARAMETERS_PATH.exists():
        with open(BEST_HYPERPARAMETERS_PATH, "r") as f:
            best_hyperparameters = json.load(f)
    else:
        best_hyperparameters = {}
    
    # Update with best hyperparameters for this model type
    best_hyperparameters[args.model] = best_params
    
    # Save updated hyperparameters
    with open(BEST_HYPERPARAMETERS_PATH, "w") as f:
        json.dump(best_hyperparameters, f, indent=2)
    
    print(f"\nBest hyperparameters saved to: {BEST_HYPERPARAMETERS_PATH}")
    print(f"Use these for final model training with: --hyperparameters flag")
    print(f"MLflow runs logged. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()

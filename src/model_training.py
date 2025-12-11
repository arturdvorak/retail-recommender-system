"""Train recommendation models (ALS or BPR) using prepared interaction matrices.

This script supports two training stages (used sequentially, not simultaneously):
- Stage 1 (tune mode): Train on train_val for hyperparameter tuning
- Stage 2 (final mode): Train on train for final model with best hyperparameters

Supports two model types:
- ALS (Alternating Least Squares): Matrix factorization using alternating optimization
- BPR (Bayesian Personalized Ranking): Optimizes for pairwise ranking using SGD
"""

import argparse
from pathlib import Path

import joblib
import mlflow
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

# Import configuration to log as parameters.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_prep import (
    TRAIN_VAL_START, TRAIN_VAL_END,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END
)

# Paths to prepared data and model artifacts.
# Using warm matrices for ALS/BPR (cold-start removed)
MATRICES_TUNE_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune_warm.pkl"
)
MATRICES_FINAL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_warm.pkl"
)
MODELS_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)

# Hyperparameters for ALS model.
ALS_FACTORS = 64  # latent dimensions
ALS_REGULARIZATION = 0.1  # controls overfitting
ALS_ITERATIONS = 20  # alternating least squares steps

# Hyperparameters for BPR model.
BPR_FACTORS = 64  # latent dimensions (same as ALS)
BPR_LEARNING_RATE = 0.01  # How fast the model learns (BPR-specific, typically 0.01-0.1)
BPR_REGULARIZATION = 0.01  # controls overfitting (typically lower than ALS)
BPR_ITERATIONS = 100  # Number of training iterations (BPR often needs more than ALS)


def main() -> None:
    """Load matrices, fit model (ALS or BPR), and save it.
    
    Supports two modes:
    - 'tune': Train on train_val for hyperparameter tuning
    - 'final': Train on train for final model
    
    Supports two model types:
    - 'als': Alternating Least Squares
    - 'bpr': Bayesian Personalized Ranking
    """

    parser = argparse.ArgumentParser(
        description="Train ALS or BPR model for hyperparameter tuning or final model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["als", "bpr"],
        default="als",
        help="Model type: 'als' for Alternating Least Squares, 'bpr' for Bayesian Personalized Ranking (default: als)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tune", "final"],
        default="final",
        help="Training mode: 'tune' for hyperparameter tuning (train_val), 'final' for final model (train) (default: final)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set model-specific paths and hyperparameters based on model type
    if args.model == "als":
        model_path = MODELS_DIR / "als_model.pkl"
        experiment_name = "ALS_Recommendation_System"
        factors = ALS_FACTORS
        regularization = ALS_REGULARIZATION
        iterations = ALS_ITERATIONS
        learning_rate = None  # Not used for ALS
    else:  # bpr
        model_path = MODELS_DIR / "bpr_model.pkl"
        experiment_name = "BPR_Recommendation_System"
        factors = BPR_FACTORS
        regularization = BPR_REGULARIZATION
        iterations = BPR_ITERATIONS
        learning_rate = BPR_LEARNING_RATE

    # Set MLflow tracking URI to local directory.
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(experiment_name)

    # Stage 1: load training sparse matrix based on mode.
    if args.mode == "tune":
        # Hyperparameter tuning: use train_val
        matrices = joblib.load(MATRICES_TUNE_PATH)
        train_matrix = matrices["train_val"].tocsr().astype("float32")
        date_start = TRAIN_VAL_START
        date_end = TRAIN_VAL_END
        run_name = "training_tune"
    else:
        # Final model: use train
        matrices = joblib.load(MATRICES_FINAL_PATH)
        train_matrix = matrices["train"].tocsr().astype("float32")
        date_start = TRAIN_START
        date_end = TRAIN_END
        run_name = "training_final"

    # Start MLflow run for training.
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters (model-specific).
        params = {
            "model_type": args.model.upper(),
            "factors": factors,
            "regularization": regularization,
            "iterations": iterations,
            "date_window_start": date_start,
            "date_window_end": date_end,
            "mode": args.mode,
        }
        if learning_rate is not None:  # BPR-specific parameter
            params["learning_rate"] = learning_rate
        mlflow.log_params(params)

        # Stage 2: train the model on user-item data.
        if args.model == "als":
            # Train Alternating Least Squares model
            model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                dtype=train_matrix.dtype,
            )
        else:  # bpr
            # Train Bayesian Personalized Ranking model
            model = BayesianPersonalizedRanking(
                factors=factors,
                learning_rate=learning_rate,
                regularization=regularization,
                iterations=iterations,
                random_state=42,  # For reproducibility
            )
        model.fit(train_matrix)

        # Stage 3: persist the trained model for later use.
        joblib.dump(model, model_path)
        print(f"{args.model.upper()} model saved to {model_path}")
        print(f"Mode: {args.mode} (trained on {'train_val' if args.mode == 'tune' else 'train'})")

        # Log model as artifact.
        mlflow.log_artifact(str(model_path), "model")
        
        # Log training data info.
        mlflow.log_metrics({
            "train_users": train_matrix.shape[0],
            "train_items": train_matrix.shape[1],
            "train_interactions": train_matrix.nnz,
        })
        
        print(f"MLflow run logged. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()


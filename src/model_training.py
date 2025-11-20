"""Train an implicit ALS recommendation model using prepared interaction matrices.

This script supports two training stages (used sequentially, not simultaneously):
- Stage 1 (tune mode): Train on train_val for hyperparameter tuning
- Stage 2 (final mode): Train on train for final model with best hyperparameters
"""

import argparse
from pathlib import Path

import joblib
import mlflow
from implicit.als import AlternatingLeastSquares

# Import configuration to log as parameters.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_prep import (
    TRAIN_VAL_START, TRAIN_VAL_END,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END
)

# Paths to prepared data and model artifacts.
MATRICES_TUNE_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune.pkl"
)
MATRICES_FINAL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final.pkl"
)
MODEL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models/als_model.pkl"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)

# Hyperparameters for the ALS model.
FACTORS = 64  # latent dimensions
REGULARIZATION = 0.1  # controls overfitting
ITERATIONS = 20  # alternating least squares steps


def main() -> None:
    """Load matrices, fit ALS model, and save it.
    
    Supports two modes:
    - 'tune': Train on train_val for hyperparameter tuning
    - 'final': Train on train for final model
    """

    parser = argparse.ArgumentParser(
        description="Train ALS model for hyperparameter tuning or final model."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tune", "final"],
        default="final",
        help="Training mode: 'tune' for hyperparameter tuning (train_val), 'final' for final model (train) (default: final)",
    )
    args = parser.parse_args()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI to local directory.
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment("ALS_Recommendation_System")

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
        # Log hyperparameters.
        mlflow.log_params({
            "factors": FACTORS,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "date_window_start": date_start,
            "date_window_end": date_end,
            "mode": args.mode,
        })

        # Stage 2: train the Alternating Least Squares model on user-item data.
        model = AlternatingLeastSquares(
            factors=FACTORS,
            regularization=REGULARIZATION,
            iterations=ITERATIONS,
            dtype=train_matrix.dtype,
        )
        model.fit(train_matrix)

        # Stage 3: persist the trained model for later use.
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        print(f"Mode: {args.mode} (trained on {'train_val' if args.mode == 'tune' else 'train'})")

        # Log model as artifact.
        mlflow.log_artifact(str(MODEL_PATH), "model")
        
        # Log training data info.
        mlflow.log_metrics({
            "train_users": train_matrix.shape[0],
            "train_items": train_matrix.shape[1],
            "train_interactions": train_matrix.nnz,
        })
        
        print(f"MLflow run logged. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()


"""Train an implicit ALS recommendation model using prepared interaction matrices."""

from pathlib import Path

import joblib
import mlflow
from implicit.als import AlternatingLeastSquares

# Import configuration to log as parameters.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_prep import START_DATE, END_DATE

# Paths to prepared data and model artifacts.
MATRICES_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices.pkl"
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
    """Load matrices, fit ALS model, and save it."""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI to local directory.
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment("ALS_Recommendation_System")

    # Stage 1: load training sparse matrix.
    matrices = joblib.load(MATRICES_PATH)
    train_matrix = matrices["train"].tocsr().astype("float32")

    # Start MLflow run for training.
    with mlflow.start_run(run_name="training"):
        # Log hyperparameters.
        mlflow.log_params({
            "factors": FACTORS,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "date_window_start": START_DATE,
            "date_window_end": END_DATE,
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


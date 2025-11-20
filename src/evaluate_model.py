"""Evaluate ALS model performance and generate metrics report."""

import argparse
from pathlib import Path
import datetime as dt

import joblib
import mlflow
import numpy as np
from implicit.als import AlternatingLeastSquares

# Import configuration from other modules to generate experiment name.
import sys

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import (
    TRAIN_VAL_START, TRAIN_VAL_END,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END
)
from src.model_training import FACTORS, REGULARIZATION, ITERATIONS

# Paths to model artifacts and data.
MODEL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models/als_model.pkl"
)
MATRICES_TUNE_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune.pkl"
)
MATRICES_FINAL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final.pkl"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)

# Evaluation configuration.
TOP_K = 10  # number of items evaluated per user
MAX_USERS_EVAL = 1000  # limit users for quick evaluation


def generate_experiment_name(eval_set: str) -> str:
    """Generate a logical experiment name from configuration values.
    
    Format: startdate_enddate_factors_reg_iterations
    Example: may3_jun30_f64_r0.1_i20
    """
    # Choose date range based on evaluation set
    if eval_set == "validation":
        # Hyperparameter tuning: use train_val dates
        start = dt.datetime.strptime(TRAIN_VAL_START, "%Y-%m-%d")
        end = dt.datetime.strptime(TRAIN_VAL_END, "%Y-%m-%d")
    else:
        # Final evaluation: use train dates
        start = dt.datetime.strptime(TRAIN_START, "%Y-%m-%d")
        end = dt.datetime.strptime(TRAIN_END, "%Y-%m-%d")
    
    start_str = f"{start.strftime('%b').lower()}{start.day}"  # "may3"
    end_str = f"{end.strftime('%b').lower()}{end.day}"  # "jun30"
    
    # Format hyperparameters
    factors_str = f"f{FACTORS}"
    reg_str = f"r{REGULARIZATION}".replace(".", "_")  # "r0_1"
    iter_str = f"i{ITERATIONS}"
    
    return f"{start_str}_{end_str}_{factors_str}_{reg_str}_{iter_str}"


def precision_at_k(
    model: AlternatingLeastSquares,
    train_user_items,
    test_user_items,
    K: int,
    max_users: int | None = None,
) -> float:
    """Compute average precision@K over users with test interactions."""

    # Identify users who have at least one held-out item.
    candidate_users = np.where(test_user_items.getnnz(axis=1) > 0)[0]
    if max_users is not None and candidate_users.size > max_users:
        candidate_users = candidate_users[:max_users]

    precisions = []
    for user in candidate_users:
        user_training_profile = train_user_items[user]
        # Recommend top-K items while filtering those already seen in training.
        recommended, _ = model.recommend(
            user,
            user_training_profile,
            N=K,
            filter_already_liked_items=True,
        )
        target_items = set(test_user_items[user].indices)
        hits = sum(1 for item in recommended if item in target_items)
        precisions.append(hits / K)

    return float(np.mean(precisions)) if precisions else 0.0


def main() -> None:
    """Load model and matrices, evaluate performance, and generate report."""

    parser = argparse.ArgumentParser(
        description="Evaluate ALS model on validation or test set."
    )
    parser.add_argument(
        "--set",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Which dataset to evaluate on: 'validation' for hyperparameter tuning (valid), 'test' for final evaluation (default: validation)",
    )
    args = parser.parse_args()

    # Stage 1: load trained model and interaction matrices based on evaluation set.
    model = joblib.load(MODEL_PATH)
    
    if args.set == "validation":
        # Hyperparameter tuning: use train_val and valid
        matrices = joblib.load(MATRICES_TUNE_PATH)
        train_matrix = matrices["train_val"].tocsr().astype("float32")
        eval_matrix = matrices["validation"].tocsr().astype("float32")
        date_start = TRAIN_VAL_START
        date_end = TRAIN_VAL_END
    else:
        # Final evaluation: use train and test
        matrices = joblib.load(MATRICES_FINAL_PATH)
        train_matrix = matrices["train"].tocsr().astype("float32")
        eval_matrix = matrices["test"].tocsr().astype("float32")
        date_start = TRAIN_START
        date_end = TRAIN_END

    # Stage 2: compute precision@K metric.
    eval_precision = precision_at_k(
        model,
        train_matrix,
        eval_matrix,
        K=TOP_K,
        max_users=MAX_USERS_EVAL,
    )

    # Set MLflow tracking and start run for evaluation.
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment("ALS_Recommendation_System")
    
    experiment_name = generate_experiment_name(args.set)
    run_name = f"evaluation_{args.set}_{experiment_name}"
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters (same as training).
        mlflow.log_params({
            "factors": FACTORS,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "date_window_start": date_start,
            "date_window_end": date_end,
            "eval_set": args.set,
            "top_k": TOP_K,
        })
        
        # Log metrics.
        mlflow.log_metrics({
            f"{args.set}_precision_at_k": eval_precision,
        })
        
        # Log evaluation data info.
        mlflow.log_metrics({
            f"{args.set}_users": eval_matrix.shape[0],
            f"{args.set}_items": eval_matrix.shape[1],
            f"{args.set}_interactions": eval_matrix.nnz,
        })

    # Stage 3: Print summary of results.
    print(f"\n{args.set.capitalize()} Precision@{TOP_K}: {eval_precision:.4f}")
    print(f"Experiment: {experiment_name}")
    print(f"Trained on: {'train_val' if args.set == 'validation' else 'train'}")
    print(f"Evaluated on: {'valid' if args.set == 'validation' else 'test'}")
    print(f"Metrics logged to MLflow. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
"""Evaluate ALS or BPR model performance and generate metrics report."""

import argparse
from pathlib import Path
import datetime as dt

import joblib
import mlflow
import numpy as np

# Import configuration from other modules to generate experiment name.
import sys

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import (
    TRAIN_VAL_START, TRAIN_VAL_END,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END
)
from src.model_training import (
    ALS_FACTORS, ALS_REGULARIZATION, ALS_ITERATIONS,
    BPR_FACTORS, BPR_LEARNING_RATE, BPR_REGULARIZATION, BPR_ITERATIONS
)
from src.evaluate_utils import evaluate_precision_and_recall_at_k

# Paths to model artifacts and data.
MODELS_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models"
)
# Using warm matrices for ALS/BPR (cold-start removed)
MATRICES_TUNE_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune_warm.pkl"
)
MATRICES_FINAL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_warm.pkl"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)

# Evaluation configuration.
TOP_K = 10  # number of items evaluated per user


def generate_experiment_name(eval_set: str, model_type: str, factors: int, regularization: float, iterations: int) -> str:
    """Generate a logical experiment name from configuration values.
    
    Format: startdate_enddate_factors_reg_iterations
    Example: may3_jun30_f64_r0_1_i20
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
    factors_str = f"f{factors}"
    reg_str = f"r{regularization}".replace(".", "_")  # "r0_1"
    iter_str = f"i{iterations}"
    
    return f"{start_str}_{end_str}_{factors_str}_{reg_str}_{iter_str}"


def main() -> None:
    """Load model and matrices, evaluate performance, and generate report.
    
    Supports both ALS and BPR models.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate ALS or BPR model on validation or test set."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["als", "bpr"],
        default="als",
        help="Model type: 'als' for Alternating Least Squares, 'bpr' for Bayesian Personalized Ranking (default: als)",
    )
    parser.add_argument(
        "--set",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Which dataset to evaluate on: 'validation' for hyperparameter tuning (valid), 'test' for final evaluation (default: validation)",
    )
    parser.add_argument(
        "--recalculate-user",
        action="store_true",
        help="If set, recalculates user embeddings on-the-fly instead of using stored embeddings. "
             "Useful for cold-start users or when users have new interactions after training.",
    )
    args = parser.parse_args()

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

    # Stage 1: load trained model and interaction matrices based on evaluation set.
    model = joblib.load(model_path)
    
    if args.set == "validation":
        # Hyperparameter tuning: use train_val and valid_warm
        matrices = joblib.load(MATRICES_TUNE_PATH)
        train_matrix = matrices["train_val"].tocsr().astype("float32")
        eval_matrix = matrices["validation_warm"].tocsr().astype("float32")
        date_start = TRAIN_VAL_START
        date_end = TRAIN_VAL_END
    else:
        # Final evaluation: use train and test_warm
        matrices = joblib.load(MATRICES_FINAL_PATH)
        train_matrix = matrices["train"].tocsr().astype("float32")
        eval_matrix = matrices["test_warm"].tocsr().astype("float32")
        date_start = TRAIN_START
        date_end = TRAIN_END

    # Stage 2: compute Precision@K and Recall@K using custom evaluation function.
    # This function works with any model and supports recalculate_user parameter.
    metrics = evaluate_precision_and_recall_at_k(
        model=model,
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        k=TOP_K,
        recalculate_user=args.recalculate_user,
        combine_train_eval=False,
        show_progress=True,
    )
    eval_precision = metrics["precision_at_k"]
    eval_recall = metrics["recall_at_k"]

    # Set MLflow tracking and start run for evaluation.
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(experiment_name)
    
    experiment_name_str = generate_experiment_name(args.set, args.model, factors, regularization, iterations)
    run_name = f"evaluation_{args.set}_{experiment_name_str}"
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters (same as training).
        params = {
            "model_type": args.model.upper(),
            "factors": factors,
            "regularization": regularization,
            "iterations": iterations,
            "date_window_start": date_start,
            "date_window_end": date_end,
            "eval_set": args.set,
            "top_k": TOP_K,
            "recalculate_user": args.recalculate_user,
        }
        if learning_rate is not None:  # BPR-specific parameter
            params["learning_rate"] = learning_rate
        mlflow.log_params(params)
        
        # Log metrics (both Precision@K and Recall@K).
        mlflow.log_metrics({
            f"{args.set}_precision_at_k": eval_precision,
            f"{args.set}_recall_at_k": eval_recall,
        })
        
        # Log evaluation data info.
        mlflow.log_metrics({
            f"{args.set}_users": eval_matrix.shape[0],
            f"{args.set}_items": eval_matrix.shape[1],
            f"{args.set}_interactions": eval_matrix.nnz,
        })

    # Stage 3: Print summary of results.
    print(f"\n{args.model.upper()} Model - {args.set.capitalize()} Metrics:")
    print(f"  Precision@{TOP_K}: {eval_precision:.4f}")
    print(f"  Recall@{TOP_K}: {eval_recall:.4f}")
    print(f"Experiment: {experiment_name_str}")
    print(f"Model type: {args.model.upper()}")
    print(f"Trained on: {'train_val' if args.set == 'validation' else 'train'}")
    print(f"Evaluated on: {'valid' if args.set == 'validation' else 'test'}")
    print(f"Recalculate user: {args.recalculate_user}")
    print(f"Metrics logged to MLflow. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
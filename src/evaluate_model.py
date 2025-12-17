"""Evaluate ALS, BPR, or LightFM model performance and generate metrics report."""

import argparse
import json
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
    BPR_FACTORS, BPR_LEARNING_RATE, BPR_REGULARIZATION, BPR_ITERATIONS,
    LIGHTFM_NO_COMPONENTS, LIGHTFM_LEARNING_RATE, LIGHTFM_LOSS, LIGHTFM_EPOCHS,
    BEST_HYPERPARAMETERS_PATH
)
from src.evaluate_utils import evaluate_precision_and_recall_at_k, evaluate_lightfm_precision_and_recall_at_k

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
# Using full matrices for LightFM (includes cold-start users)
MATRICES_TUNE_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune_full.pkl"
)
MATRICES_FINAL_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_full.pkl"
)
ENCODERS_TUNE_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_tune_full.pkl"
)
ENCODERS_FINAL_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_final_full.pkl"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)

# Evaluation configuration.
TOP_K = 10  # number of items evaluated per user


def generate_experiment_name(eval_set: str, model_type: str, **hyperparams) -> str:
    """Generate a logical experiment name from configuration values.
    
    Format: startdate_enddate_hyperparams
    Example: may3_jun30_f64_r0_1_i20 (for ALS/BPR)
    Example: may3_jun30_c64_lr0_05_warp_e20 (for LightFM)
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
    
    # Format hyperparameters based on model type
    if model_type == "lightfm":
        # LightFM: c64_lr0_05_warp_e20
        components_str = f"c{hyperparams.get('no_components', 64)}"
        lr_str = f"lr{hyperparams.get('learning_rate', 0.05)}".replace(".", "_")
        loss_str = hyperparams.get('loss', 'warp')
        epochs_str = f"e{hyperparams.get('epochs', 20)}"
        return f"{start_str}_{end_str}_{components_str}_{lr_str}_{loss_str}_{epochs_str}"
    else:
        # ALS/BPR: f64_r0_1_i20
        factors_str = f"f{hyperparams.get('factors', 64)}"
        reg_str = f"r{hyperparams.get('regularization', 0.1)}".replace(".", "_")
        iter_str = f"i{hyperparams.get('iterations', 20)}"
        return f"{start_str}_{end_str}_{factors_str}_{reg_str}_{iter_str}"


def main() -> None:
    """Load model and matrices, evaluate performance, and generate report.
    
    Supports ALS, BPR, and LightFM models.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate ALS, BPR, or LightFM model on validation or test set."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["als", "bpr", "lightfm"],
        default="als",
        help="Model type: 'als' for Alternating Least Squares, 'bpr' for Bayesian Personalized Ranking, 'lightfm' for LightFM (default: als)",
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
    parser.add_argument(
        "--hyperparameters",
        action="store_true",
        help="Load best hyperparameters from saved file (models/best_hyperparameters.json). "
             "If not set, uses default hyperparameters.",
    )
    args = parser.parse_args()

    # Load hyperparameters: either from saved file or use defaults
    if args.hyperparameters:
        # Try to load best hyperparameters from saved file
        if BEST_HYPERPARAMETERS_PATH.exists():
            with open(BEST_HYPERPARAMETERS_PATH, "r") as f:
                best_hyperparameters = json.load(f)
            
            if args.model in best_hyperparameters:
                # Use saved best hyperparameters
                saved_params = best_hyperparameters[args.model]
                print(f"Loading hyperparameters from {BEST_HYPERPARAMETERS_PATH}")
                print(f"Hyperparameters: {saved_params}")
                
                if args.model == "als":
                    factors = saved_params["factors"]
                    regularization = saved_params["regularization"]
                    iterations = saved_params["iterations"]
                    learning_rate = None
                elif args.model == "bpr":
                    factors = saved_params["factors"]
                    learning_rate = saved_params["learning_rate"]
                    regularization = saved_params["regularization"]
                    iterations = saved_params["iterations"]
                else:  # lightfm
                    no_components = saved_params["no_components"]
                    learning_rate = saved_params["learning_rate"]
                    loss = saved_params["loss"]
                    epochs = saved_params["epochs"]
            else:
                # Model type not found in saved file, use defaults
                print(f"Warning: No saved hyperparameters found for {args.model}. Using defaults.")
                if args.model == "als":
                    factors = ALS_FACTORS
                    regularization = ALS_REGULARIZATION
                    iterations = ALS_ITERATIONS
                    learning_rate = None
                elif args.model == "bpr":
                    factors = BPR_FACTORS
                    regularization = BPR_REGULARIZATION
                    iterations = BPR_ITERATIONS
                    learning_rate = BPR_LEARNING_RATE
                else:  # lightfm
                    no_components = LIGHTFM_NO_COMPONENTS
                    learning_rate = LIGHTFM_LEARNING_RATE
                    loss = LIGHTFM_LOSS
                    epochs = LIGHTFM_EPOCHS
        else:
            # File doesn't exist, use defaults
            print(f"Warning: {BEST_HYPERPARAMETERS_PATH} not found. Using default hyperparameters.")
            if args.model == "als":
                factors = ALS_FACTORS
                regularization = ALS_REGULARIZATION
                iterations = ALS_ITERATIONS
                learning_rate = None
            elif args.model == "bpr":
                factors = BPR_FACTORS
                regularization = BPR_REGULARIZATION
                iterations = BPR_ITERATIONS
                learning_rate = BPR_LEARNING_RATE
            else:  # lightfm
                no_components = LIGHTFM_NO_COMPONENTS
                learning_rate = LIGHTFM_LEARNING_RATE
                loss = LIGHTFM_LOSS
                epochs = LIGHTFM_EPOCHS
    else:
        # Use default hyperparameters
        if args.model == "als":
            factors = ALS_FACTORS
            regularization = ALS_REGULARIZATION
            iterations = ALS_ITERATIONS
            learning_rate = None
        elif args.model == "bpr":
            factors = BPR_FACTORS
            regularization = BPR_REGULARIZATION
            iterations = BPR_ITERATIONS
            learning_rate = BPR_LEARNING_RATE
        else:  # lightfm
            no_components = LIGHTFM_NO_COMPONENTS
            learning_rate = LIGHTFM_LEARNING_RATE
            loss = LIGHTFM_LOSS
            epochs = LIGHTFM_EPOCHS

    # Set model-specific paths based on model type
    if args.model == "als":
        model_path = MODELS_DIR / "als_model.pkl"
        experiment_name = "ALS_Recommendation_System"
    elif args.model == "bpr":
        model_path = MODELS_DIR / "bpr_model.pkl"
        experiment_name = "BPR_Recommendation_System"
    else:  # lightfm
        model_path = MODELS_DIR / "lightfm_model.pkl"
        experiment_name = "LightFM_Recommendation_System"

    # Stage 1: load trained model and interaction matrices based on evaluation set and model type.
    if args.model == "lightfm":
        # LightFM: load model artifact (contains model, dataset, encoders)
        model_artifact = joblib.load(model_path)
        model = model_artifact["model"]
        dataset = model_artifact["dataset"]
        encoders = model_artifact["encoders"]
        
        # Use full splits for LightFM
        if args.set == "validation":
            matrices = joblib.load(MATRICES_TUNE_FULL_PATH)
            train_matrix = matrices["train_val"].tocsr()
            eval_matrix = matrices["validation_full"].tocsr()
            date_start = TRAIN_VAL_START
            date_end = TRAIN_VAL_END
        else:
            matrices = joblib.load(MATRICES_FINAL_FULL_PATH)
            train_matrix = matrices["train"].tocsr()
            eval_matrix = matrices["test_full"].tocsr()
            date_start = TRAIN_START
            date_end = TRAIN_END
    else:
        # ALS/BPR: load model directly
        model = joblib.load(model_path)
        
        # Use warm splits for ALS/BPR
        if args.set == "validation":
            matrices = joblib.load(MATRICES_TUNE_PATH)
            train_matrix = matrices["train_val"].tocsr().astype("float32")
            eval_matrix = matrices["validation_warm"].tocsr().astype("float32")
            date_start = TRAIN_VAL_START
            date_end = TRAIN_VAL_END
        else:
            matrices = joblib.load(MATRICES_FINAL_PATH)
            train_matrix = matrices["train"].tocsr().astype("float32")
            eval_matrix = matrices["test_warm"].tocsr().astype("float32")
            date_start = TRAIN_START
            date_end = TRAIN_END

    # Stage 2: compute Precision@K and Recall@K using model-specific evaluation function.
    if args.model == "lightfm":
        # LightFM uses predict() method, needs Dataset object
        metrics = evaluate_lightfm_precision_and_recall_at_k(
            model=model,
            dataset=dataset,
            train_matrix=train_matrix,
            eval_matrix=eval_matrix,
            k=TOP_K,
            show_progress=True,
        )
    else:
        # ALS/BPR use recommend() method
        metrics = evaluate_precision_and_recall_at_k(
            model=model,
            train_matrix=train_matrix,
            eval_matrix=eval_matrix,
            k=TOP_K,
            recalculate_user=args.recalculate_user,
            show_progress=True,
        )
    eval_precision = metrics["precision_at_k"]
    eval_recall = metrics["recall_at_k"]

    # Set MLflow tracking and start run for evaluation.
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(experiment_name)
    
    # Generate experiment name based on model type
    if args.model == "lightfm":
        experiment_name_str = generate_experiment_name(
            args.set, args.model, 
            no_components=no_components, 
            learning_rate=learning_rate, 
            loss=loss, 
            epochs=epochs
        )
    else:
        experiment_name_str = generate_experiment_name(
            args.set, args.model, 
            factors=factors, 
            regularization=regularization, 
            iterations=iterations
        )
    
    run_name = f"evaluation_{args.set}_{experiment_name_str}"
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters (same as training).
        if args.model == "lightfm":
            params = {
                "model_type": args.model.upper(),
                "no_components": no_components,
                "learning_rate": learning_rate,
                "loss": loss,
                "epochs": epochs,
                "date_window_start": date_start,
                "date_window_end": date_end,
                "eval_set": args.set,
                "top_k": TOP_K,
            }
        else:
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
    if args.model != "lightfm":
        print(f"Recalculate user: {args.recalculate_user}")
    print(f"Metrics logged to MLflow. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
"""Train recommendation models (ALS, BPR, LightFM, or SVD) using prepared interaction matrices or ratings.

This script supports two training stages (used sequentially, not simultaneously):
- Stage 1 (tune mode): Train on train_val for hyperparameter tuning
- Stage 2 (final mode): Train on train for final model with best hyperparameters

Supports four model types:
- ALS (Alternating Least Squares): Matrix factorization using alternating optimization
- BPR (Bayesian Personalized Ranking): Optimizes for pairwise ranking using SGD
- LightFM: Hybrid recommendation algorithm that can handle cold-start users
- SVD (Singular Value Decomposition): Matrix factorization for explicit ratings using Surprise library
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from lightfm import LightFM
from lightfm.data import Dataset

# Import custom SVD implementation (Surprise-compatible)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from svd_model import SVD, Dataset as SurpriseDataset

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
# Using full matrices for LightFM (includes cold-start users)
MATRICES_TUNE_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_tune_full.pkl"
)
MATRICES_FINAL_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_full.pkl"
)
# Encoder paths
ENCODERS_TUNE_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_tune_full.pkl"
)
ENCODERS_FINAL_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_final_full.pkl"
)
# Rating CSV paths for SVD
RATINGS_TUNE_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/ratings_train_val.csv"
)
RATINGS_FINAL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/ratings_train.csv"
)
MODELS_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)
BEST_HYPERPARAMETERS_PATH = MODELS_DIR / "best_hyperparameters.json"

# Hyperparameters for ALS model.
ALS_FACTORS = 64  # latent dimensions
ALS_REGULARIZATION = 0.1  # controls overfitting
ALS_ITERATIONS = 20  # alternating least squares steps

# Hyperparameters for BPR model.
BPR_FACTORS = 64  # latent dimensions (same as ALS)
BPR_LEARNING_RATE = 0.01  # How fast the model learns (BPR-specific, typically 0.01-0.1)
BPR_REGULARIZATION = 0.01  # controls overfitting (typically lower than ALS)
BPR_ITERATIONS = 100  # Number of training iterations (BPR often needs more than ALS)

# Hyperparameters for LightFM model.
LIGHTFM_NO_COMPONENTS = 64  # latent dimensions (same as ALS/BPR)
LIGHTFM_LEARNING_RATE = 0.05  # How fast the model learns (typically 0.01-0.1)
LIGHTFM_LOSS = 'warp'  # Loss function: 'warp' for implicit feedback (recommended)
LIGHTFM_EPOCHS = 20  # Number of training epochs

# Hyperparameters for SVD model.
SVD_N_FACTORS = 50  # Number of latent factors
SVD_N_EPOCHS = 20  # Number of training epochs
SVD_LR_ALL = 0.005  # Learning rate for all parameters
SVD_REG_ALL = 0.02  # Regularization term for all parameters


def main() -> None:
    """Load matrices, fit model (ALS, BPR, or LightFM), and save it.
    
    Supports two modes:
    - 'tune': Train on train_val for hyperparameter tuning
    - 'final': Train on train for final model
    
    Supports four model types:
    - 'als': Alternating Least Squares
    - 'bpr': Bayesian Personalized Ranking
    - 'lightfm': LightFM hybrid recommendation algorithm
    - 'svd': Singular Value Decomposition for explicit ratings
    """

    parser = argparse.ArgumentParser(
        description="Train ALS, BPR, LightFM, or SVD model for hyperparameter tuning or final model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["als", "bpr", "lightfm", "svd"],
        default="als",
        help="Model type: 'als' for Alternating Least Squares, 'bpr' for Bayesian Personalized Ranking, 'lightfm' for LightFM, 'svd' for SVD (default: als)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tune", "final"],
        default="final",
        help="Training mode: 'tune' for hyperparameter tuning (train_val), 'final' for final model (train) (default: final)",
    )
    parser.add_argument(
        "--hyperparameters",
        action="store_true",
        help="Load best hyperparameters from saved file (models/best_hyperparameters.json). "
             "If not set, uses default hyperparameters.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load hyperparameters: either from saved file or use defaults
    if args.hyperparameters:
        # Try to load best hyperparameters from saved file
        if BEST_HYPERPARAMETERS_PATH.exists():
            with open(BEST_HYPERPARAMETERS_PATH, "r") as f:
                best_hyperparameters = json.load(f)
            
            if args.model in best_hyperparameters:
                # Use saved best hyperparameters
                saved_params = best_hyperparameters[args.model]
                print(f"Loading best hyperparameters from {BEST_HYPERPARAMETERS_PATH}")
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
                elif args.model == "lightfm":
                    no_components = saved_params["no_components"]
                    learning_rate = saved_params["learning_rate"]
                    loss = saved_params["loss"]
                    epochs = saved_params["epochs"]
                else:  # svd
                    n_factors = saved_params["n_factors"]
                    n_epochs = saved_params["n_epochs"]
                    lr_all = saved_params["lr_all"]
                    reg_all = saved_params["reg_all"]
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
                elif args.model == "lightfm":
                    no_components = LIGHTFM_NO_COMPONENTS
                    learning_rate = LIGHTFM_LEARNING_RATE
                    loss = LIGHTFM_LOSS
                    epochs = LIGHTFM_EPOCHS
                else:  # svd
                    n_factors = SVD_N_FACTORS
                    n_epochs = SVD_N_EPOCHS
                    lr_all = SVD_LR_ALL
                    reg_all = SVD_REG_ALL
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
            elif args.model == "lightfm":
                no_components = LIGHTFM_NO_COMPONENTS
                learning_rate = LIGHTFM_LEARNING_RATE
                loss = LIGHTFM_LOSS
                epochs = LIGHTFM_EPOCHS
            else:  # svd
                n_factors = SVD_N_FACTORS
                n_epochs = SVD_N_EPOCHS
                lr_all = SVD_LR_ALL
                reg_all = SVD_REG_ALL
    else:
        # Use default hyperparameters
        if args.model == "als":
            factors = ALS_FACTORS
            regularization = ALS_REGULARIZATION
            iterations = ALS_ITERATIONS
            learning_rate = None  # Not used for ALS
        elif args.model == "bpr":
            factors = BPR_FACTORS
            regularization = BPR_REGULARIZATION
            iterations = BPR_ITERATIONS
            learning_rate = BPR_LEARNING_RATE
        elif args.model == "lightfm":
            no_components = LIGHTFM_NO_COMPONENTS
            learning_rate = LIGHTFM_LEARNING_RATE
            loss = LIGHTFM_LOSS
            epochs = LIGHTFM_EPOCHS
        else:  # svd
            n_factors = SVD_N_FACTORS
            n_epochs = SVD_N_EPOCHS
            lr_all = SVD_LR_ALL
            reg_all = SVD_REG_ALL
    
    # Set model-specific paths based on model type
    if args.model == "als":
        model_path = MODELS_DIR / "als_model.pkl"
        experiment_name = "ALS_Recommendation_System"
    elif args.model == "bpr":
        model_path = MODELS_DIR / "bpr_model.pkl"
        experiment_name = "BPR_Recommendation_System"
    elif args.model == "lightfm":
        model_path = MODELS_DIR / "lightfm_model.pkl"
        experiment_name = "LightFM_Recommendation_System"
    else:  # svd
        model_path = MODELS_DIR / "svd_model.pkl"
        experiment_name = "SVD_Recommendation_System"

    # Set MLflow tracking URI to local directory.
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(experiment_name)

    # Stage 1: load training data based on mode and model type.
    # LightFM uses full splits (includes cold-start), ALS/BPR use warm splits, SVD uses rating CSV files
    if args.model == "svd":
        # SVD uses rating CSV files
        if args.mode == "tune":
            ratings_path = RATINGS_TUNE_PATH
            date_start = TRAIN_VAL_START
            date_end = TRAIN_VAL_END
            run_name = "training_tune"
        else:
            ratings_path = RATINGS_FINAL_PATH
            date_start = TRAIN_START
            date_end = TRAIN_END
            run_name = "training_final"
    elif args.model == "lightfm":
        # LightFM uses full splits
        if args.mode == "tune":
            matrices = joblib.load(MATRICES_TUNE_FULL_PATH)
            train_matrix = matrices["train_val"].tocsr()
            encoders = joblib.load(ENCODERS_TUNE_FULL_PATH)
            date_start = TRAIN_VAL_START
            date_end = TRAIN_VAL_END
            run_name = "training_tune"
        else:
            matrices = joblib.load(MATRICES_FINAL_FULL_PATH)
            train_matrix = matrices["train"].tocsr()
            encoders = joblib.load(ENCODERS_FINAL_FULL_PATH)
            date_start = TRAIN_START
            date_end = TRAIN_END
            run_name = "training_final"
    else:
        # ALS/BPR use warm splits
        if args.mode == "tune":
            matrices = joblib.load(MATRICES_TUNE_PATH)
            train_matrix = matrices["train_val"].tocsr().astype("float32")
            date_start = TRAIN_VAL_START
            date_end = TRAIN_VAL_END
            run_name = "training_tune"
        else:
            matrices = joblib.load(MATRICES_FINAL_PATH)
            train_matrix = matrices["train"].tocsr().astype("float32")
            date_start = TRAIN_START
            date_end = TRAIN_END
            run_name = "training_final"

    # Start MLflow run for training.
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters (model-specific).
        if args.model == "lightfm":
            params = {
                "model_type": args.model.upper(),
                "no_components": no_components,
                "learning_rate": learning_rate,
                "loss": loss,
                "epochs": epochs,
                "date_window_start": date_start,
                "date_window_end": date_end,
                "mode": args.mode,
            }
        elif args.model == "svd":
            params = {
                "model_type": args.model.upper(),
                "n_factors": n_factors,
                "n_epochs": n_epochs,
                "lr_all": lr_all,
                "reg_all": reg_all,
                "date_window_start": date_start,
                "date_window_end": date_end,
                "mode": args.mode,
            }
        else:
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
        if args.model == "svd":
            # Train SVD model using custom implementation (Surprise-compatible)
            # Load rating CSV file
            data = SurpriseDataset.load_from_file(str(ratings_path))
            trainset = data.build_full_trainset()
            
            # Create and train SVD model
            model = SVD(
                n_factors=n_factors,
                n_epochs=n_epochs,
                lr_all=lr_all,
                reg_all=reg_all,
                random_state=42  # For reproducibility
            )
            model.fit(trainset)
            
            # Save model and trainset (needed for predictions)
            model_artifact = {
                "model": model,
                "trainset": trainset,
            }
            joblib.dump(model_artifact, model_path)
            
            # Log training data info
            mlflow.log_metrics({
                "train_users": trainset.n_users,
                "train_items": trainset.n_items,
                "train_ratings": trainset.n_ratings,
            })
        elif args.model == "als":
            # Train Alternating Least Squares model
            model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                dtype=train_matrix.dtype,
            )
            model.fit(train_matrix)
            # Stage 3: persist the trained model for later use.
            joblib.dump(model, model_path)
        elif args.model == "bpr":
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
        else:  # lightfm
            # Train LightFM model
            # Step 1: Create LightFM Dataset object
            # Get unique user and item IDs from the matrix
            n_users, n_items = train_matrix.shape
            dataset = Dataset()
            # Build dataset with all users and items (0 to n_users-1, 0 to n_items-1)
            dataset.fit(
                users=range(n_users),
                items=range(n_items)
            )
            
            # Step 2: Build interactions from sparse matrix
            # Convert sparse matrix to list of tuples (user_id, item_id, weight)
            coo_matrix = train_matrix.tocoo()
            interactions = list(zip(coo_matrix.row, coo_matrix.col, coo_matrix.data))
            
            # Step 3: Build interaction matrix in LightFM format
            (interactions_matrix, weights_matrix) = dataset.build_interactions(interactions)
            
            # Step 4: Train LightFM model
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
                verbose=True,
            )
            
            # Stage 3: persist the trained model, Dataset, and encoders together
            # LightFM needs Dataset object for predictions, so we save everything together
            model_artifact = {
                "model": model,
                "dataset": dataset,
                "encoders": encoders,
            }
            joblib.dump(model_artifact, model_path)
        
        print(f"{args.model.upper()} model saved to {model_path}")
        print(f"Mode: {args.mode} (trained on {'train_val' if args.mode == 'tune' else 'train'})")

        # Log model as artifact.
        mlflow.log_artifact(str(model_path), "model")
        
        # Log training data info (if not already logged for SVD).
        if args.model == "svd":
            # Already logged above
            pass
        elif args.model == "lightfm":
            # For LightFM, log from the interactions matrix
            mlflow.log_metrics({
                "train_users": train_matrix.shape[0],
                "train_items": train_matrix.shape[1],
                "train_interactions": train_matrix.nnz,
            })
        else:
            # For ALS/BPR, log from the sparse matrix
            mlflow.log_metrics({
                "train_users": train_matrix.shape[0],
                "train_items": train_matrix.shape[1],
                "train_interactions": train_matrix.nnz,
            })
        
        print(f"MLflow run logged. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()


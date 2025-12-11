"""Evaluate baseline model (popular items) performance and generate metrics report.

This script implements a simple baseline recommender that recommends the most
popular items to all users. It's used as a benchmark to compare against more
sophisticated models like ALS.
"""

import argparse
from pathlib import Path
from itertools import product  # For creating all user-item combinations

import joblib
import mlflow
import numpy as np
import pandas as pd

# Import configuration from other modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import (
    TRAIN_VAL_START, TRAIN_VAL_END,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END
)

# Paths to data files
DATA_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)

# Evaluation configuration
TOP_K = 10  # Number of items to recommend per user


def calculate_precision_at_k_popular_items(train_df, valid_df, k=10):
    """
    Calculate Precision@K for a baseline recommender based on popular items.
    
    This function implements a simple baseline: recommend the most popular items
    to all users, excluding items they've already purchased.
    
    Precision@K measures: out of K recommendations, how many did the user
    actually buy? Formula: (recommended items that were bought) / K
    
    Args:
        train_df (pd.DataFrame): Training data with 'visitorid' and 'itemid' columns.
                                 Used to determine popular items and filter out
                                 items users already bought.
        valid_df (pd.DataFrame): Validation data with 'visitorid' and 'itemid' columns.
                                 Contains the actual items users bought (ground truth).
        k (int): Number of top items to recommend for each user. Defaults to 10.
    
    Returns:
        float: The mean Precision@K value across all users in validation set.
    """
    
    # Step 1: Find the user who bought the most items in training data
    # This helps us determine how large our popular items pool needs to be
    print("Finding the user who bought the most products...")
    user_purchase_counts = train_df['visitorid'].value_counts()
    most_purchased_count = user_purchase_counts.max()
    most_active_user_id = user_purchase_counts.idxmax()
    
    print(f"User ID {most_active_user_id} purchased the most products ({most_purchased_count} products).")
    
    # Step 2: Determine the size of popular items pool
    # We need enough items so that even the most active user can get K recommendations
    # after we filter out items they've already bought
    # Example: if most active user bought 50 items, and K=10, we need at least 50+10=60 items
    candidate_pool_size = most_purchased_count + k
    print(f"Setting popular item pool size to: {candidate_pool_size}")
    
    # Step 3: Get list of most popular items from training data
    # value_counts() counts how many times each item appears (how many users bought it)
    # .index.tolist() gets the items sorted by popularity (most popular first)
    print("Determining popular items...")
    popular_items = train_df['itemid'].value_counts().index.tolist()
    popular_items_pool = popular_items[:candidate_pool_size]
    
    # Step 4: Create all possible pairs of (user, popular_item) for users in validation set
    # This gives us candidate recommendations: every user gets every popular item as a candidate
    print("Creating candidate recommendations...")
    all_users = valid_df['visitorid'].unique()
    
    # itertools.product creates all combinations: user1×item1, user1×item2, ..., user2×item1, ...
    candidates = pd.DataFrame(
        list(product(all_users, popular_items_pool)),
        columns=['visitorid', 'itemid']
    )
    
    # Step 5: Filter out items users have already seen in training data
    # We don't want to recommend items they've already bought
    print("Filtering out items users already purchased...")
    merged = pd.merge(
        candidates,           # All user-item candidate pairs
        train_df,             # Training data (what users already bought)
        on=['visitorid', 'itemid'],  # Match on both user and item
        how='left',           # Keep all candidates, add info from train_df if match found
        indicator=True        # Add '_merge' column to show if match was found
    )
    # '_merge' == 'left_only' means: in candidates but NOT in train_df (unseen items)
    unseen_recommendations = merged[merged['_merge'] == 'left_only'][['visitorid', 'itemid']]
    
    # Step 6: For each user, take the top K unseen popular items as recommendations
    # The items are already sorted by popularity, so we just take first K per user
    print(f"Selecting top {k} recommendations per user...")
    top_k_recs_df = unseen_recommendations.groupby('visitorid').head(k)
    
    # Convert to dictionary: {userID: [item1, item2, ..., itemK]}
    recs_map = top_k_recs_df.groupby('visitorid')['itemid'].apply(list).to_dict()
    
    # Step 7: Create a map of actual items users bought in validation set
    # This is our "ground truth" - what we're comparing recommendations against
    print("Preparing validation data (ground truth)...")
    actuals_map = valid_df.groupby('visitorid')['itemid'].apply(set).to_dict()
    # Using set() for fast lookup when checking if item is in actual purchases
    
    # Step 8: Calculate Precision@K for each user and then find the mean
    print("Calculating Precision@K for each user...")
    precisions = []
    
    # Iterate ONLY over users who are in the validation set (the "actuals")
    for user_id, actual_items in actuals_map.items():
        # Get recommendations for this user (empty list if user has no recommendations)
        rec_items = recs_map.get(user_id, [])
        
        # Skip users with no recommendations (shouldn't happen, but safety check)
        if not rec_items:
            continue
        
        # Count how many recommended items the user actually bought
        # set.intersection() finds common items between recommendations and actual purchases
        hits = len(set(rec_items).intersection(actual_items))
        
        # Precision@K = (number of hits) / (number of recommendations)
        precision = hits / len(rec_items)
        precisions.append(precision)
    
    # Calculate mean Precision@K across all users
    # This gives us the overall performance of the baseline model
    mean_precision = np.mean(precisions) if precisions else 0.0
    print(f"\nPrecision@{k} for users in validation set: {mean_precision:.4f}")
    print(f"Calculated for {len(precisions)} users")
    
    return mean_precision


def generate_experiment_name(eval_set: str) -> str:
    """Generate a logical experiment name for baseline evaluation.
    
    Format: baseline_startdate_enddate
    Example: baseline_may3_jun30
    
    Args:
        eval_set (str): Either "validation" or "test"
    
    Returns:
        str: Experiment name string
    """
    # Choose date range based on evaluation set
    if eval_set == "validation":
        # Hyperparameter tuning: use train_val dates
        start = pd.to_datetime(TRAIN_VAL_START)
        end = pd.to_datetime(TRAIN_VAL_END)
    else:
        # Final evaluation: use train dates
        start = pd.to_datetime(TRAIN_START)
        end = pd.to_datetime(TRAIN_END)
    
    start_str = f"{start.strftime('%b').lower()}{start.day}"  # "may3"
    end_str = f"{end.strftime('%b').lower()}{end.day}"  # "jun30"
    
    return f"baseline_{start_str}_{end_str}"


def main() -> None:
    """Load data, evaluate baseline model, and log results to MLflow."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate baseline (popular items) model on validation or test set."
    )
    parser.add_argument(
        "--set",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Which dataset to evaluate on: 'validation' for hyperparameter tuning, "
             "'test' for final evaluation (default: validation)",
    )
    args = parser.parse_args()
    
    # Stage 1: Load training and validation/test DataFrames based on evaluation set
    print(f"\nLoading data for {args.set} evaluation...")
    
    if args.set == "validation":
        # Hyperparameter tuning workflow: use train_val and valid
        train_df = joblib.load(DATA_DIR / "events_train_val.pkl")
        valid_df = joblib.load(DATA_DIR / "events_validation.pkl")
        date_start = TRAIN_VAL_START
        date_end = TRAIN_VAL_END
        train_set_name = "train_val"
        eval_set_name = "valid"
    else:
        # Final evaluation workflow: use train and test
        train_df = joblib.load(DATA_DIR / "events_train.pkl")
        valid_df = joblib.load(DATA_DIR / "events_test.pkl")
        date_start = TRAIN_START
        date_end = TRAIN_END
        train_set_name = "train"
        eval_set_name = "test"
    
    print(f"Training data: {len(train_df)} events, {train_df['visitorid'].nunique()} users, "
          f"{train_df['itemid'].nunique()} items")
    print(f"Evaluation data: {len(valid_df)} events, {valid_df['visitorid'].nunique()} users, "
          f"{valid_df['itemid'].nunique()} items")
    
    # Stage 2: Extract only the columns we need (visitorid and itemid)
    # The baseline function only needs user-item pairs
    train_df_clean = train_df[['visitorid', 'itemid']].copy()
    valid_df_clean = valid_df[['visitorid', 'itemid']].copy()
    
    # Stage 3: Calculate Precision@K using baseline recommender
    print(f"\nEvaluating baseline model (recommending top {TOP_K} popular items)...")
    eval_precision = calculate_precision_at_k_popular_items(
        train_df_clean,
        valid_df_clean,
        k=TOP_K
    )
    
    # Stage 4: Log results to MLflow for experiment tracking
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment("ALS_Recommendation_System")
    
    experiment_name = generate_experiment_name(args.set)
    run_name = f"baseline_evaluation_{args.set}_{experiment_name}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters (baseline has no hyperparameters, but log configuration)
        mlflow.log_params({
            "model_type": "baseline_popular_items",
            "date_window_start": date_start,
            "date_window_end": date_end,
            "eval_set": args.set,
            "top_k": TOP_K,
        })
        
        # Log metrics
        mlflow.log_metrics({
            f"{args.set}_precision_at_k": eval_precision,
        })
        
        # Log evaluation data info
        mlflow.log_metrics({
            f"{args.set}_users": valid_df['visitorid'].nunique(),
            f"{args.set}_items": valid_df['itemid'].nunique(),
            f"{args.set}_interactions": len(valid_df),
        })
    
    # Stage 5: Print summary of results
    print(f"\n{'='*60}")
    print(f"Baseline Model Evaluation Results")
    print(f"{'='*60}")
    print(f"{args.set.capitalize()} Precision@{TOP_K}: {eval_precision:.4f}")
    print(f"Experiment: {experiment_name}")
    print(f"Trained on: {train_set_name}")
    print(f"Evaluated on: {eval_set_name}")
    print(f"Metrics logged to MLflow.")
    print(f"View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


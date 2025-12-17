"""Evaluate combined recommendations (Borda Count) on all test users.

This script evaluates the combined recommendation approach by:
1. Getting recommendations from ALS, BPR, and LightFM for all test users
2. Combining them using Borda Count method
3. Calculating Precision@10 for combined recommendations
4. Comparing with individual algorithm performance
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import mlflow
import numpy as np
from scipy import sparse
from tqdm import tqdm

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import TRAIN_START, TRAIN_END
from src.combine_recommendations import borda_count_combine, calculate_precision_at_k
from src.recommend import recommend

# Paths to model artifacts and data
MODELS_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models"
)
MATRICES_WARM_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_warm.pkl"
)
MATRICES_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_full.pkl"
)
ENCODERS_WARM_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_final_warm.pkl"
)
ENCODERS_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_final_full.pkl"
)
MLFLOW_TRACKING_URI = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/mlruns"
)

# Evaluation configuration
TOP_K = 10  # number of items evaluated per user


def evaluate_combined_precision_at_k(
    k: int = 10,
    eval_set: str = "test",
    show_progress: bool = True
) -> Dict[str, float]:
    """Evaluate Precision@K for combined recommendations and individual algorithms.
    
    Args:
        k: Number of top recommendations to evaluate (default: 10)
        eval_set: Which evaluation set to use - 'test' or 'validation' (default: 'test')
        show_progress: Show progress bar if True
    
    Returns:
        Dictionary with precision@k scores for combined, als, bpr, and lightfm
    """
    # Load test matrices and encoders
    # Use warm splits for ALS/BPR, full splits for LightFM
    matrices_warm = joblib.load(MATRICES_WARM_PATH)
    matrices_full = joblib.load(MATRICES_FULL_PATH)
    encoders_warm = joblib.load(ENCODERS_WARM_PATH)
    encoders_full = joblib.load(ENCODERS_FULL_PATH)
    
    # Get evaluation matrix based on eval_set
    if eval_set == "test":
        eval_matrix_warm = matrices_warm["test_warm"].tocsr()
        eval_matrix_full = matrices_full["test_full"].tocsr()
    else:  # validation
        eval_matrix_warm = matrices_warm["validation_warm"].tocsr()
        eval_matrix_full = matrices_full["validation_full"].tocsr()
    
    visitor_encoder_warm = encoders_warm["visitorid"]
    item_encoder_warm = encoders_warm["itemid"]
    visitor_encoder_full = encoders_full["visitorid"]
    item_encoder_full = encoders_full["itemid"]
    
    # Get users with evaluation interactions (ground truth)
    # Use warm matrix to get user list (covers ALS/BPR users)
    eval_user_indices_warm = np.unique(eval_matrix_warm.nonzero()[0])
    
    # Also get users from full matrix (covers LightFM users)
    eval_user_indices_full = np.unique(eval_matrix_full.nonzero()[0])
    
    # Combine and get unique users
    all_user_indices_warm = set(eval_user_indices_warm)
    all_user_indices_full = set(eval_user_indices_full)
    
    # Convert user indices back to visitor IDs
    warm_visitor_ids = visitor_encoder_warm.inverse_transform(list(eval_user_indices_warm))
    full_visitor_ids = visitor_encoder_full.inverse_transform(list(eval_user_indices_full))
    
    # Create mapping: visitor_id -> (warm_user_index, full_user_index, warm_encoder, full_encoder)
    visitor_to_indices = {}
    for visitor_id, warm_idx in zip(warm_visitor_ids, eval_user_indices_warm):
        visitor_to_indices[visitor_id] = {
            'warm_index': warm_idx,
            'full_index': None,
            'has_warm': True,
            'has_full': False
        }
    
    for visitor_id, full_idx in zip(full_visitor_ids, eval_user_indices_full):
        if visitor_id in visitor_to_indices:
            visitor_to_indices[visitor_id]['full_index'] = full_idx
            visitor_to_indices[visitor_id]['has_full'] = True
        else:
            visitor_to_indices[visitor_id] = {
                'warm_index': None,
                'full_index': full_idx,
                'has_warm': False,
                'has_full': True
            }
    
    # Lists to store precision scores
    combined_precisions = []
    als_precisions = []
    bpr_precisions = []
    lightfm_precisions = []
    
    # Process each user
    user_iterator = tqdm(visitor_to_indices.items(), desc="Evaluating users") if show_progress else visitor_to_indices.items()
    
    for visitor_id, indices_info in user_iterator:
        # Get ground truth items for this user
        ground_truth_items = []
        
        if indices_info['has_warm']:
            warm_idx = indices_info['warm_index']
            user_eval_interactions = eval_matrix_warm[warm_idx]
            actual_item_indices = user_eval_interactions.nonzero()[1]
            if len(actual_item_indices) > 0:
                ground_truth_items.extend(item_encoder_warm.inverse_transform(actual_item_indices).tolist())
        
        if indices_info['has_full']:
            full_idx = indices_info['full_index']
            user_eval_interactions = eval_matrix_full[full_idx]
            actual_item_indices = user_eval_interactions.nonzero()[1]
            if len(actual_item_indices) > 0:
                full_items = item_encoder_full.inverse_transform(actual_item_indices).tolist()
                # Add items not already in ground_truth_items
                for item in full_items:
                    if item not in ground_truth_items:
                        ground_truth_items.append(item)
        
        if len(ground_truth_items) == 0:
            continue  # Skip users with no ground truth
        
        # Get recommendations from all three algorithms
        try:
            als_recs = recommend(visitor_id, model_type="als", top_n=k)
            bpr_recs = recommend(visitor_id, model_type="bpr", top_n=k)
            lightfm_recs = recommend(visitor_id, model_type="lightfm", top_n=k)
        except Exception:
            continue  # Skip if recommendation fails
        
        # Skip if no recommendations from any algorithm
        if not als_recs and not bpr_recs and not lightfm_recs:
            continue
        
        # Combine recommendations using Borda Count
        recommendations_dict = {
            "als": als_recs,
            "bpr": bpr_recs,
            "lightfm": lightfm_recs
        }
        combined_ranking = borda_count_combine(recommendations_dict)
        combined_item_ids = [item_id for item_id, _ in combined_ranking[:k]]
        
        # Calculate precision@k for each approach
        # Use 0.0 precision if algorithm returned no recommendations (consistent sample size)
        combined_precision = calculate_precision_at_k(combined_item_ids, ground_truth_items, k=k)
        als_precision = calculate_precision_at_k(als_recs, ground_truth_items, k=k) if als_recs else 0.0
        bpr_precision = calculate_precision_at_k(bpr_recs, ground_truth_items, k=k) if bpr_recs else 0.0
        lightfm_precision = calculate_precision_at_k(lightfm_recs, ground_truth_items, k=k) if lightfm_recs else 0.0
        
        # Always append all precisions to maintain consistent sample size across all algorithms
        # This ensures fair comparison - all metrics are averaged over the same set of users
        combined_precisions.append(combined_precision)
        als_precisions.append(als_precision)
        bpr_precisions.append(bpr_precision)
        lightfm_precisions.append(lightfm_precision)
    
    # Calculate mean precision@k across all users
    mean_combined_precision = np.mean(combined_precisions) if combined_precisions else 0.0
    mean_als_precision = np.mean(als_precisions) if als_precisions else 0.0
    mean_bpr_precision = np.mean(bpr_precisions) if bpr_precisions else 0.0
    mean_lightfm_precision = np.mean(lightfm_precisions) if lightfm_precisions else 0.0
    
    # All precision lists now have the same length (consistent sample size)
    num_users_evaluated = len(combined_precisions)
    
    return {
        'combined_precision_at_k': mean_combined_precision,
        'als_precision_at_k': mean_als_precision,
        'bpr_precision_at_k': mean_bpr_precision,
        'lightfm_precision_at_k': mean_lightfm_precision,
        'num_users_evaluated': num_users_evaluated,
        'num_users_als': num_users_evaluated,  # Same as num_users_evaluated (consistent sample size)
        'num_users_bpr': num_users_evaluated,  # Same as num_users_evaluated (consistent sample size)
        'num_users_lightfm': num_users_evaluated,  # Same as num_users_evaluated (consistent sample size)
    }


def main() -> None:
    """Evaluate combined recommendations on all test users and compare with individual algorithms."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate combined recommendations (Borda Count) on all test users."
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        choices=["test", "validation"],
        default="test",
        help="Which dataset to evaluate on: 'test' or 'validation' (default: test)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to evaluate per user (default: 10)",
    )
    
    args = parser.parse_args()
    
    print(f"Evaluating combined recommendations on {args.eval_set} set...")
    print(f"Top-K: {args.top_k}")
    print("This may take several minutes as it evaluates all users...")
    print()
    
    # Evaluate combined recommendations
    metrics = evaluate_combined_precision_at_k(
        k=args.top_k,
        eval_set=args.eval_set,
        show_progress=True
    )
    
    # Display results
    print("\n" + "="*80)
    print(f"Combined Recommendations Evaluation Results ({args.eval_set.upper()} Set)")
    print("="*80)
    print(f"\nUsers evaluated: {metrics['num_users_evaluated']:,}")
    print(f"  - ALS:      {metrics['num_users_als']:,} users")
    print(f"  - BPR:      {metrics['num_users_bpr']:,} users")
    print(f"  - LightFM:  {metrics['num_users_lightfm']:,} users")
    print("\n" + "-"*80)
    print("Precision@K Scores:")
    print("-"*80)
    print(f"  Combined (Borda Count): {metrics['combined_precision_at_k']:.6f}")
    print(f"  ALS:                    {metrics['als_precision_at_k']:.6f}")
    print(f"  BPR:                    {metrics['bpr_precision_at_k']:.6f}")
    print(f"  LightFM:                {metrics['lightfm_precision_at_k']:.6f}")
    print("-"*80)
    
    # Calculate improvement over best individual algorithm
    individual_scores = [
        metrics['als_precision_at_k'],
        metrics['bpr_precision_at_k'],
        metrics['lightfm_precision_at_k']
    ]
    best_individual = max(individual_scores)
    improvement = metrics['combined_precision_at_k'] - best_individual
    improvement_pct = (improvement / best_individual * 100) if best_individual > 0 else 0.0
    
    print(f"\nBest Individual Algorithm: {best_individual:.6f}")
    print(f"Combined Improvement:      {improvement:+.6f} ({improvement_pct:+.2f}%)")
    print("="*80)
    
    # Log to MLflow
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment("Combined_Recommendations")
    
    with mlflow.start_run(run_name=f"combined_borda_count_{args.eval_set}"):
        mlflow.log_params({
            "method": "Borda Count",
            "eval_set": args.eval_set,
            "top_k": args.top_k,
            "date_window_start": TRAIN_START,
            "date_window_end": TRAIN_END,
        })
        
        mlflow.log_metrics({
            f"{args.eval_set}_combined_precision_at_k": metrics['combined_precision_at_k'],
            f"{args.eval_set}_als_precision_at_k": metrics['als_precision_at_k'],
            f"{args.eval_set}_bpr_precision_at_k": metrics['bpr_precision_at_k'],
            f"{args.eval_set}_lightfm_precision_at_k": metrics['lightfm_precision_at_k'],
            f"{args.eval_set}_improvement_over_best": improvement,
            "num_users_evaluated": metrics['num_users_evaluated'],
        })
        
        print(f"\nMetrics logged to MLflow. View at: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()


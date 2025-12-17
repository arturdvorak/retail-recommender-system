"""Combine recommendations from ALS, BPR, and LightFM models.

This script shows top-N recommendations from all three algorithms side by side
for easy comparison. Useful for understanding how different algorithms
recommend different items to the same user.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from collections import defaultdict

from src.recommend import recommend

# Paths to evaluation data (for precision@k calculation)
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


def calculate_precision_at_k(
    recommended_items: List[int],
    actual_items: List[int],
    k: int = 10
) -> float:
    """Calculate Precision@K for a single user.
    
    Precision@K = (number of recommended items that user actually interacted with) / K
    
    Args:
        recommended_items: List of recommended item IDs (top K items)
        actual_items: List of item IDs user actually interacted with (ground truth)
        k: Number of recommendations to evaluate (default: 10)
    
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    # Take top K recommendations
    top_k_recommended = set(recommended_items[:k])
    actual_set = set(actual_items)
    
    # Count how many recommended items match actual items
    hits = len(top_k_recommended.intersection(actual_set))
    
    # Precision@K = hits / K
    precision = hits / k
    
    return precision


def get_ground_truth_items(visitorid: int, eval_set: str = "test") -> Optional[List[int]]:
    """Get ground truth items (items user actually interacted with) from evaluation set.
    
    Args:
        visitorid: Raw visitor ID from the original dataset
        eval_set: Which evaluation set to use - 'test' or 'validation' (default: 'test')
    
    Returns:
        List of item IDs user interacted with in evaluation period, or None if visitor not found
    """
    # Map eval_set to actual matrix key names
    # Warm splits use: 'test_warm' or 'validation_warm'
    # Full splits use: 'test_full' or 'validation_full'
    matrix_key_warm = f"{eval_set}_warm"
    matrix_key_full = f"{eval_set}_full"
    
    # Collect ground truth items from both warm and full splits
    # This ensures complete ground truth for users who appear in both splits
    ground_truth_items = []
    
    # Try warm splits first (ALS/BPR)
    try:
        matrices = joblib.load(MATRICES_WARM_PATH)
        encoders = joblib.load(ENCODERS_WARM_PATH)
        visitor_encoder = encoders["visitorid"]
        item_encoder = encoders["itemid"]
        
        if visitorid in visitor_encoder.classes_:
            user_index = visitor_encoder.transform([visitorid])[0]
            # Use correct key name: test_warm or validation_warm
            if matrix_key_warm in matrices:
                eval_matrix = matrices[matrix_key_warm].tocsr()
                
                # Get items user interacted with in evaluation period
                user_eval_interactions = eval_matrix[user_index]
                actual_item_indices = user_eval_interactions.nonzero()[1]
                
                if len(actual_item_indices) > 0:
                    # Decode item indices back to original item IDs
                    warm_itemids = item_encoder.inverse_transform(actual_item_indices)
                    ground_truth_items.extend(warm_itemids.tolist())
    except Exception:
        pass
    
    # Try full splits (LightFM) - combine with warm items
    try:
        matrices = joblib.load(MATRICES_FULL_PATH)
        encoders = joblib.load(ENCODERS_FULL_PATH)
        visitor_encoder = encoders["visitorid"]
        item_encoder = encoders["itemid"]
        
        if visitorid in visitor_encoder.classes_:
            user_index = visitor_encoder.transform([visitorid])[0]
            # Use correct key name: test_full or validation_full
            if matrix_key_full in matrices:
                eval_matrix = matrices[matrix_key_full].tocsr()
                
                # Get items user interacted with in evaluation period
                user_eval_interactions = eval_matrix[user_index]
                actual_item_indices = user_eval_interactions.nonzero()[1]
                
                if len(actual_item_indices) > 0:
                    # Decode item indices back to original item IDs
                    full_itemids = item_encoder.inverse_transform(actual_item_indices)
                    # Add items not already in ground_truth_items (avoid duplicates)
                    for item_id in full_itemids.tolist():
                        if item_id not in ground_truth_items:
                            ground_truth_items.append(item_id)
    except Exception:
        pass
    
    # Return combined ground truth items if any were found
    return ground_truth_items if ground_truth_items else None


def borda_count_combine(recommendations: Dict[str, List[int]]) -> List[tuple]:
    """Combine recommendations from multiple algorithms using Borda Count method.
    
    Borda Count assigns points based on rank position:
    - Rank 1 gets highest points (equal to top_n), rank 2 gets (top_n-1), etc.
    - Items appearing in multiple lists get points from each list
    - Final ranking is sorted by total points (descending)
    
    Args:
        recommendations: Dictionary with algorithm names as keys and lists of item IDs as values.
                         Example: {'als': [1, 2, 3], 'bpr': [2, 4, 5]}
    
    Returns:
        List of tuples (item_id, total_points) sorted by total points descending.
    """
    # Dictionary to store total points for each item
    item_scores = defaultdict(int)
    
    # Get the maximum list length to determine point values
    max_length = max(len(recs) for recs in recommendations.values()) if recommendations else 0
    
    # Assign points to each item based on its rank in each algorithm's list
    for algorithm_name, item_list in recommendations.items():
        # For each algorithm, assign points: rank 1 gets max_length points, rank 2 gets (max_length-1), etc.
        for rank, item_id in enumerate(item_list, start=1):
            # Rank 1 gets max_length points, rank 2 gets (max_length-1), etc.
            points = max_length - rank + 1
            item_scores[item_id] += points
    
    # Sort items by total points (descending), then by item_id for tie-breaking (ascending)
    # Use negative score to sort descending, positive item_id to sort ascending
    sorted_items = sorted(item_scores.items(), key=lambda x: (-x[1], x[0]))
    
    return sorted_items


def combine_recommendations(visitorid: int, top_n: int = 10) -> Dict[str, List[int]]:
    """Get recommendations from all three algorithms (ALS, BPR, LightFM).
    
    Args:
        visitorid: Raw visitor ID from the original dataset.
        top_n: Number of recommendations to return from each algorithm (default: 10).
    
    Returns:
        Dictionary with keys 'als', 'bpr', 'lightfm', each containing a list
        of recommended item IDs. Empty list if visitor not found.
    """
    # Get recommendations from each algorithm
    als_recs = recommend(visitorid, model_type="als", top_n=top_n)
    bpr_recs = recommend(visitorid, model_type="bpr", top_n=top_n)
    lightfm_recs = recommend(visitorid, model_type="lightfm", top_n=top_n)
    
    return {
        "als": als_recs,
        "bpr": bpr_recs,
        "lightfm": lightfm_recs
    }


def display_recommendations(visitorid: int, recommendations: Dict[str, List[int]], top_n: int = 10) -> None:
    """Display recommendations from all three algorithms in a formatted table.
    
    Args:
        visitorid: The visitor ID being recommended for.
        recommendations: Dictionary with 'als', 'bpr', 'lightfm' keys containing lists of item IDs.
        top_n: Number of recommendations shown (for display purposes).
    """
    print(f"\n{'='*80}")
    print(f"Recommendations for Visitor ID: {visitorid}")
    print(f"{'='*80}\n")
    
    # Get the maximum length to handle different number of recommendations
    max_len = max(len(recommendations["als"]), len(recommendations["bpr"]), len(recommendations["lightfm"]))
    
    # Print header
    print(f"{'Rank':<6} {'ALS':<20} {'BPR':<20} {'LightFM':<20}")
    print("-" * 80)
    
    # Print recommendations row by row
    for i in range(max_len):
        rank = i + 1
        
        # Get item ID from each algorithm (or empty string if not available)
        als_item = recommendations["als"][i] if i < len(recommendations["als"]) else "-"
        bpr_item = recommendations["bpr"][i] if i < len(recommendations["bpr"]) else "-"
        lightfm_item = recommendations["lightfm"][i] if i < len(recommendations["lightfm"]) else "-"
        
        print(f"{rank:<6} {str(als_item):<20} {str(bpr_item):<20} {str(lightfm_item):<20}")
    
    # Print summary
    print("\n" + "-" * 80)
    print(f"Summary:")
    print(f"  ALS:      {len(recommendations['als'])} recommendations")
    print(f"  BPR:      {len(recommendations['bpr'])} recommendations")
    print(f"  LightFM:  {len(recommendations['lightfm'])} recommendations")
    print(f"{'='*80}\n")


def display_combined_recommendations(
    visitorid: int, 
    recommendations: Dict[str, List[int]], 
    top_n: int = 10,
    ground_truth: Optional[List[int]] = None
) -> None:
    """Display combined recommendations using Borda Count method.
    
    Args:
        visitorid: The visitor ID being recommended for.
        recommendations: Dictionary with 'als', 'bpr', 'lightfm' keys containing lists of item IDs.
        top_n: Number of top combined recommendations to display.
        ground_truth: Optional list of item IDs user actually interacted with (for precision calculation).
    """
    # Combine recommendations using Borda Count
    combined_ranking = borda_count_combine(recommendations)
    
    # Extract just the item IDs from combined ranking
    combined_item_ids = [item_id for item_id, _ in combined_ranking[:top_n]]
    
    print(f"\n{'='*80}")
    print(f"Combined Recommendations (Borda Count) for Visitor ID: {visitorid}")
    print(f"{'='*80}\n")
    
    # Display top-N combined recommendations
    print(f"{'Rank':<6} {'Item ID':<15} {'Total Points':<15} {'Appears In':<30} {'Hit':<6}")
    print("-" * 80)
    
    # Track which algorithms recommended each item
    algorithm_sets = {}
    for alg_name, item_list in recommendations.items():
        for item_id in item_list:
            if item_id not in algorithm_sets:
                algorithm_sets[item_id] = []
            algorithm_sets[item_id].append(alg_name.upper())
    
    # Track hits if ground truth is available
    ground_truth_set = set(ground_truth) if ground_truth else set()
    
    # Display top-N items
    for rank, (item_id, total_points) in enumerate(combined_ranking[:top_n], start=1):
        appears_in = ", ".join(algorithm_sets.get(item_id, []))
        hit_marker = "YES" if ground_truth and item_id in ground_truth_set else "-"
        print(f"{rank:<6} {item_id:<15} {total_points:<15} {appears_in:<30} {hit_marker:<6}")
    
    # Calculate and display precision@k if ground truth is available
    if ground_truth:
        precision = calculate_precision_at_k(combined_item_ids, ground_truth, k=top_n)
        hits_count = len(set(combined_item_ids).intersection(ground_truth_set))
        print(f"\n{'='*80}")
        print(f"Precision@{top_n}: {precision:.4f} ({hits_count} hits out of {top_n} recommendations)")
        print(f"Ground truth items: {len(ground_truth)} items")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}\n")


def main() -> None:
    """Get and display combined recommendations from all three algorithms."""
    
    parser = argparse.ArgumentParser(
        description="Get recommendations from ALS, BPR, and LightFM models for comparison."
    )
    parser.add_argument(
        "--visitorid",
        type=int,
        required=True,
        help="Visitor ID to get recommendations for"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return from each algorithm (default: 10)"
    )
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Also show individual algorithm recommendations side by side"
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        choices=["test", "validation"],
        default="test",
        help="Evaluation set to use for precision calculation: 'test' or 'validation' (default: test)"
    )
    parser.add_argument(
        "--calculate-precision",
        action="store_true",
        help="Calculate and display Precision@K by comparing recommendations to ground truth"
    )
    
    args = parser.parse_args()
    
    # Get recommendations from all three algorithms
    recommendations = combine_recommendations(args.visitorid, top_n=args.top_n)
    
    # Check if visitor was found
    if not any(recommendations.values()):
        print(f"Warning: Visitor {args.visitorid} not found in training data for any algorithm.")
        return
    
    # Get ground truth items if precision calculation is requested
    ground_truth = None
    if args.calculate_precision:
        ground_truth = get_ground_truth_items(args.visitorid, eval_set=args.eval_set)
        if ground_truth is None:
            print(f"Warning: Could not find ground truth data for visitor {args.visitorid} in {args.eval_set} set.")
            print("Precision calculation will be skipped.")
        else:
            print(f"Found {len(ground_truth)} ground truth items in {args.eval_set} set.")
    
    # Display individual recommendations if requested
    if args.show_individual:
        display_recommendations(args.visitorid, recommendations, top_n=args.top_n)
    
    # Display combined recommendations using Borda Count
    display_combined_recommendations(
        args.visitorid, 
        recommendations, 
        top_n=args.top_n,
        ground_truth=ground_truth
    )


if __name__ == "__main__":
    main()


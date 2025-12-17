"""Generate recommendations for visitors using trained ALS, BPR, LightFM, or SVD models.

This script loads the trained model and encoders, then provides functions
to get top-N item recommendations for single visitor or a list of visitors.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

# Paths to saved artifacts.
MODELS_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models"
)
# Paths for ALS/BPR (warm splits)
ENCODERS_WARM_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_final_warm.pkl"
)
MATRICES_WARM_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_warm.pkl"
)
# Paths for LightFM (full splits)
ENCODERS_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders_final_full.pkl"
)
MATRICES_FULL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices_final_full.pkl"
)
# Paths for SVD (rating CSV files)
RATINGS_TRAIN_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/ratings_train.csv"
)


def recommend(visitorid: int, model_type: str = "als", top_n: int = 10) -> List[int]:
    """Get top-N item recommendations for a given visitor using ALS, BPR, LightFM, or SVD model.

    Args:
        visitorid: Raw visitor ID from the original dataset.
        model_type: Model type to use - 'als', 'bpr', 'lightfm', or 'svd' (default: 'als').
        top_n: Number of recommendations to return (default: 10).

    Returns:
        List of recommended item IDs (original item IDs, not encoded indices).
        Returns empty list if visitorid is not found in training data (for ALS/BPR/SVD).
        For LightFM, can handle cold-start users.
    """
    if model_type == "svd":
        # SVD: load model artifact (contains model and trainset)
        model_path = MODELS_DIR / "svd_model.pkl"
        model_artifact = joblib.load(model_path)
        model = model_artifact["model"]
        trainset = model_artifact["trainset"]
        
        # Load rating CSV to get user-item mapping
        ratings_df = pd.read_csv(RATINGS_TRAIN_PATH, header=None, names=['user_id', 'item_id', 'rating'])
        
        # Check if visitorid exists in training data
        if visitorid not in ratings_df['user_id'].values:
            return []
        
        # Get all unique items from training data
        all_items = ratings_df['item_id'].unique()
        
        # Get items user has already rated
        user_rated_items = set(ratings_df[ratings_df['user_id'] == visitorid]['item_id'].values)
        
        # Get predictions for all items user hasn't rated
        item_scores = []
        for item_id in all_items:
            if item_id not in user_rated_items:
                try:
                    # Get internal user and item IDs from trainset
                    inner_user_id = trainset.to_inner_uid(str(visitorid))
                    inner_item_id = trainset.to_inner_iid(str(item_id))
                    prediction = model.predict(str(visitorid), str(item_id))
                    item_scores.append((item_id, prediction.est))
                except (ValueError, KeyError):
                    # User or item not in trainset, skip
                    continue
        
        # Sort by predicted rating (descending) and get top-N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_itemids = [item_id for item_id, _ in item_scores[:top_n]]
        
        return recommended_itemids
    elif model_type == "lightfm":
        # LightFM: load model artifact (contains model, dataset, encoders)
        model_path = MODELS_DIR / "lightfm_model.pkl"
        model_artifact = joblib.load(model_path)
        model = model_artifact["model"]
        dataset = model_artifact["dataset"]
        encoders = model_artifact["encoders"]
        
        # Load full matrices for LightFM
        matrices = joblib.load(MATRICES_FULL_PATH)
        train_matrix = matrices["train"].tocsr()
        
        visitor_encoder = encoders["visitorid"]
        item_encoder = encoders["itemid"]
        
        # Check if visitorid exists in encoders
        # LightFM can handle cold-start, but we need the visitor to be in the encoder
        # (which includes all users from train+test periods)
        if visitorid not in visitor_encoder.classes_:
            return []
        
        # Encode visitorid to get the user index
        user_index = visitor_encoder.transform([visitorid])[0]
        
        # Get all item indices
        n_items = train_matrix.shape[1]
        all_item_indices = np.arange(n_items)
        
        # Get user's training interactions (items they've already seen)
        user_train_interactions = train_matrix[user_index]
        seen_item_indices = set(user_train_interactions.nonzero()[1])
        
        # Get predictions for all items
        user_ids = np.full(n_items, user_index, dtype=np.int32)
        predictions = model.predict(user_ids, all_item_indices)
        
        # Filter out seen items and sort by score (descending)
        item_scores = list(zip(all_item_indices, predictions))
        unseen_item_scores = [
            (item_idx, score) 
            for item_idx, score in item_scores 
            if item_idx not in seen_item_indices
        ]
        unseen_item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-N item indices
        recommended_item_indices = np.array([item_idx for item_idx, _ in unseen_item_scores[:top_n]])
        
        if len(recommended_item_indices) == 0:
            return []
        
        # Decode item indices back to original item IDs
        recommended_itemids = item_encoder.inverse_transform(recommended_item_indices)
        
        return recommended_itemids.tolist()
    else:
        # ALS/BPR: use warm splits
        if model_type == "als":
            model_path = MODELS_DIR / "als_model.pkl"
        else:  # bpr
            model_path = MODELS_DIR / "bpr_model.pkl"
        
        model = joblib.load(model_path)
        encoders = joblib.load(ENCODERS_WARM_PATH)
        matrices = joblib.load(MATRICES_WARM_PATH)
        train_matrix = matrices["train"].tocsr().astype("float32")

        visitor_encoder = encoders["visitorid"]
        item_encoder = encoders["itemid"]

        # Check if visitorid exists in the training data.
        if visitorid not in visitor_encoder.classes_:
            return []

        # Encode visitorid to get the user index used by the model.
        user_index = visitor_encoder.transform([visitorid])[0]

        # Get the user's training profile (items they've already interacted with).
        user_training_profile = train_matrix[user_index]

        # Get top-N recommendations from the model.
        # filter_already_liked_items=True excludes items the user already saw.
        recommended_item_indices, scores = model.recommend(
            user_index,
            user_training_profile,
            N=top_n,
            filter_already_liked_items=True,
        )

        # Filter to only items that are in the encoder (handle out-of-range indices)
        max_item_index = len(item_encoder.classes_) - 1
        valid_indices = recommended_item_indices[recommended_item_indices <= max_item_index]
        
        if len(valid_indices) == 0:
            return []
        
        # Decode item indices back to original item IDs.
        recommended_itemids = item_encoder.inverse_transform(valid_indices)

        return recommended_itemids.tolist()


def recommend_batch(visitorids: List[int], model_type: str = "als", top_n: int = 10) -> Dict[int, List[int]]:
    """Get top-N item recommendations for a list of visitors.

    Args:
        visitorids: List of raw visitor IDs from the original dataset.
        model_type: Model type to use - 'als', 'bpr', 'lightfm', or 'svd' (default: 'als').
        top_n: Number of recommendations to return per user (default: 10).

    Returns:
        Dictionary mapping visitor ID to list of recommended item IDs.
        Visitors not found in training data will have empty lists.
    """
    results = {}
    
    if model_type == "svd":
        # SVD: load model artifact
        model_path = MODELS_DIR / "svd_model.pkl"
        model_artifact = joblib.load(model_path)
        model = model_artifact["model"]
        trainset = model_artifact["trainset"]
        
        # Load rating CSV to get user-item mapping
        ratings_df = pd.read_csv(RATINGS_TRAIN_PATH, header=None, names=['user_id', 'item_id', 'rating'])
        all_items = ratings_df['item_id'].unique()
        
        for visitorid in visitorids:
            if visitorid not in ratings_df['user_id'].values:
                results[visitorid] = []
                continue
            
            try:
                # Get items user has already rated
                user_rated_items = set(ratings_df[ratings_df['user_id'] == visitorid]['item_id'].values)
                
                # Get predictions for all items user hasn't rated
                item_scores = []
                for item_id in all_items:
                    if item_id not in user_rated_items:
                        try:
                            prediction = model.predict(str(visitorid), str(item_id))
                            item_scores.append((item_id, prediction.est))
                        except (ValueError, KeyError):
                            continue
                
                # Sort by predicted rating and get top-N
                item_scores.sort(key=lambda x: x[1], reverse=True)
                recommended_itemids = [item_id for item_id, _ in item_scores[:top_n]]
                results[visitorid] = recommended_itemids
            except Exception:
                results[visitorid] = []
    elif model_type == "lightfm":
        # LightFM: load model artifact
        model_path = MODELS_DIR / "lightfm_model.pkl"
        model_artifact = joblib.load(model_path)
        model = model_artifact["model"]
        dataset = model_artifact["dataset"]
        encoders = model_artifact["encoders"]
        
        matrices = joblib.load(MATRICES_FULL_PATH)
        train_matrix = matrices["train"].tocsr()
        
        visitor_encoder = encoders["visitorid"]
        item_encoder = encoders["itemid"]
        n_items = train_matrix.shape[1]
        all_item_indices = np.arange(n_items)
        
        for visitorid in visitorids:
            if visitorid not in visitor_encoder.classes_:
                results[visitorid] = []
                continue
            
            try:
                user_index = visitor_encoder.transform([visitorid])[0]
                user_train_interactions = train_matrix[user_index]
                seen_item_indices = set(user_train_interactions.nonzero()[1])
                
                # Get predictions for all items
                user_ids = np.full(n_items, user_index, dtype=np.int32)
                predictions = model.predict(user_ids, all_item_indices)
                
                # Filter out seen items and sort by score
                item_scores = list(zip(all_item_indices, predictions))
                unseen_item_scores = [
                    (item_idx, score) 
                    for item_idx, score in item_scores 
                    if item_idx not in seen_item_indices
                ]
                unseen_item_scores.sort(key=lambda x: x[1], reverse=True)
                
                recommended_item_indices = np.array([item_idx for item_idx, _ in unseen_item_scores[:top_n]])
                
                if len(recommended_item_indices) == 0:
                    results[visitorid] = []
                else:
                    recommended_itemids = item_encoder.inverse_transform(recommended_item_indices)
                    results[visitorid] = recommended_itemids.tolist()
            except Exception:
                results[visitorid] = []
    else:
        # ALS/BPR: use warm splits
        if model_type == "als":
            model_path = MODELS_DIR / "als_model.pkl"
        else:  # bpr
            model_path = MODELS_DIR / "bpr_model.pkl"
        
        model = joblib.load(model_path)
        encoders = joblib.load(ENCODERS_WARM_PATH)
        matrices = joblib.load(MATRICES_WARM_PATH)
        train_matrix = matrices["train"].tocsr().astype("float32")

        visitor_encoder = encoders["visitorid"]
        item_encoder = encoders["itemid"]
        
        for visitorid in visitorids:
            # Check if visitorid exists in the training data.
            if visitorid not in visitor_encoder.classes_:
                results[visitorid] = []
                continue

            # Encode visitorid to get the user index used by the model.
            user_index = visitor_encoder.transform([visitorid])[0]

            # Get the user's training profile (items they've already interacted with).
            user_training_profile = train_matrix[user_index]

            # Get top-N recommendations from the model.
            # filter_already_liked_items=True excludes items the user already saw.
            try:
                recommended_item_indices, scores = model.recommend(
                    user_index,
                    user_training_profile,
                    N=top_n,
                    filter_already_liked_items=True,
                )

                # Filter to only items that are in the encoder (handle out-of-range indices)
                max_item_index = len(item_encoder.classes_) - 1
                valid_indices = recommended_item_indices[recommended_item_indices <= max_item_index]
                
                if len(valid_indices) == 0:
                    results[visitorid] = []
                else:
                    # Decode item indices back to original item IDs.
                    recommended_itemids = item_encoder.inverse_transform(valid_indices)
                    results[visitorid] = recommended_itemids.tolist()
            except Exception:
                # If recommendation fails for any reason, return empty list
                results[visitorid] = []
    
    return results


def main() -> None:
    """Get recommendations for visitor ID(s) provided via command line."""

    parser = argparse.ArgumentParser(
        description="Get item recommendations for visitor(s) using trained ALS, BPR, LightFM, or SVD model."
    )
    parser.add_argument(
        "--visitorids",
        type=str,
        help="Comma-separated list of visitor IDs (e.g., '123,456,789') or path to file with one ID per line",
    )
    parser.add_argument(
        "--visitorid",
        type=int,
        help="Single visitor ID to get recommendations for (alternative to --visitorids)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["als", "bpr", "lightfm", "svd"],
        default="als",
        help="Model type to use: 'als', 'bpr', 'lightfm', or 'svd' (default: als)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return per user (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Path to save results as CSV file",
    )

    args = parser.parse_args()

    # Determine which visitor IDs to process
    visitorids = []
    
    if args.visitorids:
        # Check if it's a file path or comma-separated list
        visitorids_path = Path(args.visitorids)
        if visitorids_path.exists():
            # Read from file (one ID per line)
            with open(visitorids_path, "r") as f:
                visitorids = [int(line.strip()) for line in f if line.strip()]
        else:
            # Parse comma-separated list
            visitorids = [int(x.strip()) for x in args.visitorids.split(",")]
    elif args.visitorid:
        visitorids = [args.visitorid]
    else:
        parser.error("Either --visitorid or --visitorids must be provided")

    # Get recommendations
    if len(visitorids) == 1:
        # Single user - use simple function
        recommendations = recommend(visitorids[0], model_type=args.model, top_n=args.top_n)
        
        if recommendations:
            print(f"\n{args.model.upper()} Model - Top {args.top_n} recommendations for visitor {visitorids[0]}:")
            for i, itemid in enumerate(recommendations, 1):
                print(f"  {i}. Item ID: {itemid}")
        else:
            print(f"Visitor {visitorids[0]} not found in training data.")
    else:
        # Multiple users - use batch function
        results = recommend_batch(visitorids, model_type=args.model, top_n=args.top_n)
        
        print(f"\n{args.model.upper()} Model - Recommendations for {len(visitorids)} visitors:")
        print("-" * 60)
        
        for visitorid, recs in results.items():
            if recs:
                print(f"\nVisitor {visitorid}:")
                for i, itemid in enumerate(recs, 1):
                    print(f"  {i}. Item ID: {itemid}")
            else:
                print(f"\nVisitor {visitorid}: Not found in training data or no recommendations available")
        
        # Save to CSV if output path provided
        if args.output:
            output_data = []
            for visitorid, recs in results.items():
                for i, itemid in enumerate(recs, 1):
                    output_data.append({
                        "visitorid": visitorid,
                        "rank": i,
                        "itemid": itemid
                    })
            
            df = pd.DataFrame(output_data)
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()


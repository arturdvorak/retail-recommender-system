"""Generate recommendations for visitors using the trained ALS model.

This script loads the trained model and encoders, then provides a function
to get top-N item recommendations for any visitor ID.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

# Paths to saved artifacts.
MODEL_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/models/als_model.pkl"
)
ENCODERS_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/encoders.pkl"
)
MATRICES_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/interaction_matrices.pkl"
)


def recommend(visitorid: int, top_n: int = 10) -> list[int]:
    """Get top-N item recommendations for a given visitor.

    Args:
        visitorid: Raw visitor ID from the original dataset.
        top_n: Number of recommendations to return (default: 10).

    Returns:
        List of recommended item IDs (original item IDs, not encoded indices).
        Returns empty list if visitorid is not found in training data.
    """
    # Load artifacts (model, encoders, training matrix).
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    matrices = joblib.load(MATRICES_PATH)
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

    # Decode item indices back to original item IDs.
    recommended_itemids = item_encoder.inverse_transform(recommended_item_indices)

    return recommended_itemids.tolist()


def main() -> None:
    """Get recommendations for a visitor ID provided via command line."""

    parser = argparse.ArgumentParser(
        description="Get item recommendations for a visitor using the trained ALS model."
    )
    parser.add_argument(
        "visitorid",
        type=int,
        help="Visitor ID to get recommendations for",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10)",
    )

    args = parser.parse_args()

    recommendations = recommend(args.visitorid, top_n=args.top_n)

    if recommendations:
        print(f"Top {args.top_n} recommendations for visitor {args.visitorid}:")
        for i, itemid in enumerate(recommendations, 1):
            print(f"  {i}. Item ID: {itemid}")
    else:
        print(f"Visitor {args.visitorid} not found in training data.")


if __name__ == "__main__":
    main()


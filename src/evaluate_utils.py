"""Generic evaluation utilities for recommendation models.

This module provides reusable functions to evaluate any recommendation model
(ALS, BPR, etc.) with support for Precision@K, Recall@K, and recalculate_user option.
"""

import numpy as np
from scipy import sparse
from tqdm import tqdm


def evaluate_precision_and_recall_at_k(
    model,
    train_matrix,
    eval_matrix,
    k=10,
    recalculate_user=False,
    combine_train_eval=False,
    show_progress=True,
):
    """Evaluate Precision@K and Recall@K for a recommendation model.
    
    This function works with any model from the implicit library (ALS, BPR, etc.)
    that implements the recommend() method.
    
    Precision@K: Of the K recommendations, how many are relevant?
    Formula: (relevant items in top K) / K
    
    Recall@K: Of all relevant items, how many did we find in top K?
    Formula: (relevant items in top K) / (total relevant items)
    
    Args:
        model: Trained recommendation model (ALS, BPR, etc.) with recommend() method.
        train_matrix: Sparse matrix of user-item interactions from training period.
                     Shape: (num_users, num_items). Used to get user's historical interactions.
        eval_matrix: Sparse matrix of user-item interactions from evaluation period.
                    Shape: (num_users, num_items). Contains ground truth (what users actually did).
        k: Number of top recommendations to evaluate. Defaults to 10.
        recalculate_user: If True, recalculates user embedding on-the-fly using combined
                         train + eval interactions (simulating users with new interactions).
                         If False, uses stored embeddings from training only.
                         Defaults to False.
        combine_train_eval: DEPRECATED. This option causes data leakage by using evaluation
                           data as input to the model. Evaluation data should only be used as
                           ground truth, never as input. This parameter is kept for backward
                           compatibility but will raise an error if set to True.
        show_progress: If True, shows progress bar during evaluation. Defaults to True.
    
    Returns:
        dict: Dictionary with two keys:
            - 'precision_at_k': Average Precision@K across all users (float)
            - 'recall_at_k': Average Recall@K across all users (float)
    
    Raises:
        ValueError: If combine_train_eval=True is set, as this causes data leakage.
    """
    
    # Validate inputs to prevent data leakage
    if combine_train_eval:
        raise ValueError(
            "combine_train_eval=True causes data leakage by using evaluation data as input. "
            "Evaluation data should only be used as ground truth, never as input to the model. "
            "This violates train/test separation and artificially inflates metrics. "
            "Set combine_train_eval=False for proper evaluation."
        )
    
    # Convert matrices to CSR format (efficient row access)
    train_matrix = train_matrix.tocsr()
    eval_matrix = eval_matrix.tocsr()
    
    # Lists to store Precision@K and Recall@K for each user
    precisions = []
    recalls = []
    
    # Get list of users who have interactions in evaluation set (ground truth)
    # These are the users we can evaluate
    eval_user_indices = np.unique(eval_matrix.nonzero()[0])
    
    # Progress bar for tracking evaluation progress
    user_iterator = tqdm(eval_user_indices, desc="Evaluating users") if show_progress else eval_user_indices
    
    # Evaluate each user
    for user_index in user_iterator:
        # Step 1: Get user's training interactions (what they did during training)
        user_train_interactions = train_matrix[user_index]
        
        # Step 2: Get user's evaluation interactions (ground truth - what they actually did)
        user_eval_interactions = eval_matrix[user_index]
        
        # Step 3: Determine which interactions to use for recommendations
        # When recalculate_user=True: Combine train + eval interactions to simulate
        # users with new interactions (this tests if recalculating helps with updated profiles).
        # When recalculate_user=False: Use only training interactions (stored embeddings).
        # Note: Eval interactions are still used as ground truth for evaluation.
        if recalculate_user:
            # Combine train + eval interactions for recalculation
            # This simulates the scenario where users have new interactions after training
            # and we want to see if recalculating embeddings improves recommendations
            # Both matrices are already CSR format, so we can add them directly
            # Note: If same item appears in both, values will be summed
            user_interactions_for_rec = user_train_interactions + user_eval_interactions
        else:
            # Use only training interactions (default behavior - uses stored embedding)
            user_interactions_for_rec = user_train_interactions
        
        # Step 4: Generate recommendations for this user
        # model.recommend() returns: (item_indices, scores)
        try:
            # Generate recommendations using only training interactions
            # Evaluation interactions are never used as input (they're ground truth only)
            recommended_item_indices, _ = model.recommend(
                user_index,
                user_interactions_for_rec,
                N=k,
                filter_already_liked_items=True,
                recalculate_user=recalculate_user,
            )
        except Exception as e:
            # If recommendation fails (e.g., user not in model, or recalculate fails), skip this user
            # Note: This can happen with recalculate_user=True if the combined interactions cause issues
            continue
        
        # Step 5: Get ground truth - which items did user actually interact with?
        # nonzero()[1] gets the column indices (item indices) where user interacted
        actual_item_indices = user_eval_interactions.nonzero()[1]
        
        # Skip users with no ground truth interactions (can't evaluate)
        if len(actual_item_indices) == 0:
            continue
        
        # Step 6: Calculate Precision@K
        # Precision@K = (relevant items in top K) / K
        # Find intersection: which recommended items are in actual items?
        recommended_set = set(recommended_item_indices)
        actual_set = set(actual_item_indices)
        hits = len(recommended_set.intersection(actual_set))
        
        precision = hits / k if k > 0 else 0.0
        precisions.append(precision)
        
        # Step 7: Calculate Recall@K
        # Recall@K = (relevant items in top K) / (total relevant items)
        recall = hits / len(actual_set) if len(actual_set) > 0 else 0.0
        recalls.append(recall)
    
    # Step 8: Calculate average Precision@K and Recall@K across all users
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        'precision_at_k': mean_precision,
        'recall_at_k': mean_recall,
    }


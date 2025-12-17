"""Evaluation utilities for recommendation models (ALS, BPR, LightFM, SVD, etc.)."""

import numpy as np
from pathlib import Path
from scipy import sparse
from tqdm import tqdm


def evaluate_precision_and_recall_at_k(
    model,
    train_matrix,
    eval_matrix,
    k=10,
    recalculate_user=False,
    show_progress=True,
):
    """Evaluate Precision@K and Recall@K for a recommendation model.
    
    Precision@K = (relevant items in top K) / K
    Recall@K = (relevant items in top K) / (total relevant items)
    
    Args:
        model: Trained model with recommend() method (ALS, BPR, etc.)
        train_matrix: Training user-item interactions (num_users, num_items)
        eval_matrix: Evaluation user-item interactions (ground truth)
        k: Number of top recommendations to evaluate
        recalculate_user: If True, recalculate user embedding using train+eval interactions
        show_progress: Show progress bar if True
    
    Returns:
        dict: {'precision_at_k': float, 'recall_at_k': float}
    """
    
    # Convert to CSR format for efficient row access
    train_matrix = train_matrix.tocsr()
    eval_matrix = eval_matrix.tocsr()
    
    precisions = []
    recalls = []
    
    # Get users with evaluation interactions (ground truth)
    eval_user_indices = np.unique(eval_matrix.nonzero()[0])
    
    user_iterator = tqdm(eval_user_indices, desc="Evaluating users") if show_progress else eval_user_indices
    
    for user_index in user_iterator:
        user_train_interactions = train_matrix[user_index]
        user_eval_interactions = eval_matrix[user_index]
        
        # Choose interactions for recommendations
        if recalculate_user:
            # Combine train+eval to simulate new interactions
            user_interactions_for_rec = user_train_interactions + user_eval_interactions
        else:
            # Use only training interactions (stored embeddings)
            user_interactions_for_rec = user_train_interactions
        
        # Generate recommendations
        try:
            recommended_item_indices, _ = model.recommend(
                user_index,
                user_interactions_for_rec,
                N=k,
                filter_already_liked_items=True,
                recalculate_user=recalculate_user,
            )
        except Exception:
            # Skip if recommendation fails
            continue
        
        # Get ground truth items
        actual_item_indices = user_eval_interactions.nonzero()[1]
        
        if len(actual_item_indices) == 0:
            continue
        
        # Calculate Precision@K and Recall@K
        recommended_set = set(recommended_item_indices)
        actual_set = set(actual_item_indices)
        hits = len(recommended_set.intersection(actual_set))
        
        precision = hits / k if k > 0 else 0.0
        precisions.append(precision)
        
        recall = hits / len(actual_set) if len(actual_set) > 0 else 0.0
        recalls.append(recall)
    
    # Calculate averages across all users
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        'precision_at_k': mean_precision,
        'recall_at_k': mean_recall,
    }


def evaluate_lightfm_precision_and_recall_at_k(
    model,
    dataset,
    train_matrix,
    eval_matrix,
    k=10,
    show_progress=True,
    batch_size=1000,
    max_candidate_items=10000,
):
    """Evaluate Precision@K and Recall@K for LightFM model (OPTIMIZED VERSION).
    
    Optimizations:
    1. Item sampling: Only predict items in validation set + top popular items
    2. Batch predictions: Process multiple users at once
    
    LightFM uses predict() method instead of recommend(), so we need to:
    1. Get predictions for candidate items for batches of users
    2. Sort by score and take top-K
    3. Compare with ground truth
    
    Precision@K = (relevant items in top K) / K
    Recall@K = (relevant items in top K) / (total relevant items)
    
    Args:
        model: Trained LightFM model with predict() method
        dataset: LightFM Dataset object (needed for predictions)
        train_matrix: Training user-item interactions (num_users, num_items) - sparse matrix
        eval_matrix: Evaluation user-item interactions (ground truth) - sparse matrix
        k: Number of top recommendations to evaluate
        show_progress: Show progress bar if True
        batch_size: Number of users to process in each batch (default: 1000)
        max_candidate_items: Maximum number of candidate items to consider (default: 10000)
    
    Returns:
        dict: {'precision_at_k': float, 'recall_at_k': float}
    """
    
    # Convert to CSR format for efficient row access
    train_matrix = train_matrix.tocsr()
    eval_matrix = eval_matrix.tocsr()
    
    precisions = []
    recalls = []
    
    # Get users with evaluation interactions (ground truth)
    eval_user_indices = np.unique(eval_matrix.nonzero()[0])
    n_users = len(eval_user_indices)
    
    # OPTIMIZATION 1: Item Sampling
    # Only predict items that appear in validation set + top popular items
    # This dramatically reduces the number of predictions needed
    
    # Get all items that appear in validation set (likely to be relevant)
    validation_items = np.unique(eval_matrix.nonzero()[1])
    
    # Get top popular items from training set (items with most interactions)
    # This helps catch popular items that might be recommended
    item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    top_popular_indices = np.argsort(item_popularity)[-max_candidate_items:][::-1]
    
    # Combine validation items and top popular items
    candidate_items = np.unique(np.concatenate([validation_items, top_popular_indices]))
    
    # Limit to max_candidate_items to control memory usage
    if len(candidate_items) > max_candidate_items:
        candidate_items = candidate_items[:max_candidate_items]
    
    n_candidate_items = len(candidate_items)
    
    if show_progress:
        print(f"Optimization: Predicting for {n_candidate_items:,} candidate items instead of {train_matrix.shape[1]:,} items")
        print(f"Reduction: {train_matrix.shape[1] / n_candidate_items:.1f}x fewer predictions")
    
    # OPTIMIZATION 2: Batch Processing
    # Process users in batches to reduce function call overhead
    
    user_iterator = range(0, n_users, batch_size)
    if show_progress:
        user_iterator = tqdm(user_iterator, desc="Evaluating user batches")
    
    for batch_start in user_iterator:
        batch_end = min(batch_start + batch_size, n_users)
        batch_user_indices = eval_user_indices[batch_start:batch_end]
        batch_size_actual = len(batch_user_indices)
        
        # Create matrices for batch prediction
        # Shape: (batch_size * n_candidate_items,)
        batch_user_ids = np.repeat(batch_user_indices, n_candidate_items).astype(np.int32)
        batch_item_ids = np.tile(candidate_items, batch_size_actual).astype(np.int32)
        
        # Batch prediction: predict for all user-item pairs at once
        batch_predictions = model.predict(batch_user_ids, batch_item_ids)
        
        # Reshape predictions: (batch_size, n_candidate_items)
        predictions_matrix = batch_predictions.reshape(batch_size_actual, n_candidate_items)
        
        # Process each user in the batch
        for i, user_index in enumerate(batch_user_indices):
            # Get user's training interactions (items they've already seen)
            user_train_interactions = train_matrix[user_index]
            user_eval_interactions = eval_matrix[user_index]
            
            # Get ground truth items (what user actually interacted with in eval period)
            actual_item_indices = user_eval_interactions.nonzero()[1]
            
            if len(actual_item_indices) == 0:
                continue
            
            # Get predictions for this user (from batch predictions)
            user_predictions = predictions_matrix[i]
            
            # Get items user has already seen (from training) to filter them out
            seen_item_indices = set(user_train_interactions.nonzero()[1])
            
            # Use numpy operations for filtering and sorting (much faster than Python lists)
            # Create boolean mask for unseen items
            unseen_mask = ~np.isin(candidate_items, list(seen_item_indices))
            
            # Filter predictions and items
            unseen_predictions = user_predictions[unseen_mask]
            unseen_items = candidate_items[unseen_mask]
            
            # Sort by prediction score (descending) and get top-K
            top_k_indices = np.argsort(unseen_predictions)[::-1][:k]
            recommended_item_indices = unseen_items[top_k_indices]
            
            # Calculate Precision@K and Recall@K
            recommended_set = set(recommended_item_indices)
            actual_set = set(actual_item_indices)
            hits = len(recommended_set.intersection(actual_set))
            
            precision = hits / k if k > 0 else 0.0
            precisions.append(precision)
            
            recall = hits / len(actual_set) if len(actual_set) > 0 else 0.0
            recalls.append(recall)
    
    # Calculate averages across all users
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        'precision_at_k': mean_precision,
        'recall_at_k': mean_recall,
    }


def evaluate_svd_precision_and_recall_at_k(
    model,
    trainset,
    eval_matrix,
    visitor_encoder,
    item_encoder,
    k=10,
    show_progress=True,
):
    """Evaluate Precision@K and Recall@K for SVD model.
    
    SVD uses Surprise's predict() method, so we need to:
    1. Get predictions for all items for each user
    2. Sort by predicted rating and take top-K
    3. Compare with ground truth
    
    Precision@K = (relevant items in top K) / K
    Recall@K = (relevant items in top K) / (total relevant items)
    
    Args:
        model: Trained SVD model from Surprise library
        trainset: Surprise Trainset object (needed for predictions)
        eval_matrix: Evaluation user-item interactions (ground truth) - sparse matrix
        visitor_encoder: LabelEncoder for visitor IDs
        item_encoder: LabelEncoder for item IDs
        k: Number of top recommendations to evaluate
        show_progress: Show progress bar if True
    
    Returns:
        dict: {'precision_at_k': float, 'recall_at_k': float}
    """
    import pandas as pd
    
    # Convert to CSR format for efficient row access
    eval_matrix = eval_matrix.tocsr()
    
    precisions = []
    recalls = []
    
    # Get users with evaluation interactions (ground truth)
    eval_user_indices = np.unique(eval_matrix.nonzero()[0])
    
    # Load rating CSV to get all items
    ratings_path = Path(
        "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/ratings_train.csv"
    )
    ratings_df = pd.read_csv(ratings_path, header=None, names=['user_id', 'item_id', 'rating'])
    all_items = ratings_df['item_id'].unique()
    
    user_iterator = tqdm(eval_user_indices, desc="Evaluating users") if show_progress else eval_user_indices
    
    for user_index in user_iterator:
        # Get visitor ID from encoder
        visitor_id = visitor_encoder.inverse_transform([user_index])[0]
        
        # Get user's evaluation interactions (ground truth)
        user_eval_interactions = eval_matrix[user_index]
        actual_item_indices = user_eval_interactions.nonzero()[1]
        
        if len(actual_item_indices) == 0:
            continue
        
        # Decode actual item indices to original item IDs
        actual_item_ids = set(item_encoder.inverse_transform(actual_item_indices))
        
        # Get items user has already rated (from training)
        user_rated_items = set(ratings_df[ratings_df['user_id'] == visitor_id]['item_id'].values)
        
        # Get predictions for all items user hasn't rated
        item_scores = []
        for item_id in all_items:
            if item_id not in user_rated_items:
                try:
                    prediction = model.predict(str(visitor_id), str(item_id))
                    # Get item index for comparison with ground truth
                    if item_id in item_encoder.classes_:
                        item_scores.append((item_id, prediction.est))
                except (ValueError, KeyError):
                    # User or item not in trainset, skip
                    continue
        
        # Sort by predicted rating (descending) and get top-K
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_item_ids = [item_id for item_id, _ in item_scores[:k]]
        
        if len(recommended_item_ids) == 0:
            continue
        
        # Calculate Precision@K and Recall@K
        recommended_set = set(recommended_item_ids)
        hits = len(recommended_set.intersection(actual_item_ids))
        
        precision = hits / k if k > 0 else 0.0
        precisions.append(precision)
        
        recall = hits / len(actual_item_ids) if len(actual_item_ids) > 0 else 0.0
        recalls.append(recall)
    
    # Calculate averages across all users
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        'precision_at_k': mean_precision,
        'recall_at_k': mean_recall,
    }


def evaluate_svd_precision_and_recall_at_k(
    model,
    trainset,
    eval_matrix,
    visitor_encoder,
    item_encoder,
    k=10,
    show_progress=True,
):
    """Evaluate Precision@K and Recall@K for SVD model.
    
    SVD uses Surprise's predict() method, so we need to:
    1. Get predictions for all items for each user
    2. Sort by predicted rating and take top-K
    3. Compare with ground truth
    
    Precision@K = (relevant items in top K) / K
    Recall@K = (relevant items in top K) / (total relevant items)
    
    Args:
        model: Trained SVD model from Surprise library
        trainset: Surprise Trainset object (needed for predictions)
        eval_matrix: Evaluation user-item interactions (ground truth) - sparse matrix
        visitor_encoder: LabelEncoder for visitor IDs
        item_encoder: LabelEncoder for item IDs
        k: Number of top recommendations to evaluate
        show_progress: Show progress bar if True
    
    Returns:
        dict: {'precision_at_k': float, 'recall_at_k': float}
    """
    import pandas as pd
    
    # Convert to CSR format for efficient row access
    eval_matrix = eval_matrix.tocsr()
    
    precisions = []
    recalls = []
    
    # Get users with evaluation interactions (ground truth)
    eval_user_indices = np.unique(eval_matrix.nonzero()[0])
    
    # Load rating CSV to get all items
    ratings_path = Path(
        "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed/ratings_train.csv"
    )
    ratings_df = pd.read_csv(ratings_path, header=None, names=['user_id', 'item_id', 'rating'])
    all_items = ratings_df['item_id'].unique()
    
    user_iterator = tqdm(eval_user_indices, desc="Evaluating users") if show_progress else eval_user_indices
    
    for user_index in user_iterator:
        # Get visitor ID from encoder
        visitor_id = visitor_encoder.inverse_transform([user_index])[0]
        
        # Get user's evaluation interactions (ground truth)
        user_eval_interactions = eval_matrix[user_index]
        actual_item_indices = user_eval_interactions.nonzero()[1]
        
        if len(actual_item_indices) == 0:
            continue
        
        # Decode actual item indices to original item IDs
        actual_item_ids = set(item_encoder.inverse_transform(actual_item_indices))
        
        # Get items user has already rated (from training)
        user_rated_items = set(ratings_df[ratings_df['user_id'] == visitor_id]['item_id'].values)
        
        # Get predictions for all items user hasn't rated
        item_scores = []
        for item_id in all_items:
            if item_id not in user_rated_items:
                try:
                    prediction = model.predict(str(visitor_id), str(item_id))
                    # Get item index for comparison with ground truth
                    if item_id in item_encoder.classes_:
                        item_scores.append((item_id, prediction.est))
                except (ValueError, KeyError):
                    # User or item not in trainset, skip
                    continue
        
        # Sort by predicted rating (descending) and get top-K
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_item_ids = [item_id for item_id, _ in item_scores[:k]]
        
        if len(recommended_item_ids) == 0:
            continue
        
        # Calculate Precision@K and Recall@K
        recommended_set = set(recommended_item_ids)
        hits = len(recommended_set.intersection(actual_item_ids))
        
        precision = hits / k if k > 0 else 0.0
        precisions.append(precision)
        
        recall = hits / len(actual_item_ids) if len(actual_item_ids) > 0 else 0.0
        recalls.append(recall)
    
    # Calculate averages across all users
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        'precision_at_k': mean_precision,
        'recall_at_k': mean_recall,
    }

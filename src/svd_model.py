"""SVD-based collaborative filtering model (Surprise-compatible implementation).

This module provides an SVD model implementation that mimics Surprise's API
but works with Python 3.13. Uses matrix factorization with stochastic gradient descent.
"""

import numpy as np
from typing import Dict, Tuple


class SVDModel:
    """SVD model for collaborative filtering (Surprise-compatible).
    
    Implements matrix factorization using stochastic gradient descent (SGD)
    similar to Surprise's SVD algorithm.
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42
    ):
        """Initialize SVD model.
        
        Args:
            n_factors: Number of latent factors (default: 50)
            n_epochs: Number of training epochs (default: 20)
            lr_all: Learning rate for all parameters (default: 0.005)
            reg_all: Regularization term for all parameters (default: 0.02)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        
        # Model parameters (will be initialized during fit)
        self.bu = None  # User biases
        self.bi = None  # Item biases
        self.pu = None  # User factors
        self.qi = None  # Item factors
        self.global_mean = None
        self.trainset = None
        
    def fit(self, trainset):
        """Train the SVD model.
        
        Args:
            trainset: Trainset object with all_ratings() method
                      Returns list of (user_id, item_id, rating) tuples
        """
        self.trainset = trainset
        
        # Get number of users and items
        n_users = trainset.n_users
        n_items = trainset.n_items
        
        # Initialize global mean
        all_ratings = [r for (_, _, r) in trainset.all_ratings()]
        self.global_mean = np.mean(all_ratings) if all_ratings else 0.0
        
        # Initialize parameters with small random values
        np.random.seed(self.random_state)
        self.bu = np.zeros(n_users, dtype=np.double)
        self.bi = np.zeros(n_items, dtype=np.double)
        self.pu = np.random.normal(0, 0.1, (n_users, self.n_factors)).astype(np.double)
        self.qi = np.random.normal(0, 0.1, (n_items, self.n_factors)).astype(np.double)
        
        # Train using SGD
        for epoch in range(self.n_epochs):
            # Shuffle ratings for each epoch
            ratings = list(trainset.all_ratings())
            np.random.shuffle(ratings)
            
            for u, i, r in ratings:
                # Get inner user and item IDs
                u_inner = u
                i_inner = i
                
                # Predict current rating
                pred = self.global_mean + self.bu[u_inner] + self.bi[i_inner] + np.dot(self.pu[u_inner], self.qi[i_inner])
                
                # Calculate error
                err = r - pred
                
                # Update biases
                self.bu[u_inner] += self.lr_all * (err - self.reg_all * self.bu[u_inner])
                self.bi[i_inner] += self.lr_all * (err - self.reg_all * self.bi[i_inner])
                
                # Update factors
                puf = self.pu[u_inner].copy()
                self.pu[u_inner] += self.lr_all * (err * self.qi[i_inner] - self.reg_all * self.pu[u_inner])
                self.qi[i_inner] += self.lr_all * (err * puf - self.reg_all * self.qi[i_inner])
    
    def predict(self, user_id: str, item_id: str) -> 'Prediction':
        """Predict rating for a user-item pair.
        
        Args:
            user_id: User ID (as string)
            item_id: Item ID (as string)
        
        Returns:
            Prediction object with 'est' attribute (predicted rating)
        """
        # Convert to inner IDs
        try:
            u_inner = self.trainset.to_inner_uid(user_id)
            i_inner = self.trainset.to_inner_iid(item_id)
        except ValueError:
            # User or item not in training set, return global mean
            return Prediction(user_id, item_id, self.global_mean, True)
        
        # Predict rating
        pred = self.global_mean + self.bu[u_inner] + self.bi[i_inner] + np.dot(self.pu[u_inner], self.qi[i_inner])
        
        # Clip to valid rating range (1-5)
        pred = np.clip(pred, 1.0, 5.0)
        
        return Prediction(user_id, item_id, pred, False)


class Prediction:
    """Prediction object (Surprise-compatible)."""
    
    def __init__(self, uid: str, iid: str, est: float, was_impossible: bool = False):
        """Initialize prediction.
        
        Args:
            uid: User ID
            iid: Item ID
            est: Estimated rating
            was_impossible: Whether prediction was impossible (default: False)
        """
        self.uid = uid
        self.iid = iid
        self.est = est
        self.was_impossible = was_impossible


class Trainset:
    """Trainset object (Surprise-compatible).
    
    Wraps rating data and provides mapping between raw and inner IDs.
    """
    
    def __init__(self, ratings_df):
        """Initialize trainset from ratings DataFrame.
        
        Args:
            ratings_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        self.ratings_df = ratings_df.copy()
        
        # Create mappings
        unique_users = sorted(self.ratings_df['user_id'].unique())
        unique_items = sorted(self.ratings_df['item_id'].unique())
        
        self._raw2inner_id_users = {uid: i for i, uid in enumerate(unique_users)}
        self._raw2inner_id_items = {iid: i for i, iid in enumerate(unique_items)}
        self._inner2raw_id_users = {i: uid for uid, i in self._raw2inner_id_users.items()}
        self._inner2raw_id_items = {i: iid for iid, i in self._raw2inner_id_items.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        self.n_ratings = len(self.ratings_df)
        
        # Convert to inner IDs for efficient access
        self.ratings_df['user_inner'] = self.ratings_df['user_id'].map(self._raw2inner_id_users)
        self.ratings_df['item_inner'] = self.ratings_df['item_id'].map(self._raw2inner_id_items)
    
    def all_ratings(self):
        """Return all ratings as (user_inner, item_inner, rating) tuples.
        
        Returns:
            Generator of (user_inner, item_inner, rating) tuples
        """
        for _, row in self.ratings_df.iterrows():
            yield (int(row['user_inner']), int(row['item_inner']), float(row['rating']))
    
    def to_inner_uid(self, raw_uid):
        """Convert raw user ID to inner user ID.
        
        Args:
            raw_uid: Raw user ID (can be string or int)
        
        Returns:
            Inner user ID (int)
        
        Raises:
            ValueError: If user ID not found
        """
        # Convert to same type as keys
        if isinstance(raw_uid, str):
            raw_uid = int(raw_uid)
        return self._raw2inner_id_users[raw_uid]
    
    def to_inner_iid(self, raw_iid):
        """Convert raw item ID to inner item ID.
        
        Args:
            raw_iid: Raw item ID (can be string or int)
        
        Returns:
            Inner item ID (int)
        
        Raises:
            ValueError: If item ID not found
        """
        # Convert to same type as keys
        if isinstance(raw_iid, str):
            raw_iid = int(raw_iid)
        return self._raw2inner_id_items[raw_iid]
    
    def to_raw_uid(self, inner_uid):
        """Convert inner user ID to raw user ID.
        
        Args:
            inner_uid: Inner user ID (int)
        
        Returns:
            Raw user ID
        """
        return self._inner2raw_id_users[inner_uid]
    
    def to_raw_iid(self, inner_iid):
        """Convert inner item ID to raw item ID.
        
        Args:
            inner_iid: Inner item ID (int)
        
        Returns:
            Raw item ID
        """
        return self._inner2raw_id_items[inner_iid]


def load_from_file(file_path: str, reader=None) -> 'Dataset':
    """Load dataset from CSV file (Surprise-compatible).
    
    Args:
        file_path: Path to CSV file (format: user_id,item_id,rating)
        reader: Reader object (ignored, kept for compatibility)
    
    Returns:
        Dataset object
    """
    import pandas as pd
    
    # Read CSV file (no header, comma-separated)
    ratings_df = pd.read_csv(file_path, header=None, names=['user_id', 'item_id', 'rating'])
    
    return Dataset(ratings_df)


class Dataset:
    """Dataset object (Surprise-compatible)."""
    
    def __init__(self, ratings_df):
        """Initialize dataset from ratings DataFrame.
        
        Args:
            ratings_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        self.ratings_df = ratings_df
    
    def build_full_trainset(self) -> Trainset:
        """Build full trainset from dataset.
        
        Returns:
            Trainset object
        """
        return Trainset(self.ratings_df)


# Alias for compatibility
SVD = SVDModel
Dataset.load_from_file = staticmethod(load_from_file)


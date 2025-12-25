"""Simplified feature preparation script for the Retail Rocket baseline."""

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import preprocessing


EVENTS_PATH = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/raw/retailrocket/events.csv"
)
OUTPUT_DIR = Path(
    "/Users/arturdvorak/Desktop/ML course/Notebooks/retail-recommender-system/data/processed"
)
# Date ranges for splitting data chronologically
# Using all available data: May 3, 2015 to Sep 18, 2015 (~4.5 months)
# Split into 3 roughly equal periods (~1.5 months each)
TRAIN_VAL_START = "2015-05-03"  # Oldest period: training data for hyperparameter tuning
TRAIN_VAL_END = "2015-06-18"    # End of first period
VALID_START = "2015-06-19"      # Middle period start (same as TRAIN_START)
VALID_END = "2015-07-31"        # Middle period end (same as TRAIN_END)
TRAIN_START = "2015-06-19"      # Same period as VALID but with all data
TRAIN_END = "2015-07-31"        # Training data for final model
TEST_START = "2015-08-01"       # Latest period: final evaluation
TEST_END = "2015-09-18"         # End of available data


def add_random_ratings(purchases_df: pd.DataFrame, rating_percentage: float = 0.2, random_seed: int = 42) -> pd.Series:
    """Add realistic random ratings to a percentage of purchased products.
    
    Simulates post-purchase reviews where users rate products they bought.
    Rating distribution: 60% positive (4-5 stars), 20% neutral (3 stars), 20% negative (1-2 stars)
    
    Args:
        purchases_df: DataFrame containing purchase events (transaction events)
        rating_percentage: Percentage of purchases to assign explicit ratings (default: 0.2 = 20%)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Series with explicit ratings (1-5) for selected purchases, NaN for others
    """
    np.random.seed(random_seed)
    
    # Randomly select purchases to rate
    n_purchases = len(purchases_df)
    n_to_rate = int(n_purchases * rating_percentage)
    
    # Create array of NaN (no explicit rating)
    explicit_ratings = pd.Series([np.nan] * n_purchases, index=purchases_df.index)
    
    if n_to_rate > 0:
        # Randomly select indices to rate
        indices_to_rate = np.random.choice(purchases_df.index, size=n_to_rate, replace=False)
        
        # Generate ratings with realistic distribution
        # 60% positive (4-5), 20% neutral (3), 20% negative (1-2)
        rating_probs = [0.2, 0.2, 0.2, 0.3, 0.1]  # Probabilities for ratings 1-5
        ratings = np.random.choice([1, 2, 3, 4, 5], size=n_to_rate, p=rating_probs)
        
        # Assign ratings to selected purchases
        explicit_ratings.loc[indices_to_rate] = ratings
    
    return explicit_ratings


def create_explicit_ratings(events_df: pd.DataFrame, split_name: str, random_seed: int = 42) -> pd.DataFrame:
    """Convert events to explicit ratings with hybrid implicit+explicit approach.
    
    Step 1: Convert all events to base implicit ratings (view=1, addtocart=2, transaction=3)
    Step 2: Add realistic explicit ratings to 20% of purchased products
    
    Args:
        events_df: DataFrame with columns ['visitorid', 'itemid', 'event']
        split_name: Name of the split (e.g., 'train_val', 'train', 'validation', 'test')
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        DataFrame with columns ['user_id', 'item_id', 'rating'] ready for Surprise
    """
    # Step 1: Convert events to base implicit ratings
    event_to_rating = {
        'view': 1,
        'addtocart': 2,
        'transaction': 3
    }
    
    # Map events to ratings
    events_df = events_df.copy()
    events_df['base_rating'] = events_df['event'].map(event_to_rating)
    
    # Group by (visitorid, itemid) and take maximum rating (strongest signal wins)
    ratings_df = events_df.groupby(['visitorid', 'itemid'])['base_rating'].max().reset_index()
    ratings_df.columns = ['visitorid', 'itemid', 'rating']
    
    # Step 2: Identify purchased products and add explicit ratings
    # Get all purchases (items with transaction events)
    purchases = events_df[events_df['event'] == 'transaction'][['visitorid', 'itemid']].drop_duplicates()
    
    if len(purchases) > 0:
        # Merge to find which ratings correspond to purchases
        purchases_with_ratings = ratings_df.merge(
            purchases, 
            on=['visitorid', 'itemid'], 
            how='inner',
            indicator=True
        )
        
        # Add explicit ratings to 20% of purchases
        explicit_ratings_series = add_random_ratings(purchases_with_ratings, rating_percentage=0.2, random_seed=random_seed)
        
        # Update ratings: replace base rating (3) with explicit rating where available
        purchases_with_ratings['explicit_rating'] = explicit_ratings_series
        
        # Create mapping: (visitorid, itemid) -> explicit_rating
        explicit_rating_map = purchases_with_ratings[
            purchases_with_ratings['explicit_rating'].notna()
        ].set_index(['visitorid', 'itemid'])['explicit_rating'].to_dict()
        
        # Update ratings_df: use explicit rating if available, otherwise keep base rating
        ratings_df['rating'] = ratings_df.apply(
            lambda row: explicit_rating_map.get((row['visitorid'], row['itemid']), row['rating']),
            axis=1
        )
    
    # Rename columns for Surprise format (user_id, item_id, rating)
    ratings_df = ratings_df.rename(columns={'visitorid': 'user_id', 'itemid': 'item_id'})
    
    # Ensure rating is integer (explicit ratings are 1-5, implicit are 1-3)
    ratings_df['rating'] = ratings_df['rating'].astype(int)
    
    return ratings_df


def main() -> None:
    """Load events, split chronologically by date ranges, encode ids, and persist artifacts.
    
    Creates multiple splits to support different model types:
    - ALS/BPR: Use warm splits (cold-start removed)
    - LightFM/Explicit: Use warm + cold splits (can handle cold-start)
    
    Splits created:
    Hyperparameter tuning: train_val, valid_warm, valid_cold, valid_full
    Final model: train, test_warm, test_cold, test_full
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: load the raw dataset and convert timestamps.
    events = pd.read_csv(EVENTS_PATH)
    
    # Convert timestamp to datetime (keeping full time information)
    events["datetime"] = pd.to_datetime(
        events["timestamp"], unit="ms", utc=True
    ).dt.tz_convert(None)
    # Extract date from datetime
    events["date"] = events["datetime"].dt.date
    # Extract time-based features for future use
    events["hour"] = events["datetime"].dt.hour  # Hour of day (0-23)
    events["day_of_week"] = events["datetime"].dt.dayofweek  # Monday=0, Sunday=6
    events["is_weekend"] = events["day_of_week"].isin([5, 6])  # Saturday=5, Sunday=6
    # Sort all events by datetime
    events.sort_values("datetime", inplace=True)
    events.reset_index(drop=True, inplace=True)
    # Keep all columns including timestamp and time-based features
    events = events[["visitorid", "itemid", "event", "timestamp", "date", "datetime", "hour", "day_of_week", "is_weekend"]]

    # Stage 2: split into chronological periods by date ranges.
    # HYPERPARAMETER TUNING PHASE (May 3 - July 31):
    # 1. train_val (Period 1): Training data for all models
    #    - First step: Train ALS/BPR with default hyperparameters on train_val (Period 1)
    # 2. valid_warm (Period 2, cold-start removed): Validation for ALS/BPR
    # 3. valid_cold (Period 2, cold-start only): Validation for LightFM/Explicit
    # 4. valid_full (Period 2, ALL data): Combined validation
    #
    # FINAL MODEL PHASE (June 19 - Sep 18):
    # 5. train (Period 2, ALL data): Final training data
    # 6. test_warm (Period 3, cold-start removed): Final test for ALS/BPR
    # 7. test_cold (Period 3, cold-start only): Final test for LightFM/Explicit
    # 8. test_full (Period 3, ALL data): Combined test
    
    # Convert date strings to date objects
    train_val_start = dt.datetime.strptime(TRAIN_VAL_START, "%Y-%m-%d").date()
    train_val_end = dt.datetime.strptime(TRAIN_VAL_END, "%Y-%m-%d").date()
    valid_start = dt.datetime.strptime(VALID_START, "%Y-%m-%d").date()
    valid_end = dt.datetime.strptime(VALID_END, "%Y-%m-%d").date()
    train_start = dt.datetime.strptime(TRAIN_START, "%Y-%m-%d").date()
    train_end = dt.datetime.strptime(TRAIN_END, "%Y-%m-%d").date()
    test_start = dt.datetime.strptime(TEST_START, "%Y-%m-%d").date()
    test_end = dt.datetime.strptime(TEST_END, "%Y-%m-%d").date()
    
    # Create train_val: oldest period - training data for hyperparameter tuning
    train_val = events.loc[
        (events["date"] >= train_val_start) & (events["date"] <= train_val_end)
    ].copy()
    
    # Create validation splits: middle period (June 19 - July 31)
    valid_full = events.loc[
        (events["date"] >= valid_start) & (events["date"] <= valid_end)
    ].copy()
    
    # valid_warm: only users/items seen in train_val (for ALS/BPR)
    valid_warm_mask = valid_full["visitorid"].isin(train_val["visitorid"]) & valid_full["itemid"].isin(
        train_val["itemid"]
    )
    valid_warm = valid_full.loc[valid_warm_mask].copy()
    
    # valid_cold: only users NOT in train_val (cold-start users for LightFM/Explicit)
    # Items can be in train_val (items are usually not cold-start)
    valid_cold_mask = ~valid_full["visitorid"].isin(train_val["visitorid"])
    valid_cold = valid_full.loc[valid_cold_mask].copy()
    
    # Create train: same period as valid but ALL data - training for final model
    train = events.loc[
        (events["date"] >= train_start) & (events["date"] <= train_end)
    ].copy()  # No filtering: includes all users/items for maximum training data
    
    # Create test splits: latest period (Aug 1 - Sep 18)
    test_full = events.loc[
        (events["date"] >= test_start) & (events["date"] <= test_end)
    ].copy()
    
    # test_warm: only users/items seen in train (for ALS/BPR)
    test_warm_mask = test_full["visitorid"].isin(train["visitorid"]) & test_full["itemid"].isin(
        train["itemid"]
    )
    test_warm = test_full.loc[test_warm_mask].copy()
    
    # test_cold: only users NOT in train (cold-start users for LightFM/Explicit)
    # Items can be in train (items are usually not cold-start)
    test_cold_mask = ~test_full["visitorid"].isin(train["visitorid"])
    test_cold = test_full.loc[test_cold_mask].copy()

    # Stage 3: fit encoders and transform splits.
    # We need multiple sets of encoders:
    # 1. encoders_tune_warm: For train_val/valid_warm (ALS/BPR tuning)
    # 2. encoders_tune_full: For train_val/valid_full (LightFM/Explicit tuning)
    # 3. encoders_final_warm: For train/test_warm (ALS/BPR final)
    # 4. encoders_final_full: For train/test_full (LightFM/Explicit final)
    
    # Encoders for hyperparameter tuning - warm splits (ALS/BPR)
    encoders_tune_warm = {}
    train_val_user_warm = preprocessing.LabelEncoder().fit(train_val["visitorid"])
    encoders_tune_warm["visitorid"] = train_val_user_warm
    train_val_user_ids = train_val_user_warm.transform(train_val["visitorid"])
    valid_warm_user_ids = train_val_user_warm.transform(valid_warm["visitorid"])

    train_val_item_warm = preprocessing.LabelEncoder().fit(train_val["itemid"])
    encoders_tune_warm["itemid"] = train_val_item_warm
    train_val_item_ids = train_val_item_warm.transform(train_val["itemid"])
    valid_warm_item_ids = train_val_item_warm.transform(valid_warm["itemid"])

    train_val_event_warm = preprocessing.LabelEncoder().fit(train_val["event"])
    encoders_tune_warm["event"] = train_val_event_warm
    train_val_events = train_val_event_warm.transform(train_val["event"])
    valid_warm_events = train_val_event_warm.transform(valid_warm["event"])

    # Encoders for hyperparameter tuning - full splits (LightFM/Explicit)
    # Need to fit on combined train_val + valid_full to handle cold-start users
    encoders_tune_full = {}
    tune_all_users = pd.concat([train_val["visitorid"], valid_full["visitorid"]]).unique()
    tune_all_items = pd.concat([train_val["itemid"], valid_full["itemid"]]).unique()
    train_val_user_full = preprocessing.LabelEncoder().fit(tune_all_users)
    encoders_tune_full["visitorid"] = train_val_user_full
    valid_full_user_ids = train_val_user_full.transform(valid_full["visitorid"])
    valid_cold_user_ids = train_val_user_full.transform(valid_cold["visitorid"])

    train_val_item_full = preprocessing.LabelEncoder().fit(tune_all_items)
    encoders_tune_full["itemid"] = train_val_item_full
    valid_full_item_ids = train_val_item_full.transform(valid_full["itemid"])
    valid_cold_item_ids = train_val_item_full.transform(valid_cold["itemid"])

    train_val_event_full = preprocessing.LabelEncoder().fit(train_val["event"])
    encoders_tune_full["event"] = train_val_event_full
    # Also encode train_val with full encoders for LightFM/Explicit training
    train_val_user_ids_full = train_val_user_full.transform(train_val["visitorid"])
    train_val_item_ids_full = train_val_item_full.transform(train_val["itemid"])
    train_val_events_full = train_val_event_full.transform(train_val["event"])
    valid_full_events = train_val_event_full.transform(valid_full["event"])
    valid_cold_events = train_val_event_full.transform(valid_cold["event"])

    # Encoders for final model - warm splits (ALS/BPR)
    encoders_final_warm = {}
    train_user_warm = preprocessing.LabelEncoder().fit(train["visitorid"])
    encoders_final_warm["visitorid"] = train_user_warm
    train_user_ids = train_user_warm.transform(train["visitorid"])
    test_warm_user_ids = train_user_warm.transform(test_warm["visitorid"])

    train_item_warm = preprocessing.LabelEncoder().fit(train["itemid"])
    encoders_final_warm["itemid"] = train_item_warm
    train_item_ids = train_item_warm.transform(train["itemid"])
    test_warm_item_ids = train_item_warm.transform(test_warm["itemid"])

    train_event_warm = preprocessing.LabelEncoder().fit(train["event"])
    encoders_final_warm["event"] = train_event_warm
    train_events = train_event_warm.transform(train["event"])
    test_warm_events = train_event_warm.transform(test_warm["event"])

    # Encoders for final model - full splits (LightFM/Explicit)
    # Need to fit on combined train + test_full to handle cold-start users
    encoders_final_full = {}
    final_all_users = pd.concat([train["visitorid"], test_full["visitorid"]]).unique()
    final_all_items = pd.concat([train["itemid"], test_full["itemid"]]).unique()
    train_user_full = preprocessing.LabelEncoder().fit(final_all_users)
    encoders_final_full["visitorid"] = train_user_full
    test_full_user_ids = train_user_full.transform(test_full["visitorid"])
    test_cold_user_ids = train_user_full.transform(test_cold["visitorid"])

    train_item_full = preprocessing.LabelEncoder().fit(final_all_items)
    encoders_final_full["itemid"] = train_item_full
    test_full_item_ids = train_item_full.transform(test_full["itemid"])
    test_cold_item_ids = train_item_full.transform(test_cold["itemid"])

    train_event_full = preprocessing.LabelEncoder().fit(train["event"])
    encoders_final_full["event"] = train_event_full
    # Also encode train with full encoders for LightFM/Explicit training
    train_user_ids_full = train_user_full.transform(train["visitorid"])
    train_item_ids_full = train_item_full.transform(train["itemid"])
    train_events_full = train_event_full.transform(train["event"])
    test_full_events = train_event_full.transform(test_full["event"])
    test_cold_events = train_event_full.transform(test_cold["event"])

    # Stage 4: build sparse matrices for all splits.
    
    # Matrices for hyperparameter tuning - warm splits (ALS/BPR)
    n_users_tune_warm = int(train_val_user_ids.max() + 1)
    n_items_tune_warm = int(train_val_item_ids.max() + 1)
    train_val_matrix = sparse.coo_matrix(
        (train_val_events, (train_val_user_ids, train_val_item_ids)), 
        shape=(n_users_tune_warm, n_items_tune_warm)
    )
    valid_warm_matrix = sparse.coo_matrix(
        (valid_warm_events, (valid_warm_user_ids, valid_warm_item_ids)), 
        shape=(n_users_tune_warm, n_items_tune_warm)
    )
    
    # Matrices for hyperparameter tuning - full splits (LightFM/Explicit)
    # Need to include train_val in shape calculation for full encoders
    n_users_tune_full = max(int(train_val_user_ids_full.max() + 1), int(valid_full_user_ids.max() + 1))
    n_items_tune_full = max(int(train_val_item_ids_full.max() + 1), int(valid_full_item_ids.max() + 1))
    train_val_matrix_full = sparse.coo_matrix(
        (train_val_events_full, (train_val_user_ids_full, train_val_item_ids_full)), 
        shape=(n_users_tune_full, n_items_tune_full)
    )
    valid_full_matrix = sparse.coo_matrix(
        (valid_full_events, (valid_full_user_ids, valid_full_item_ids)), 
        shape=(n_users_tune_full, n_items_tune_full)
    )
    valid_cold_matrix = sparse.coo_matrix(
        (valid_cold_events, (valid_cold_user_ids, valid_cold_item_ids)), 
        shape=(n_users_tune_full, n_items_tune_full)
    )
    
    # Matrices for final model - warm splits (ALS/BPR)
    n_users_final_warm = int(train_user_ids.max() + 1)
    n_items_final_warm = int(train_item_ids.max() + 1)
    train_matrix = sparse.coo_matrix(
        (train_events, (train_user_ids, train_item_ids)), 
        shape=(n_users_final_warm, n_items_final_warm)
    )
    test_warm_matrix = sparse.coo_matrix(
        (test_warm_events, (test_warm_user_ids, test_warm_item_ids)), 
        shape=(n_users_final_warm, n_items_final_warm)
    )
    
    # Matrices for final model - full splits (LightFM/Explicit)
    # Need to include train in shape calculation for full encoders
    n_users_final_full = max(int(train_user_ids_full.max() + 1), int(test_full_user_ids.max() + 1))
    n_items_final_full = max(int(train_item_ids_full.max() + 1), int(test_full_item_ids.max() + 1))
    train_matrix_full = sparse.coo_matrix(
        (train_events_full, (train_user_ids_full, train_item_ids_full)), 
        shape=(n_users_final_full, n_items_final_full)
    )
    test_full_matrix = sparse.coo_matrix(
        (test_full_events, (test_full_user_ids, test_full_item_ids)), 
        shape=(n_users_final_full, n_items_final_full)
    )
    test_cold_matrix = sparse.coo_matrix(
        (test_cold_events, (test_cold_user_ids, test_cold_item_ids)), 
        shape=(n_users_final_full, n_items_final_full)
    )

    # Stage 5: persist datasets, encoders, and matrices for later steps.
    
    # Save all event datasets
    joblib.dump(train_val, OUTPUT_DIR / "events_train_val.pkl")
    joblib.dump(valid_warm, OUTPUT_DIR / "events_validation_warm.pkl")
    joblib.dump(valid_cold, OUTPUT_DIR / "events_validation_cold.pkl")
    joblib.dump(valid_full, OUTPUT_DIR / "events_validation_full.pkl")
    joblib.dump(train, OUTPUT_DIR / "events_train.pkl")
    joblib.dump(test_warm, OUTPUT_DIR / "events_test_warm.pkl")
    joblib.dump(test_cold, OUTPUT_DIR / "events_test_cold.pkl")
    joblib.dump(test_full, OUTPUT_DIR / "events_test_full.pkl")
    
    # Save all encoder sets
    joblib.dump(encoders_tune_warm, OUTPUT_DIR / "encoders_tune_warm.pkl")
    joblib.dump(encoders_tune_full, OUTPUT_DIR / "encoders_tune_full.pkl")
    joblib.dump(encoders_final_warm, OUTPUT_DIR / "encoders_final_warm.pkl")
    joblib.dump(encoders_final_full, OUTPUT_DIR / "encoders_final_full.pkl")
    
    # Save matrices for hyperparameter tuning - warm (ALS/BPR)
    joblib.dump(
        {
            "train_val": train_val_matrix, 
            "validation_warm": valid_warm_matrix
        },
        OUTPUT_DIR / "interaction_matrices_tune_warm.pkl",
    )
    
    # Save matrices for hyperparameter tuning - full (LightFM/Explicit)
    joblib.dump(
        {
            "train_val": train_val_matrix_full,
            "validation_full": valid_full_matrix,
            "validation_cold": valid_cold_matrix
        },
        OUTPUT_DIR / "interaction_matrices_tune_full.pkl",
    )
    
    # Save matrices for final model - warm (ALS/BPR)
    joblib.dump(
        {
            "train": train_matrix, 
            "test_warm": test_warm_matrix
        },
        OUTPUT_DIR / "interaction_matrices_final_warm.pkl",
    )
    
    # Save matrices for final model - full (LightFM/Explicit)
    joblib.dump(
        {
            "train": train_matrix_full,
            "test_full": test_full_matrix,
            "test_cold": test_cold_matrix
        },
        OUTPUT_DIR / "interaction_matrices_final_full.pkl",
    )
    
    # Print summary statistics
    print("\n=== Dataset Split Summary ===")
    print(f"\nHyperparameter Tuning Phase:")
    print(f"  train_val: {len(train_val):,} events, {train_val['visitorid'].nunique():,} users, {train_val['itemid'].nunique():,} items")
    print(f"  valid_warm: {len(valid_warm):,} events, {valid_warm['visitorid'].nunique():,} users, {valid_warm['itemid'].nunique():,} items")
    print(f"  valid_cold: {len(valid_cold):,} events, {valid_cold['visitorid'].nunique():,} users, {valid_cold['itemid'].nunique():,} items")
    print(f"  valid_full: {len(valid_full):,} events, {valid_full['visitorid'].nunique():,} users, {valid_full['itemid'].nunique():,} items")
    print(f"\nFinal Model Phase:")
    print(f"  train: {len(train):,} events, {train['visitorid'].nunique():,} users, {train['itemid'].nunique():,} items")
    print(f"  test_warm: {len(test_warm):,} events, {test_warm['visitorid'].nunique():,} users, {test_warm['itemid'].nunique():,} items")
    print(f"  test_cold: {len(test_cold):,} events, {test_cold['visitorid'].nunique():,} users, {test_cold['itemid'].nunique():,} items")
    print(f"  test_full: {len(test_full):,} events, {test_full['visitorid'].nunique():,} users, {test_full['itemid'].nunique():,} items")
    print("\nAll splits saved successfully!")
    
    # Stage 6: Create explicit ratings for SVD model
    print("\n=== Creating Explicit Ratings for SVD ===")
    
    # Create ratings for hyperparameter tuning phase
    train_val_ratings = create_explicit_ratings(train_val, split_name='train_val', random_seed=42)
    train_val_ratings_path = OUTPUT_DIR / "ratings_train_val.csv"
    train_val_ratings.to_csv(train_val_ratings_path, index=False, header=False)
    print(f"  Created ratings_train_val.csv: {len(train_val_ratings):,} ratings")
    
    # Create ratings for final model phase
    train_ratings = create_explicit_ratings(train, split_name='train', random_seed=42)
    train_ratings_path = OUTPUT_DIR / "ratings_train.csv"
    train_ratings.to_csv(train_ratings_path, index=False, header=False)
    print(f"  Created ratings_train.csv: {len(train_ratings):,} ratings")
    
    # Note: We don't create ratings for validation/test sets as they're only used for evaluation
    print("  Rating files saved successfully!")


if __name__ == "__main__":
    main()


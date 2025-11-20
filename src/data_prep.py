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


def main() -> None:
    """Load events, split chronologically by date ranges, encode ids, and persist artifacts.
    
    Following tutorial approach: creates train_val, valid (cold-start removed), 
    train (all data), and test (cold-start removed) splits.
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
    # 1. train_val (May 3 - June 18): Training data for hyperparameter tuning
    # 2. valid (June 19 - July 31, cold-start removed): Validation set for hyperparameter tuning
    # 3. train (June 19 - July 31, ALL data): Training data for final model
    # 4. test (Aug 1 - Sep 18, cold-start removed): Final evaluation set
    # Workflow: Step 1 (tune): train_val → model → valid | Step 2 (final): train → model → test
    
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
    
    # Create valid: middle period with cold-start removed - validation for hyperparameter tuning
    valid_full = events.loc[
        (events["date"] >= valid_start) & (events["date"] <= valid_end)
    ].copy()
    valid_mask = valid_full["visitorid"].isin(train_val["visitorid"]) & valid_full["itemid"].isin(
        train_val["itemid"]
    )  # Filter: only users/items seen in train_val (ALS can't predict cold-start)
    valid = valid_full.loc[valid_mask].copy()
    
    # Create train: same period as valid but ALL data - training for final model
    train = events.loc[
        (events["date"] >= train_start) & (events["date"] <= train_end)
    ].copy()  # No filtering: includes all users/items for maximum training data
    
    # Create test: latest period with cold-start removed - final evaluation
    test_full = events.loc[
        (events["date"] >= test_start) & (events["date"] <= test_end)
    ].copy()
    test_mask = test_full["visitorid"].isin(train["visitorid"]) & test_full["itemid"].isin(
        train["itemid"]
    )  # Filter: only users/items seen in train (ALS can't predict cold-start)
    test = test_full.loc[test_mask].copy()

    # Stage 3: fit encoders and transform splits.
    # We need two sets of encoders: one for hyperparameter tuning (train_val/valid),
    # and one for final model (train/test)
    
    # Encoders for hyperparameter tuning workflow (train_val -> valid)
    encoders_tune = {}
    train_val_user = preprocessing.LabelEncoder().fit(train_val["visitorid"])
    encoders_tune["visitorid"] = train_val_user
    train_val_user_ids = train_val_user.transform(train_val["visitorid"])
    valid_user_ids = train_val_user.transform(valid["visitorid"])

    train_val_item = preprocessing.LabelEncoder().fit(train_val["itemid"])
    encoders_tune["itemid"] = train_val_item
    train_val_item_ids = train_val_item.transform(train_val["itemid"])
    valid_item_ids = train_val_item.transform(valid["itemid"])

    train_val_event = preprocessing.LabelEncoder().fit(train_val["event"])
    encoders_tune["event"] = train_val_event
    train_val_events = train_val_event.transform(train_val["event"])
    valid_events = train_val_event.transform(valid["event"])

    # Encoders for final model workflow (train -> test)
    encoders_final = {}
    train_user = preprocessing.LabelEncoder().fit(train["visitorid"])
    encoders_final["visitorid"] = train_user
    train_user_ids = train_user.transform(train["visitorid"])
    test_user_ids = train_user.transform(test["visitorid"])

    train_item = preprocessing.LabelEncoder().fit(train["itemid"])
    encoders_final["itemid"] = train_item
    train_item_ids = train_item.transform(train["itemid"])
    test_item_ids = train_item.transform(test["itemid"])

    train_event = preprocessing.LabelEncoder().fit(train["event"])
    encoders_final["event"] = train_event
    train_events = train_event.transform(train["event"])
    test_events = train_event.transform(test["event"])

    # Stage 4: build sparse matrices required by ALS.
    # Matrices for hyperparameter tuning
    n_users_tune = int(train_val_user_ids.max() + 1)
    n_items_tune = int(train_val_item_ids.max() + 1)
    train_val_matrix = sparse.coo_matrix(
        (train_val_events, (train_val_user_ids, train_val_item_ids)), 
        shape=(n_users_tune, n_items_tune)
    )
    valid_matrix = sparse.coo_matrix(
        (valid_events, (valid_user_ids, valid_item_ids)), 
        shape=(n_users_tune, n_items_tune)
    )
    
    # Matrices for final model
    n_users_final = int(train_user_ids.max() + 1)
    n_items_final = int(train_item_ids.max() + 1)
    train_matrix = sparse.coo_matrix(
        (train_events, (train_user_ids, train_item_ids)), 
        shape=(n_users_final, n_items_final)
    )
    test_matrix = sparse.coo_matrix(
        (test_events, (test_user_ids, test_item_ids)), 
        shape=(n_users_final, n_items_final)
    )

    # Stage 5: persist datasets, encoders, and matrices for later steps.
    joblib.dump(train_val, OUTPUT_DIR / "events_train_val.pkl")
    joblib.dump(valid, OUTPUT_DIR / "events_validation.pkl")
    joblib.dump(train, OUTPUT_DIR / "events_train.pkl")
    joblib.dump(test, OUTPUT_DIR / "events_test.pkl")
    
    # Save both sets of encoders
    joblib.dump(encoders_tune, OUTPUT_DIR / "encoders_tune.pkl")
    joblib.dump(encoders_final, OUTPUT_DIR / "encoders_final.pkl")
    
    # Save matrices for both workflows
    joblib.dump(
        {
            "train_val": train_val_matrix, 
            "validation": valid_matrix
        },
        OUTPUT_DIR / "interaction_matrices_tune.pkl",
    )
    joblib.dump(
        {
            "train": train_matrix, 
            "test": test_matrix
        },
        OUTPUT_DIR / "interaction_matrices_final.pkl",
    )


if __name__ == "__main__":
    main()


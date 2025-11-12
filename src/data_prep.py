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
START_DATE = "2015-05-03"
END_DATE = "2015-06-30"
TRAIN_FRACTION = 0.7  # 70% for training
VAL_FRACTION = 0.15  # 15% for validation
# Remaining 15% goes to test


def main() -> None:
    """Load events, split chronologically, encode ids, and persist artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: load the raw dataset and assign column names for clarity.
    events = pd.read_csv(EVENTS_PATH)

    # Stage 2: convert timestamps into Python dates and filter the requested window.
    events["date"] = pd.to_datetime(
        events["timestamp"], unit="ms", utc=True
    ).dt.tz_convert(None).dt.date
    start = dt.datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end = dt.datetime.strptime(END_DATE, "%Y-%m-%d").date()
    events = events.loc[(events["date"] >= start) & (events["date"] <= end)].copy()
    events.sort_values("date", inplace=True)
    events.reset_index(drop=True, inplace=True)
    events = events[["visitorid", "itemid", "event", "date"]]

    # Stage 3: split into chronological train/validation/test portions.
    train_idx = int(round(events.shape[0] * TRAIN_FRACTION))
    val_idx = train_idx + int(round(events.shape[0] * VAL_FRACTION))
    
    train = events.iloc[:train_idx].copy()
    validation = events.iloc[train_idx:val_idx].copy()
    test = events.iloc[val_idx:].copy()
    
    # Filter validation and test to only include users/items seen in training.
    val_mask = validation["visitorid"].isin(train["visitorid"]) & validation["itemid"].isin(
        train["itemid"]
    )
    validation = validation.loc[val_mask].copy()
    
    test_mask = test["visitorid"].isin(train["visitorid"]) & test["itemid"].isin(
        train["itemid"]
    )
    test = test.loc[test_mask].copy()

    # Stage 4: fit encoders on training data and transform all splits.
    encoders = {}
    train_user = preprocessing.LabelEncoder().fit(train["visitorid"])
    encoders["visitorid"] = train_user
    train_user_ids = train_user.transform(train["visitorid"])
    val_user_ids = train_user.transform(validation["visitorid"])
    test_user_ids = train_user.transform(test["visitorid"])

    train_item = preprocessing.LabelEncoder().fit(train["itemid"])
    encoders["itemid"] = train_item
    train_item_ids = train_item.transform(train["itemid"])
    val_item_ids = train_item.transform(validation["itemid"])
    test_item_ids = train_item.transform(test["itemid"])

    train_event = preprocessing.LabelEncoder().fit(train["event"])
    encoders["event"] = train_event
    train_events = train_event.transform(train["event"])
    val_events = train_event.transform(validation["event"])
    test_events = train_event.transform(test["event"])

    # Stage 5: build sparse matrices required by ALS.
    n_users = int(train_user_ids.max() + 1)
    n_items = int(train_item_ids.max() + 1)
    train_matrix = sparse.coo_matrix(
        (train_events, (train_user_ids, train_item_ids)), shape=(n_users, n_items)
    )
    val_matrix = sparse.coo_matrix(
        (val_events, (val_user_ids, val_item_ids)), shape=(n_users, n_items)
    )
    test_matrix = sparse.coo_matrix(
        (test_events, (test_user_ids, test_item_ids)), shape=(n_users, n_items)
    )

    # Stage 6: persist datasets, encoders, and matrices for later steps.
    joblib.dump(train, OUTPUT_DIR / "events_train.pkl")
    joblib.dump(validation, OUTPUT_DIR / "events_validation.pkl")
    joblib.dump(test, OUTPUT_DIR / "events_test.pkl")
    joblib.dump(encoders, OUTPUT_DIR / "encoders.pkl")
    joblib.dump(
        {"train": train_matrix, "validation": val_matrix, "test": test_matrix},
        OUTPUT_DIR / "interaction_matrices.pkl",
    )


if __name__ == "__main__":
    main()


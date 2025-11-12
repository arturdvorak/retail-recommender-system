# Retail Recommender System - Project Status

## Project Overview

This is a collaborative filtering recommendation system built for the Retail Rocket e-commerce dataset. The system uses ALS (Alternating Least Squares) to recommend products to visitors based on their interaction history.

## What Has Been Completed

### 1. Data Pipeline (`src/data_prep.py`)
- Downloads Retail Rocket dataset from Kaggle
- Filters events from May 3, 2015 to June 30, 2015
- Splits data chronologically: 70% train, 15% validation, 15% test
- Encodes visitor and item IDs to numeric indices
- Creates sparse interaction matrices for efficient model training
- Saves processed data to `data/processed/`

### 2. Exploratory Data Analysis (`src/eda_data_exploration.py`)
- Analyzed 2.76M events with 1.41M visitors and 235k items
- Found event imbalance: 96.7% views, 2.5% add-to-cart, 0.8% transactions
- Generated visualizations in `reports/eda/`
- Summary saved in `reports/eda/eda_summary.md`

### 3. Model Training (`src/model_training.py`)
- Trains ALS model with hyperparameters:
  - Factors: 64 (latent dimensions)
  - Regularization: 0.1 (prevents overfitting)
  - Iterations: 20 (training steps)
- Saves trained model to `models/als_model.pkl`
- Logs training metrics to MLflow for experiment tracking

### 4. Model Evaluation (`src/evaluate_model.py`)
- Evaluates model using precision@K metric (K=10)
- Can evaluate on validation or test set
- Logs evaluation metrics to MLflow
- Generates experiment names based on configuration

### 5. Recommendation Generation (`src/recommend.py`)
- Provides function to get top-N recommendations for any visitor ID
- Can be used via command line or imported as a module
- Handles visitors not in training data gracefully

### 6. Experiment Tracking
- MLflow integration for tracking experiments
- Tracks hyperparameters, metrics, and model artifacts
- View experiments: `mlflow ui --backend-store-uri mlruns`

## Current Project Structure

```
retail-recommender-system/
├── data/
│   ├── raw/retailrocket/          # Original dataset files
│   └── processed/                  # Preprocessed data and encoders
├── models/                         # Trained model files
├── mlruns/                         # MLflow experiment tracking data
├── reports/eda/                    # EDA visualizations and summary
├── src/                            # Source code modules
│   ├── download_retailrocket.py   # Download dataset
│   ├── data_prep.py               # Data preprocessing
│   ├── eda_data_exploration.py    # Exploratory analysis
│   ├── model_training.py          # Train ALS model
│   ├── evaluate_model.py          # Evaluate model performance
│   └── recommend.py               # Generate recommendations
├── requirements.txt               # Python dependencies
├── MLFLOW_USAGE.md               # MLflow usage guide
└── PROJECT_STATUS.md             # This file
```

## How to Use

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Download Data
```bash
python src/download_retailrocket.py
```

### 3. Prepare Data
```bash
python src/data_prep.py
```

### 4. Train Model
```bash
python src/model_training.py
```

### 5. Evaluate Model
```bash
# Evaluate on validation set
python src/evaluate_model.py --set validation

# Evaluate on test set
python src/evaluate_model.py --set test
```

### 6. Get Recommendations
```bash
python src/recommend.py <visitorid> --top-n 10
```

### 7. View Experiments
```bash
mlflow ui --backend-store-uri mlruns
```

## Current Configuration

- **Date Window**: May 3, 2015 to June 30, 2015
- **Train/Val/Test Split**: 70% / 15% / 15% (chronological)
- **Model**: Alternating Least Squares (ALS)
- **Hyperparameters**:
  - Factors: 64
  - Regularization: 0.1
  - Iterations: 20
- **Evaluation Metric**: Precision@10

## Key Files and Their Purpose

- `src/data_prep.py`: Prepares raw data into train/val/test splits and creates interaction matrices
- `src/model_training.py`: Trains the ALS recommendation model
- `src/evaluate_model.py`: Evaluates model performance on validation or test sets
- `src/recommend.py`: Generates recommendations for specific visitors
- `data/processed/interaction_matrices.pkl`: Sparse matrices for training/evaluation
- `data/processed/encoders.pkl`: Mappings between original IDs and model indices
- `models/als_model.pkl`: Trained recommendation model

## Next Steps (Potential Improvements)

1. Hyperparameter tuning (try different factors, regularization values)
2. Feature engineering (incorporate item properties, categories)
3. Cold start handling (recommendations for new visitors/items)
4. A/B testing framework
5. Model deployment (API or web service)
6. Real-time recommendation updates

## Notes for Future Development

- All paths use absolute paths - consider making them configurable
- Model currently handles only visitors seen in training data
- Evaluation uses precision@K - could add recall, NDCG, or other metrics
- Dataset is sparse (many users with few interactions) - consider popularity-based fallbacks


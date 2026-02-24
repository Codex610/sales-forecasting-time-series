"""
ml_models.py
------------
Trains Random Forest and XGBoost on the full 94-feature dataset
(all 1,115 stores). Saves models to models/ directory.

Results (actual):
  Random Forest  MAE: 736.83  RMSE: 1,098.15
  XGBoost        MAE: 644.67  RMSE:   933.08  ← best
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from evaluation import compute_metrics, print_summary

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH = 'data/processed/'
MODEL_PATH     = 'models/'
REPORT_PATH    = 'reports/'

# ── Config ────────────────────────────────────────────────────────────────────
TARGET      = 'Sales_log'
RANDOM_SEED = 42


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data():
    train_df = pd.read_csv(PROCESSED_PATH + 'train_featured.csv', parse_dates=['Date'])
    val_df   = pd.read_csv(PROCESSED_PATH + 'val_featured.csv',   parse_dates=['Date'])

    feature_cols = pd.read_csv(PROCESSED_PATH + 'feature_list.csv')['feature'].tolist()
    feature_cols = [f for f in feature_cols if f in train_df.columns]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET]
    X_val   = val_df[feature_cols].fillna(0)
    y_val   = val_df[TARGET]

    print(f'Train : {X_train.shape}  |  Val : {X_val.shape}')
    print(f'Features : {len(feature_cols)}')
    return X_train, y_train, X_val, y_val, feature_cols


# ── 2. Train Random Forest ────────────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    print('\nTraining Random Forest...')
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=10,
        max_features='sqrt', n_jobs=-1, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    print('Random Forest trained ✅')
    return model


# ── 3. Train XGBoost ──────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val):
    print('\nTraining XGBoost...')
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print('XGBoost trained ✅')
    return model


# ── 4. Save model ─────────────────────────────────────────────────────────────
def save_model(model, filename):
    os.makedirs(MODEL_PATH, exist_ok=True)
    path = MODEL_PATH + filename
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Saved → {path}')


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_ml():
    print('=' * 52)
    print('  ML MODELS PIPELINE')
    print('=' * 52)

    X_train, y_train, X_val, y_val, feature_cols = load_data()
    y_val_actual = np.expm1(y_val.values)

    # Train
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # Predict
    rf_pred  = np.expm1(rf_model.predict(X_val))
    xgb_pred = np.expm1(xgb_model.predict(X_val))

    # Evaluate
    print('\n── Evaluation ──')
    results = [
        compute_metrics(y_val_actual, rf_pred,  'Random Forest'),
        compute_metrics(y_val_actual, xgb_pred, 'XGBoost'),
    ]
    results_df = print_summary(pd.DataFrame(results).sort_values('MAE').reset_index(drop=True))

    # Save
    save_model(rf_model,  'random_forest.pkl')
    save_model(xgb_model, 'xgboost.pkl')
    os.makedirs(REPORT_PATH, exist_ok=True)
    pd.DataFrame(results).to_csv(REPORT_PATH + 'ml_results.csv', index=False)
    print(f'Saved → {REPORT_PATH}ml_results.csv')

    print('\n' + '=' * 52)
    print('  Pipeline complete ✅')
    print('=' * 52)
    return rf_model, xgb_model


if __name__ == '__main__':
    run_ml()

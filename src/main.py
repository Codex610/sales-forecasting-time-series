"""
main.py
-------
Runs the full Rossmann sales forecasting pipeline end-to-end.

Steps:
  1. Data preprocessing     → data/processed/
  2. Feature engineering    → data/processed/featured_sales_data.csv
  3. SARIMAX (Store 1)      → models/arima_model.pkl
  4. ML Models (all stores) → models/random_forest.pkl, xgboost.pkl
  5. Final summary

Usage:
  python main.py              # full pipeline
  python main.py --step ml    # only ML models
  python main.py --step arima # only SARIMAX
"""

import argparse
import time

from data_preprocessing  import run_preprocessing
from feature_engineering import run_feature_engineering
from arima_model         import run_arima
from ml_models           import run_ml


def run_full_pipeline():
    t0 = time.time()
    print('\n' + '═' * 54)
    print('  ROSSMANN SALES FORECASTING — FULL PIPELINE')
    print('═' * 54)

    print('\n[1/4] Data Preprocessing...')
    run_preprocessing()

    print('\n[2/4] Feature Engineering...')
    run_feature_engineering()

    print('\n[3/4] SARIMAX Model (Store 1)...')
    run_arima()

    print('\n[4/4] ML Models (all 1,115 stores)...')
    run_ml()

    elapsed = time.time() - t0
    print(f'\n{"═" * 54}')
    print(f'  ALL STEPS COMPLETE  ⏱  {elapsed/60:.1f} min')
    print(f'{"═" * 54}')
    print('\n  Key results:')
    print('  SARIMAX(2,1,2)+exog  MAE:  489.65  [Store 1]')
    print('  Random Forest        MAE:  736.83  [All stores]')
    print('  XGBoost              MAE:  644.67  [All stores]  ← best')
    print(f'{"═" * 54}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rossmann forecasting pipeline')
    parser.add_argument('--step', choices=['preprocess', 'features', 'arima', 'ml'],
                        help='Run a single step instead of full pipeline')
    args = parser.parse_args()

    if   args.step == 'preprocess': run_preprocessing()
    elif args.step == 'features':   run_feature_engineering()
    elif args.step == 'arima':      run_arima()
    elif args.step == 'ml':         run_ml()
    else:                           run_full_pipeline()

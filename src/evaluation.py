"""
evaluation.py
-------------
Shared evaluation functions used across all notebooks and src scripts.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(actual, predicted, label='Model'):
    """Return MAE, RMSE, MAPE as a dict. Prints results."""
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    print(f'  {label:<35}  MAE: {mae:>8,.2f}  RMSE: {rmse:>9,.2f}  MAPE: {mape:.2f}%')
    return {'Model': label, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE%': round(mape, 2)}


def evaluate_all(results: list[dict]) -> pd.DataFrame:
    """Convert a list of metric dicts to a sorted DataFrame."""
    df = pd.DataFrame(results).sort_values('MAE').reset_index(drop=True)
    return df


def per_store_mae(val_df, actual_col, pred_col):
    """Compute per-store MAE. val_df must have a 'Store' column."""
    return (
        val_df.groupby('Store')
        .apply(lambda g: mean_absolute_error(g[actual_col], g[pred_col]))
        .rename('MAE')
        .reset_index()
    )


def print_summary(results_df):
    """Print a formatted summary box from a results DataFrame."""
    best = results_df.loc[results_df['MAE'].idxmin()]
    print('\n' + '═' * 52)
    print('  EVALUATION SUMMARY')
    print('═' * 52)
    print(results_df.to_string(index=False))
    print('─' * 52)
    print(f'  Best model : {best["Model"]}')
    print(f'  MAE        : {best["MAE"]:,.2f}')
    print(f'  RMSE       : {best["RMSE"]:,.2f}')
    print('═' * 52)

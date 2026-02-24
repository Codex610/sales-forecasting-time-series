"""
arima_model.py
Saves best model → models/arima_model.pkl
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH = 'data/processed/'
MODEL_PATH     = 'models/'

# ── Config ────────────────────────────────────────────────────────────────────
STORE_ID    = 1
SPLIT_RATIO = 0.80
ORDER       = (2, 1, 2)    # (p, d, q)

# Exogenous features passed into SARIMAX
# Strongest non-lag predictors from EDA + ML feature importance
EXOG_FEATURES = [
    'Promo',                   # strongest categorical predictor
    'SchoolHoliday',
    'IsPublicHoliday',
    'IsChristmas',
    'IsMonday',                # Monday = highest sales (EDA)
    'IsSaturday',              # Saturday = lowest sales (EDA)
    'IsMonthEnd',
    'IsQ4',                    # Christmas quarter
    'BeforeHoliday',
    'AfterHoliday',
    'Promo_x_Monday',
    'Promo_x_SchoolHol',
    'fourier_weekly_sin_7_1',  # weekly seasonality
    'fourier_weekly_cos_7_1',
    'fourier_annual_sin_365_1',# annual seasonality
    'fourier_annual_cos_365_1',
]


# ── 1. Load store data ────────────────────────────────────────────────────────
def load_store_data(store_id=STORE_ID):
    """Load full featured data for a single store."""
    df = pd.read_csv(PROCESSED_PATH + 'featured_sales_data.csv',parse_dates=['Date'], low_memory=False)
    store_df = (df[df['Store'] == store_id].sort_values('Date').set_index('Date').dropna(subset=['Sales_log']))
    print(f'Store {store_id} : {len(store_df)} days  'f'({store_df.index.min().date()} → {store_df.index.max().date()})')
    return store_df


# ── 2. Prepare series + exog ──────────────────────────────────────────────────
def prepare_data(store_df):
    """Split log(Sales) series and exog features into train/val."""
    series    = store_df['Sales_log']
    exog_cols = [c for c in EXOG_FEATURES if c in store_df.columns]
    exog      = store_df[exog_cols].fillna(0).astype(float)

    print(f'Exog features : {len(exog_cols)}  → {exog_cols}')

    idx        = int(len(series) * SPLIT_RATIO)
    train_y    = series.iloc[:idx]
    val_y      = series.iloc[idx:]
    train_exog = exog.iloc[:idx]
    val_exog   = exog.iloc[idx:]

    print(f'Train : {len(train_y)} days  |  Val : {len(val_y)} days')
    return train_y, val_y, train_exog, val_exog


# ── 3. Stationarity check ─────────────────────────────────────────────────────
def check_stationarity(series, label='Series'):
    result = adfuller(series.dropna(), autolag='AIC')
    pval   = result[1]
    status = 'Stationary ✅' if pval < 0.05 else 'Non-stationary ❌'
    print(f'ADF [{label}] p={pval:.4f} → {status}')
    return pval < 0.05


# ── 4. Fit plain ARIMA ────────────────────────────────────────────────────────
def fit_arima(train_y, order=ORDER):
    """Univariate ARIMA — ignores all exogenous features."""
    print(f'\nFitting ARIMA{order} ...')
    model = ARIMA(train_y, order=order).fit()
    print(f'  AIC : {model.aic:.2f}')
    return model


# ── 5. Fit SARIMAX with exogenous features ────────────────────────────────────
def fit_sarimax(train_y, train_exog, order=ORDER):
    """
    SARIMAX — same (p,d,q) as ARIMA but now the model also learns
    the effect of Promo, holidays, Fourier terms etc.
    This is the key improvement over plain ARIMA.
    """
    print(f'\nFitting SARIMAX{order} + {train_exog.shape[1]} exog features ...')
    model = SARIMAX(
        train_y,
        exog=train_exog,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    print(f'  AIC : {model.aic:.2f}')
    return model


# ── 6. Forecast ───────────────────────────────────────────────────────────────
def forecast_arima(model, n):
    return np.expm1(model.forecast(steps=n).values)

def forecast_sarimax(model, val_exog):
    return np.expm1(model.forecast(steps=len(val_exog), exog=val_exog).values)


# ── 7. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(actual, predicted, label='Model'):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    print(f'\n── {label} ──')
    print(f'  MAE  : {mae:>10,.2f}')
    print(f'  RMSE : {rmse:>10,.2f}')
    print(f'  MAPE : {mape:>9.2f}%')
    return {'Model': label, 'MAE': round(mae, 2),'RMSE': round(rmse, 2), 'MAPE%': round(mape, 2)}


# ── 8. Save model ─────────────────────────────────────────────────────────────
def save_model(model, filename='arima_model.pkl'):
    os.makedirs(MODEL_PATH, exist_ok=True)
    path = MODEL_PATH + filename
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nSaved → {path} ✅')


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_arima(store_id=STORE_ID):
    print('=' * 54)
    print('  ARIMA / SARIMAX PIPELINE')
    print('=' * 54)

    store_df = load_store_data(store_id)
    train_y, val_y, train_exog, val_exog = prepare_data(store_df)

    # Stationarity
    print()
    is_stat = check_stationarity(train_y, 'log(Sales)')
    if not is_stat:
        check_stationarity(train_y.diff().dropna(), '1st Difference')

    actual = np.expm1(val_y.values)

    # Fit models
    arima_model   = fit_arima(train_y)
    sarimax_model = fit_sarimax(train_y, train_exog)

    # Forecast
    arima_pred   = forecast_arima(arima_model, len(val_y))
    sarimax_pred = forecast_sarimax(sarimax_model, val_exog)
    naive_pred   = np.full(len(val_y), np.expm1(train_y.iloc[-1]))

    # Evaluate
    print('\n── Evaluation (original Sales scale) ──')
    r1 = evaluate(actual, arima_pred,   f'ARIMA{ORDER}         [no exog]')
    r2 = evaluate(actual, sarimax_pred, f'SARIMAX{ORDER}+exog  [improved]')
    r3 = evaluate(actual, naive_pred,    'Naive Baseline')

    # Improvement summary
    mae_gain  = (r1['MAE']  - r2['MAE'])  / r1['MAE']  * 100
    rmse_gain = (r1['RMSE'] - r2['RMSE']) / r1['RMSE'] * 100
    print(f'\n── ARIMA → SARIMAX Improvement ──')
    print(f'  MAE  improved by : {mae_gain:+.1f}%')
    print(f'  RMSE improved by : {rmse_gain:+.1f}%')

    # Summary table
    results_df = pd.DataFrame([r1, r2, r3])
    print('\n── Summary ──')
    print(results_df.to_string(index=False))

    # Save SARIMAX (best model)
    save_model(sarimax_model, 'arima_model.pkl')

    print('\n' + '=' * 54)
    print('  Pipeline complete ✅')
    print('=' * 54)
    return sarimax_model, results_df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_arima()
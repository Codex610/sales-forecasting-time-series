# Rossmann Store Sales Forecasting

End-to-end sales forecasting pipeline for 1,115 Rossmann stores using SARIMAX, Random Forest, and XGBoost.

## Results

| Model | Scope | MAE | RMSE |
|---|---|---|---|
| SARIMAX(2,1,2)+exog | Store 1 | **489.65** | 653.29 |
| Random Forest | All 1,115 stores | 736.83 | 1,098.15 |
| **XGBoost** | **All 1,115 stores** | **644.67** | **933.08** |

> XGBoost is the best model at scale. SARIMAX outperforms ML models on a single store when given the right exogenous features.

---

## Project Structure

```
rossmann-forecasting/
│
├── data/
│   ├── raw/                    # train.csv, store.csv (from Kaggle)
│   └── processed/              # featured_sales_data.csv, train/val splits
│
├── models/
│   ├── arima_model.pkl         # SARIMAXResultsWrapper (Store 1)
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_arima_model.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_model_comparison.ipynb
│
├── reports/
│   ├── figures/                # all saved plots
│   ├── final_report.pdf
│   ├── arima_results.csv
│   ├── ml_results.csv
│   └── final_comparison.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── arima_model.py
│   ├── ml_models.py
│   ├── evaluation.py
│   ├── utils.py
│   └── main.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data from Kaggle and place in data/raw/
#    https://www.kaggle.com/competitions/rossmann-store-sales/data

# 3. Run full pipeline
python src/main.py

# 4. Or run a single step
python src/main.py --step preprocess
python src/main.py --step features
python src/main.py --step arima
python src/main.py --step ml
```

---

## Pipeline Steps

### 1. Data Preprocessing (`data_preprocessing.py`)
- Merges `train.csv` + `store.csv`
- Handles missing values (CompetitionDistance, Promo2 dates)
- Removes closed-store rows (Sales=0, Open=0)
- Creates `Sales_log = log1p(Sales)` target

### 2. Feature Engineering (`feature_engineering.py`)
- **Lag features**: Sales_lag_1, Sales_lag_7, Sales_lag_14, Sales_lag_28
- **Rolling stats**: roll_mean_7, roll_mean_28, roll_std_7, ewm_7
- **Calendar**: DayOfWeek dummies, IsMonday, IsSaturday, IsMonthEnd, IsQ4
- **Holiday flags**: IsPublicHoliday, IsChristmas, BeforeHoliday, AfterHoliday
- **Interaction terms**: Promo_x_Monday, Promo_x_SchoolHol
- **Fourier terms**: weekly (period=7) and annual (period=365) sin/cos pairs
- **Store aggregates**: Store_SalesMean, Store_CustomersMean
- **Total**: 94 features

### 3. SARIMAX Model (`arima_model.py`)
- Fits on Store 1 only (illustrative — not scalable)
- Order: (2,1,2) — chosen via ACF/PACF
- 16 exogenous features passed explicitly
- ADF test confirms stationarity after 1st differencing
- Saves `SARIMAXResultsWrapper` → `models/arima_model.pkl`

> ⚠️ When loading this model elsewhere, do NOT call `.forecast()` directly.
> Refit on the exact train split to avoid `ValueError: end must be after start`.

### 4. ML Models (`ml_models.py`)
- Random Forest: 200 trees, max_depth=10
- XGBoost: 500 rounds, lr=0.05, max_depth=6
- Both trained on all 1,115 stores with 94 features

### 5. Evaluation (`evaluation.py`)
- `compute_metrics(actual, pred, label)` → MAE, RMSE, MAPE
- `per_store_mae(val_df, actual_col, pred_col)` → per-store breakdown
- `print_summary(results_df)` → formatted summary box

### 6. Utilities (`utils.py`)
- `plot_forecast()` — actual vs predicted line chart
- `plot_metrics_bar()` — MAE/RMSE bar chart
- `plot_error_distributions()` — residual histograms
- `plot_feature_importance()` — horizontal bar charts
- `plot_scatter()` — actual vs predicted scatter
- `plot_error_by_period()` — MAE by day/month
- `plot_per_store_mae()` — per-store scatter + hardest stores

---

## Key Findings

- **Lag features** (Sales_lag_1, Sales_lag_7) are the strongest predictors — strong autocorrelation in daily sales
- **Promo** is the most important categorical feature — confirmed in EDA, SARIMAX, and ML
- **SARIMAX >> plain ARIMA** — adding 16 exogenous features reduces MAE by ~30%
- **December and Monday** have the highest prediction error across all models
- **XGBoost edges out Random Forest** by ~12% MAE — better at capturing feature interactions

---

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) — Kaggle competition dataset.

- 1,115 stores across Germany
- Daily sales from Jan 2013 – Jul 2015
- Features: store type, assortment, promotions, competition, school/public holidays

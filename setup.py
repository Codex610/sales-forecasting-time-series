import os

# List of files to create
files = [
    "data/raw/sales_data.csv",
    "data/processed/cleaned_sales_data.csv",
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_arima_model.ipynb",
    "notebooks/04_ml_models.ipynb",
    "notebooks/05_model_comparison.ipynb",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/arima_model.py",
    "src/ml_models.py",
    "src/evaluation.py",
    "src/utils.py",
    "models/arima_model.pkl",
    "models/random_forest.pkl",
    "models/xgboost.pkl",
    "reports/figures/trend_plot.png",
    "reports/figures/seasonality_plot.png",
    "reports/figures/forecast_plot.png",
    "reports/final_report.pdf",
    "app/app.py",
    "app/requirements.txt",
    "requirements.txt",
    "config.yaml",
    "README.md",
    "main.py"
]

for file in files:
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # Create empty file if it doesn't exist
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass

print("✅ Folder structure generated inside existing base folder!")
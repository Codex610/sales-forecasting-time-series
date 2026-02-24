"""
utils.py
--------
Shared plotting helpers and small utilities used across notebooks and scripts.
All functions are self-contained — just pass in arrays and get a plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Colours (consistent across all notebooks) ─────────────────────────────────
COLORS = {
    'sarimax': 'seagreen',
    'rf'     : 'steelblue',
    'xgb'    : 'tomato',
    'actual' : 'black',
    'train'  : 'lightsteelblue',
}


# ── 1. Forecast line plot ──────────────────────────────────────────────────────
def plot_forecast(dates, actuals, preds: dict, title='Forecast', save_path=None):
    """
    Plot actual vs one or more model predictions over time.

    preds : {'Model Name': pred_array, ...}
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(dates, actuals, color=COLORS['actual'], linewidth=1.5, label='Actual')

    model_colors = [COLORS['sarimax'], COLORS['rf'], COLORS['xgb'],
                    'purple', 'darkorange']
    for (label, pred), color in zip(preds.items(), model_colors):
        ax.plot(dates, pred, color=color, linewidth=1.1,
                linestyle='--', label=label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 2. MAE / RMSE bar chart ────────────────────────────────────────────────────
def plot_metrics_bar(results_df, title='Model Comparison', save_path=None):
    """
    Bar chart of MAE and RMSE side by side.
    results_df must have columns: Model, MAE, RMSE
    """
    colors = [COLORS['rf'], COLORS['xgb'], COLORS['sarimax'],
              'purple', 'darkorange'][:len(results_df)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ['MAE', 'RMSE']):
        bars = ax.bar(results_df['Model'], results_df[metric],
                      color=colors, edgecolor='white', width=0.5)
        ax.set_title(f'{metric} — Lower is Better')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=12)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    f'{val:,.0f}', ha='center', fontweight='bold', fontsize=10)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 3. Error distribution histograms ──────────────────────────────────────────
def plot_error_distributions(errors: dict, title='Error Distributions', save_path=None):
    """
    errors : {'Model Name': residuals_array, ...}
    """
    palette = [COLORS['sarimax'], COLORS['rf'], COLORS['xgb'],
               'purple', 'darkorange']
    n = len(errors)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, err), color in zip(axes, errors.items(), palette):
        ax.hist(err, bins=40, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(0,          color='black', linestyle='--', linewidth=1.2)
        ax.axvline(err.mean(), color='red',   linestyle='-',  linewidth=1.5,
                   label=f'Mean = {err.mean():.0f}')
        ax.set_title(label)
        ax.set_xlabel('Actual − Predicted')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 4. Feature importance bar chart ───────────────────────────────────────────
def plot_feature_importance(importances: dict, top_n=20, save_path=None):
    """
    importances : {'Model Name': pd.Series(importance, index=feature_names), ...}
    """
    colors = [COLORS['rf'], COLORS['xgb'], 'purple', 'darkorange']
    n      = len(importances)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (label, imp), color in zip(axes, importances.items(), colors):
        imp.nlargest(top_n).sort_values().plot(kind='barh', ax=ax, color=color)
        ax.set_title(f'{label} — Top {top_n} Features')
        ax.set_xlabel('Importance')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 5. Actual vs Predicted scatter ────────────────────────────────────────────
def plot_scatter(actual, predicted, label='Model', sample=20_000, save_path=None):
    """Scatter of actual vs predicted (sampled for speed)."""
    idx = np.random.choice(len(actual), size=min(sample, len(actual)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual[idx], predicted[idx], alpha=0.15, s=5, color=COLORS['xgb'])
    m = max(actual.max(), predicted.max())
    ax.plot([0, m], [0, m], 'k--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Sales')
    ax.set_ylabel('Predicted Sales')
    ax.set_title(f'{label} — Actual vs Predicted')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 6. MAE by time period ──────────────────────────────────────────────────────
def plot_error_by_period(val_df, abs_err_col, save_path=None):
    """Bar charts of mean absolute error by day-of-week and month."""
    dow_map   = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

    dow_mae   = val_df.groupby(val_df['Date'].dt.dayofweek)[abs_err_col].mean().rename(index=dow_map)
    month_mae = val_df.groupby(val_df['Date'].dt.month)[abs_err_col].mean().rename(index=month_map)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    dow_mae.plot(  kind='bar', ax=axes[0], color=COLORS['xgb'],   edgecolor='white', rot=0)
    axes[0].set_title('MAE by Day of Week')
    axes[0].set_ylabel('Mean Absolute Error')
    month_mae.plot(kind='bar', ax=axes[1], color=COLORS['rf'],    edgecolor='white', rot=0)
    axes[1].set_title('MAE by Month')
    axes[1].set_ylabel('Mean Absolute Error')

    plt.suptitle('Error Patterns by Time Period', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── 7. Per-store MAE scatter ───────────────────────────────────────────────────
def plot_per_store_mae(store_errors, x_col='RF_MAE', y_col='XGB_MAE', save_path=None):
    """Scatter of per-store MAE for two models + top 10 hardest stores."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].scatter(store_errors[x_col], store_errors[y_col],
                    alpha=0.5, s=30, color=COLORS['rf'], edgecolors='white')
    m = max(store_errors[x_col].max(), store_errors[y_col].max())
    axes[0].plot([0, m], [0, m], 'r--', linewidth=1.2, label='Equal performance')
    better = (store_errors[y_col] < store_errors[x_col]).sum()
    axes[0].set_title(f'Per-Store MAE\n(XGBoost better on {better}/{len(store_errors)} stores)')
    axes[0].set_xlabel(x_col.replace('_', ' '))
    axes[0].set_ylabel(y_col.replace('_', ' '))
    axes[0].legend()

    hard = store_errors.sort_values(y_col, ascending=False).head(10)
    axes[1].barh(hard['Store'].astype(str), hard[y_col], color=COLORS['xgb'])
    axes[1].set_title('Top 10 Hardest Stores (XGBoost)')
    axes[1].set_xlabel('MAE')
    axes[1].invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

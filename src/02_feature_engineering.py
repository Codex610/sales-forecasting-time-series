"""
feature_engineering.py
"""

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH = 'data/processed/'

# ── Config ────────────────────────────────────────────────────────────────────
LAG_DAYS     = [1, 2, 3, 7, 14, 21, 28]
ROLL_WINDOWS = [7, 14, 28]
SPLIT_RATIO  = 0.80


# ── 1. Load cleaned data ──────────────────────────────────────────────────────
def load_cleaned():
    path = PROCESSED_PATH + 'cleaned_sales_data.csv'
    print(f'Loading {path} ...')
    df = pd.read_csv(path, parse_dates=['Date'], low_memory=False)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    print(f'  Shape : {df.shape}')
    return df


# ── 2. Target transform ───────────────────────────────────────────────────────
def add_target(df):
    """log1p(Sales) — reduces right-skew confirmed in EDA."""
    df['Sales_log'] = np.log1p(df['Sales'])
    return df


# ── 3. Date / calendar features ───────────────────────────────────────────────
def add_date_features(df):
    """Year, Month, Day, flags for key days found in EDA."""
    df['Year']         = df['Date'].dt.year
    df['Month']        = df['Date'].dt.month
    df['Day']          = df['Date'].dt.day
    df['DayOfWeek']    = df['Date'].dt.dayofweek       # 0=Mon … 6=Sun
    df['DayOfYear']    = df['Date'].dt.dayofyear
    df['WeekOfYear']   = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter']      = df['Date'].dt.quarter

    # Boolean flags
    df['IsWeekend']    = (df['DayOfWeek'] >= 5).astype(int)
    df['IsMonday']     = (df['DayOfWeek'] == 0).astype(int)  # EDA: Monday peak
    df['IsSaturday']   = (df['DayOfWeek'] == 5).astype(int)  # EDA: Saturday low
    df['IsMonthStart'] = (df['Day'] <= 5).astype(int)
    df['IsMonthEnd']   = (df['Day'] >= 25).astype(int)
    df['IsDecember']   = (df['Month'] == 12).astype(int)     # Christmas peak
    df['IsQ4']         = (df['Quarter'] == 4).astype(int)

    # Days to year-end (Christmas proximity)
    df['DaysToYearEnd'] = df['Date'].apply(lambda d: (pd.Timestamp(d.year, 12, 31) - d).days)

    # Cyclical encodings
    df['Month_sin']       = np.sin(2 * np.pi * df['Month']     / 12)
    df['Month_cos']       = np.cos(2 * np.pi * df['Month']     / 12)
    df['WeekOfYear_sin']  = np.sin(2 * np.pi * df['WeekOfYear']/ 52)
    df['WeekOfYear_cos']  = np.cos(2 * np.pi * df['WeekOfYear']/ 52)

    print('  Date features          ✅')
    return df


# ── 4. Fourier terms ──────────────────────────────────────────────────────────
def add_fourier_terms(df):
    """Sin/cos pairs to capture weekly and annual seasonality explicitly."""

    def fourier(df, period, n_terms, col, prefix):
        for k in range(1, n_terms + 1):
            df[f'{prefix}_sin_{period}_{k}'] = np.sin(2 * np.pi * k * df[col] / period)
            df[f'{prefix}_cos_{period}_{k}'] = np.cos(2 * np.pi * k * df[col] / period)
        return df

    df = fourier(df, period=7,   n_terms=3, col='DayOfWeek', prefix='fourier_weekly')
    df = fourier(df, period=365, n_terms=3, col='DayOfYear',  prefix='fourier_annual')

    print('  Fourier terms          ✅')
    return df


# ── 5. Lag features ───────────────────────────────────────────────────────────
def add_lag_features(df):
    """Per-store lags — no cross-store leakage."""
    for lag in LAG_DAYS:
        df[f'Sales_lag_{lag}']    = df.groupby('Store')['Sales'].shift(lag)
        df[f'LogSales_lag_{lag}'] = df.groupby('Store')['Sales_log'].shift(lag)

    print(f'  Lag features ({len(LAG_DAYS)*2})      ✅')
    return df


# ── 6. Rolling statistics ─────────────────────────────────────────────────────
def add_rolling_features(df):
    """
    Rolling mean, std, max, min computed on the SHIFTED series
    (shift by 1 day) to prevent look-ahead leakage.
    """
    for w in ROLL_WINDOWS:
        shifted = df.groupby('Store')['Sales'].shift(1)
        grp     = shifted.groupby(df['Store'])
        df[f'Sales_roll_mean_{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'Sales_roll_std_{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f'Sales_roll_max_{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f'Sales_roll_min_{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).min())

    # Exponentially weighted mean
    df['Sales_ewm_7']  = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).ewm(span=7).mean())
    df['Sales_ewm_28'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).ewm(span=28).mean())

    print(f'  Rolling features ({len(ROLL_WINDOWS)*4+2})    ✅')
    return df


# ── 7. Competition features ───────────────────────────────────────────────────
def add_competition_features(df):
    """log-transform distance (EDA: non-linear effect) + months open."""
    df['CompDist_log']    = np.log1p(df['CompetitionDistance'])
    df['CompDist_WasNull']= (df['CompetitionDistance'] == df['CompetitionDistance'].median()).astype(int)

    df['CompOpen_Months'] = np.where(
        df['CompetitionOpenSinceYear'] > 0,
        (df['Year']  - df['CompetitionOpenSinceYear']) * 12 +
        (df['Month'] - df['CompetitionOpenSinceMonth']),
        -1
    )
    df['CompOpen_Months'] = df['CompOpen_Months'].clip(lower=-1)
    df['HasCompetitor']   = (df['CompetitionOpenSinceYear'] > 0).astype(int)

    print('  Competition features   ✅')
    return df


# ── 8. Promo2 features ────────────────────────────────────────────────────────
def add_promo2_features(df):
    """Whether Promo2 is active for a given store-month combination."""

    promo_month_map = {
        'Jan,Apr,Jul,Oct' : [1, 4, 7, 10],
        'Feb,May,Aug,Nov' : [2, 5, 8, 11],
        'Mar,Jun,Sept,Dec': [3, 6, 9, 12],
        'None'            : [],
    }

    def is_active(row):
        if row['Promo2'] == 0:
            return 0
        months = promo_month_map.get(str(row['PromoInterval']), [])
        if row['Month'] in months and row['Promo2SinceYear'] > 0:
            start = pd.Timestamp(int(row['Promo2SinceYear']), 1, 1)
            if row['Date'] >= start:
                return 1
        return 0

    df['IsPromo2Active'] = df.apply(is_active, axis=1)

    # Only compute weeks for rows where Promo2SinceYear is valid (> 0)
    # to avoid parsing "0-01-01" as a date which causes ValueError
    has_promo2 = (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0)

    df['WeeksSincePromo2'] = -1  # default: not participating

    if has_promo2.any():
        promo2_rows = df[has_promo2].copy()
        promo2_start = pd.to_datetime(promo2_rows['Promo2SinceYear'].astype(int).astype(str) + '-01-01',format='%Y-%m-%d')
        weeks = (promo2_rows['Date'] - promo2_start).dt.days // 7 \
                - promo2_rows['Promo2SinceWeek']
        df.loc[has_promo2, 'WeeksSincePromo2'] = weeks.values

    df['WeeksSincePromo2'] = df['WeeksSincePromo2'].clip(lower=-1)

    print('  Promo2 features        ✅')
    return df


# ── 9. Holiday features ───────────────────────────────────────────────────────
def add_holiday_features(df):
    """Encode state holidays and add before/after holiday flags."""
    df['StateHoliday_enc'] = df['StateHoliday'].astype(str).map(
        {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    ).fillna(0).astype(int)

    df['IsPublicHoliday'] = (df['StateHoliday_enc'] == 1).astype(int)
    df['IsEasterHoliday'] = (df['StateHoliday_enc'] == 2).astype(int)
    df['IsChristmas']     = (df['StateHoliday_enc'] == 3).astype(int)
    df['IsAnyHoliday']    = (df['StateHoliday_enc'] > 0).astype(int)

    # Day before and after any holiday (±1 window)
    df['BeforeHoliday'] = df.groupby('Store')['IsAnyHoliday'].shift(-1).fillna(0).astype(int)
    df['AfterHoliday']  = df.groupby('Store')['IsAnyHoliday'].shift(1).fillna(0).astype(int)

    print('  Holiday features       ✅')
    return df


# ── 10. Interaction features ──────────────────────────────────────────────────
def add_interaction_features(df):
    """Multiplicative interactions between strong predictors."""
    df['Promo_x_Monday']    = df['Promo'] * df['IsMonday']
    df['Promo_x_Weekend']   = df['Promo'] * df['IsWeekend']
    df['Promo_x_SchoolHol'] = df['Promo'] * df['SchoolHoliday']
    df['Promo_x_Promo2']    = df['Promo'] * df['IsPromo2Active']
    df['Q4_x_Promo']        = df['IsQ4']  * df['Promo']
    df['Holiday_x_Weekend'] = df['IsAnyHoliday'] * df['IsWeekend']
    df['Assort_x_CompDist'] = df['Assortment_enc'] * df['CompDist_log']

    print('  Interaction features   ✅')
    return df


# ── 11. Categorical encoding ──────────────────────────────────────────────────
def add_categorical_features(df):
    """One-hot StoreType, ordinal Assortment, one-hot DayOfWeek."""

    # StoreType → one-hot
    df = pd.get_dummies(df, columns=['StoreType'], prefix='StoreType', drop_first=False)

    # Assortment → ordinal rank (from EDA: b > c > a)
    df['Assortment_enc'] = df['Assortment'].map({'a': 1, 'b': 3, 'c': 2}).fillna(1).astype(int)

    # DayOfWeek → one-hot
    df = pd.get_dummies(df, columns=['DayOfWeek'], prefix='DayOfWeek', drop_first=False)

    print('  Categorical encoding   ✅')
    return df


# ── 12. Store aggregate features ─────────────────────────────────────────────
def add_store_aggregates(df):
    """Per-store historical statistics as features (store-level baseline)."""
    store_stats = df.groupby('Store')['Sales'].agg(
        Store_SalesMean   ='mean',
        Store_SalesMedian ='median',
        Store_SalesStd    ='std',
        Store_SalesMax    ='max',
        Store_SalesMin    ='min',
    ).reset_index()

    store_promo = df.groupby('Store')['Promo'].mean().reset_index()\
                    .rename(columns={'Promo': 'Store_PromoRate'})

    df = df.merge(store_stats, on='Store', how='left')
    df = df.merge(store_promo, on='Store', how='left')

    print('  Store aggregates       ✅')
    return df


# ── 13. Build final feature list ──────────────────────────────────────────────
def get_feature_cols(df):
    """Return list of all feature columns (excludes identifiers and targets)."""
    exclude = {'Store', 'Date', 'Sales', 'Sales_log',
               'StateHoliday', 'PromoInterval', 'Assortment',
               'Open', 'Customers'}
    feature_cols = [
        c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, np.uint8, bool]]
    return feature_cols


# ── 14. Train / val split ─────────────────────────────────────────────────────
def split_data(df):
    """Strict time-based split — never random for time series."""
    split_date  = df['Date'].quantile(SPLIT_RATIO)
    split_date  = pd.Timestamp(split_date)

    train_df    = df[df['Date'] <= split_date].copy()
    val_df      = df[df['Date']  > split_date].copy()

    print(f'\n  Split date : {split_date.date()}')
    print(f'  Train      : {len(train_df):,} rows')
    print(f'  Val        : {len(val_df):,} rows')
    return train_df, val_df


# ── 15. Save outputs ──────────────────────────────────────────────────────────
def save_outputs(df, train_df, val_df, feature_cols):
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    df.to_csv(PROCESSED_PATH + 'featured_sales_data.csv', index=False)
    train_df.to_csv(PROCESSED_PATH + 'train_featured.csv',    index=False)
    val_df.to_csv(PROCESSED_PATH + 'val_featured.csv',        index=False)

    pd.Series(feature_cols).to_csv(
        PROCESSED_PATH + 'feature_list.csv', index=False, header=['feature'])

    print(f'\n  ✅ featured_sales_data.csv  ({df.shape})')
    print(f'  ✅ train_featured.csv        ({train_df.shape})')
    print(f'  ✅ val_featured.csv          ({val_df.shape})')
    print(f'  ✅ feature_list.csv          ({len(feature_cols)} features)')


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_feature_engineering():
    print('='*52)
    print('  FEATURE ENGINEERING PIPELINE')
    print('='*52)

    df = load_cleaned()

    print('\nEngineering features...')
    df = add_target(df)
    df = add_date_features(df)
    df = add_fourier_terms(df)
    df = add_categorical_features(df)   # must run before interactions
    df = add_competition_features(df)
    df = add_promo2_features(df)
    df = add_holiday_features(df)
    df = add_interaction_features(df)   # must run after promo2 + holiday
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_store_aggregates(df)

    feature_cols = get_feature_cols(df)
    print(f'\n  Total features : {len(feature_cols)}')

    print('\nSplitting data...')
    train_df, val_df = split_data(df)

    print('\nSaving outputs...')
    save_outputs(df, train_df, val_df, feature_cols)

    print('\n' + '='*52)
    print('  Feature engineering complete ✅')
    print('='*52)
    return df, train_df, val_df, feature_cols


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_feature_engineering()
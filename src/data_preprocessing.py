import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH       = 'data/raw/'
PROCESSED_PATH = 'data/processed/'


# ── 1. Load raw files ─────────────────────────────────────────────────────────
def load_raw_data():
    """Load train.csv and store.csv from data/raw/"""

    print('Loading raw data...')
    train = pd.read_csv(RAW_PATH + 'train.csv',parse_dates=['Date'], low_memory=False)
    store = pd.read_csv(RAW_PATH + 'store.csv')

    print(f'  train.csv : {train.shape}')
    print(f'  store.csv : {store.shape}')
    return train, store


# ── 2. Clean train.csv ────────────────────────────────────────────────────────
def clean_train(train):
    """
    Basic cleaning on the train dataframe:
    - Remove rows where store is closed (Open == 0)
    - Remove rows with zero or negative sales
    - Drop duplicates
    """

    print('\nCleaning train data...')
    before = len(train)

    # Keep only open stores with positive sales
    train = train[(train['Open'] == 1) & (train['Sales'] > 0)].copy()
    train = train.drop_duplicates()

    after = len(train)
    print(f'  Rows before : {before:,}')
    print(f'  Rows after  : {after:,}  (removed {before - after:,} rows)')
    return train


# ── 3. Clean store.csv ────────────────────────────────────────────────────────
def clean_store(store):
    """
    Impute missing values in store.csv:
    - CompetitionDistance     → median fill
    - CompetitionOpenSince*   → 0 (no competitor recorded)
    - Promo2Since*            → 0 (not participating)
    - PromoInterval           → 'None'
    """

    print('\nCleaning store data...')

    # Missing before
    missing = store.isnull().sum()
    missing = missing[missing > 0]
    print(f'  Missing before imputation:\n{missing.to_string()}')

    # Impute
    store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

    store['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    store['CompetitionOpenSinceYear'].fillna(0, inplace=True)

    store['Promo2SinceWeek'].fillna(0, inplace=True)
    store['Promo2SinceYear'].fillna(0, inplace=True)

    store['PromoInterval'].fillna('None', inplace=True)

    remaining = store.isnull().sum().sum()
    print(f'  Missing after  imputation: {remaining}')
    return store


# ── 4. Merge train + store ────────────────────────────────────────────────────
def merge_data(train, store):
    """Left-join train with store on Store column."""

    print('\nMerging train + store...')
    df = train.merge(store, on='Store', how='left')
    print(f'  Merged shape : {df.shape}')
    return df


# ── 5. Basic type fixes ───────────────────────────────────────────────────────
def fix_dtypes(df):
    """
    Ensure correct dtypes:
    - StateHoliday : string  (Kaggle stores it as mixed 0 / 'a' / 'b' / 'c')
    - Open, Promo  : int
    - Date         : datetime (already parsed)
    """

    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['Open']         = df['Open'].astype(int)
    df['Promo']        = df['Promo'].astype(int)
    df['SchoolHoliday']= df['SchoolHoliday'].astype(int)

    # Cast store-level int columns that got float due to NaN filling
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek', 'Promo2SinceYear']:
        df[col] = df[col].astype(int)
    return df


# ── 6. Sort ───────────────────────────────────────────────────────────────────
def sort_data(df):
    """Sort by Store then Date — required for correct lag computation later."""
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    return df


# ── 7. Save ───────────────────────────────────────────────────────────────────
def save_cleaned(df):
    """Save cleaned dataframe to data/processed/"""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    out = PROCESSED_PATH + 'cleaned_sales_data.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved → {out}')
    return out


# ── 8. Summary printout ───────────────────────────────────────────────────────
def print_summary(df):
    print('\n── Cleaned Data Summary ──────────────────────────────')
    print(f'  Shape         : {df.shape}')
    print(f'  Date range    : {df.Date.min().date()} → {df.Date.max().date()}')
    print(f'  Unique stores : {df.Store.nunique()}')
    print(f'  Sales min/max : {df.Sales.min():,} / {df.Sales.max():,}')
    print(f'  Null values   : {df.isnull().sum().sum()}')
    print('──────────────────────────────────────────────────────')


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing():
    train, store = load_raw_data()
    train        = clean_train(train)
    store        = clean_store(store)
    df           = merge_data(train, store)
    df           = fix_dtypes(df)
    df           = sort_data(df)
    print_summary(df)
    save_cleaned(df)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_preprocessing()
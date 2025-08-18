import os
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configuration
INDEX = 'nifty_50'
INTERVAL = 'day'
SPLIT_DATE = '2025-06-15'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data' / 'historical'
PLOTS_DIR = BASE_DIR / 'plots' / 'combined' / TIMESTAMP
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load raw data and scalers
with open(DATA_DIR / f"{INDEX}_{INTERVAL}.pkl", 'rb') as f:
    payload = dill.load(f)
raw_df = payload['data'].rename(columns={'date':'ds','symbol':'unique_id','close':'y'})
raw_df['ds'] = pd.to_datetime(raw_df['ds']).dt.tz_localize(None)
scalers = payload.get('scalers', {})

# 2. Build train/test index per symbol
def get_test_idx(df, split_date):
    tests = {}
    for uid, grp in df.groupby('unique_id'):
        idx = pd.to_datetime(grp['ds'])
        tests[uid] = idx[idx >= split_date]
    return tests

test_idx = get_test_idx(raw_df, SPLIT_DATE)

# 3. Load baseline predictions and metrics
# Baseline predictions are saved as plots; we need to regenerate forecasts
from .base_models import load_data, train_test_split, naive_forecast, rolling_mean_forecast, arima_forecast

df, scalers = load_data(INDEX, INTERVAL)
trains, tests = train_test_split(df, SPLIT_DATE)
methods = {
    'naive': naive_forecast,
    'rolling_mean': lambda tr,h: rolling_mean_forecast(tr,h, window=5),
    'arima': arima_forecast
}

# 4. Load N-BeatsX predictions (assumes you have saved them to CSV)
# CSV should have columns: unique_id, ds, y_pred (and optionally y_lower/y_upper)
nb_file = BASE_DIR / 'nbeatsx_results' / 'nbeatsx_predictions.csv'
nb_df = pd.read_csv(nb_file, parse_dates=['ds'])
nb_df['model'] = 'nbeatsx'
nb_df.rename(columns={'y_pred':'prediction'}, inplace=True)

# 5. Generate baseline predictions into a unified DataFrame
records = []
for model_name, func in methods.items():
    for uid, train in trains.items():
        test = tests[uid]
        h = len(test)
        if h == 0:
            continue
        # inverse scale train/test
        scaler = scalers.get(f"{uid}_y")
        test_raw = pd.Series(
            scaler.inverse_transform(test.values.reshape(-1,1)).flatten(),
            index=test.index
        ) if scaler else test
        preds = func(test_raw.iloc[:0].append(test_raw.shift(1).dropna()), h)
        preds.index = test.index
        for ds, y_pred in preds.items():
            records.append({'unique_id': uid,
                            'ds': ds,
                            'model': model_name,
                            'prediction': y_pred})

base_preds = pd.DataFrame(records)

# 6. Extract actuals for test period
actuals = raw_df[raw_df['ds'] >= SPLIT_DATE][['unique_id','ds','y']].rename(columns={'y':'actual'})

# 7. Combine all into one CSV
combined = pd.concat([
    actuals,
    base_preds.drop(columns=[]).merge(actuals, on=['unique_id','ds']),
    nb_df[['unique_id','ds','model','prediction']].merge(actuals, on=['unique_id','ds'])
], sort=False)
combined.to_csv(BASE_DIR / 'combined_results' / f'combined_{TIMESTAMP}.csv', index=False)

# 8. Plot overlays for a few stocks
example_ids = combined['unique_id'].unique()[:5]
for uid in example_ids:
    df_uid = combined[combined['unique_id']==uid]
    plt.figure(figsize=(12,6))
    plt.plot(df_uid['ds'], df_uid['actual'], label='Actual', marker='o')
    for model in df_uid['model'].unique():
        dfm = df_uid[df_uid['model']==model]
        plt.plot(dfm['ds'], dfm['prediction'], label=model, marker='x')
    plt.title(f'Forecast comparison for {uid}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{uid}_comparison.png', dpi=300)
    plt.close()

print(f"Combined CSV and comparison plots saved under {BASE_DIR}/combined_results and {PLOTS_DIR}")

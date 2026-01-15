import os
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from datetime import datetime


def load_data(index: str, interval: str, data_dir: Path = Path("data/historical/")):
    """
    Load the preprocessed data for the given index and interval.
    Returns:
        df: DataFrame with ['ds','unique_id','y']
        scalers: dict of fitted scalers (if available in payload)
    """
    file_path = data_dir / f"{index}_{interval}.pkl"
    with open(file_path, 'rb') as f:
        payload = dill.load(f)
    df = payload['data'].dropna().copy()
    # rename to standard
    df = df.rename(columns={
        'date': 'ds',
        'symbol': 'unique_id',
        'close': 'y'
    })
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    scalers = payload.get('scalers', {})
    return df, scalers


def train_test_split(df: pd.DataFrame, split_date: str):
    """
    Split each series into train/test.
    Returns train/test dicts of pd.Series indexed by datetime.
    """
    trains, tests = {}, {}
    df = df.sort_values(['unique_id', 'ds'])
    for uid, grp in df.groupby('unique_id'):
        ts = grp.set_index('ds')['y']
        trains[uid] = ts[ts.index < split_date]
        tests[uid] = ts[ts.index >= split_date]
    return trains, tests


def naive_forecast(train: pd.Series, h: int):
    if len(train) == 0 or h <= 0:
        return pd.Series([], dtype=float)
    last = train.iloc[-1]
    return pd.Series([last] * h, index=range(h), dtype=float)


def rolling_mean_forecast(train: pd.Series, h: int, window: int = 5):
    if len(train) == 0 or h <= 0:
        return pd.Series([], dtype=float)
    window = min(window, len(train))
    mu = train.iloc[-window:].mean()
    return pd.Series([mu] * h, index=range(h), dtype=float)


def arima_forecast(train: pd.Series, h: int):
    if len(train) == 0 or h <= 0:
        return pd.Series([], dtype=float)
    try:
        model = pm.auto_arima(
            train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        fc = model.predict(n_periods=h)
        fc = np.asarray(fc, dtype=float)
    except Exception as e:
        print(f"ARIMA failed for series {train.name if hasattr(train, 'name') else ''}: {e}")
        last = train.iloc[-1]
        fc = np.array([last] * h, dtype=float)
    return pd.Series(fc, index=range(h), dtype=float)


def evaluate_forecasts(trains, tests, methods):
    """Produce forecasts dict[model] -> dict[uid] -> pd.Series(preds)"""
    all_preds = {name: {} for name in methods}
    metrics = []
    for uid, train in trains.items():
        test = tests.get(uid)
        if test is None or len(test) == 0:
            continue
        h = len(test)
        for name, func in methods.items():
            preds = func(train, h)
            preds.index = test.index
            all_preds[name][uid] = preds
            mae = mean_absolute_error(test.values, preds.values)
            rmse = mean_squared_error(test.values, preds.values)
            metrics.append({'unique_id': uid,
                            'model': name,
                            'MAE': mae,
                            'RMSE': rmse})
    metrics_df = pd.DataFrame(metrics)
    return all_preds, metrics_df


def plot_series(df: pd.DataFrame, preds: pd.Series, scaler, output_dir: Path, uid: str, model_name: str):
    """
    Plot actual vs predicted series, unscaled if scaler provided.
    """
    dates = df['ds']
    y = df['y'].values
    y_pred = preds.values
    # unscale if needed
    if scaler is not None:
        y = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y, marker='o', label='Actual')
    plt.plot(dates, y_pred, marker='x', label='Prediction')
    plt.title(f"{model_name.upper()} Forecast for {uid}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # save
    out_file = output_dir / f"{uid}_forecast.png"
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    # Config
    INDEX = 'nifty_50'
    INTERVAL = 'day'
    SPLIT_DATE = '2025-06-15'

    # Load data + scalers
    df, scalers = load_data(INDEX, INTERVAL)
    trains, tests = train_test_split(df, SPLIT_DATE)

    # Methods
    methods = {
        'naive': naive_forecast,
        'rolling_mean': lambda tr, h: rolling_mean_forecast(tr, h, window=5),
        'arima': arima_forecast
    }

    # Forecast + metrics
    all_preds, metrics_df = evaluate_forecasts(trains, tests, methods)
    print("Metrics summary:")
    print(metrics_df.groupby('model')[['MAE','RMSE']].describe())

    # Plotting setup
    base_output = Path('plots')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for model_name, preds_dict in all_preds.items():
        model_dir = base_output / model_name / timestamp
        model_dir.mkdir(parents=True, exist_ok=True)
        for uid, preds in preds_dict.items():
            # subset df to test period for this uid
            df_uid = df[(df['unique_id'] == uid) & (df['ds'] >= SPLIT_DATE)].copy()
            # get scaler for this uid & target y
            scaler = scalers.get(f"{uid}_y", None)
            plot_series(df_uid, preds, scaler, model_dir, uid, model_name)

    print(f"Plots saved under {base_output}/<model_name>/{timestamp}/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate baseline forecasting models on historical data.")
    parser.add_argument('--index', type=str, default='nifty_50', help="Index to evaluate (default: 'nifty_50')")
    parser.add_argument('--cutoff_date', type=str, default='2025-05-01', help="Cutoff date for training data (default: '2025-01-01')")
    parser.add_argument('--interval', type=str, default='day', help="Data interval (default: 'day')")
    parser.add_argument('--target', type=str, default='close', help="Target column for predictions (default: 'close')")
    args = parser.parse_args()
    INDEX = args.index
    SPLIT_DATE = args.cutoff_date
    INTERVAL = args.interval
    TARGET = args.target
    logging.info(f"Evaluating {INDEX} with {INTERVAL} interval, split at {SPLIT_DATE}...")
    main()

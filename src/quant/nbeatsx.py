import os
import dill
import torch
from pathlib import Path
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from neuralforecast.neuralforecast import NeuralForecast
from neuralforecast.neuralforecast.models import NBEATSx
from neuralforecast.neuralforecast.losses.pytorch import DistributionLoss

from utils.fetch_data import load
from utils.logging import setup_logger

logger = setup_logger(__name__, log_subdir="nbeatsx", include_console=True)
logger.info("NBEATSx script initialized")

 
def shift_weekend_predictions(df: pd.DataFrame, date_col='ds') -> pd.DataFrame:
    """
    Shift any weekend dates to the next business day.
    """
    def adjust_date(ds):
        if ds.weekday() in (5, 6):  # Saturday=5, Sunday=6
            return ds + pd.offsets.BDay(1)
        return ds
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None).apply(adjust_date)
    return df


def plot_predictions_vs_actual(results_df, model_name="NBEATSx", hyperparams=None, title_suffix="", save_path=''):
    """
    For each stock, plots a daily line chart of true vs predictions values 
    and saves them in an organized folder structure.
    
    Instead of using low and high confidence intervals, this function aggregates 
    multiple predictions for the same date (using min, max, and median) and shades 
    the area between the minimum and maximum predictions values, with the median 
    predictions connected by a line.
    
    Args:
        results_df: DataFrame containing columns 'ds', 'unique_id', 'y', 'predictions'
        model_name: Name of the model used (default: "NBEATSx")
        hyperparams: Dictionary of hyperparameters used in the model
        title_suffix: Optional suffix for plot titles
        save_path: Base path to save plots (default: creates a timestamped folder)
    """

    # Create a timestamped folder if save_path is not provided
    if not save_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a folder name that includes model and key hyperparameters
        hyper_str = ""
        if hyperparams:
            hyper_str = "_" + "_".join([f"{k}-{v}" for k, v in hyperparams.items() 
                                         if k in ['input_size', 'window_size']])
        folder_name = f"plots/{model_name}/{hyper_str}_{timestamp}"
        save_path = folder_name
    logger.info(f"Saving plots to: {save_path}")
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    unique_stocks = results_df['unique_id'].unique()
    summary_stats = {}
    
    for stock in unique_stocks:
        stock_data = results_df[results_df['unique_id'] == stock].copy()
        stock_data['ds'] = pd.to_datetime(stock_data['ds'])
        
        # Calculate error metrics for this stock
        mae = mean_absolute_error(stock_data['y'], stock_data['predictions'])
        rmse = np.sqrt(mean_squared_error(stock_data['y'], stock_data['predictions']))
        
        # Store statistics
        summary_stats[stock] = {'MAE': mae, 'RMSE': rmse}
        
        # Create a stock-specific subfolder
        stock_folder = os.path.join(save_path, stock.replace('.', '_'))
        os.makedirs(stock_folder, exist_ok=True)
        
        # Line plot for true vs predictions values
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Plot actual values
        sns.lineplot(x='ds', y='y', data=stock_data, marker="o", 
                     label='y', linewidth=2, ax=ax)
        
        # Aggregate multiple predictions per date:
        # For each date, compute the min, max, and median predictions values
        pred_summary = stock_data.groupby(by='ds').agg(
            min_pred=('predictions', 'min'),
            max_pred=('predictions', 'max'),
            median_pred=('predictions', 'median')
        ).reset_index()
        
        # Shade the area between the minimum and maximum predictions
        ax.fill_between(pred_summary['ds'], 
                        pred_summary['min_pred'], 
                        pred_summary['max_pred'],
                        alpha=0.2, color='orange', label='Prediction range')
        
        # Plot the median predictions as a line
        sns.lineplot(x='ds', y='median_pred', data=pred_summary, marker="x", 
                     label='Median Prediction', color='orange', linewidth=2, ax=ax)
        
        plt.title(f"Actual vs predictions for {stock} {title_suffix}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}")
        plt.xlabel("Date")
        plt.ylabel("Percent Change")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the line plot
        line_plot_path = os.path.join(stock_folder, f"{stock}_line_plot.png")
        plt.savefig(line_plot_path, dpi=300)
        plt.close()
        
        # Create a distribution plot of errors
        plt.figure(figsize=(10, 6))
        logger.info(f'stock: {stock} with Columns: {stock_data.columns}')
        stock_data['Error'] = stock_data['predictions'] - stock_data['y']
        ax = sns.barplot(x='ds', y='Error', data=stock_data, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        
        # Simplify x-axis labels
        dates = stock_data['ds'].dt.strftime('%m-%d').values
        step = max(1, len(dates) // 10)  # Show at most ~10 dates
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels(dates[::step], rotation=45)
        
        plt.title(f"Prediction Error for {stock} {title_suffix}")
        plt.xlabel("Date")
        plt.ylabel("Error (predictions - Actual)")
        plt.tight_layout()
        
        # Save the distribution plot
        dist_plot_path = os.path.join(stock_folder, f"{stock}_error_plot.png")
        plt.savefig(dist_plot_path, dpi=300)
        plt.close()
        
    # Create a summary file with metrics
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    summary_path = os.path.join(save_path, "model_performance_summary.csv")
    summary_df.to_csv(summary_path)
    logger.info(f"Plots and summary saved to: {save_path}")
    return summary_df



def modify_predictions(df:pd.DataFrame):
    """
    Modify predictions to only consider business days 
    """
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    unique_dates = sorted(df['ds'].unique())
    new_dates = pd.date_range(start=unique_dates[0], periods=len(unique_dates), freq='B')
    df['ds'] = df['ds'].map(dict(zip(unique_dates, new_dates)))
    return df

 

def rolling_forecast_evaluation(
    df: pd.DataFrame,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
    window_size: int = 5,
    input_size: int = 50
):
    """
    Performs rolling window evaluation for the NBEATSx model.
    Args:
        df: DataFrame with columns ['ds', 'unique_id', 'y', 'upper_shadow', 'lower_shadow', 'tick_body']
        window_size: Number of days to forecast
        input_size: Number of historical days to use for training
    Returns:
        results_df: DataFrame with actual vs predicted values
        preds_df: DataFrame with predictions
        metrics_df: DataFrame with evaluation metrics for each window
    """

    
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df = df[(df['ds'] >= from_date) & (df['ds'] <= to_date)]
    all_results = []
    all_predictions = []
    metrics = []

    unique_dates = sorted(df['ds'].unique())
    n_dates = len(unique_dates)


    if n_dates <= input_size:
        logger.error(f"Not enough unique dates ({n_dates}) for input_size {input_size}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    current_start_training_window_idx = 0
    current_end_training_window_idx = input_size
    

    while current_end_training_window_idx + window_size - 1 < len(unique_dates):

    
        train_window_start = unique_dates[current_start_training_window_idx]
        train_window_end = unique_dates[current_end_training_window_idx]
        test_window_end = unique_dates[current_end_training_window_idx + window_size - 1]

        logger.info(f"Processing training window: {train_window_start} to {train_window_end}, test window end: {test_window_end}")  

        train_df = df[( df['ds'] >= train_window_start ) & (df['ds'] < train_window_end)].copy()

        assert train_df['ds'].nunique()==input_size, f"Expected {input_size} unique dates in training data, got {train_df['ds'].nunique()}"


        test_df = df[(df['ds'] >= train_window_end) & (df['ds'] <= test_window_end)].copy()
        assert test_df['ds'].nunique() == window_size, f"Expected {window_size} unique dates in test data, got {test_df['ds'].nunique()}"



        # initialize model
        model = NBEATSx(
            input_size=input_size,
            h=window_size,
            activation='ReLU',
            hist_exog_list=['upper_shadow', 'lower_shadow', 'tick_body'],
            loss=DistributionLoss(distribution='Poisson', level=[80, 90]),
            stack_types=['identity', 'trend'],
            max_steps=1000,
            num_lr_decays=10,
            val_check_steps=100,
            early_stop_patience_steps=5,
            batch_size=128,
            start_padding_enabled=True,
            random_seed=13,
        )
        nf = NeuralForecast(models=[model], freq='D')

        try:
            logger.info(f"Training window starting {train_window_start} to {train_window_end} with {len(train_df)} training samples")
            nf.fit(df=train_df, val_size=12)
        except Exception as e:
            if 'Time series is too short' in str(e):
                logger.warning(f"Skipping training at {train_window_start}: {e}")
                continue
            else:
                logger.error(f"Unexpected error at training window {train_window_start}: {e}")
                continue

        # forecasting
        preds = nf.predict(df=test_df)
        preds = modify_predictions(preds)
        all_predictions.append(preds)

        merged = pd.merge(
            df[['unique_id', 'ds', 'y']],
            preds[['unique_id', 'ds', 'NBEATSx']],
            on=['unique_id', 'ds'], how='inner'
        )
        all_results.append(merged)
        if not merged.empty:
            mae = mean_absolute_error(merged['y'], merged['NBEATSx'])
            rmse = np.sqrt(mean_squared_error(merged['y'], merged['NBEATSx']))
            metrics.append({'window_start': train_window_start, 'window_end': test_window_end, 'mae': mae, 'rmse': rmse})

        current_start_training_window_idx += 1
        current_end_training_window_idx += 1
        

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    preds_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics)
    return results_df, preds_df, metrics_df



def evaluate_results(metrics_df):
    """
    Evaluates the overall performance across all rolling windows.
    """
    overall_metrics = {
        'mean_mae': metrics_df['mae'].mean(),
        'std_mae': metrics_df['mae'].std(),
        'mean_rmse': metrics_df['rmse'].mean(),
        'std_rmse': metrics_df['rmse'].std(),
    }
    return overall_metrics


def train(
    index: str, 
    interval: str, 
    target: str, 
    historical_data: Path = Path("data/historical/"),
):
    """
    Load preprocessed data, apply scalers, run rolling forecasts, inverse-transform, and plot.
    """
    # Load data
    file_path = historical_data / f"{index}_{interval}.pkl"
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'rb') as f:
        data = dill.load(f)


    # Extract DataFrame and scalers
    df = data.get('data')
    scalers = data.get('scalers', {})

    # Drop rows with any NaN values
    df = df.dropna(axis=0, how='any')  

    logger.info(f"Loaded {len(df)} rows for {index} at {interval} interval.")

    # Ensure required columns exist
    required_cols = [
        'date', 'open', 'high', 'low', 'close', 'volume', 'symbol',
        'upper_shadow', 'lower_shadow', 'tick_body', 'diff', 'percent_change'
    ]

    # Check if all required columns are present
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    df = df.rename(columns={
        'date': 'ds',
        'symbol': 'unique_id',
        target: 'y'
    })
    # Ensure ds tz-naive
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    # Run rolling evaluation
    results, predictions, window_metrics = rolling_forecast_evaluation(
        df=df,
        window_size=5,
        input_size=50, 
        from_date=datetime.datetime(2025, 1, 1),  # Adjust as needed
        to_date=datetime.datetime(2025, 7, 1)  # Adjust as needed
    )

    # Inverse-transform target and predictions
    for stock in df.unique_id.unique():
        mask = results['unique_id'] == stock
        key = f"{stock}_{target}"
        if key in scalers and scalers[key] is not None:
            results.loc[mask, 'y'] = scalers[key].inverse_transform(results.loc[mask, 'y'].values.reshape(-1, 1)).flatten()
            if 'NBEATSx' in results.columns:
                results.loc[mask, 'predictions'] = scalers[key].inverse_transform(results.loc[mask, 'NBEATSx'].values.reshape(-1, 1)).flatten()
    
    # Plot results
    summary_df = plot_predictions_vs_actual(results)
    logger.info("Done. Summary metrics:", summary_df)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Train NBEATSx model on historical data.")
    parser.add_argument('--index', type=str, required=True, help="Index to train on (e.g., 'nifty_50', 'banknifty', 'midcap_150')")
    parser.add_argument('--interval', type=str, required=True, help="Data interval (e.g., 'day', 'hour', etc.)")
    parser.add_argument('--cutoff_date', type=str, default='2025-05-01', help="Cutoff date for training data (default: '2025-01-01')")
    parser.add_argument('--target', type=str, default='close', help="Target column for predictions (default: 'close')")
    args = parser.parse_args()

    train(
        index=args.index,
        interval=args.interval,
        target=args.target
    )

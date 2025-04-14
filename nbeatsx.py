import os
import datetime

from neuralforecast.neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import DistributionLoss, MAPE
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from fetch_data import load
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/nbeatsx/{ts}.log"

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
logging.info("Log initialized")


def shift_weekend_predictions(df, date_col='ds'):
    def adjust_date(ds):
        # If Saturday or Sunday, shift to next business day
        if ds.weekday() == 5 or ds.weekday() == 6:
            return ds + pd.offsets.BDay(1)
        return ds
    df[date_col] = df[date_col].apply(adjust_date)
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
        folder_name = f"plots/{model_name}{hyper_str}_{timestamp}"
        save_path = folder_name

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
        print(f'stock: {stock} with Columns: {stock_data.columns}')
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
    print(f"Plots and summary saved to: {save_path}")
    return summary_df



def modify_predictions(df:pd.DataFrame):
    """
    Modify predictions to only consider business days 
    """
    unique_dates = sorted(df['ds'].unique())
    forecast_start = unique_dates[0] 
    # Create a new range of business days starting at forecast_start
    new_dates = pd.date_range(start=forecast_start, periods=len(unique_dates), freq='B')
    
    # Map the original predictions dates to the new business dates.
    mapping = dict(zip(unique_dates, new_dates))
    df['ds'] = df['ds'].map(mapping)

    return df 

 
def rolling_forecast_evaluation(df:pd.DataFrame, window_size=5, input_size=50):
    """
    Rolling forecast evaluation that ensures a full `input_size` context is available for prediction.
    """

    all_predictions, window_metrics, results = [], [], []
    #start_date, end_date = Y_test_df['ds'].min(), Y_test_df['ds'].max()
    #current_start = start_date
    #test_date = '2025-01-01'
    current_start = pd.Timestamp('2025-01-01')
    end_date = df.ds.max() 


    while current_start <= end_date - pd.Timedelta(days=window_size - 1):
        

        current_train = df.loc[df['ds']<current_start].copy()
        test_start = sorted(current_train.ds.unique())[-input_size]
        current_test = df.loc[(df['ds']<current_start) & (df['ds']>= test_start)] #current_start-pd.Timedelta(days=input_size+1))] 
        
        try: 
            print(f"The start of test data: {current_test.ds.min()} The end date: {current_test.ds.max()}")
            assert len(current_test.ds.unique())==input_size, f"inp.days input"
        except: 
            import ipdb;ipdb.set_trace()


        window_end = current_start + pd.Timedelta(days=window_size - 1)
        #print(f"Starting training from {current_start}")
        
        # Initialize model for this window
        model = NBEATSx(
            h=window_size,
            input_size=input_size,
            activation='ReLU',
            hist_exog_list=['upper_shadow', 'lower_shadow', 'tick_body'],
            loss=DistributionLoss(distribution='Poisson', level=[80, 90]),
            stack_types=['identity', 'trend'],
            max_steps=1000,
            num_lr_decays=10,
            val_check_steps=100,
            early_stop_patience_steps=5,
            batch_size=128,
            random_seed=13,
        )
        fcst = NeuralForecast(models=[model], freq='D')

        # Use training data up to one day before the forecast start
        #train_cutoff = current_start - pd.Timedelta(days=1)
        #current_train = Y_train_df[Y_train_df['ds'] <= train_cutoff].copy()
        # Skip window if we lack full context
        #if len(current_train) < input_size:
        #    current_start += pd.Timedelta(days=1)
        #    continue

        logger.info(f"Training range {current_train['ds'].min()} - {current_train['ds'].max()}")
        print(f"Training range {current_train['ds'].min()} - {current_train['ds'].max()}") 
        fcst.fit(df=current_train, val_size=12)

        # Predict on the full series then filter out only the forecast window
        predictions_full = fcst.predict(current_test)
        predictions = predictions_full[predictions_full['ds'] >= current_start].copy()
    
        predictions = modify_predictions(predictions) 
        all_predictions.append(predictions)
        
        # Merge predictions with actuals for metric calculation
        result = pd.merge(df[['unique_id','ds', 'y']], predictions[['unique_id', 'ds', 'NBEATSx']], on=['unique_id', 'ds'], how='inner')
        results.append(result) 
        
        print(f"Test predictions date range: {result['ds'].min()} - {result['ds'].max()}")
        logger.info(f"Test predictions date range: {result['ds'].min()} - {result['ds'].max()}")
        if result.empty: 
            continue
        actuals, preds = result['y'].values, result['NBEATSx'].values
        window_metrics.append({
            'window_start': result.ds.min(),
            'window_end': result.ds.max(),
            'mae': mean_absolute_error(actuals, preds),
            'rmse': np.sqrt(mean_squared_error(actuals, preds))
        })
        if current_start.weekday()==5: 
            current_start+=pd.Timedelta(days=2)
        current_start += pd.Timedelta(days=1)

    return pd.concat(results, ignore_index=True), pd.concat(all_predictions, ignore_index=True), pd.DataFrame(window_metrics)


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


def train():

    nse = load()

    # Define hyperparameters for better documentation
    hyperparams = {
        'input_size': 50,
        'window_size': 5,
        'max_steps': 500,
        'stack_types': ['identity', 'trend'] #, 'seasonality']
    }
    
    target = 'Close'
    df = nse.data
    stocks = df.Ticker.unique()

    for stock in stocks:     
        for col in [
                'Close',    
                'diff', 
                'percent_change',
                'classification_marker', 
                'upper_shadow', 
                'lower_shadow', 
                'tick_body'
                ]: 
            if key:=f"{stock}_{col}" not in nse.scalers: 
                logger.error(f"Scaler Not Found for the {col} attribute of stock {stock}")
                continue

            mask = df['Ticker'] == stock
            scaler = nse.scalers[f"{stock}_{col}"]
            if not scaler:
                continue
            df.loc[mask, col] = scaler.transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()

    df = df.rename(columns={'Date': 'ds', 'Ticker': 'unique_id', target: 'y'})

    # Split data into training and test sets
    #Y_train_df = df.loc[df['ds'] < '2025-01-01'].copy()
    #Y_test_df = df.loc[df['ds'] >= '2025-01-01'].copy()

    results, predictions, window_metrics = rolling_forecast_evaluation(
            #Y_train_df, 
            #Y_test_df,
            df, 
            window_size=hyperparams['window_size'],
            input_size=hyperparams['input_size']
    )
    overall_metrics = evaluate_results(window_metrics)
    
    #import ipdb;ipdb.set_trace()
    for stock in stocks:     
        if key:=f"{stock}_{target}" not in nse.scalers: 
            logger.error(f"Scaler Not Found for the {col} attribute of stock {stock}")
            continue

        mask = results['unique_id'] == stock
        scaler = nse.scalers[f"{stock}_{target}"]
        if not scaler:
            continue
        results.loc[mask, 'y'] = scaler.inverse_transform(results.loc[mask, 'y'].values.reshape(-1, 1)).flatten()
        results.loc[mask, 'predictions'] = scaler.inverse_transform(results.loc[mask, 'NBEATSx'].values.reshape(-1, 1)).flatten()
    plot_predictions_vs_actual(results)  
    import ipdb;ipdb.set_trace()

if __name__=='__main__': 
    train()

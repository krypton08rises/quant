import datetime
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import dill
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

os.makedirs("logs", exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/fetch_data/{ts}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
print(f"Logging to {log_file}")


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logging.info("Log initialized")


class TokenBucket:
    def __init__(self, rate_per_sec, capacity):
        self.capacity = capacity
        self.tokens = capacity
        self.rate = rate_per_sec
        self.lock = threading.Lock()
        self.last_check = time.time()

    def consume(self, tokens=1):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_check
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_check = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
            time.sleep(0.1)


# Global token bucket shared across threads
token_bucket = TokenBucket(rate_per_sec=0.5, capacity=50)


def fetch_with_throttle(ticker, start, end, interval, max_retries=5):
    for attempt in range(max_retries):
        try:
            token_bucket.consume()
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

            # Check if we got valid data
            if df is None or df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            df["Ticker"] = ticker
            df.columns = df.columns.get_level_values(0)
            logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
            return df.reset_index()

        except Exception as e:
            wait = 2**attempt
            logger.warning(f"Error fetching {ticker} (attempt {attempt+1}/{max_retries}): {e}")

            # If this is the last attempt, log as error and return None
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {ticker} after {max_retries} attempts: {e}")
                return None
            else:
                logger.info(f"Retrying {ticker} in {wait}s...")
                time.sleep(wait)

    return None


def load(
    file_path: str = "data/NSE_1d.pkl",
):
    with open(file_path, "rb") as inp:
        nse: exchangeData = dill.load(inp)
    return nse


class exchangeData:
    def __init__(self, exchange: str = "NSE", interval: str = "1d", days: int = 365):
        self.exchange = exchange
        self.interval = interval
        self.days = days
        self.data: Optional[pd.DataFrame] | None = None

        self.scaling_columns = [
            "Close",
            "diff",
            "percent_change",
            "classification_marker",
            "upper_shadow",
            "lower_shadow",
            "tick_body",
        ]

        self.end_dt = date.today()
        self.start_dt = self._estimate_start_date()
        self.scalers = {}
        self.tickers = self._get_tickers()
        logger.info(
            f"Initialized exchangeData for {self.exchange} with interval {self.interval} from {self.start_dt} to {self.end_dt}"
        )

    def _estimate_start_date(self) -> date:
        if self.interval in ["5m", "15m", "30m"]:
            return self.end_dt - timedelta(days=min(60, self.days) - 1)
        if self.interval == "1h":
            return self.end_dt - timedelta(days=min(730, self.days) - 1)
        if self.interval == "1d":
            return self.end_dt - timedelta(days=self.days - 1)
        raise ValueError("Invalid interval. Choose from ['5m', '15m', '30m', '1h', '1d'].")

    def _get_tickers(self) -> List[str]:
        return pd.read_csv("data/exchange/nifty50.csv")["Symbol"].tolist()

    def fetch_data(self) -> None:
        if self.start_dt >= self.end_dt:
            logger.info("Dataset is upto date...")
            return
        dataset = self._combine_dataframes(self.tickers)
        self.data = dataset if self.data is None else pd.concat([self.data, dataset], axis=0)

    def _combine_dataframes(self, stocks: List[str]) -> pd.DataFrame:
        dfs = []
        failed_tickers = []

        def worker(ticker):
            logger.info(f"Starting fetch for {ticker}")
            df = fetch_with_throttle(ticker, self.start_dt, self.end_dt, self.interval)

            if df is not None and not df.empty:
                logger.info(f"Processing {ticker} with {len(df)} rows")
                try:
                    df = self._generate_candlestick_data(df)
                    df = self._normalize_data(df, ticker)
                    df.drop(columns=["High", "Low", "Open", "shifted_close"], inplace=True)
                    df.dropna(inplace=True)
                    logger.info(f"Processed {ticker} successfully, final shape: {df.shape}")
                    return df
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    return None
            else:
                logger.warning(f"No valid data for {ticker}")
                return None

        # Reduce max_workers to avoid overwhelming the API
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(worker, stock): stock for stock in stocks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching data"):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        logger.info(f"Successfully processed data for {ticker}")
                        dfs.append(result)
                    else:
                        logger.warning(f"Failed to get data for {ticker}")
                        failed_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Exception processing {ticker}: {e}")
                    failed_tickers.append(ticker)

        logger.info(f"Successfully fetched data for {len(dfs)} stocks out of {len(stocks)}")
        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _normalize_data(self, df: pd.DataFrame, stock: str) -> pd.DataFrame:
        df["Volume"] = df["Volume"].apply(lambda x: np.log10(x) if x > 0 else 0)
        df["diff"] = df["Close"].diff()
        df["shifted_close"] = df["Close"].shift(1)
        df["percent_change"] = (df["diff"] * 100 / df["shifted_close"].replace(0, np.nan)).fillna(0)
        df["classification_marker"] = df["percent_change"].astype(int)

        for col in self.scaling_columns:
            ky = f"{stock}_{col}"
            self.scalers[ky] = MinMaxScaler((0, 1))
            self.scalers[ky].fit(df[[col]])
        return df

    @staticmethod
    def _generate_candlestick_data(df: pd.DataFrame) -> pd.DataFrame:
        df["upper_shadow"] = df.apply(
            lambda row: row["High"] - max(row["Open"], row["Close"]), axis=1
        )
        df["lower_shadow"] = df.apply(
            lambda row: min(row["Open"], row["Close"]) - row["Low"], axis=1
        )
        df["tick_body"] = df.apply(lambda row: abs(row["Open"] - row["Close"]), axis=1)
        return df

    def save_data(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as outp:
            dill.dump(self, outp, dill.HIGHEST_PROTOCOL)


def run(exchange: str = "NSE") -> None:
    for interval in ["5m", "15m", "30m", "1h", "1d"]:
        logger.info(f"Fetching {interval} data for {exchange}")
        file_path = Path(f"data/{exchange}_{interval}.pkl")

        if file_path.exists():
            try:
                nse = load(file_path)
                nse.start_dt = nse.end_dt
                nse.end_dt = date.today()
            except Exception as e:
                logger.error(f"Failed to load existing data from {file_path}: {e}")
                nse = exchangeData(exchange, interval)

        else:
            nse = exchangeData(exchange, interval)

        print(f"Fetching {interval} interval data from {nse.start_dt} to {nse.end_dt}")
        nse.fetch_data()
        if nse.data is None or nse.data.empty:
            logger.warning(f"No data fetched for {exchange} at {interval} interval.")
            continue
        nse.start_dt = nse.data.Datetime.min().date()
        nse.end_dt = nse.data.Datetime.max().date()

        nse.save_data(file_path)


if __name__ == "__main__":
    print("Starting data fetching process")
    run()

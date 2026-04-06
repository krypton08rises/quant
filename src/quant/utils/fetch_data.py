"""Depracated: Use src/quant/data/kite_data.py instead; cron handler could be useful"""
import datetime
import logging
import os
from datetime import date, timedelta
from typing import List, Optional

import dill
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

os.makedirs("logs", exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/fetch_data_{ts}.log"

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logging.info("Log initialized")


def load(
    file_path: str = "data/NSE_1d.pkl",
):
    with open(file_path, "rb") as inp:
        nse: exchangeData = dill.load(inp)

    return nse


class exchangeData:
    """
    Fetches stock price data from Yahoo Finance for a given exchange and interval.

    Parameters
    ----------
    exchange : str
        The stock exchange name (default: 'NSE').
    interval : str
        Data interval, one of ['5m', '15m', '30m', '1h', '1d'] (default: '1d').
    days : int
        Number of days of historical data to fetch (default: 365).
    """

    def __init__(self, exchange: str = "NSE", interval: str = "1d", days: int = 365):
        self.exchange = exchange
        self.interval = interval
        self.days = days
        self.data: Optional[pd.DataFrame] = None
        self.scaling_columns = [
            "Close",
            "diff",
            "percent_change",
            "classification_marker",
            "upper_shadow",
            "lower_shadow",
            "tick_body"
            # ratio of shadow-body / shadow-shadow
        ]

        self.end_dt = date.today()
        self.start_dt = self._estimate_start_date()
        self.scalers = {}
        self.tickers = self._get_tickers()

    def _estimate_start_date(self) -> date:
        """
        Determines the valid start date based on Yahoo Finance API limitations.

        Returns
        -------
        date
            The computed start date.
        """
        if self.interval in ["5m", "15m", "30m"]:
            return self.end_dt - timedelta(days=min(60, self.days) - 1)
        if self.interval == "1h":
            return self.end_dt - timedelta(days=min(730, self.days) - 1)
        if self.interval == "1d":
            return self.end_dt - timedelta(days=self.days - 1)
        raise ValueError("Invalid interval. Choose from ['5m', '15m', '30m', '1h', '1d'].")

    def _get_tickers(self) -> List[str]:
        """
        Reads stock ticker symbols from a CSV file.

        Returns
        -------
        List[str]
            A list of ticker symbols.
        """
        return pd.read_csv("data/exchange/nifty50.csv")["Symbol"].tolist()

    def fetch_data(self) -> None:
        """
        Fetches stock data and updates the dataset.
        """
        if self.start_dt >= self.end_dt:
            logger.info("Dataset is up to date...")
            return
        dataset = self._combine_dataframes(self.tickers)
        if self.data is None:
            self.data = dataset
        else:
            # Determine the time column name ('Datetime' if it exists, else 'Date')
            time_col = "Datetime" if "Datetime" in self.data.columns else "Date"
            self.data = pd.concat([self.data, dataset], axis=0)
            self.data = self.data.drop_duplicates(subset=[time_col, "Ticker"]).reset_index(
                drop=True
            )

    # def fetch_data(self) -> None:
    #    """
    #    Fetches stock data and updates the dataset.
    #    """
    #    if self.start_dt>=self.end_dt:
    #        logger.info("Dataset is upto date...")
    #        return
    #    dataset = self._combine_dataframes(self.tickers)  # [5:])
    #    self.data = dataset if self.data is None else pd.concat([self.data, dataset], axis=0)

    def _combine_dataframes(self, stocks: List[str]) -> pd.DataFrame:
        """
        Combines multiple stock dataframes into a single dataset.

        Parameters
        ----------
        stocks : List[str]
            List of stock ticker symbols.

        Returns
        -------
        pd.DataFrame
            The combined stock data.
        """
        dfs = []
        for stock in (pbar := tqdm(stocks)):
            pbar.set_description(f"Processing {stock}")
            df = self._fetch_ticker_data(stock)
            if df is not None and not df.empty:
                df = self._generate_candlestick_data(df)
                df = self._normalize_data(df, stock)
                df.drop(columns=["High", "Low", "Open", "shifted_close"], inplace=True)
                df.dropna(inplace=True)
                dfs.append(df)
            else:
                logger.warn(f"Failed to download {stock}", stock=stock)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _fetch_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetches data for a given ticker from Yahoo Finance.
        """
        try:
            df = yf.download(
                ticker, start=self.start_dt, end=self.end_dt, interval=self.interval, progress=False
            )
            df["Ticker"] = ticker
            df.columns = df.columns.get_level_values(0)
            return df.reset_index()
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                logger.error(f"Rate limit error when fetching data for {ticker}: {e}")
            else:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None

    def _normalize_data(self, df: pd.DataFrame, stock: str) -> pd.DataFrame:
        """
        Adds calculated features like percent change and classification marker.
        """
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
        """
        Computes candlestick features.
        """
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


class CronJobHandler:
    """
    Handles scheduling and execution of the stock data fetcher.
    """

    @staticmethod
    def run(exchange: str = "NSE") -> None:
        """
        Runs the stock data fetcher for multiple intervals.
        """
        # logging.basicConfig(filename='myapp.log', level=logging.INFO)

        for interval in ["5m", "15m", "30m", "1h", "1d"]:
            logger.info(f"Fetching {interval} data for {exchange}")
            file_path = f"data/{exchange}_{interval}.pkl"

            if os.path.exists(file_path):
                with open(file_path, "rb") as inp:
                    nse: exchangeData = dill.load(inp)

                if nse.data is not None and not nse.data.empty:
                    logger.info(f"Found existing data for {interval}")
                    time_col = "Datetime" if "Datetime" in nse.data.columns else "Date"
                    last_timestamp = pd.to_datetime(nse.data[time_col]).max()
                    # Advance the start date by one period (adjust based on the interval)
                    if nse.interval in ["5m", "15m", "30m"]:
                        new_start = last_timestamp + pd.Timedelta(minutes=int(nse.interval[:-1]))
                    elif nse.interval == "1h":
                        new_start = last_timestamp + pd.Timedelta(hours=1)
                    elif nse.interval == "1d":
                        new_start = last_timestamp + pd.Timedelta(days=1)
                    nse.start_dt = new_start.date()

                else:
                    nse.start_dt = nse.end_dt
                nse.end_dt = date.today()

            else:
                nse = exchangeData(exchange, interval)
                logger.info("Instantiated new exchangeData object...")

            logger.info(f"Fetching {interval} interval data from {nse.start_dt} to {nse.end_dt}")
            nse.fetch_data()

            nse.save_data(file_path)


if __name__ == "__main__":
    CronJobHandler.run()

import os
import json
import logging
import time
from pathlib import Path
from datetime import date, timedelta

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect
import dill

from pydantic import BaseModel, PrivateAttr
from ._common import INTERVAL_LOOKBACK, MAX_DAYS_PER_CALL
# Configure logging
timestamp = date.today().isoformat()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


timestamp = date.today().isoformat()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class KiteDataHandler:
    """
    Handler for fetching, processing, saving, and loading historical data from Kite Connect API.
    """

    def __init__(
        self,
        api_key: str = None,
        token_path: Path = Path("access_token.json"),
        inst_csv: Path = Path("data/historical/kite_nse_instruments.csv"),
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        self.kite = self._get_kite_session(api_key, token_path)
        self.instrument_map = self._load_instruments(inst_csv)
        self.scalers = {}
        self.data = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_kite_session(self, api_key: str, token_path: Path) -> KiteConnect:
        """
        Gets a Kite Connect session.
        Arguments
        ----------
        api_key : str
            API key for Kite Connect.
        token_path : Path
            Path to the JSON file containing the access token.
        Returns
        -------
        kite : KiteConnect
            Authenticated Kite Connect session.
        """
        with open(token_path, 'r') as f:
            session = json.load(f)
        api_key = api_key or os.getenv("ZERODHA_API_KEY")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(session["access_token"].strip())
        profile = kite.profile()
        logger.info(f"Logged in as {profile['user_name']}")
        return kite

    def _load_instruments(self, path: Path) -> dict:
        """
        Loads instrument tokens from CSV file.
        Arguments
        ----------
        path : Path
            Path to the CSV file containing instrument data.
        Returns
        -------
        mapping : dict
            Mapping of tradingsymbol to instrument_token.
        """

        df = pd.read_csv(path)
        df = df[(df.instrument_type == 'EQ') & (df.exchange == 'NSE')]
        mapping = dict(zip(df.tradingsymbol, df.instrument_token))
        logger.info(f"Loaded {len(mapping)} NSE instruments")
        return mapping

    async def fetch_historical(self,
                         symbols: list,
                         interval: str = 'day',
                         start: date = None,
                         end: date = None) -> pd.DataFrame:
        """
        Fetch historical data for symbols between start and end using chunking.
        ToDo: Batched async requests for multiple chunks and symbols.
        Arguments
        ----------
        symbols : list
            List of symbol strings.
        interval : str
            Data interval (e.g., 'minute', '5minute', 'day').
        start : date
            Start date for fetching data.
        end : date
            End date for fetching data.
        Returns
        -------
        df : pd.DataFrame
        DataFrame containing historical data with features and normalized values.
        """
        end = end or date.today()
        start = start or (end - timedelta(days=INTERVAL_LOOKBACK.get(interval, 365)))
        frames = []
        logger.info(f"Fetching historical data for {len(symbols)} symbols from {start} to {end} with interval '{interval}'")

        for sym in symbols:

            # Get instrument token for api request 
            token = self.instrument_map.get(sym)
            if not token:
                logger.warning(f"No token for {sym}")
                continue

            
            # chunk dates
            chunk_start = start
            while chunk_start <= end:    
                chunk_end = min(chunk_start + timedelta(days=MAX_DAYS_PER_CALL - 1), end)
                logger.info(f"Fetching {sym} {interval} from {chunk_start} to {chunk_end}")
                data = None

                # Simple retry logic with exponential backoff 
                for i in range(1, self.max_retries + 1):
                    try:
                        # can I do batched async requests here? Historical_data seems sync only
                        data = self.kite.historical_data(
                            instrument_token=token,
                            from_date=chunk_start,
                            to_date=chunk_end,
                            interval=interval,
                            continuous=False
                        )
                        break
                    except Exception as e:
                        logger.warning(f"Attempt {i} failed: {e}")
                        if i < self.max_retries:
                            time.sleep(self.retry_delay * 2**(i-1))
                if not data:
                    logger.error(f"Failed to fetch {sym} chunk {chunk_start} to {chunk_end}")
                    chunk_start = chunk_end + timedelta(days=1)
                    continue

                # convert to DataFrame - remember this is per symbol per chunk at this
                df = pd.DataFrame(data) 
                if df.empty:
                    logger.info(f"No data for chunk {chunk_start} to {chunk_end}")
                else:
                    df['symbol'] = sym
                    frames.append(df)

                chunk_start = chunk_end + timedelta(days=1)

        # concatenate all frames for all chunks and symbols
        if frames:
            full_df = pd.concat(frames, ignore_index=True)
            full_df.sort_values(by=['symbol', 'date'], inplace=True)

            # process features/scalers
            full_df = self._generate_features(full_df)
            # full_df = self._normalize(full_df)
            self.data = full_df
            return full_df
        else:
            return pd.DataFrame()

    @staticmethod
    def _generate_features(df: pd.DataFrame) -> pd.DataFrame:
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['tick_body'] = (df['open'] - df['close']) #.abs() -- DO NOT USE ABSOLUTE; wouldn't make sense to only have positives --
        df['diff'] = df['close'].diff()
        df['shifted_close'] = df['close'].shift(1)
        df['percent_change'] = (df['diff'] * 100 / df['shifted_close'].replace(0, np.nan)).fillna(0)
        df['classification_marker'] = df['percent_change'].astype(int)
        return df


    def save(self, file_path: Path):
        if self.data is None:
            raise ValueError("No data to save.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving data+metadata to {file_path.with_suffix('.pkl')}")
        payload = {'data': self.data, 'scalers': self.scalers}
        with open(file_path.with_suffix('.pkl'), 'wb') as f:
            dill.dump(payload, f)
        logger.info(f"Saved data+metadata to {file_path.with_suffix('.pkl')}")

    @classmethod
    def load(cls, file_path: Path):
        with open(file_path.with_suffix('.pkl'), 'rb') as f:
            payload = dill.load(f)
        data = payload.get('data')
        scalers = payload.get('scalers', {})
        logger.info(f"Loaded data+metadata from {file_path.with_suffix('.pkl')}")
        return data




# -------------------------------
# Bronze → Silver
# -------------------------------
class SilverConfig(BaseModel):
    """
    Configuration class for Silver features.
    """
    exchange: str = "NSE"
    index: str = "nifty_50"
    interval: str = "1d"
    base_dir: Path = Path("data/historical/")
    features_version: str = "v1"
    # indicator params (override if you like)
    rsi_period: int = 14
    bb_window: int = 20
    bb_k: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_windows: Tuple[int, ...] = (5, 10, 20)
    vol_window: int = 10  # rolling stdev on percent_change
    label_version: str = "1"
    horizons: int = 1
    up: float = 3.0
    down: float = -3.0



# -------------------------------
# Silver → Gold (labels
# -------------------------------
class GoldConfig(BaseModel):
    """
    Configuration class for gold feature extraction.
    """
    exchange: str = "NSE"
    interval: str = "1d"
    base_dir: Path = Path("data/historical/")
    feature_version: str = "v1"
    label_version: str = "1"
    up_thresh: float = 2.0
    down_thresh: float = -2.0
    label_unit: str = "pct"  # "pct" points (e.g., 2.0) or "fraction"
    horizons: Tuple[int, ...] = (1,4)

    @staticmethod
    def _compute_label_from_pct(x: float, down: float, up: float) -> int:
        """
        Compute the label for a given percent change value.
        """
        if x < down:
            return 1
        if x > up:
            return 2
        return 0
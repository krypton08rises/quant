import dill
import pandas as pd 

from pathlib import Path
from typing import Optional, overload
from dataclasses import dataclass

from .models import SilverConfig, GoldConfig
from ._common import Indices, Interval

from logging import getLogger
logger = getLogger(__name__)


def save_dataset(df: pd.DataFrame, cfg: SilverConfig | GoldConfig, file_name: str) -> Path:
    """
    Write down a silver DataFrame as parquet files.
    Arguments
    ---------
    df: pd.DataFrame
        The silver DataFrame to write.
    cfg: SilverConfig
        The configuration object containing feature parameters.
    Returns
    -------
    Path
        The path to the directory containing the written parquet files.
    """

    directory = cfg.base_dir / "silver" if isinstance(cfg, SilverConfig) else cfg.base_dir / "gold"
    directory.mkdir(parents=True, exist_ok=True)
    out_file = directory / file_name

    logger.info(f"Saving dataset to {out_file} with {len(df)} rows.")
    
    # Overwrite in-place to avoid loading the historical file into RAM
    df.to_parquet(out_file, index=False)
    return out_file


def ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensure the specified column is in datetime format.
    Arguments
    ---------
    df: DataFrame
        The DataFrame containing the column to check.
    col: str
        The name of the column to ensure is in datetime format.
    Returns
    -------
    DataFrame
        The DataFrame with the specified column converted to datetime format.
    """
    # if not np.issubdtype(df[col].dtype, np.datetime64):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
    return df


def group_sort(df: pd.DataFrame, sym_col: str, dt_col: str) -> pd.DataFrame:
    """
    Sort the DataFrame by symbol and date.
    Arguments
    ---------
    df: DataFrame
        The DataFrame to sort.
    sym_col: str
        The name of the column containing the symbol.
    dt_col: str
        The name of the column containing the date.
    Returns
    -------
    DataFrame
        The sorted DataFrame.
    """
    df = df.sort_values([sym_col, dt_col]).reset_index(drop=True)
    return df


def load_bronze(
    index: str, 
    interval: str, 
    base_dir: Path,
) -> tuple[pd.DataFrame, Optional[dict]]:
    """
    Load a bronze pickle file.
    This file contains per index stock data with raw prices and engineered candle features.
    Arguments
    ---------
    index: str
        The market index to load (e.g., "nifty_50").
    interval: str
        The time interval of the data (e.g., "1d").
    base_dir: Path
        The base directory where the pickle files are stored.
    Returns
    -------
    pd.DataFrame
        The loaded bronze DataFrame.
    """
    parquet_file = base_dir / f"{index}_{interval}.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Bronze parquet not found: {parquet_file}")
    df = pd.read_parquet(parquet_file)
            
    # Normalize column names used downstream
    # Expecting columns: Date, Open, High, Low, Close, Volume, Ticker and engineered fields
    rename = {
        "Date": "date",
        "Ticker": "symbol",
        "Close": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }


    for k, v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
        # inverse transform to get raw prices -- only 'close' and 'percent_change' are normalized for now
    df = ensure_dt(df, "date")
    df = group_sort(df, "symbol", "date")

    return df.dropna(axis=0)


def load_silver(symbol: str, gc: GoldConfig) -> Optional[pd.DataFrame]:
    """
    Load silver data for a given symbol.
    Arguments
    ---------
    symbol: str
        The stock symbol to load.
    gc: GoldConfig
        The gold configuration object containing parameters.
    Returns
    -------
    Optional[pd.DataFrame]
        The loaded silver DataFrame for the symbol, or None if not found.
    """
    silver_dir = gc.base_dir / "silver"
    file_name = f"{symbol}_{gc.interval}_{gc.label_version}.parquet"
    silver_path = silver_dir / file_name

    if not silver_path.exists():
        logger.warning(f"Silver data not found for symbol {symbol} at {silver_path}")
        return None

    sdf = pd.read_parquet(silver_path)
    sdf = ensure_dt(sdf, "date")
    sdf = group_sort(sdf, "symbol", "date")
    return sdf

def read_gold(gold_dir: Path) -> pd.DataFrame:
    """
    Stream GOLD parquet files from the specified directory.
    Arguments
    ---------
    gold_dir: Path
        The path to the GOLD directory containing parquet files.
    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame containing all GOLD data.
    """
    parts = sorted(Path(gold_dir).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {gold_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    
    # normalize timezones: make naive datetime
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)

    # if utc convert fails because it's already tz-naive with offset string, try tz_localize(None)
    if df["date"].isna().any():
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


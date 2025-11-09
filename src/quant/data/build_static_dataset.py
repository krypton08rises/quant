"""
Build static, versioned datasets from your existing pickles.

Pipeline stages (Bronze → Silver → Gold):
- Bronze: load raw/pickled bars you already produce (e.g., data/NSE_1d.pkl).
- Silver: compute time-safe technical indicators & rolling features per symbol.
- Gold: compute classification labels for one or more horizons and thresholds.
- Manifests: write a date-stamped manifest that points to the exact partitions.

Outputs (default base_dir = data):
  data/historical/index/silver/exchange=NSE/interval=1d/features_v1/part-*.parquet
  data/historical/index/gold/exchange=NSE/interval=1d/h=1/labels_v1/part-*.parquet
  data/historical/index/gold/exchange=NSE/interval=1d/h=5/labels_v1/part-*.parquet
  data/historical/index/napshots/dataset_YYYY-MM-DD.json

Usage (CLI):
  python build_static_datasets.py --exchange NSE --interval 1d --features-version v1 --labels-version v1 --horizons 1 5 \
      --up 2.0 --down -2.0

You can schedule this to run daily via cron.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import dill

import numpy as np
import pandas as pd
from ._common import Indices, Interval
from .models import SilverConfig, GoldConfig

# logging 
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------
# Utils: safe rolling/group ops
# -------------------------------

def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
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


def _group_sort(df: pd.DataFrame, sym_col: str, dt_col: str) -> pd.DataFrame:
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


# -------------------------------
# Indicators (time-safe)
# -------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Compute the Exponential Moving Average (EMA) of a pandas Series.
    Arguments
    ---------
    series: pd.Series
        The input time series.
    span: int
        The span (window) for the EMA.
    Returns
    -------
    pd.Series
        The EMA of the input series.
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing.
    Arguments
    ---------
    close: pd.Series
        The input time series of closing prices.
    period: int
        The lookback period for the RSI.
    Returns
    -------
    pd.Series
        The RSI of the input series.
    """
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute the Moving Average Convergence Divergence (MACD) of a pandas Series.
    Arguments
    ---------
    close: pd.Series
        The input time series of closing prices.
    fast: int
        The fast EMA window.
    slow: int
        The slow EMA window.
    signal: int
        The signal line EMA window.
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        The MACD line, signal line, and histogram.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute the Bollinger Bands for a pandas Series.
    Arguments
    ---------
    close: pd.Series
        The input time series of closing prices.
    window: int
        The lookback period for the Bollinger Bands.
    k: float
        The number of standard deviations to use for the bands.
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        The middle band, upper band, lower band, and position of close within the bands.
    """
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + k * std
    lower = mid - k * std
    # position of close within the band (useful feature)
    pct = (close - mid) / (upper - lower)
    return mid, upper, lower, pct


def atr(high: pd.Series, low: pd.Series, prev_close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Average True Range (ATR) of a pandas Series.
    Arguments
    ---------
    high: pd.Series
        The input time series of high prices.
    low: pd.Series
        The input time series of low prices.
    prev_close: pd.Series
        The input time series of previous close prices.
    period: int
        The lookback period for the ATR.
    Returns
    -------
    pd.Series
        The ATR of the input series.
    """
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def load_bronze_pickle(index: str, interval: str, base_dir: Path) -> pd.DataFrame:
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
    pkl = base_dir / f"{index}_{interval}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Bronze pickle not found: {pkl}")
    with open(pkl, "rb") as f:
        obj = dill.load(f)
    df: pd.DataFrame = obj.get('data', pd.DataFrame()).copy()
    scalers: Optional[dict] = obj.get('scalers', None)
    
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
    for sym, g in df.groupby("symbol"): 
        close_key = f"{sym}_close"
        pct_change_key = f"{sym}_percent_change"
        assert close_key in scalers, f"Missing scaler for 'close' prices of {close_key}"
        assert pct_change_key in scalers, f"Missing scaler for 'percent_change' of {pct_change_key}"

        # inverse transform to get raw prices -- only 'close' and 'percent_change' are normalized for now
        df.loc[g.index, "raw_close"] = scalers[close_key].inverse_transform(g[["close"]])
        df.loc[g.index, "raw_percent_change"] = scalers[pct_change_key].inverse_transform(g[["percent_change"]])
    df = _ensure_dt(df, "date")
    df = _group_sort(df, "symbol", "date")

    return df.dropna(axis=0)


def make_silver(df: pd.DataFrame, cfg: SilverConfig) -> pd.DataFrame:
    """
    Compute time-safe indicators and rolling features per symbol.
    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing historical price data.
    cfg: SilverConfig
        The configuration object containing feature parameters.
    Returns
    -------
    pd.DataFrame
        The DataFrame with computed silver features.
    """
    g = df.groupby("symbol", group_keys=False)

    def _per_symbol(sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a single symbol DataFrame.
        Args
        ----
        sdf: pd.DataFrame
            The input DataFrame for a single symbol.
        Returns
        -------
        pd.DataFrame
            The DataFrame with computed features for the symbol.
        """
        sdf = sdf.copy()

        # RSI
        sdf[f"rsi_{cfg.rsi_period}"] = rsi_wilder(sdf["raw_close"], cfg.rsi_period)

        # MACD
        macd_line, signal_line, hist = macd(sdf["raw_close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        sdf["macd_line"], sdf["macd_signal"], sdf["macd_hist"] = macd_line, signal_line, hist

        # Bollinger
        mid, up, lo, pct = bollinger(sdf["raw_close"], cfg.bb_window, cfg.bb_k)
        sdf["bb_mid"], sdf["bb_up"], sdf["bb_lo"], sdf["bb_pct"] = mid, up, lo, pct

        # ATR (use previous close)
        sdf["prev_close"] = sdf["raw_close"].shift(1)
        sdf["atr_14"] = atr(sdf["high"], sdf["low"], sdf["prev_close"], period=14)

        # Moving averages & EMA deltas
        for w in cfg.ma_windows:
            sdf[f"sma_{w}"] = sdf["raw_close"].rolling(w, min_periods=w).mean()
            sdf[f"ema_{w}"] = ema(sdf["raw_close"], w)
            sdf[f"close_over_sma_{w}"] = sdf["raw_close"] / sdf[f"sma_{w}"] - 1

        # Rolling volatility on percent_change
        sdf["volatility_10"] = sdf["raw_percent_change"].rolling(cfg.vol_window, min_periods=cfg.vol_window).std()

        # Keep existing engineered candles if present
        # tick_body, upper_shadow, lower_shadow, diff should already exist in bronze
        return sdf

    out = g.apply(_per_symbol)
    # Drop warmup rows with insufficient history (simple rule: drop rows with any NaNs in new features)
    feature_cols = [c for c in out.columns if c not in ("open","high","low","close","volume","date","symbol", "raw_close", "raw_percent_change")]
    out = out.dropna(subset=feature_cols)
    return out


def save_dataset(df: pd.DataFrame, cfg: SilverConfig, file_name: str) -> Path:
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


    out_file = cfg.base_dir / file_name #/ f"{cfg.exchange}_{cfg.interval}_{cfg.features_version}"

    logger.info(f"Saving dataset to {out_file} with {len(df)} rows.")
    # Open file if exists, read existing data and append new data
    if out_file.exists():
        existing_data = pd.read_parquet(out_file)
        df = pd.concat([existing_data, df], ignore_index=True)

    df.to_parquet(out_file, index=False)
    return out_file

    # out_dir.mkdir(parents=True, exist_ok=True)
    # Write one big file or partitioned by symbol; here: by symbol
    # paths: List[str] = []
    # Save all data in one file? Why are we saving per symbol? 
    # for sym, sdf in df.groupby("symbol"):
    #     path = out_dir / f"{sym.replace('/', '_').replace('.', '_')}.parquet"
    #     sdf.to_parquet(path, index=False)
    #     paths.append(str(path))
    # Return directory path
    return out_dir



def _compute_label_from_pct(x: float, down: float, up: float) -> int:
    """
    Compute the label for a given percent change value.
    """
    if x < down:
        return 1
    if x > up:
        return 2
    return 0


def make_gold_from_silver(df: pd.DataFrame, gc: GoldConfig) -> List[Path]:
    """
    Create gold labels from silver features.
    Arguments
    ---------
    df: pd.DataFrame
        The silver DataFrame containing features.
    gc: GoldConfig
        The configuration object containing label parameters.
    Returns
    -------
    List[Path]
        A list of paths to the created gold parquet files.
    """
    master_gold_df: pd.DataFrame = pd.DataFrame()

    # Iterate each symbol partition
    for sym, sdf in df.groupby("symbol"):
        sdf = _ensure_dt(sdf, "date")
        sdf = _group_sort(sdf, "symbol", "date")

        # Build future percent_change for horizons and derive labels
        for h in gc.horizons:
            dfh = sdf.copy()
            # Use existing percent_change (daily %), compute forward change over horizon h
            # Here we sum h 1-day changes ~ approx horizon move; alternatively use (close.shift(-h)/close - 1)*100
            # Prefer log return aggregation for horizon; implement robustly:

            logret = np.log(dfh["raw_close"]).diff()
            fwd_logret = logret.rolling(window=h, min_periods=h).sum().shift(-h)
            fwd_pct = (np.exp(fwd_logret) - 1.0) * 100.0
            dfh[f"fwd_pct_h{h}"] = fwd_pct
            dfh[f"label_h{h}"] = dfh[f"fwd_pct_h{h}"].apply(lambda v: _compute_label_from_pct(v, gc.down_thresh, gc.up_thresh)).astype("int8")
            # label_ready_date = date + h days; drop rows where future not available
            dfh = dfh.iloc[:-h] if h > 0 else dfh
            
            cols = [
                "symbol","date","close","volume","percent_change","tick_body","upper_shadow","lower_shadow","diff",
                # silver features
                "rsi_14","macd_line","macd_signal","macd_hist","bb_mid","bb_up","bb_lo","bb_pct","prev_close","atr_14",
                "sma_5","ema_5","close_over_sma_5","sma_10","ema_10","close_over_sma_10","sma_20","ema_20","close_over_sma_20","vol_10",
                # labels
                f"fwd_pct_h{h}", f"label_h{h}",
            ]
            cols = [c for c in cols if c in dfh.columns]
            dfh = dfh[cols].copy()
            master_gold_df = pd.concat([master_gold_df, dfh], ignore_index=True)

    return master_gold_df


# -------------------------------
# Manifests / snapshots
# -------------------------------

def write_manifest(
    base_dir: Path, 
    exchange: str, 
    interval: str, 
    features_dir: Path, 
    gold_dirs: List[Path] 
) -> Path:
    """
    Write a manifest file for the dataset.
    Arguments
    -------
    base_dir: Path
        The base directory for the dataset.
    exchange: str
        The exchange name.
    index: str
        The index name.
    interval: str
        The time interval.
    features_dir: Path
        The directory containing feature files.
    gold_dirs: List[Path]
        A list of directories containing gold label files.
    Returns
    -------
    Path
        The path to the created manifest file.
    """
    snap_date = date.today().isoformat()
    manifest = {
        "snapshot_date": snap_date,
        "exchange": exchange,
        "interval": interval,
        "silver_dir": str(features_dir),
        "gold_dirs": [str(d) for d in gold_dirs],
    }
    snap_dir = base_dir / "gold"
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_path = snap_dir / f"dataset_{snap_date}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path


# -------------------------------
# Main
# -------------------------------
def run_build(
    config: SilverConfig
) -> None:
    
    # Bronze → Silver
    bronze = load_bronze_pickle(config.index, config.interval, config.base_dir.joinpath("bronze"))
    bronze["index"] = config.index  
    
    silver = make_silver(bronze, config)
    save_dataset(silver, config, "silver.parquet")

    # Silver → Gold
    g_cfg = GoldConfig(
        exchange=config.exchange, 
        interval=config.interval, 
        base_dir=config.base_dir, 
        labels_version=config.label_version,
        horizons=(1,), 
        up_thresh=config.up, 
        down_thresh=config.down
    )
    gold_dirs = make_gold_from_silver(silver, g_cfg)

    save_dataset(gold_dirs, g_cfg, "gold.parquet")  
    # Manifest
    # manifest_path = write_manifest(base_dir=config.base_dir, exchange=config.exchange, interval=config.interval, features_dir=silver_dir, gold_dirs=gold_dirs)

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the dataset from a Parquet file.
    """
    return pd.read_parquet(path)

def handler():
    """
    Main function to parse arguments and trigger dataset build.
    """
    for index in Indices:
        for interval in Interval:
            try: 
                config = SilverConfig(
                    exchange="NSE",
                    index=index.value,
                    interval=interval.value,
                    base_dir=Path("data/historical/"),
                    features_version="v1",
                    label_version="1",
                    horizons=(1),
                    up=3.0,
                    down=-3.0
                )
                run_build(
                    config=config,
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping {index.value} at {interval.value}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing {index.value} at {interval.value}: {e}")
                continue

if __name__ == "__main__":
    handler()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exchange", default="NSE")
    # parser.add_argument("--index", default="nifty_50", help="\t".join([f"{i}. {idx.value}" for i, idx in enumerate(Indices)]))
    # parser.add_argument("--interval", default="day")
    # parser.add_argument("--base-dir", default="data/historical/index/")
    # parser.add_argument("--features-version", default="v1")
    # parser.add_argument("--labels-version", default="1")
    # parser.add_argument("--horizons", nargs="+", type=int, default=[1])
    # parser.add_argument("--up", type=float, default=2.0, help="upper threshold in pct points")
    # parser.add_argument("--down", type=float, default=-2.0, help="lower threshold in pct points")
    # args = parser.parse_args()
    # config = SilverConfig(
    #     exchange=args.exchange,
    #     index=args.index,
    #     interval=args.interval,
    #     base_dir=Path(args.base_dir),
    #     features_version=args.features_version,
    #     label_version=args.labels_version,
    #     horizons=args.horizons,
    #     up=args.up,
    #     down=args.down
    # )
    # run_build(
    #     config=config,
    # )

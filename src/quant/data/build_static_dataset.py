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
import gc
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
from .utils import save_dataset
from .indicators import rsi_wilder, macd, bollinger, atr, ema

# logging 
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

"""
Base Columns: 
'date', 'open', 'high', 'low', 'close'(N), 'volume'(lgN), 'symbol', 'upper_shadow'(N), 'lower_shadow'(N), 'tick_body'(N), 'diff'(N), 'percent_change'(N),
       'classification_marker', 'raw_close', 'raw_percent_change'
Details about bronze data: 

"""

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

        # Ensure no NaNs
        sdf = sdf.dropna(axis=0)
        
        # Save here per symbol
        save_dataset(sdf, cfg, f"silver_{sdf['symbol'].iloc[0]}_{cfg.interval}_{cfg.label_version}.parquet")

        return sdf

    return g.apply(_per_symbol).reset_index(drop=True)
    



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
    gold_chunks: list[pd.DataFrame] = []

    def _per_symbol(sdf: pd.DataFrame) -> pd.DataFrame:
        """
        Compute labels for a single symbol DataFrame.
        Args
        ----
        sdf: pd.DataFrame
            The input DataFrame for a single symbol.
        Returns
        -------
        pd.DataFrame
            The DataFrame with computed labels for the symbol.
        """
        sdf = sdf.copy()
        # Build future percent_change for horizons and derive labels
        for h in gc.horizons:
            logret = np.log(sdf["raw_close"]).diff()
            fwd_logret = logret.rolling(window=h, min_periods=h).sum().shift(-h)
            fwd_pct = (np.exp(fwd_logret) - 1.0) * 100.0
            sdf[f"fwd_pct_h{h}"] = fwd_pct
            sdf[f"label_h{h}"] = sdf[f"fwd_pct_h{h}"].apply(lambda v: gc._compute_label_from_pct(v, gc.down_thresh, gc.up_thresh)).astype("int8")

        save_dataset(sdf, gc, f"gold_{sdf['symbol'].iloc[0]}_{gc.interval}_h{'_'.join(map(str,gc.horizons))}_{gc.labels_version}.parquet")

        return sdf
    df = df.groupby("symbol", group_keys=False).apply(_per_symbol).reset_index(drop=True)


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
    
    make_silver(bronze, config)
    gc.collect()
    
    del bronze

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
    del silver
    gc.collect()
    
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


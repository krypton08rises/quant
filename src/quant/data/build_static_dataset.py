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
import gc
from pathlib import Path
from typing import List 

import time

import numpy as np
import pandas as pd
from ._common import Indices, Interval
from .models import SilverConfig, GoldConfig
from .utils import save_dataset, load_bronze , load_silver
from .indicators import rsi_wilder, macd, bollinger, atr, ema

# logging 
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

"""
Base Columns: 
'date', 'open', 'high', 'low', 'close'(N), 'volume'(lgN), 'symbol', 'upper_shadow'(N), 'lower_shadow'(N), 'tick_body'(N), 'diff'(N), 'percent_change'(N),
       'classification_marker', 'raw_close', 'raw_percent_change'
Tests : 
- check if volume is 0 ( log normalized )  
- upper shadow, lower shadow, tick body can be 0; just not all 3 at the same time
- need to mark tick body as positive / negative next time I download data
Details about bronze data: 
TODO: 
- Identify simple rule based signals like MA crossover, RSI overbought/oversold, etc; besides computing values.
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
        silver_file = f"{sdf['symbol'].iloc[0]}_{cfg.interval}_{cfg.label_version}.parquet"
        # if silver already exists, skip this;  
        if (cfg.base_dir / "silver" / silver_file).exists():
            logger.info(f"Silver data already exists for {silver_file}, skipping.")
            return sdf

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
        save_dataset(sdf, cfg, f"{sdf['symbol'].iloc[0]}_{cfg.interval}_{cfg.label_version}.parquet")

    g.apply(_per_symbol)
    del g

def make_silver(df: pd.DataFrame, cfg: SilverConfig) -> List[str]:
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
    List[str]
        List of symbols that were processed.
    """
    symbols = df['symbol'].unique().tolist()
    
    for symbol in symbols:
        sdf = df[df['symbol'] == symbol].copy()
        
        silver_file = f"{symbol}_{cfg.interval}_{cfg.label_version}.parquet"
        # if silver already exists, skip this;  
        if (cfg.base_dir / "silver" / silver_file).exists():
            logger.info(f"Silver data already exists for {silver_file}, skipping.")
            continue

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
        save_dataset(sdf, cfg, f"{symbol}_{cfg.interval}_{cfg.label_version}.parquet")
        
        # Clean up
        del sdf
        
    return symbols


def make_gold_from_silver(symbols: list[str], gc: GoldConfig) -> None:
    """
    Create gold labels from silver features.
    Loads silver data per symbol; per interval and computes labels for each horizon. 
    Holding all symbols in memory crashes vscode, so we stream them. 
    Arguments
    ---------
    symbols: list[str]
        List of symbols to process.
    gc: GoldConfig
        The configuration object containing label parameters.
    Returns
    -------
    None
    """
    time.sleep(5)
    logger.info(f"Creating gold data for {len(symbols)} symbols at interval {gc.interval}")
    for symbol in symbols:
        sdf = load_silver(symbol, gc)
        if sdf is None:
            logger.warning(f"Could not load silver data for {symbol}")
            continue
        
        sdf = sdf.copy()
        # Build future percent_change for horizons and derive labels
        for h in gc.horizons:

            logret = np.log(sdf["raw_close"]).diff()
            fwd_logret = logret.rolling(window=h, min_periods=h).sum().shift(-h)
            fwd_pct = (np.exp(fwd_logret) - 1.0) * 100.0
            sdf[f"fwd_pct_h{h}"] = fwd_pct
            sdf[f"label_h{h}"] = sdf[f"fwd_pct_h{h}"].apply(lambda v: gc._compute_label_from_pct(v, gc.down_thresh, gc.up_thresh)).astype("int8")
            sdf = sdf.iloc[:-h]  # drop last h rows with NaN labels

            # Save here per symbol
            logger.info(f"Saving gold data for {symbol} horizon {h}")
            save_dataset(sdf, gc, f"{symbol}_{gc.interval}_horizon({h})_{gc.label_version}.parquet")
        
        # Clean up
    
    gc.collect()


def run_build(
    config: SilverConfig
) -> None:
    
    # Bronze → Silver
    bronze = load_bronze(config.index, config.interval, config.base_dir.joinpath("bronze"))
    bronze["index"] = config.index  
    symbols = bronze['symbol'].unique().tolist()

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
    gold_dirs = make_gold_from_silver(symbols=symbols, gc=g_cfg)
    del silver
    gc.collect()
    
    save_dataset(gold_dirs, g_cfg, "gold.parquet")  


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
                logger.info(f"Processing {index.value} at {interval.value}")
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


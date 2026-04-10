"""
Builds the bronze layer (data/historical/bronze) from raw Kite data (data/historical/raw).

For each raw pkl file, applies:
  1. filter_active_universe  — drop symbols with no data in 2026 (delisted / inactive)
  2. remove_phantom_ticks    — truncate pre-gap history per symbol
  3. flag_null_volume        — mark zero-volume bars (flag_null_volume)
  4. process_symbol          — mark log-return, z-score, volume-price disparity outliers
                               (flag_log_ret, flag_z_score, flag_volume_price_disparity)

Run:
    python build_bronze.py --interval day
    python build_bronze.py --interval all
"""

import argparse
from pathlib import Path

import dill
import pandas as pd
from quant.data._common import BRONZE_DIR, RAW_DIR, FlagColumns, RawColumns
from quant.data.audit.dates_and_volumes.missing_dates import (
    filter_active_universe,
    remove_phantom_ticks,
)
from quant.data.audit.returns.log_returns import LogReturnsConfig, process_symbol
from quant.data.kite.kite_handler import KiteDataHandler
from quant.logs.logging import logger

SUPPORTED_INTERVALS = ["day", "60minute", "30minute", "15minute", "5minute"]


def _save(df: pd.DataFrame, out_path: Path) -> None:
    """Save a DataFrame using the same dill payload format as KiteDataHandler."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".pkl"), "wb") as f:
        dill.dump({"data": df}, f)


def build_bronze(interval: str) -> None:
    """
    Transform all raw pkl files for the given interval into bronze pkl files.
    """
    raw_paths = sorted(RAW_DIR.glob(f"*_{interval}.pkl"))
    if not raw_paths:
        logger.warning("No raw files found for interval '%s' in %s", interval, RAW_DIR)
        return

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    config = LogReturnsConfig()

    for pth in raw_paths:
        logger.info("Building bronze for %s", pth.name)
        df = KiteDataHandler.load(pth)

        if interval == "day":
            df[RawColumns.DATE.value] = df[RawColumns.DATE.value].dt.normalize()
            n_before = len(df)
            df = df.drop_duplicates(
                subset=[RawColumns.SYMBOL.value, RawColumns.DATE.value], keep="first"
            )
            n_dupes = n_before - len(df)
            if n_dupes:
                logger.warning(
                    "%s: dropped %d duplicate (symbol, date) rows after normalization "
                    "(mixed midnight/intraday timestamps in raw data)",
                    pth.name,
                    n_dupes,
                )

        n_before = len(df)
        df = df[df["close"] > 0]
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(
                "%s: dropped %d rows with close <= 0 (zero/phantom price bars from API)",
                pth.name,
                n_dropped,
            )

        df = filter_active_universe(df)
        df = remove_phantom_ticks(df)

        # flag_null_volume is a simple per-row check — no per-symbol loop needed
        df[FlagColumns.NULL_VOLUME] = df[RawColumns.VOLUME.value] == 0

        enriched: list[pd.DataFrame] = []
        for symbol, group in df.groupby(RawColumns.SYMBOL.value, sort=False):
            _, flagged = process_symbol(
                group.copy(),
                symbol=symbol,
                config=config,
                mutate=True,
            )
            if flagged is not None:
                enriched.append(flagged)
            else:
                # symbol had too few observations to compute flags — keep rows unflagged
                group[FlagColumns.LOG_RET] = False
                group[FlagColumns.Z_SCORE] = False
                group[FlagColumns.VOLUME_PRICE_DISPARITY] = False
                enriched.append(group)

        df_bronze = pd.concat(enriched).sort_values(
            [RawColumns.SYMBOL.value, RawColumns.DATE.value]
        )

        out_path = BRONZE_DIR / pth.name
        _save(df_bronze, out_path)
        logger.info("Saved bronze to %s (%d rows)", out_path, len(df_bronze))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bronze layer from raw Kite data.")
    parser.add_argument(
        "--interval",
        default="day",
        choices=SUPPORTED_INTERVALS + ["all"],
        help="Data interval to process, or 'all' for every supported interval.",
    )
    args = parser.parse_args()

    intervals = SUPPORTED_INTERVALS if args.interval == "all" else [args.interval]
    for iv in intervals:
        build_bronze(iv)


if __name__ == "__main__":
    main()

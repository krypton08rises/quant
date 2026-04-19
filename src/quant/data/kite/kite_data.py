"""
Eventually to be made into a service ran everyday as a cronjob to backfill data for indices
and update data for symbols.
Saves data and metadata to pickle files in data/historical directory.
Can be used to keep data updated and model trained on a daily basis.
"""

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant.data._common import RAW_DIR, Indices, RawColumns
from quant.data.kite.kite_handler import KiteDataHandler

INDICES_DIR = Path(__file__).resolve().parents[1] / "indices"

timestamp = date.today().isoformat()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _collect_symbols(index_names: list[str], indices_dir: Path) -> list[str]:
    """
    Read each index CSV and return the deduped union of their ``Symbol`` columns.
    Missing CSVs are warned and skipped so partial universes still work.
    """
    universe: set[str] = set()
    for name in index_names:
        f = indices_dir / f"{name}.csv"
        if not f.exists():
            logger.warning("Index CSV not found, skipping: %s", f)
            continue
        universe.update(pd.read_csv(f)["Symbol"].dropna().unique())
    return sorted(universe)


def _symbol_path(out_dir: Path, symbol: str, interval: str) -> Path:
    return out_dir / f"{symbol}_{interval}.pkl"


async def backfill_symbols(
    symbols: list[str],
    interval: str,
    from_date: date,
    out_dir: Path = RAW_DIR,
    handler: KiteDataHandler | None = None,
) -> None:
    """
    Fetch and persist historical data per symbol. For each symbol:
      - If ``raw/<symbol>_<interval>.pkl`` exists, resume from the day after its max date.
      - Otherwise, fetch from ``from_date`` through today.

    Storage is per-symbol, so overlapping index memberships never trigger duplicate fetches.
    """
    handler = handler or KiteDataHandler()
    today = date.today()

    for sym in symbols:
        out_path = _symbol_path(out_dir, sym, interval)
        existing = handler.load(out_path) if out_path.exists() else None
        has_existing = existing is not None and not existing.empty

        if has_existing:
            last_date = pd.to_datetime(existing[RawColumns.DATE.value]).max().date()
            sym_start = last_date + timedelta(days=1)
        else:
            sym_start = from_date

        if sym_start > today:
            logger.info(
                "%s: already current (last=%s); skipping", sym, sym_start - timedelta(days=1)
            )
            continue

        df_new = await handler.fetch_historical([sym], interval, sym_start, today)
        if df_new is None or df_new.empty:
            logger.info("%s: no new rows from %s to %s", sym, sym_start, today)
            continue

        if has_existing:
            handler.data = (
                pd.concat([existing, df_new], ignore_index=True)
                .drop_duplicates(
                    subset=[RawColumns.SYMBOL.value, RawColumns.DATE.value], keep="last"
                )
                .sort_values([RawColumns.SYMBOL.value, RawColumns.DATE.value])
                .reset_index(drop=True)
            )
        else:
            handler.data = df_new

        handler.save(out_path)


async def backfill_index(
    index_name: str,
    interval: str,
    from_date: date,
    indices_dir: Path = INDICES_DIR,
    out_dir: Path = RAW_DIR,
) -> None:
    """
    Backfill a single index by reading its constituent CSV and fetching each symbol.
    """
    idx_file = indices_dir / f"{index_name}.csv"
    if not idx_file.exists():
        raise FileNotFoundError(f"Index file not found: {idx_file}")

    symbols = _collect_symbols([index_name], indices_dir)
    logger.info("Backfilling %d symbols from %s at interval=%s", len(symbols), index_name, interval)
    await backfill_symbols(symbols, interval, from_date, out_dir)


async def handler() -> None:
    """
    CLI entry: fetch raw data per-symbol for a named index (or ``all``) and interval(s).

    Example
    -------
        python kite_data.py --index nifty_50 --interval day --date 2020-01-01
        python kite_data.py --index all --interval all --date 2020-01-01
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index slug (see Indices enum) or 'all'")
    parser.add_argument(
        "--interval", default="day", help="Bar interval, or 'all' for {5,15,60}minute + day"
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Backfill from this date (YYYY-MM-DD)",
        default=None,
    )
    args = parser.parse_args()

    intervals = (
        ["5minute", "15minute", "60minute", "day"] if args.interval == "all" else [args.interval]
    )

    if args.index == "all":
        index_names = [idx.value for idx in Indices]
    elif args.index:
        index_names = [Indices(args.index).value]
    else:
        logger.error("No --index provided")
        return

    if not args.date:
        logger.error("No --date provided; cold-start backfill requires a from_date")
        return

    # Dedup the union of symbols once per run so overlapping indices don't double-fetch.
    symbols = _collect_symbols(index_names, INDICES_DIR)
    logger.info(
        "Collected %d unique symbols across %d indices (dedup applied)",
        len(symbols),
        len(index_names),
    )

    kite_handler = KiteDataHandler()
    for interval in intervals:
        logger.info("Processing interval=%s (from %s)", interval, args.date)
        await backfill_symbols(symbols, interval, args.date, handler=kite_handler)


if __name__ == "__main__":
    try:
        asyncio.run(handler())
    except KeyboardInterrupt:
        logger.warning("Process Interrupted by user...")

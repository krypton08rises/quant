import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from quant.data._common import (
    BRONZE_DIR,
    CUTOFF_DATE,
    EMBARGOED_DATE_START,
    EntryPriceMode,
    Interval,
    RawColumns,
    SilverColumns,
    TieBreaking,
    TripleBarrierSpec,
    VolMethod,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_entry_price(df: pd.DataFrame, t: int, spec: TripleBarrierSpec) -> float:
    """
    Compute the entry price at time t based on the specification.
    Arguments
    ---------
    df: pd.DataFrame
        The DataFrame containing price data.
    t: int
        The index of the data point.
    spec: TripleBarrierSpec
        The specification for the triple barrier method.
    Returns
    -------
    float
        The entry price at time t.
    """
    assert (
        "open" in df.columns and "close" in df.columns
    ), f"DataFrame must contain 'open' and 'close' columns. Only have: {df.columns.tolist()}"
    if spec.entry == EntryPriceMode.TODAYS_CLOSE:
        return df["close"].iloc[t]
    elif spec.entry == EntryPriceMode.NEXT_OPEN:
        return df["open"].iloc[t + 1]
    raise ValueError(f"Unsupported entry price mode: {spec.entry}")


def compute_atr(df: pd.DataFrame, t: int, spec: TripleBarrierSpec) -> float:
    """
    Compute the volatility measure at time t based on the specification.
    Arguments
    ---------
    df: pd.DataFrame
        The DataFrame containing price data and indicators.
    t: int
        The index of the data point.
    spec: TripleBarrierSpec
        The specification for the triple barrier method.
    Returns
    -------
    float
        The volatility measure at time t.
    """
    if spec.vol_method == VolMethod.ATR:
        return df[SilverColumns.ATR].iloc[t]
    raise ValueError(f"Unsupported volatility method: {spec.vol_method}")


def label_at_t(df: pd.DataFrame, t: int, spec: TripleBarrierSpec) -> int | None:
    """
    Label the data point at time t using the triple barrier method.
    Arguments
    ---------
    df: pd.DataFrame
        The DataFrame containing price data and indicators.
    t: int
        The index of the data point to label.
    spec: TripleBarrierSpec
        The specification for the triple barrier method.
    Returns
    -------
    int | None
        The label for the data point at time t: +1 for profit target hit, -1 for stop loss hit, 0 for neither.
    """

    df = df.reset_index(drop=True)
    entry_price = compute_entry_price(df, t, spec)
    vol = compute_atr(df, t, spec)

    # logger.info(f"At index {t}, entry price: {entry_price}, volatility: {vol}")
    if vol is None or np.isnan(vol) or vol == 0:
        logger.warning(f"Volatility is invalid at index {t}, cannot compute label.")
        return None  # Cannot compute label without valid volatility

    pt_barrier = entry_price + (spec.pt_k * vol)
    sl_barrier = entry_price - (spec.sl_k * vol)

    # start checking from t+1 regardless of entry mode (entry is at t or t+1, barriers from t+1 onward)
    start_idx = t + 1

    # start checking our labels from Jan 2nd
    end_idx = min(t + spec.H, len(df))  # if H=5, check till Jan 6th (exclusive)
    if t + spec.H >= len(df):
        date_val = df[RawColumns.DATE].iloc[t] if RawColumns.DATE in df.columns else t
        logger.warning(
            f"Not enough future data to apply vertical barrier of H={spec.H} days at index {t}. Only have {len(df)-t-1} days ahead. Date is {date_val}."
        )

    # logger.info(f"Checking barriers from index {start_idx} to {end_idx-1} for entry at index {t}. PT: {pt_barrier}, SL: {sl_barrier}")
    for i in range(start_idx, end_idx):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # logger.info(f"Index {i}: High={high}, Low={low}")
        if spec.use_high_low:
            if high >= pt_barrier and low <= sl_barrier:
                # both hit same day, apply tie-breaking rule
                if spec.tie_breaking == TieBreaking.CONSERVATIVE:
                    return -1  # treat as stop loss hit
                elif spec.tie_breaking == TieBreaking.ZERO:
                    return 0  # treat as neither
                # alternatively, can load intraday data to resolve tie
            elif high >= pt_barrier:
                return +1  # profit target hit
            elif low <= sl_barrier:
                return -1  # stop loss hit
        else:
            close = df["close"].iloc[i]
            # logger.info(f"Index {i}: Close={close}")
            if close >= pt_barrier:
                return +1
            elif close <= sl_barrier:
                return -1
    return 0  # neither barrier hit within H days


PER_SYMBOL_SUBDIR = "_per_symbol"
""" Subdirectory under ``<interval>__<config>/`` holding per-symbol labelled caches. """


def _symbol_from_bronze_path(path: Path, interval: Interval) -> str:
    """Derive the symbol name from a bronze filename like ``<SYMBOL>_<interval>.pkl``."""
    suffix = f"_{interval.value}"
    stem = path.stem
    assert stem.endswith(suffix), f"Unexpected bronze filename: {path.name}"
    return stem[: -len(suffix)]


def build_symbol_silver(
    bronze_path: Path,
    interval: Interval,
    spec: TripleBarrierSpec,
    cache_dir: Path,
    force: bool,
) -> Path | None:
    """
    Build labelled silver data for a single (symbol, interval) pair and cache it.

    Returns the path to the per-symbol parquet, or ``None`` if the bronze was
    empty/unreadable. Existing caches are reused unless ``force=True``.
    """
    from quant.data.kite.kite_handler import KiteDataHandler
    from quant.data.processing.indicators import generate_indicators_from_df

    symbol = _symbol_from_bronze_path(bronze_path, interval)
    out_path = cache_dir / f"{symbol}.parquet"
    if out_path.exists() and not force:
        logger.info("Cached silver exists for %s [%s]; skipping.", symbol, interval.value)
        return out_path

    data = KiteDataHandler.load(bronze_path)
    if data is None or data.empty:
        logger.warning("Skipping %s [%s]; bronze empty or missing.", symbol, interval.value)
        return None

    sym_df = data.sort_values(by=RawColumns.DATE).reset_index(drop=True)
    if RawColumns.SYMBOL.value not in sym_df.columns:
        sym_df[RawColumns.SYMBOL.value] = symbol

    sym_df = generate_indicators_from_df(sym_df, tag=RawColumns.CLOSE)
    assert SilverColumns.ATR in sym_df.columns, "ATR indicator must be computed before labelling."

    label_col = f"label_{spec.H}"
    labels = np.full(len(sym_df), np.nan, dtype=float)
    end = max(len(sym_df) - spec.H - 1, 0)
    for t in tqdm(range(end), desc=f"Labeling {symbol} [{interval.value}] H={spec.H}"):
        lab = label_at_t(sym_df, t, spec)
        if lab is not None:
            labels[t] = lab
    sym_df[label_col] = labels

    cache_dir.mkdir(parents=True, exist_ok=True)
    sym_df.to_parquet(out_path)
    logger.info("Wrote per-symbol silver for %s → %s", symbol, out_path)
    return out_path


def assemble_splits(cache_dir: Path, spec_dir: Path) -> None:
    """
    Concatenate every per-symbol silver parquet under ``cache_dir`` and write
    train/test/embargoed splits to ``spec_dir``.
    """
    parts = sorted(cache_dir.glob("*.parquet"))
    if not parts:
        logger.warning("No per-symbol silver files in %s; skipping assembly.", cache_dir)
        return

    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as e:
            logger.error("Failed to read %s: %s", p, e)
    if not frames:
        return

    df_static = pd.concat(frames, ignore_index=True)
    df_static.drop_duplicates(subset=[RawColumns.SYMBOL.value, RawColumns.DATE.value], inplace=True)

    date_col = RawColumns.DATE.value
    train_df = df_static[df_static[date_col] <= CUTOFF_DATE]
    test_df = df_static[
        (df_static[date_col] > CUTOFF_DATE) & (df_static[date_col] < EMBARGOED_DATE_START)
    ]
    embargoed_df = df_static[df_static[date_col] >= EMBARGOED_DATE_START]

    logger.info(
        "Split sizes — train: %d, test: %d, embargoed: %d",
        len(train_df),
        len(test_df),
        len(embargoed_df),
    )
    for df, name in [(train_df, "train"), (test_df, "test"), (embargoed_df, "embargoed")]:
        path = spec_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info("Static labelled dataset saved to %s.", path)


def test_silver_data(df: pd.DataFrame) -> bool:
    """
    Tests the silver data for correctness.
    Arguments
    --------
    df: pd.DataFrame
        The DataFrame containing the labelled data.
    Returns
    -------
    bool
        True if the data passes all tests, False otherwise.
    """
    # assert RSI is between 0 and 100
    if SilverColumns.RSI in df.columns:
        if not df[SilverColumns.RSI].between(0, 100).all():
            logger.error("RSI values out of bounds [0, 100].")
            return False

    return True


SUPPORTED_INTERVALS = ["day", "60minute", "30minute", "15minute", "5minute"]


def build_silver(interval: Interval, force: bool = False) -> None:
    """
    Build the silver dataset for the given interval, one symbol at a time.

    Per-symbol labelled parquets are cached under
    ``<spec_dir>/_per_symbol/<SYMBOL>.parquet``. Cached symbol-interval pairs
    are skipped unless ``force=True``. After all symbols are processed, the
    caches are concatenated and split into train/test/embargoed parquets:

      train     — date <= CUTOFF_DATE
      test      — CUTOFF_DATE < date < EMBARGOED_DATE_START
      embargoed — date >= EMBARGOED_DATE_START
    """
    spec = TripleBarrierSpec(H=5)
    spec_dir = spec.silver_dir(interval)
    cache_dir = spec_dir / PER_SYMBOL_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    bronze_paths = sorted(BRONZE_DIR.glob(f"*_{interval.value}.pkl"))
    if not bronze_paths:
        logger.warning("No bronze files for interval %s in %s.", interval.value, BRONZE_DIR)
        return

    logger.info(
        "Building silver for interval=%s across %d bronze files; force=%s.",
        interval.value,
        len(bronze_paths),
        force,
    )

    for bronze_path in bronze_paths:
        try:
            build_symbol_silver(bronze_path, interval, spec, cache_dir, force=force)
        except Exception as e:
            logger.error("Failed processing %s: %s", bronze_path.name, e, exc_info=True)
            # Continue so a single bad symbol doesn't abort the whole run.

    assemble_splits(cache_dir, spec_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build silver layer from bronze data.")
    parser.add_argument(
        "--interval",
        default="day",
        choices=SUPPORTED_INTERVALS + ["all"],
        help="Data interval to process, or 'all' for every supported interval.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess symbol-interval pairs even if a cached silver parquet exists.",
    )
    args = parser.parse_args()

    intervals = SUPPORTED_INTERVALS if args.interval == "all" else [args.interval]
    for iv in intervals:
        build_silver(Interval(iv), force=args.force)


if __name__ == "__main__":
    main()

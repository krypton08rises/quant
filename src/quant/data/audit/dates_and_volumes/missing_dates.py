from pathlib import Path

import numpy as np
import pandas as pd

from quant.data._common import RAW_DIR, Interval, RawColumns
from quant.data.kite.kite_handler import KiteDataHandler
from quant.logs.logging import logger

# NSE normal session bounds (IST offset)
_NSE_OPEN = pd.Timedelta(hours=9, minutes=15)
_NSE_CLOSE = pd.Timedelta(hours=15, minutes=30)


def _expected_intraday_bars(
    start: pd.Timestamp, end: pd.Timestamp, interval: Interval
) -> pd.DatetimeIndex:
    """
    Generate all expected bar timestamps within NSE trading hours
    (09:15–15:30 IST) on business days between *start* and *end*.
    """
    freq = f"{interval.minutes}min"
    tz = start.tz
    trading_days = pd.bdate_range(
        start=start.normalize(),
        end=end.normalize(),
        tz=tz,
    )
    sessions = [
        pd.date_range(start=day + _NSE_OPEN, end=day + _NSE_CLOSE, freq=freq)
        for day in trading_days
    ]
    if not sessions:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([s.values for s in sessions]))


def find_missing_dates(
    df: pd.DataFrame, date_col: RawColumns, interval: Interval
) -> pd.DatetimeIndex:
    """
    Identify missing bars in a DataFrame for the given interval.

    For daily data: checks for missing business days (freq="B").
    For intraday: checks for missing bars within NSE trading hours (09:15–15:30 IST).

    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing a date/timestamp column.
    date_col: RawColumns
        The name of the date column in the DataFrame.
    interval: Interval
        The bar interval — determines what counts as a missing bar.

    Returns
    -------
    pd.DatetimeIndex
        Missing bar timestamps (dates for daily, full timestamps for intraday).
    """
    actual = pd.DatetimeIndex(df[date_col.value])

    if interval == Interval.DAY:
        actual_dates = actual.normalize()
        all_dates = pd.date_range(start=actual_dates.min(), end=actual_dates.max(), freq="B")
        return pd.DatetimeIndex(all_dates.difference(actual_dates))

    expected = _expected_intraday_bars(actual.min(), actual.max(), interval)
    return pd.DatetimeIndex(expected.difference(actual))


def resolve_holidays(grouped_missing: pd.Series) -> pd.Series:
    """
    Consensus-based approach to identify holidays from missing bar patterns.

    Aggregates missing bars to date-level (so intraday and daily are treated
    consistently) and counts how many symbols are missing each date.

    Arguments
    ---------
    grouped_missing: pd.Series
        A Series whose values are DatetimeIndex of missing bars per symbol.

    Returns
    -------
    pd.Series
        Value counts of missing dates (date-normalised), descending.
    """
    all_missing = np.concatenate([idx.normalize().unique() for idx in grouped_missing.values])
    return pd.Series(all_missing).value_counts().sort_values(ascending=False)


def remove_phantom_ticks(
    df: pd.DataFrame,
    date_col: RawColumns = RawColumns.DATE,
    symbol_col: RawColumns = RawColumns.SYMBOL,
    min_phantom_gap_days: int = 45,
) -> pd.DataFrame:
    """
    Drop pre-IPO / phantom-tick history by removing rows before the latest large gap per symbol.

    A gap of ``min_phantom_gap_days`` far exceeds any normal overnight or weekend break at
    any supported interval, so it reliably identifies pre-IPO or back-filled artefact data.

    Arguments
    ---------
    df : pd.DataFrame
        Input DataFrame containing at least *date_col* and *symbol_col*.
    date_col : RawColumns
        Column holding the bar timestamp.
    symbol_col : RawColumns
        Column holding the ticker symbol.
    min_phantom_gap_days : int
        Minimum gap (in calendar days) that is treated as a phantom-tick boundary.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with pre-gap rows removed; helper columns dropped.
    """
    df = df.sort_values([symbol_col.value, date_col.value])
    df["time_jump"] = df.groupby(symbol_col.value)[date_col.value].diff()

    logger.debug(
        f"Top 10 biggest time jumps:\n"
        f"{df[[symbol_col.value, date_col.value, 'time_jump']].nlargest(10, 'time_jump')}"
    )

    threshold = pd.Timedelta(days=min_phantom_gap_days)
    big_gaps = df[df["time_jump"] >= threshold]

    if big_gaps.empty:
        logger.info("No phantom ticks detected. Returning original DataFrame.")
        return df.drop(columns=["time_jump"])

    true_starts = big_gaps.groupby(symbol_col.value)[date_col.value].max()
    logger.info(f"Identified {len(true_starts)} stocks with phantom histories.")

    safe_min_date = df[date_col.value].min()
    df["true_start"] = df[symbol_col.value].map(true_starts).fillna(safe_min_date)
    clean_df = df[df[date_col.value] >= df["true_start"]].copy()

    return clean_df.drop(columns=["time_jump", "true_start"])


def filter_active_universe(
    df: pd.DataFrame,
    symbol_col: RawColumns = RawColumns.SYMBOL,
    date_col: RawColumns = RawColumns.DATE,
) -> pd.DataFrame:
    """
    Remove symbols whose last trading bar predates 2026 (delisted / inactive stocks).

    Arguments
    ---------
    df : pd.DataFrame
        Input DataFrame containing at least *symbol_col* and *date_col*.
    symbol_col : RawColumns
        Column holding the ticker symbol.
    date_col : RawColumns
        Column holding the bar timestamp.

    Returns
    -------
    pd.DataFrame
        Filtered copy retaining only symbols with at least one bar on or after 2026-01-01.
    """
    last_active = df.groupby(symbol_col.value)[date_col.value].max()
    active_symbols = last_active[last_active >= pd.Timestamp("2026-01-01", tz="Asia/Kolkata")].index
    logger.info(f"Filtering universe to {len(active_symbols)} active stocks with data in 2026.")
    return df[df[symbol_col.value].isin(active_symbols)].copy()


def get_market_holidays(
    df: pd.DataFrame, missing_dates: pd.Series, interval: Interval
) -> tuple[set, set]:
    """
    Identify market holidays using a dynamic-denominator consensus approach.

    A date is classified as a holiday when ≥ 90 % of stocks that were active on that
    date have no bars for it. Operates at date granularity so intraday and daily data
    are handled consistently.

    Arguments
    ---------
    df : pd.DataFrame
        Full DataFrame used to determine each symbol's first and last traded date.
    missing_dates : pd.Series
        Per-symbol Series whose values are ``DatetimeIndex`` objects of missing bars
        (output of a ``groupby(...).apply(find_missing_dates)`` call).
    interval : Interval
        Data interval; intraday timestamps are normalised to dates before comparison.

    Returns
    -------
    tuple[set, set]
        ``(holidays, unresolved_dates)`` where *holidays* are dates explained by
        market closure and *unresolved_dates* are potential data quality issues.
    """
    first_traded = df.groupby(RawColumns.SYMBOL.value)[RawColumns.DATE.value].min()
    last_traded = df.groupby(RawColumns.SYMBOL.value)[RawColumns.DATE.value].max()

    if interval != Interval.DAY:
        first_traded = first_traded.dt.normalize()
        last_traded = last_traded.dt.normalize()

    dates_vc = resolve_holidays(missing_dates)
    identified, unresolved = 0, 0
    unresolved_dates: set = set()

    for date, count in dates_vc.items():
        norm_date = date.normalize() if interval != Interval.DAY else date
        active_symbols = ((first_traded <= norm_date) & (last_traded >= norm_date)).sum()
        if active_symbols and count / active_symbols >= 0.9:
            identified += 1
        else:
            unresolved += 1
            unresolved_dates.add(date)
            if not active_symbols:
                logger.warning(
                    f"Date {date.date()} missing for {count} symbols but only "
                    f"{active_symbols} were active — possible data issue."
                )

    logger.info(
        f"Identified holidays: {identified} | "
        f"Unresolved missing (potential data issues): {unresolved} / {len(dates_vc)}"
    )
    return set(dates_vc.index) - unresolved_dates, unresolved_dates


def audit_data_quality(missing_dates: pd.Series, holidays: set) -> list:
    """
    Flag symbols whose missing bars are not explained by known holidays.

    Comparison is at date level so intraday bars on holiday dates are excluded.

    Arguments
    ---------
    missing_dates: pd.Series
        Per-symbol Series of missing bar DatetimeIndex.
    holidays: set
        Dates identified as market holidays.

    Returns
    -------
    list
        Top 10 (symbol, unexplained_gap_count) tuples.
    """
    holiday_dates = {h.normalize() for h in holidays}
    bad_apples = {}
    for symbol, missing in missing_dates.items():
        unexplained = len(set(missing.normalize()) - holiday_dates)
        if unexplained > 0:
            bad_apples[symbol] = unexplained

    worst_stocks = sorted(bad_apples.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"Top 10 symbols with unexplained missing bars: {worst_stocks}")
    return worst_stocks


def main(interval: Interval = Interval.DAY) -> None:
    """
    Run the missing-bars audit for the given interval across all matching bronze files.
    """
    for pth in Path(RAW_DIR).glob(f"*_{interval.value}.pkl"):
        logger.info(f"Auditing {pth.name} ...")
        df = KiteDataHandler.load(pth)

        # For daily data, strip the intraday time component before date comparisons
        if interval == Interval.DAY:
            df[RawColumns.DATE.value] = df[RawColumns.DATE.value].dt.normalize()

        df = filter_active_universe(df)
        df = remove_phantom_ticks(df)

        missing = df.groupby(RawColumns.SYMBOL.value).apply(
            lambda x: find_missing_dates(x, date_col=RawColumns.DATE, interval=interval)
        )
        holidays, _ = get_market_holidays(df, missing, interval)
        audit_data_quality(missing, holidays)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd 
from pathlib import Path
from functools import reduce

from quant.logs.logging import logger
from quant.data._common import BRONZE_DIR, BronzeColumns
from quant.data.kite.kite_handler import KiteDataHandler

def find_missing_dates(df: pd.Series, date_col: BronzeColumns) -> pd.DatetimeIndex:
    """
    Identify missing dates in a DataFrame excluding weekends and holidays.
    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing a date column.
    date_col: BronzeColumns
        The name of the date column in the DataFrame.
    """    
    all_dates = pd.date_range(start=df[date_col.value].min(), end=df[date_col.value].max(), freq='B')  # Business days only
    missing_dates = all_dates.difference(df[date_col.value])

    # Ensure what we return is a DatetimeIndex for consistency
    return pd.DatetimeIndex(missing_dates)


def resolve_holidays(grouped_missing_dates:pd.Series) -> pd.Series:
    """
    Consensus based approach to identify and remove known holidays from the list of missing dates. If most stocks are missing the same date, it's likely a holiday. This can be refined with an actual holiday calendar in the future.
    
    Arguments
    ---------
    grouped_missing_dates: pd.Series
        A series of DatetimeIndex for each symbol.
    Returns
    -------
    pd.DatetimeIndex
        The list of missing dates after removing known holidays.
    """
    all_missing = np.concatenate(list(grouped_missing_dates.values))
    missing_series = pd.Series(all_missing)

    return missing_series.value_counts().sort_values(ascending=False)


def last_traded_date(df: pd.Series, date_col: BronzeColumns) -> pd.Timestamp:
    """
    Get the last traded date from the DataFrame.
    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing a date column.
    date_col: BronzeColumns
        The name of the date column in the DataFrame.
    Returns
    -------
    pd.Timestamp
        The last traded date in the DataFrame.
    """
    return df[date_col.value].max()


def main():
    """
    Parse through all bronze data files, identify missing dates for each symbol besides weekends and holidays, and log the results.

    """
    for pth in Path(BRONZE_DIR).glob("*day.pkl"):
        logger.info(f"Processing Symbol: {pth.stem.split('_')[0]}")
        df = KiteDataHandler.load(pth)

        # Test last traded date was in 2026 to ensure we have recent data
        last_dates = df.groupby(BronzeColumns.SYMBOL.value).apply(lambda x: last_traded_date(x, date_col=BronzeColumns.DATE))
        if (last_dates < pd.Timestamp("2026-01-01", tz='Asia/Kolkata')).any():
            # filter out symbols with last traded date before 2026 for more accurate missing date analysis
            last_dates = last_dates[last_dates >= pd.Timestamp("2026-01-01", tz='Asia/Kolkata')]
            logger.warning(f"Last traded date for some symbols is before 2026: {len(last_dates)} symbols will be included in missing date analysis, {len(df[BronzeColumns.SYMBOL.value].unique()) - len(last_dates)} symbols will be excluded")
        # Following line converts the missing_dates to a numpy.ndarray, which is failing the resolve_holidays function. We need to ensure it remains a DatetimeIndex for consistency.

        missing_dates = df.groupby(BronzeColumns.SYMBOL.value).apply(lambda x: find_missing_dates(x, date_col=BronzeColumns.DATE))

        resolved_holidays = resolve_holidays(missing_dates)

        # We have a pandas series with value counts of missing dates across all symbols. 
        # See if 95% of the symbols are missing the same date, 
        # Also see if the remaining 5% of the missing dates are < 2010, which is fine because only a small number of stocks would have been actively traded back then. We are concerned with all data.
        total_symbols = len(df[BronzeColumns.SYMBOL.value].unique())
        holiday_threshold = total_symbols * 0.9
        holidays = resolved_holidays[resolved_holidays >= holiday_threshold].index
        logger.info(f"Identified {len(holidays)} holidays based on missing date consensus ")    
        logger.info(f"Number of total symbols:{total_symbols} \n Number of days missing from 95% of symbols: {len(holidays)} \n Remaining missing dates: {len(resolved_holidays) - len(holidays)}")


if __name__ == "__main__":
    main()
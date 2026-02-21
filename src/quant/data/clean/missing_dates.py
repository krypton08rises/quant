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

        
def remove_phantom_ticks(df: pd.DataFrame, date_col: BronzeColumns = BronzeColumns.DATE, symbol_col: BronzeColumns = BronzeColumns.SYMBOL, min_phantom_gap_days: int = 45) -> pd.DataFrame:
    """
    Identifies massive gaps (phantom ticks -> IPO date) and removes all rogue data before the *latest* massive gap.
    """
    df = df.sort_values([symbol_col.value, date_col.value])
    df['time_jump'] = df.groupby(symbol_col.value)[date_col.value].diff()

    # log rows of top 5 biggest time jumps for debugging
    logger.debug(f"Top 5 biggest time jumps:\n{df[[symbol_col.value, date_col.value, 'time_jump']].nlargest(10, 'time_jump')}")

    # 1. Find ALL gaps that exceed the threshold
    threshold = pd.Timedelta(days=min_phantom_gap_days)
    big_gaps = df[df['time_jump'] >= threshold]

    if big_gaps.empty:
        logger.info("No phantom ticks detected based on the current threshold. Returning original DataFrame.")
        return df.drop(columns=['time_jump'])
    
    # 2. If a stock has multiple big gaps, the TRUE start is the date of the LATEST big gap
    true_starts = big_gaps.groupby(symbol_col.value)[date_col.value].max()
    
    logger.info(f"Identified {len(true_starts)} stocks with phantom histories.")
    
    # 3. Map the true start dates to the main dataframe. Fill healthy stocks with an ancient date.
    # Use the earliest date in the entire dataset as the safe default
    safe_min_date = df[date_col.value].min()

    df['true_start'] = df[symbol_col.value].map(true_starts).fillna(safe_min_date)    
    # 4. Vectorized Filtering: Keep only rows where date >= true_start
    clean_df = df[df[date_col.value] >= df['true_start']].copy()
    
    return clean_df.drop(columns=['time_jump', 'true_start'])        
        
def filter_active_universe(
    df: pd.DataFrame, 
    symbol_col: BronzeColumns = BronzeColumns.SYMBOL, 
    date_col: BronzeColumns = BronzeColumns.DATE
) -> pd.DataFrame:
    """
    Filters out stock with last active trading day before 2026.
    
    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing stock data.
    symbol_col: BronzeColumns
        The column name representing the stock symbol.
    date_col: BronzeColumns
        The column name representing the trading date.

    Returns
    -------
    pd.DataFrame    
        A filtered DataFrame containing only active stocks with trading data in 2026.
    """
    last_active = df.groupby(symbol_col.value)[date_col.value].max()
    active_symbols = last_active[last_active >= pd.Timestamp("2026-01-01", tz='Asia/Kolkata')].index
    logger.info(f"Filtering universe to {len(active_symbols)} active stocks with trading data in 2026.")
    return df[df[symbol_col.value].isin(active_symbols)].copy()


def get_market_holidays(df: pd.DataFrame, missing_dates: pd.Series) -> set:
    """
    Does the dynamic denominator math to return a verified list of market holidays based on active stocks and missing date patterns.
    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing stock data.
    missing_dates: pd.Series    
        A series of missing dates across stocks.
    Returns
    ------- 
    set
        A set of dates that are likely market holidays based on the analysis.
    """ 
    first_traded_dates = df.groupby(BronzeColumns.SYMBOL.value)[BronzeColumns.DATE.value].min()
    last_traded_dates = df.groupby(BronzeColumns.SYMBOL.value)[BronzeColumns.DATE.value].max()

    dates_vc = resolve_holidays(missing_dates)
    identified, unresolved = 0, 0 
    unresolved_dates = set()
    for date, count in dates_vc.items():
        # for symbols having data before this date, because obviously 'missing_dates' would not be present for the stock
        active_symbols = ((first_traded_dates <= date) & (last_traded_dates >= date)).sum()
        if active_symbols and count / active_symbols >= 0.9:
            identified +=1
        else:
            unresolved+=1
            unresolved_dates.add(date)
            if not active_symbols:
                logger.warning(f"Date {date.date()} is missing for {count} symbols but only {active_symbols} were active. This may indicate a data issue rather than a holiday.")
    logger.info(f"Total unresolved missing dates (potential data issues): {unresolved} out of {len(dates_vc)} total missing dates. | Identified holidays: {identified}")
    return set(dates_vc.index) - unresolved_dates, unresolved_dates


def audit_data_quality(missing_dates: pd.Series, holidays: set) -> list:
    """
    Audits the missing dates against identified holidays to flag stocks with potential data quality issues.
    Arguments
    ---------
    missing_dates: pd.Series
        A series of missing dates across stocks.
    holidays: set
        A set of dates that are likely market holidays based on the analysis.
    Returns
    -------
    list
        A list of stocks with the most missing dates that are not explained by holidays, indicating potential data quality issues.
    """
    bad_apples = {}
    for symbol, missing in missing_dates.items():
        gaps = len(missing.difference(holidays))
        if gaps > 0:
            bad_apples[symbol] = gaps
    worst_stocks = sorted(bad_apples.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"Top 10 stocks with most missing dates (potential data issues): {worst_stocks}")
    return worst_stocks

def main():
    for pth in Path(BRONZE_DIR).glob("*day.pkl"):
        df = KiteDataHandler.load(pth)
        df[BronzeColumns.DATE.value] = df[BronzeColumns.DATE.value].dt.normalize()
        
        # 1. Filter out dead stocks
        df = filter_active_universe(df)
        
        # 2. Clean known data artifacts (Phantom Ticks) FIRST
        df = remove_phantom_ticks(df)
        
        # 3. Now run your missing dates logic on the CLEANED data
        missing_dates = df.groupby(BronzeColumns.SYMBOL.value).apply(lambda x: find_missing_dates(x, date_col=BronzeColumns.DATE))
        holidays, unresolved_dates = get_market_holidays(df, missing_dates)
        
        # 4. Audit what's left
        worst_stocks = audit_data_quality(missing_dates, holidays)
        
        # 5. Save if successful
        # if not worst_stocks:
        #      KiteDataHandler.save(df, pth.with_name(pth.stem.replace('_day', '_day_cleaned') + |.suffix))

            
if __name__ == "__main__":
    main()
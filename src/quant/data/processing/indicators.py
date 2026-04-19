import numpy as np
import pandas as pd

from .._common import SilverColumns


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Compute the Exponential Moving Average (EMA) of a pandas Series. It responds faster to change in prices than sma, given the same span. Lower alpha means higher span, thus slower response.
    Formula: EMA_t = (Value_t * (α)) + (EMA_{t-1} * (1 - α))
    where α = 2 / (span + 1)

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


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing.
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss  (for given period)

    Arguments
    ---------
    series: pd.Series
        The input time series of closing prices.
    period: int
        The lookback period for the RSI.
    Returns
    -------
    pd.Series
        The RSI of the input series.
    """
    diff = series.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute the Moving Average Convergence Divergence (MACD) of a pandas Series.
    Formula:
        MACD Line = EMA_fast - EMA_slow
        Signal Line = EMA_signal of MACD Line
        Histogram = MACD Line - Signal Line

    Arguments
    ---------
    series: pd.Series
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
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    series: pd.Series, window: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute the Bollinger Bands for a pandas Series.
    Formula:
        Middle Band = SMA(window)
        Upper Band = Middle Band + k * stddev(window)
        Lower Band = Middle Band - k * stddev(window)
        %B = (Price - Lower Band) / (Upper Band - Lower Band)

    Arguments
    ---------
    series: pd.Series
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
    mid = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    upper = mid + k * std
    lower = mid - k * std
    # position of close within the band (useful feature)
    pct = (series - mid) / (upper - lower)
    return mid, upper, lower, pct


def atr(high: pd.Series, low: pd.Series, prev_close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Average True Range (ATR) of a pandas Series.
    Formula: TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
            ATR = EMA(TR, period)
    Arguments
    ---------
    high: pd.Series
        The input time series of high prices.
    low: pd.Series
        The input time series of low prices.
    prev_close: pd.Series
        The input time series of previous close prices.
    period: int
        The lookback period for the ATR; default is 14.
    Returns
    -------
    pd.Series
        The ATR of the input series.
    """
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1 / period, adjust=False, ignore_na=True).mean()


def generate_indicators_from_df(
    df: pd.DataFrame,
    tag: str,
) -> pd.DataFrame:
    """
    Generate a DataFrame of technical indicators from a price series.

    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing price data.
    tag: str
        The column name in df representing the price series. Should be `close` in most cases.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated indicators.
    """
    df[SilverColumns.EMA_12] = ema(df[tag], span=12)
    df[SilverColumns.EMA_26] = ema(df[tag], span=26)

    df[SilverColumns.RSI] = rsi_wilder(df[tag])
    macd_line, signal_line, hist = macd(df[tag])
    df[SilverColumns.MACD_LINE] = macd_line
    df[SilverColumns.MACD_SIGNAL] = signal_line
    df[SilverColumns.MACD_HIST] = hist
    mid, upper, lower, pct = bollinger(df[tag])
    df[SilverColumns.BB_MID] = mid
    df[SilverColumns.BB_UPPER] = upper
    df[SilverColumns.BB_LOWER] = lower
    df[SilverColumns.BB_PCT] = pct

    df[SilverColumns.ATR] = atr(
        high=df["high"],  # Using close prices as a proxy for high/low for simplicity
        low=df["low"],
        prev_close=df[tag].shift(1),
        period=14,
    )

    return df

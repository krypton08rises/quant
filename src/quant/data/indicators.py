import numpy as np
import pandas as pd
from typing import Tuple


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

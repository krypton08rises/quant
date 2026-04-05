import pandas as pd

from pathlib import Path
from enum import Enum, StrEnum 
from dataclasses import dataclass


EMBARGOED_DATE_START = pd.Timestamp('2025-01-01', tz='UTC+05:30')

AUDIT_DIR = Path('data/audit/analysis/')
""" Directory for audit data (log returns, histograms, etc.)."""

BRONZE_DIR = Path('data/historical/bronze')
""" Directory for bronze historical data (raw + candle features)."""

SILVER_DIR = Path('data/historical/silver')
""" Directory for silver historical data (with technical indicators and labels)."""

MAX_DAYS_PER_CALL = 100
""" Maximum days per API call to Kite for historical data. """

CUTOFF_DATE = pd.Timestamp('2020-12-31', tz='UTC+05:30')
""" Cutoff date for historical data processing. """

INTERVAL_LOOKBACK = {
    'day': 365,
    '60minute': 60,
    '30minute': 60,
    '15minute': 60,
    '10minute': 60,
    '5minute': 60,
    '3minute': 60,
    '1minute': 60,
}
""" Maximum lookback days per interval for historical data fetching from Kite API. """

class Indices(Enum):
    BANKNIFTY = 'banknifty'
    NIFTY = 'nifty_50'
    MIDCAP150 = 'midcap_150'
    NIFTYNEXT50 = 'nifty_next_50'
    NIFTYFINSERV = 'nifty_finserv'
""" Supported market indices. """

class Interval(Enum):
    DAY = 'day'
    MINUTE_60 = '60minute'
    MINUTE_30 = '30minute'
    MINUTE_15 = '15minute'
    MINUTE_10 = '10minute'
    MINUTE_5 = '5minute'
    MINUTE_3 = '3minute'
    MINUTE_1 = '1minute'

    @property
    def minutes(self):
        if self == Interval.DAY:
            return 1440
        return int(self.value.replace('minute_',''))
""" Minutes corresponding to the interval. """

class BronzeColumns(StrEnum):
    SYMBOL = 'symbol'
    DATE = 'date'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'
    VWAP = 'vwap'
    CANDLE_FEATURES_START = 'candle_'
""" Standardized column names for bronze data. """

class SilverColumns(StrEnum):
    RSI = 'rsi'
    MACD = 'macd'
    EMA_12 = 'ema_12'
    EMA_26 = 'ema_26'
    BB_UPPER = 'bb_upper'
    BB_LOWER = 'bb_lower'
    BB_MID = 'bb_mid'
    BB_PCT = 'bb_pct'
    MACD_LINE = 'macd_line'
    MACD_SIGNAL = 'macd_signal'
    MACD_HIST = 'macd_hist'
    ATR = 'atr'
    LABEL_3DAY = 'label_3'
    LABEL_5DAY = 'label_5'
    LABEL_7DAY = 'label_7'
""" Standardized column names for silver data. """


class BarrierMode(Enum):
    """
    Barrier calculation modes.
    """
    RETURNS_BARRIERS = "returns barriers"
    # Future extensions possible

class TieBreaking(Enum):
    """
    Tie-breaking rules when both barriers are hit on the same day.
    """
    CONSERVATIVE = "conservative"
    ZERO = "zero"

class EntryPriceMode(Enum):
    """
    Entry price calculation modes.
    """
    NEXT_OPEN = "NEXT_OPEN"
    TODAYS_CLOSE = "TODAYS_CLOSE"

class VolMethod(Enum):
    """
    Volatility calculation methods.
    """
    ATR = "ATR"
    STD = "STD"
    
@dataclass
class TripleBarrierSpec:
    """
    Specification for the triple barrier method.
    Attributes
    ----------
    use_high_low: bool
        Whether to use high/low prices to detect barrier hits.
    vol_method: VolMethod
        Method for volatility calculation ("ATR" or "STD").     
    atr_length: int
        Length for ATR calculation.
    pt_k: float
        Multiplier for profit target barrier.
    sl_k: float
        Multiplier for stop loss barrier.
    H: int
        Vertical barrier in trading days.
    entry: EntryPriceMode
        Mode for entry price calculation.
    barrier_mode: BarrierMode
        Mode for barrier calculation.
    tie_breaking: TieBreaking   
        Tie-breaking rule when both barriers are hit on the same day.
    Returns
    -------
    TripleBarrierSpec    
    """
    use_high_low: bool = True  # True (daily uses high/low to detect hits); False (intraday uses close prices to detect hits)
    vol_method: VolMethod = VolMethod.ATR  # or "STD"
    atr_length: int = 14
    pt_k: float = 1.0
    sl_k: float = 1.0
    H: int = 5  # vertical barrier in trading days
    entry: EntryPriceMode = EntryPriceMode.NEXT_OPEN
    barrier_mode: BarrierMode = BarrierMode.RETURNS_BARRIERS
    tie_breaking: TieBreaking = TieBreaking.CONSERVATIVE    

    config_name: str = ""

    def config_str(self):
        return f"H{self.H}_pt{self.pt_k}_sl{self.sl_k}_{self.vol_method.value}_{self.entry.value}_{self.barrier_mode.value}_{self.tie_breaking.value}"
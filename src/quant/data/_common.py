from enum import Enum


# Lookback days for each interval   
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

class Indices(Enum):
    BANKNIFTY = 'banknifty'
    NIFTY = 'nifty_50'
    MIDCAP150 = 'midcap_150'
    NIFTYNEXT50 = 'nifty_next_50'
    NIFTYFINSERV = 'nifty_finserv'

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

# Maximum days per API call (chunk size)
MAX_DAYS_PER_CALL = 100
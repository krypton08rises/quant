
from pathlib import Path

EXCLUDE_COLS = {
    "date", "fwd_pct_h1", "label_h1",  # time/targets
}
""" You can add anything else you want to exclude explicitly (e.g., raw open/high/low if present)"""

TARGET_COL = "label_h1"
""" target column name from gold dataset"""

GOLD_DIR = Path("data/historical/gold") 
""" Path to gold dataset directory """

HIDDEN_LAYERS:tuple[int] = (256, 128, 64, 16)
""" Default hidden layer sizes for MLP models """


NUM_COLS = ['upper_shadow', 'lower_shadow', 'tick_body', 'diff', 'raw_percent_change',
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'bb_mid', 'bb_up', 'bb_lo', 'bb_pct', 'prev_close', 'atr_14', 'sma_5',
    'ema_5', 'close_over_sma_5', 'sma_10', 'ema_10', 'close_over_sma_10',
    'sma_20', 'ema_20', 'close_over_sma_20', 'volatility_10', 
]
""" List of numeric feature columns in the gold dataset """

TO_BE_NORMALIZED_COLS = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_mid", "bb_up", "bb_lo", "attr_14", "volatility_10",
    "sma_5", "ema_5", "sma_10", "ema_10", "sma_20", "ema_20",
]
""" 
List of numeric columns to be normalized {make uniform in build_static_dataset}
df[TO_BE_NORMALIZED_COLS]=(df[TO_BE_NORMALIZED_COLS]-df[TO_BE_NORMALIZED_COLS].mean())/df[TO_BE_NORMALIZED_COLS].std()
"""
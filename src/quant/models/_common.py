
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


num_cols = ['upper_shadow', 'lower_shadow', 'tick_body', 'diff', 'percent_change',
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'bb_mid', 'bb_up', 'bb_lo', 'bb_pct', 'prev_close', 'atr_14', 'sma_5',
    'ema_5', 'close_over_sma_5', 'sma_10', 'ema_10', 'close_over_sma_10',
    'sma_20', 'ema_20', 'close_over_sma_20', 'volatility_10', 
]
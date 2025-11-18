from pathlib import Path

EXCLUDE_COLS = {
    "date", "fwd_pct_h1", "label_h1",  # time/targets
}
# You can add anything else you want to exclude explicitly (e.g., raw open/high/low if present)
GOLD_DIR = Path("data/historical/gold")
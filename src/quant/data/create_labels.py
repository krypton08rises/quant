# datasets/streaming_sequence.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import random 
import numpy as np
from typing import Optional, Tuple, Dict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Logs are for printing while code execution only, not to be saved anywhere

import pandas as pd

try:
    import torch
    from torch.utils.data import IterableDataset
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# -----------------------
# Config & Utilities
# -----------------------

@dataclass
class StreamConfig:
    """
    Configuration for streaming data processing.
    """
    folder_path: Path = Path("data/historical/")
    file_glob: str = "*day.pkl"
    label_col: str = "percent_change"     # unscaled, per your comment  
    label_unit: str = "pct"               # "pct" -> ±2 means ±2 percentage points; "fraction" -> thresholds will be scaled by 100
    up_thresh:float = 3.0
    down_thresh: float = -3.0             # < -2 => class 1
    window_len: int = 30                  # L
    horizon: int = 1                      # k steps ahead label is computed at t+horizon using info up to t
    step: int = 1                         # slide stride
    balance: bool = True                  # class-balanced sampling
    features: Tuple[str, ...] = (
        "close", 
        "volume",
        "tick_body", 
        "upper_shadow",         
        "lower_shadow", 
        "diff"
    )
    include_symbol_onehot: bool = True
    max_date: Optional[pd.Timestamp] = None   # walk-forward cutoff (inclusive)
    random_seed: int = 42
    shuffle_files: bool = True
    # If your pickles contain scalers per (symbol, column)
    use_scalers: bool = True
    scaler_key_format: str = "{sym}_{col}"

    # Optional caching of vocab for reproducibility
    symbol_vocab: Optional[Dict[str, int]] = None


def _coerce_ts(x) -> Optional[pd.Timestamp]:
    """ Converts series to pd datetime """
    if x is None:
        return None
    return pd.to_datetime(x)


def _list_pickles(cfg: StreamConfig) -> List[Path]:
    """ Lists all files of a given pattern in the historical data folder """
    files = sorted(cfg.folder_path.glob(cfg.file_glob))
    if cfg.shuffle_files:
        rnd = random.Random(cfg.random_seed)
        rnd.shuffle(files)
    return files


def _load_pickle(file: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """  Load a pickled DataFrame and its scalers from a file """
    logger.debug(f"Loading data from {file}")
    with file.open("rb") as f:
        dataset = dill.load(f)
    data = dataset.get("data", pd.DataFrame()).copy()
    scalers = dataset.get("scalers", {}) or {}
    return data.dropna(axis=0), scalers


def _build_symbol_vocab(files: List[Path], cfg: StreamConfig) -> Dict[str, int]:
    """
    Builds a vocabulary mapping from symbol names to integer IDs.
    """
    if cfg.symbol_vocab is not None:
        return cfg.symbol_vocab
    symbols: List[str] = []
    for fp in files:
        df, _ = _load_pickle(fp)
        if "symbol" in df.columns:
            symbols.extend(df["symbol"].dropna().astype(str).unique().tolist())
    vocab = {sym: i for i, sym in enumerate(sorted(set(symbols)))}
    logger.info(f"Built symbol vocabulary with {len(vocab)} unique symbols.")
    return vocab


def _compute_labels(
    df: pd.DataFrame, cfg: StreamConfig,
) -> pd.DataFrame:
    """
    Compute labels from **unscaled** percent_change, respecting unit.
    label: 0 if down_thresh <= x <= up_thresh
           1 if x < down_thresh
           2 if x > up_thresh
    """
    if cfg.label_unit == "fraction":
        # Convert fraction (e.g., 0.021) into percent points (2.1)
        raw = df["rescaled_percent_change"] * 100.0
    else:
        raw = df["rescaled_percent_change"]

    def enc(x: float) -> int:
        if x < cfg.down_thresh:
            return 1
        elif x > cfg.up_thresh:
            return 2
        return 0

    df = df.copy()
    df["label"] = raw.apply(enc).astype("int8")
    logger.debug(f"Computed labels: {df['label'].value_counts()}")
    return df


def _maybe_trim_by_date(df: pd.DataFrame, cfg: StreamConfig) -> pd.DataFrame:
    """
    Optionally trims the DataFrame by the maximum date.
    """
    if cfg.max_date is None:
        return df
    maxd = _coerce_ts(cfg.max_date)
    logger.debug(f"Trimming data to max date: {maxd}")
    if "ds" not in df.columns:
        raise ValueError("Expected a 'ds' datetime column in the data.")
    out = df[df["ds"] <= maxd].copy()
    return out


def _apply_scalers_per_symbol(
    df: pd.DataFrame,
    scalers: Dict[str, object],
    cfg: StreamConfig
) -> pd.DataFrame:
    
    """
    Loops through all symbols in dataframe  and applies inverse_transform on only the percent_change column -- we want everything else to stay normalized..  
    Arguments
    ---------
    df: The input DataFrame containing the data to be scaled.
    scalers: A dictionary mapping column names to their respective scaler objects.
    cols: The list of columns to apply scaling to.
    cfg: The configuration object containing settings for scaling.

    Returns
    -------
    pd.DataFrame: The rescaled DataFrame.
    """
    if not cfg.use_scalers or not scalers:
        return df

    df = df.copy()
    if "symbol" not in df.columns:
        return df

    for sym, group in df.groupby("symbol"):
        mask = df["symbol"] == sym
        scaler_key = f"{sym}_{cfg.label_col}"
        # Apply inverse_transform on the percent_change column
        if scaler_key in scalers:
            df.loc[mask, f"rescaled_{cfg.label_col}"] = scalers[scaler_key].inverse_transform(group[cfg.label_col].to_numpy().reshape(-1, 1)).flatten()
    # Drop all  rows with entry of min date 
    df = df[df["date"] != df["date"].min()].copy()
    return df


def _assemble_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    symbol_vocab: Optional[Dict[str, int]],
    include_symbol_onehot: bool
) -> Tuple[np.ndarray, List[str]]:
    """
    Assembles the feature matrix from the DataFrame  
    """
    cols = list(feature_cols)
    logger.info(f"Assembling feature matrix with columns: {cols} | Df Columns: {df.columns.tolist()}")
    X = df[cols].to_numpy(dtype=np.float32)

    if include_symbol_onehot and symbol_vocab is not None:
        onehot = np.zeros((len(df), len(symbol_vocab)), dtype=np.float32)
        sym_idx = df["symbol"].map(symbol_vocab).astype(int).to_numpy()
        valid_mask = (sym_idx >= 0) & (sym_idx < len(symbol_vocab))
        row_idx = np.arange(len(df))[valid_mask]
        onehot[row_idx, sym_idx[valid_mask]] = 1.0
        X = np.concatenate([X, onehot], axis=1)
        feature_names = cols + ( [f"sym::{s}" for s, _ in sorted(symbol_vocab.items(), key=lambda x: x[1])] ) 
    else:
        feature_names = cols
    logger.info("Feature names: %s", feature_names)
    return X, feature_names


def _window_indices(n: int, L: int, step: int, horizon: int) -> np.ndarray:
    """
    Valid window starts i such that window is [i, i+L) and label time is i+L-1+horizon within bounds.
    """
    last_label_idx = n - 1
    max_start = last_label_idx - (L - 1) - horizon
    if max_start < 0:
        return np.array([], dtype=int)
    return np.arange(0, max_start + 1, step, dtype=int)


def _extract_windows(
    df: pd.DataFrame,
    X: np.ndarray,
    cfg: StreamConfig
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, pd.Timestamp]]]:
    """
    Extracts sliding windows of features and labels from the DataFrame.

    Returns:
      x_windows: [num_windows, L, D]
      y: [num_windows] int labels taken at (start + L - 1 + horizon)
      meta: list of (symbol, date_at_label)
    """
    if "date" not in df.columns or "symbol" not in df.columns:
        logger.error("DataFrame must contain 'date' and 'symbol' columns. Contains only: %s", df.columns)
        raise ValueError("Expected 'date' (datetime) and 'symbol' columns.")
    L, k = cfg.window_len, cfg.horizon    # 30, 1 respectively

    idx = _window_indices(len(df), L, cfg.step, k)      # Get valid window start indices
    if len(idx) == 0:
        return np.empty((0, L, X.shape[1]), np.float32), np.empty((0,), np.int64), []

    # label at the right edge + horizon
    label_positions = idx + (L - 1) + k
    y = df["label"].to_numpy(dtype=np.int64)[label_positions]

    # windows
    D = X.shape[1]
    x = np.lib.stride_tricks.sliding_window_view(X, window_shape=(L, X.shape[1]))  # shape: [n-L+1, L, D]
    x = x[idx]  # select starts
    # sliding_window_view packs the last two dims; we ensure it is [*, L, D]
    x = x.reshape(-1, L, D)

    # meta info (symbol/date at label time)
    meta = list(zip(df["symbol"].astype(str).to_numpy()[label_positions],
                    pd.to_datetime(df["date"].to_numpy()[label_positions])))

    return x.astype(np.float32), y, meta


def _balanced_indices(y: np.ndarray, rng: random.Random) -> np.ndarray:
    """Return indices to sample a balanced set across classes present in y."""
    cls_to_idx: Dict[int, List[int]] = {}
    for i, c in enumerate(y.tolist()):
        cls_to_idx.setdefault(c, []).append(i)
    if len(cls_to_idx) == 0:
        return np.arange(0)
    # downsample to the smallest class count
    min_count = min(len(v) for v in cls_to_idx.values())
    chosen: List[int] = []
    for c, idxs in cls_to_idx.items():
        rng.shuffle(idxs)
        chosen.extend(idxs[:min_count])
    rng.shuffle(chosen)
    return np.array(chosen, dtype=int)


def stream_sequence_windows(cfg: StreamConfig) -> Iterator[Dict[str, object]]:
    """
    Yields dictionaries:
      {
        'x': np.ndarray [L, D],
        'y': int,
        'meta': {'symbol': str, 'label_time': pd.Timestamp, 'file': Path},
        'feature_names': List[str]
      }
    Class-balanced across 0/1/2 when cfg.balance=True (balanced **within each file**).
    """
    rng = random.Random(cfg.random_seed)
    files = _list_pickles(cfg)
    # Don't think there's any use to this
    symbol_vocab = _build_symbol_vocab(files, cfg) if cfg.include_symbol_onehot else None
    cfg.symbol_vocab = symbol_vocab  # save for later use
    for fp in files:
        df, scalers = _load_pickle(fp)
        if df.empty:
            continue

        # Basic expectations
        req_cols = {"date", "symbol", cfg.label_col} | set(cfg.features)
        missing = req_cols - set(df.columns)
        if missing:
            logger.debug(f"Found some missing columns: {missing}")
            continue

        # Time/order & cutoff
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df = _maybe_trim_by_date(df, cfg)


        # rescale percent change 
        df = _apply_scalers_per_symbol(df, scalers, cfg)

        # compute labels 
        df = _compute_labels(df, cfg)

        # Build feature matrix (+ optional symbol one-hot)
        X, feat_names = _assemble_feature_matrix(df, cfg.features, symbol_vocab, cfg.include_symbol_onehot)

        # We must create windows **per symbol** to avoid mixing sequences across assets
        for sym, sdf in df.groupby("symbol", sort=False):
            rows = sdf.index.to_numpy()
            try: 
                Xs = X[rows]
            except IndexError as e:
                continue
                # import ipdb;ipdb.set_trace()
            x_win, y, meta = _extract_windows(sdf, Xs, cfg)
            if len(y) == 0:
                continue

            select = np.arange(len(y))
            if cfg.balance:
                select = _balanced_indices(y, rng)
                if len(select) == 0:
                    continue

            for i in select:
                yield {
                    "x": x_win[i],                         # [L, D]
                    "y": int(y[i]),                       # 0/1/2
                    "meta": {
                        "symbol": meta[i][0],
                        "label_time": meta[i][1],
                        "file": fp
                    },
                    "feature_names": feat_names
                }


# -----------------------
# Optional: PyTorch IterableDataset
# -----------------------

class SequenceWindowIterable(IterableDataset if _HAS_TORCH else object):
    """
    Torch-friendly wrapper around `stream_sequence_windows`.
    """

    def __init__(self, cfg: StreamConfig):
        if not _HAS_TORCH:
            raise ImportError("PyTorch not available; install torch to use SequenceWindowIterable.")
        super().__init__()
        self.cfg = cfg

    def __iter__(self):
        for sample in stream_sequence_windows(self.cfg):
            x = torch.from_numpy(sample["x"])          # [L, D], float32
            y = torch.tensor(sample["y"], dtype=torch.long)
            yield x, y, sample["meta"]

    @staticmethod
    def collate_fn(batch):
        """
        Returns:
          x: [B, L, D]
          y: [B]
          meta: list of dicts
        """
        xs, ys, metas = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        return x, y, list(metas)

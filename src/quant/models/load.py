""" 
Dataset class that streams gold dataset from "data/historical/gold" and passes on to iterable dataloader objects.
"""
import os 
import random 
import pandas as pd

import numpy as np
import torch
from torch.utils.data import IterableDataset
from typing import Generator, Any

from ._common import EXCLUDE_COLS, TARGET_COL, NUM_COLS

from ._common import GOLD_DIR
from .config import SeqClassDataConfig
from ..logs.logging import logger
# make class 

class SeqClassificationDataset(IterableDataset):
    def __init__(
        self,
        config: SeqClassDataConfig,
        train: bool,
    ):
        self.interval = config.interval
        self.seed = config.seed
        self.gold = config.gold_dir

        self.filepaths = [pth for pth in os.listdir(config.gold_dir) if self.interval.value in pth]

        self.training_artifacts = []
        self.train = train
        self.symbols = list(map(lambda x: x.split("_")[0], self.filepaths))
        self.sym2id = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.train_cutoff_dt = config.val_start_dt
        self.val_cutoff_dt = config.test_start_dt

        self.input_length = config.max_seq_length
        self.exclude_cols = EXCLUDE_COLS
        self.target_col = TARGET_COL

        # optional params
        self.stride = getattr(config, "stride", 1)  # how far to move window each time
        self.num_classes = config.num_classes       # assume 3
        self.balance_classes = getattr(config, "balance_classes", False)
        self._stock_cache: dict[str, dict[str, Any]] = {}

    def _prepare_stock(self, filename: str, rnd: random.Random) -> dict:
        if filename in self._stock_cache:
            return self._stock_cache[filename]

        df = pd.read_parquet(self.gold / filename)
        df["date"] = df["date"].dt.tz_localize(None)

        if not self.train:
            df = df[df["date"] >= self.train_cutoff_dt].reset_index(drop=True)
        else:
            df = df[df["date"] < self.train_cutoff_dt].reset_index(drop=True)

        if len(df) <= self.input_length:
            self._stock_cache[filename] = {"valid": False}
            return self._stock_cache[filename]

        feat_np = df[NUM_COLS].to_numpy(dtype="float32")
        labels_np = df[self.target_col].to_numpy()
        dates = df["date"].to_numpy()

        indices_by_label: dict[int, list[int]] = {}
        for c in (0, 1, 2):
            idx = np.where(labels_np == c)[0]
            idx = idx[idx >= self.input_length]  # need history for window
            if len(idx) > 0:
                idx_list = idx.tolist()
                rnd.shuffle(idx_list)
                indices_by_label[c] = idx_list

        cache_entry = {
            "valid": True,
            "feat_np": feat_np,
            "labels_np": labels_np,
            "dates": dates,
            "indices_by_label": indices_by_label,
        }
        self._stock_cache[filename] = cache_entry
        return cache_entry

    def __iter__(self) -> Generator:
        """
        Iterator to yield data samples.
        Yields
        ------
        Generator
            Yields individual data samples.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            # derive a per-worker seed
            rnd = random.Random(self.seed + worker_id)
        else:
            rnd = random.Random(self.seed)

        filepaths = self.filepaths[:]  # local copy
        rnd.shuffle(filepaths)        
        
        # Need to add recursive loading here - if i load yahoofinance 1day in the first iteration, I want to load it again after all files are processed; since the input length of training is very small compared to 10 years worth of data; it makes sense to loop over files multiple times per epoch; the  training_track dataframe will keep track to ensure 
        for filename in self.filepaths:
            cache_entry = self._prepare_stock(filename, rnd)
            if not cache_entry.get("valid", False):
                continue

            feat_np = cache_entry["feat_np"]           # (T, num_features)
            labels_np = cache_entry["labels_np"]       # (T,)
            dates = cache_entry["dates"]               # (T,)
            indices_by_label = cache_entry["indices_by_label"]

            symbol = filename.split("_")[0]
            sym_id_val = self.sym2id[symbol]

            # --- balanced sampling if all three classes exist ---
            have_all = all(c in indices_by_label and len(indices_by_label[c]) > 0 for c in (0, 1, 2))
            if self.train and have_all:
                # equalize class counts per stock
                min_count = min(len(indices_by_label[c]) for c in (0, 1, 2))
                logger.info(f"Balanced sampling for {symbol}: min_count of labels={min_count}")
                for c in (0, 1, 2):
                    indices_by_label[c] = indices_by_label[c][:min_count]

                # interleave 0,1,2, 0,1,2, ...
                for i in range(min_count):
                    for c in (0, 1, 2):
                        label_idx = indices_by_label[c][i]          # index of "tomorrow"
                        start_idx = label_idx - self.input_length   # window starts self.input_length days before
                        end_idx = label_idx                         # exclusive

                        if start_idx < 0:
                            continue  # should not happen due to filtering, but safe

                        window = feat_np[start_idx:end_idx]         # (seq_len, num_features)
                        x_num = torch.from_numpy(window)            # float32
                        sym_tensor = torch.tensor(sym_id_val, dtype=torch.long)
                        y_tensor = torch.tensor(c, dtype=torch.long)

                        self.training_artifacts.append([
                            symbol,
                            dates[start_idx],
                            dates[end_idx - 1],  # last day of input window
                            c,                   # target label (tomorrow)
                        ])

                        yield {
                            "features": x_num,
                            "symbol": sym_tensor,
                            "target": y_tensor,
                        }

            else:
                # fallback: iterate over all valid label_idx (unbalanced)
                valid_label_indices = [
                    idx for idx in range(self.input_length, len(labels_np))
                ]
                rnd.shuffle(valid_label_indices)

                for label_idx in valid_label_indices:
                    start_idx = label_idx - self.input_length
                    end_idx = label_idx
                    if start_idx < 0:
                        continue

                    label_val = int(labels_np[label_idx])
                    window = feat_np[start_idx:end_idx]
                    x_num = torch.from_numpy(window)
                    sym_tensor = torch.tensor(sym_id_val, dtype=torch.long)
                    y_tensor = torch.tensor(label_val, dtype=torch.long)

                    self.training_artifacts.append([
                        symbol,
                        dates[start_idx],
                        dates[end_idx - 1],
                        label_val,
                    ])

                    yield {
                        "features": x_num,
                        "symbol": sym_tensor,
                        "target": y_tensor,
                    }


    def __end__(self):
        pd.DataFrame(
            self.training_artifacts, columns=["symbol", "start_date", "end_date", "label"]).to_csv("training_artifacts.csv", index=False
            )   
        logger.info("Saved training artifacts to training_artifacts.csv")   
        



    def build_feature_space(df: pd.DataFrame):
        """
        Build the feature space for the model.
        Arguments
        ---------
        df: pd.DataFrame
            The input DataFrame containing the data.
        Returns
        -------
        Tuple[str, str, list[str]]
            The target column, symbol column, and list of numeric feature columns.
        """
        # Identify target and features
        target_col = "label_h1"
        # symbol handling: keep separate so we can embedding it
        sym_col = "symbol"
        # numeric features = all float-like columns except excludes and target
        numeric_cols = [
            c for c in df.columns 
            if c not in EXCLUDE_COLS | {target_col, sym_col} and pd.api.types.is_numeric_dtype(df[c])
        ]
        return target_col, sym_col, numeric_cols


    def check_overlap(self, ): 
        """
        Basically loops over the iterator and checks if same date range is repeated across the same symbol+interval pair
        """
        pass 
""" 
Dataset class that streams gold dataset from "data/historical/gold" and passes on to iterable dataloader objects.
"""
import os 
import random 
import pandas as pd

import torch
from pathlib import Path
from torch.utils.data import IterableDataset
from typing import Generator

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
            df = pd.read_parquet(self.gold / filename)
            df['date'] = df['date'].dt.tz_localize(None)

            if not self.train: 
                df = df[df['date'] >= self.train_cutoff_dt]
            else:
                df = df[df['date'] < self.train_cutoff_dt].reset_index(drop=True)
            """
            Sample generation logic:
            - keep track of labels, ensure balanced classes if needed
            - identify a label that hasn't been used enough yet
            - randomly sample start indices for sequences with that label at the end
            - yield sequences of length self.input_length   
            """

            
            max_start_idx = len(df) - self.input_length
            if max_start_idx <= 0:
                continue  # Skip if not enough data

            # Precompute all candidate windows per label
            indices_by_label: dict[int, list[int]] = {}

            for start_idx in range(0, max_start_idx + 1):
                end_idx = start_idx + self.input_length
                if end_idx > len(df):
                    break
                label_val = int(df[self.target_col].iloc[end_idx])
                if label_val not in indices_by_label:
                    indices_by_label[label_val] = []
                indices_by_label[label_val].append(start_idx)

            # symbol id once per stock
            symbol = filename.split("_")[0]
            sym_id_val = self.sym2id[symbol]

            # Precompute feature matrix once per stock: (T, num_features)
            feat_np = df[NUM_COLS].to_numpy(dtype="float32")

            # Try to do balanced sampling across labels 0,1,2
            labels_present = set(indices_by_label.keys())
            balanced_labels = {0, 1, 2}

            if self.train and balanced_labels.issubset(labels_present):
                # We have 0,1,2 for this stock -> can balance
                # Shuffle and trim each label list to same length
                min_count = min(len(indices_by_label[l]) for l in balanced_labels)

                for l in balanced_labels:
                    rnd.shuffle(indices_by_label[l])
                    indices_by_label[l] = indices_by_label[l][:min_count]

                # Interleave 0,1,2,0,1,2,...
                for i in range(min_count):
                    for l in (0, 1, 2):
                        start_idx = indices_by_label[l][i]
                        end_idx = start_idx + self.input_length

                        window = feat_np[start_idx:end_idx]  # (seq_len, num_features)
                        x_num = torch.from_numpy(window)     # float32
                        sym_tensor = torch.tensor(sym_id_val, dtype=torch.long)
                        y_tensor = torch.tensor(l, dtype=torch.long)

                        self.training_artifacts.append(
                            [
                                symbol,
                                df["date"].iloc[start_idx],
                                df["date"].iloc[end_idx - 1],
                                l,
                            ]
                        )

                        yield {
                            "features": x_num,
                            "symbol": sym_tensor,
                            "target": y_tensor,
                        }

            else:
                # Fallback: unbalanced but still randomized windows
                candidate_indices = list(range(0, max_start_idx + 1))
                rnd.shuffle(candidate_indices)

                for start_idx in candidate_indices:
                    end_idx = start_idx + self.input_length
                    if end_idx > len(df):
                        continue

                    label_val = int(df[self.target_col].iloc[end_idx - 1])

                    window = feat_np[start_idx:end_idx]
                    x_num = torch.from_numpy(window)
                    sym_tensor = torch.tensor(sym_id_val, dtype=torch.long)
                    y_tensor = torch.tensor(label_val, dtype=torch.long)

                    self.training_artifacts.append(
                        [
                            symbol,
                            df["date"].iloc[start_idx],
                            df["date"].iloc[end_idx - 1],
                            label_val,
                        ]
                    )

                    yield {
                        "features": x_num,
                        "symbol": sym_tensor,
                        "target": y_tensor,
                    }

    def __end__(self):
        pd.DataFrame(
            self.training_artifacts, columns=["symbol", "start_date", "end_date", "label"]).to_csv("training_artifacts.csv", index=False
            )   
        logging.info("Saved training artifacts to training_artifacts.csv")   
        



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
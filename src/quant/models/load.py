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
            # Randomly pick a starting point for sequence
            max_start_idx = len(df) - self.input_length
            if max_start_idx <= 0:
                continue  # Skip if not enough data
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + self.input_length

            sym_id = self.sym2id[filename.split("_")[0]]
            symbol = filename.split("_")[0]
            self.training_artifacts.append([symbol, df['date'].iloc[start_idx], df['date'].iloc[end_idx - 1]])

            yield {
                'features': torch.from_numpy(
                    df[NUM_COLS].iloc[start_idx:end_idx].to_numpy().astype('float32')
                ),
                'symbol': torch.tensor(sym_id, dtype=torch.float16),
                'target': df[self.target_col].iloc[end_idx - 1]
            }
            

    def __end__(self):
        pd.DataFrame(
            self.training_artifacts, columns=["symbol", "start_date", "end_date"]).to_csv("training_artifacts.csv", index=False
            )   

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
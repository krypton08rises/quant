""" 
Dataset class that streams gold dataset from "data/historical/gold" and passes on to iterable dataloader objects.
"""
import os 
import random 
import pandas as pd

from torch.utils.data import IterableDataset
from typing import Generator

from wastebin.temp import data

from ._common import EXCLUDE_COLS, TARGET_COL

from ..data._common import Interval
from ._common import GOLD_DIR
from .config import SeqClassDataConfig

# make class 

class SeqClassificationDataset(IterableDataset):
    """
    Dataset class for time series classification.
    Streams data from gold dataset and prepares it for DataLoader.
    """

    def __init__(
            self, 
            interval: Interval, 
            config: SeqClassDataConfig, 
            train: bool = True,
        ): 
        self.interval = interval
        self.seed = config.seed
        self.filepaths = [pth for pth in os.listdir(GOLD_DIR) if interval.value in pth]
        self.training_artifacts = []
        self.train = train
        self.symbols = list(map(lambda x: x.split("_")[0], self.filepaths))
        self.sym2id = {sym: idx for idx, sym in enumerate(self.symbols)}
        self.train_cutoff_dt = config.val_start_dt 
        self.val_cutoff_dt = config.test_start_dt

        self.input_length = config.max_seq_length
        self.exclude_cols = EXCLUDE_COLS  # To be set after first data load
        self.target_col = TARGET_COL

    def __iter__(self) -> Generator:
        """
        Iterator to yield data samples.
        Yields
        ------
        Generator
            Yields individual data samples.
        """
        random.seed(self.seed)
        random.shuffle(self.filepaths)
        
        
        # Need to add recursive loading here - if i load yahoofinance 1day in the first iteration, I want to load it again after all files are processed; since the input length of training is very small compared to 10 years worth of data; it makes sense to loop over files multiple times per epoch; the  training_track dataframe will keep track to ensure 
        for filename in self.filepaths:
            filepath = GOLD_DIR / filename
            df = pd.read_csv(filepath, parse_dates=["date"])
            df = df[df['date'] < self.cutoff_dt].reset_index(drop=True)
            if not self.train: 
                df = df[df['date'] >= self.train_cutoff_dt]

            # Randomly pick a starting point for sequence

            if max_start_idx <= 0:
                continue  # Skip if not enough data
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + self.input_length

            symbol = filename.split("_")[0]
            self.training_artifacts.append([symbol, df['date'].iloc[start_idx], df['date'].iloc[end_idx - 1]])

            yield df.drop(columns=self.exclude_cols, axis=1).iloc[start_idx:end_idx]

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
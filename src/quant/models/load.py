""" 
Dataset class that streams gold dataset from "data/historical/gold" and passes on to iterable dataloader objects.
"""
import os 
import random 
import pandas as pd

from torch.utils.data import IterableDataset
from typing import Generator

from ._common import EXCLUDE_COLS

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
        self.training_track = pd.DataFrame(columns=["symbol", "start", "end"])  # To track loaded data for overlap checking
        self.symbols = list(map(lambda x: x.split("_")[0], self.filepaths))
        self.sym2id = {sym: idx for idx, sym in enumerate(self.symbols)}



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

        for filename in self.filepaths:
            filepath = GOLD_DIR / filename
            df = pd.read_csv(filepath, parse_dates=["date"])
            df = df.
            symbol = filename.split("_")[0]


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
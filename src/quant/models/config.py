from pathlib import Path
from datetime import datetime
from pydantic import BaseModel 
from ..data._common import Interval

from ._common import HIDDEN_LAYERS, NUM_COLS

class SeqClassDataConfig(BaseModel):
    """
    Configuration for time series classification dataset.
    """

    # Deterministic seed
    seed: int = 13

    # File paths 
    gold_dir:Path = Path("src/quant/data/historical/gold")
    artifacts: Path = Path("src/quant/models/data/cnn/")

    # Test set parameters
    val_start_dt: datetime = datetime(2024, 1, 1)
    test_start_dt: datetime = datetime(2024, 6, 1)  

    # Dataset parameters
    numeric_cols: list[str] = NUM_COLS
    interval: Interval = Interval.DAY

    # Model Parameters
    max_seq_length: int = 20 # Maximum length of days seen by the model
    kernel_size: int = 10
    num_classes: int = 3
    dropout:float = 0.3
    MLP_HIDDEN: tuple[int, ...] = HIDDEN_LAYERS
    emb_dim: int = 32

    # DataLoader parameters
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    sym2id: dict[str, int] = {}

    # Training Parameters
    num_epochs: int = 20
    lr: float = 5e-5
    wd: float = 0.01



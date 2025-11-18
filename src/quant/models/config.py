import torch

from pathlib import Path
from datetime import datetime
from pydantic import BaseModel 

from ._common import HIDDEN_LAYERS

class SeqClassDataConfig(BaseModel):
    """
    Configuration for time series classification dataset.
    """

    # Deterministic seed
    seed: int = 13

    # File paths 
    gold_dir = Path("data/historical/gold")
    artifacts: Path = Path("models/data/cnn/")

    # Test set parameters
    val_start_dt = datetime(2024, 1, 1)
    test_start_dt = datetime(2024, 6, 1)

    # Model Parameters
    max_seq_length: int = 32 # Maximum length of days seen by the model
    embed_dim: int = 16
    h1: int = 256
    h2: int = 128
    h3: int = 64
    h4: int = 16
    num_classes: int = 3
    dropout:float = 0.3
    MLP_HIDDEN: tuple[int, ...] = HIDDEN_LAYERS

    # DataLoader parameters
    batch_size: int = 512
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    # Training Parameters
    num_epochs: int = 2
    lr: float = 5e-5
    wd: float = 0.01
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Train a simple PyTorch tabular classifier on GOLD parquet files.
- Reads all parquet partitions from a GOLD directory (one row per (symbol, date)).
- Uses only past-derived features at time t to predict label_h{H} at time t (already computed in GOLD).
- Time-based split (train/val/test) using user-provided cutoff dates.
- Class-weighted CrossEntropy for imbalance, macro-F1 early stopping.
- Optional learned symbol embedding concatenated with numeric features.

Example:
python train_torch_tabular.py \
  --gold-dir data/gold/exchange=NSE/interval=1d/h=1/labels_v1 \
  --val-start 2025-01-01 --test-start 2025-04-01 \
  --batch-size 512 --epochs 50 --lr 3e-4 --use-emb
"""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from ..data.utils import read_gold
from ..data.models import GoldConfig
from .config import SeqClassDataConfig
from ._common import HIDDEN_LAYERS
# -------------------------
# Repro
# -------------------------

def set_seed(seed: int = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TabularDS(Dataset):
    """
    Tabular dataset for training and evaluation.
    """
    def __init__(self, df: pd.DataFrame, numeric_cols: list[str], target_col: str, sym2id: dict[str, int] | None = None):
        self.df = df.reset_index(drop=True)
        self.numeric_cols = numeric_cols
        self.target_col = target_col
        self.sym2id = sym2id
        # standardize numeric on-the-fly using train stats you pass in later
        self.mean = None
        self.std = None

    def set_norm(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_num = torch.tensor(row[self.numeric_cols].values.astype(np.float32))
        if self.mean is not None:
            x_num = (x_num - self.mean) / (self.std + 1e-8)
        y = int(row[self.target_col])
        if self.sym2id is not None:
            sym_id = self.sym2id.get(row["symbol"], 0)
            sym_id = torch.tensor(sym_id, dtype=torch.long)
            return x_num, sym_id, y
        else:
            return x_num, y

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model for tabular data.
    """
    def __init__(self, in_channels: int, n_classes: int = 3, hidden: tuple[int, ...] = HIDDEN_LAYERS, pdrop: float = 0.1):
        super().__init__()
        layers = []
        last = in_channels
        for h in hidden:
            layers += [nn.Conv1d(last, h, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(pdrop)]
            last = h
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(last, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_length)
        return self.net(x)
    

# -------------------------
# Training utils
# -------------------------

def compute_class_weights(y: np.ndarray, n_classes: int = 3):
    """
    Compute class weights for imbalanced datasets.
    Arguments
    ---------
    y: np.ndarray
        The target labels.
    n_classes: int
        The number of classes.
    Returns
    -------
    torch.Tensor
        The class weights.
    """
    # inverse frequency
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, use_emb: bool = False):
    """
    Evaluate the model on the given data loader.
    Arguments
    ---------
    model: nn.Module
        The model to evaluate.
    loader: DataLoader
        The data loader for the evaluation dataset.
    device: torch.device
        The device to run the evaluation on.
    use_emb: bool
        Whether to use symbol embeddings.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        The macro F1 score, true labels, and predicted labels.
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if use_emb:
                x_num, sym_id, y = batch
                x_num, sym_id = x_num.to(device), sym_id.to(device)
                logits = model(x_num, sym_id)
            else:
                x_num, y = batch
                x_num = x_num.to(device)
                logits = model(x_num)
            prob = torch.softmax(logits, dim=1)
            ps.append(prob.detach().cpu().numpy())
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return macro_f1, y_true, y_pred


def train_loop(
    model, 
    train_loader, 
    val_loader,
    device, 
    class_weights,
    epochs=50, 
    lr=3e-4, 
    use_emb=False
):
    """
    Training loop for the model.
    Arguments
    ---------
    model: nn.Module
        The model to train.
    train_loader: DataLoader
        The data loader for the training dataset.
    val_loader: DataLoader
        The data loader for the validation dataset.
    device: torch.device
        The device to run the training on.
    class_weights: torch.Tensor
        The class weights for the loss function.
    epochs: int
        The number of training epochs.
    lr: float
        The learning rate.
    use_emb: bool
        Whether to use symbol embeddings.
    Returns
    -------
    nn.Module
        The trained model.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if use_emb:
                x_num, sym_id, y = batch
                x_num, sym_id, y = x_num.to(device), sym_id.to(device), torch.tensor(y, dtype=torch.long, device=device)
                logits = model(x_num, sym_id)
            else:
                x_num, y = batch
                x_num, y = x_num.to(device), torch.tensor(y, dtype=torch.long, device=device)
                logits = model(x_num)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_f1, _, _ = evaluate(model, val_loader, device, use_emb)
        print(f"Epoch {ep:03d} | val macro-F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def handler(dataconfig: SeqClassDataConfig, gc: GoldConfig):
    """
    Handler function to run the training and evaluation.
    Arguments
    ---------
    args: argparse.Namespace
        The command line arguments.
    Returns
    -------
    None
    """
    set_seed(dataconfig.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not dataconfig.cpu else "cpu")

    df = read_gold(Path(dataconfig.gold_dir))
    # Pick the label column based on horizon inferred from directory name if needed.
    # Here we assume h=1 gold folder -> label_h1 present.

    # Basic cleaning
    df = df.dropna(subset=["label_h1"]).copy()
    target_col, sym_col, numeric_cols = build_feature_space(df)

    # Iterable Dataset: 



if __name__ == "__main__":
    gc = GoldConfig()
    dataconfig = SeqClassDataConfig()
    handler(dataconfig, gc)

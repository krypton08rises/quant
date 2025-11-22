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
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from quant.data.utils import read_gold
from quant.data.models import GoldConfig
from quant.models.config import SeqClassDataConfig
from quant.models._common import HIDDEN_LAYERS
from quant.models.load import SeqClassificationDataset
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


class CNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 kernel_size: int = 5,
                 dropout: float = 0.3,
                 mlp_hidden: tuple[int, ...] = HIDDEN_LAYERS,
                 emb_dim: int = 0,
                 num_symbols: int = 0
                 ):
        super().__init__()

        # Conv trunk: (B, C_in=input_dim, T) -> (B, 256)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=kernel_size, padding=1), # (B, 64, T) 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=1), # (B, 128, T)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=kernel_size, padding=1),    # (B, 256, T)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),  # (B, 256, 1)
            nn.Flatten(),             # (B, 256)
        )

        self.use_emb = emb_dim > 0 and num_symbols > 0 
        if self.use_emb:
            self.symbol_emb = nn.Embedding(num_symbols, emb_dim) # (num_symbols, emb_dim)
            mlp_input_dim = 256 + emb_dim
        else:
            mlp_input_dim = 256

        mlp_layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in mlp_hidden:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))      
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_num: torch.Tensor, sym_id: torch.Tensor | None = None):
        # x_num: (B, seq_len, num_features)
        x_num = x_num.permute(0, 2, 1)  # -> (B, num_features, seq_len)
        features = self.cnn(x_num)      # (B, 256)

        if self.use_emb and sym_id is not None:
            # ensure sym_id is long: sym_id.dtype == torch.long
            sym_emb = self.symbol_emb(sym_id)
            features = torch.cat([features, sym_emb], dim=1)

        logits = self.mlp(features)     # (B, num_classes)
        return logits


def evaluate(
    model: CNN,
    loader: DataLoader,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on the given data loader.
    Returns
    -------
    macro_f1: float
        Macro F1 score.
    y_true: np.ndarray
        True labels.
    y_pred: np.ndarray
        Predicted labels.
    """
    model.eval()
    # infer device from model
    device = next(model.parameters()).device

    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            # move inputs to the same device as model
            x_num = batch['features'].to(device, non_blocking=True)
            sym_id = batch['symbol'].to(device, non_blocking=True)
            # labels can stay on cpu, but if they are on gpu, move back:
            y = batch['target'].long().to("cpu")
            print("Batch shapes:", x_num.shape, sym_id.shape, y.shape)
            logits = model(x_num, sym_id)          # (B, num_classes)
            prob = torch.softmax(logits, dim=1)    # (B, num_classes)

            ps.append(prob.cpu().numpy())
            ys.append(y.cpu().numpy())
    print("Evaluation complete. Shapes: ", [p.shape for p in ps], [y.shape for y in ys])
    y_true = np.concatenate(ys, axis=0)   # (N,)
    y_prob = np.concatenate(ps, axis=0)   # (N, C)
    y_pred = y_prob.argmax(axis=1)        # (N,)

    print(f"Classification Report:{classification_report(y_true, y_pred)}")

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return macro_f1, y_true, y_pred



def train_loop(
    model: CNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: SeqClassDataConfig,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    

    best_f1, best_state = -1.0, None

    for ep in range(1, config.num_epochs + 1):
        model.train()
        for batch in train_loader:
            x_num = batch['features'].to(device)
            sym_id = batch['symbol'].to(device)
            y = batch['target'].to(device, non_blocking=True)
            logits = model(x_num, sym_id)
            y = y.long()
            loss = nn.CrossEntropyLoss()(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_f1, _, _ = evaluate(model, val_loader)
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


    train_dataset = SeqClassificationDataset(
        config=dataconfig,
        train=True,
    )
    val_dataset = SeqClassificationDataset(
        config=dataconfig,
        train=False,
    )   
    model = CNN(
        input_dim=len(dataconfig.numeric_cols),
        num_classes=dataconfig.num_classes,
        dropout=dataconfig.dropout,
        kernel_size=dataconfig.kernel_size,
        mlp_hidden=dataconfig.MLP_HIDDEN,
        emb_dim=dataconfig.emb_dim ,
        num_symbols=len(dataconfig.sym2id),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=dataconfig.lr, weight_decay=dataconfig.wd)

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataconfig.batch_size,
        shuffle=False,
        num_workers=dataconfig.num_workers,
        pin_memory=dataconfig.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataconfig.batch_size,
        shuffle=False,
        num_workers=dataconfig.num_workers,
        pin_memory=dataconfig.pin_memory,
    )   
    trained_model = train_loop(
        model,
        train_loader,
        val_loader,
        opt,
        dataconfig,
    )


if __name__ == "__main__":
    gc = GoldConfig()
    dataconfig = SeqClassDataConfig()
    handler(dataconfig, gc)

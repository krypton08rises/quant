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
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score

# -------------------------
# Repro
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Data
# -------------------------

EXCLUDE_COLS = {
    "date", "fwd_pct_h1", "label_h1",  # time/targets
}
# You can add anything else you want to exclude explicitly (e.g., raw open/high/low if present)


def read_gold(gold_dir: Path) -> pd.DataFrame:
    """
    Read GOLD parquet files from the specified directory.
    Arguments
    ---------
    gold_dir: Path
        The path to the GOLD directory containing parquet files.
    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame containing all GOLD data.
    """
    parts = sorted(Path(gold_dir).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {gold_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    
    # normalize timezones: make naive datetime
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)

    # if utc convert fails because it's already tz-naive with offset string, try tz_localize(None)
    if df["date"].isna().any():
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, val_start: str, test_start: str):
    """
    Create train/validation/test splits based on date.
    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing the data.
    val_start: str
        The start date for the validation set.
    test_start: str
        The start date for the test set.
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The train, validation, and test DataFrames.
    """
    val_start = pd.to_datetime(val_start)
    test_start = pd.to_datetime(test_start)
    train = df[df["date"] < val_start].copy()
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()
    return train, val, test


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


# -------------------------
# Model
# -------------------------

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) model.
    """
    def __init__(self, in_dim: int, n_classes: int = 3, hidden: tuple[int, ...] = (256, 128, 64), pdrop: float = 0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(pdrop)]
            last = h
        layers += [nn.Linear(last, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPWithSymbol(nn.Module):
    """
    Multi-layer perceptron (MLP) model with symbol embeddings.
    """
    def __init__(self, in_dim: int, n_symbols: int, emb_dim: int = 16, n_classes: int = 3,
                 hidden: tuple[int, ...] = (256, 128, 64), pdrop: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=n_symbols, embedding_dim=emb_dim)
        self.mlp = MLP(in_dim + emb_dim, n_classes=n_classes, hidden=hidden, pdrop=pdrop)

    def forward(self, x_num, sym_id):
        e = self.emb(sym_id)
        z = torch.cat([x_num, e], dim=1)
        return self.mlp(z)


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


def train_loop(model, train_loader, val_loader, device, class_weights, epochs=50, lr=3e-4, use_emb=False):
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


# -------------------------
# Main
# -------------------------

def handler(args):
    """
    Main function to run the training and evaluation.
    Arguments
    ---------
    args: argparse.Namespace
        The command line arguments.
    Returns
    -------
    None
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    df = read_gold(Path(args.gold_dir))
    # Pick the label column based on horizon inferred from directory name if needed.
    # Here we assume h=1 gold folder -> label_h1 present.

    # Basic cleaning
    df = df.dropna(subset=["label_h1"]).copy()
    target_col, sym_col, numeric_cols = build_feature_space(df)

    # Time-based split
    train_df, val_df, test_df = make_splits(df, args.val_start, args.test_start)

    # Build symbol vocab (from TRAIN ONLY to avoid leakage of identities — optional choice)
    if args.use_emb:
        symbols = train_df[sym_col].astype(str).unique().tolist()
        sym2id = {s: i for i, s in enumerate(symbols)}
    else:
        sym2id = None

    # Datasets
    tr_ds = TabularDS(train_df, numeric_cols, target_col, sym2id)
    va_ds = TabularDS(val_df, numeric_cols, target_col, sym2id)
    te_ds = TabularDS(test_df, numeric_cols, target_col, sym2id)

    # Fit normalization from TRAIN only
    x_tr = train_df[numeric_cols].astype(np.float32).values
    mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
    std[std == 0] = 1.0
    for ds in (tr_ds, va_ds, te_ds):
        ds.set_norm(mean, std)

    # Loaders
    kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=(device.type=="cuda"))
    if args.use_emb:
        collate = None  # default works since each __getitem__ returns (x_num, sym_id, y)
    else:
        collate = None
    tr_ld = DataLoader(tr_ds, shuffle=True, **kwargs)
    va_ld = DataLoader(va_ds, shuffle=False, **kwargs)
    te_ld = DataLoader(te_ds, shuffle=False, **kwargs)

    # Class weights from TRAIN only
    class_weights = compute_class_weights(train_df[target_col].values, n_classes=3)

    # Model
    if args.use_emb:
        model = MLPWithSymbol(in_dim=len(numeric_cols), n_symbols=len(sym2id), emb_dim=args.emb_dim,
                              n_classes=3, hidden=(args.h1, args.h2, args.h3), pdrop=args.dropout)
        use_emb = True
    else:
        model = MLP(in_dim=len(numeric_cols), n_classes=3, hidden=(args.h1, args.h2, args.h3), pdrop=args.dropout)
        use_emb = False

    # Train
    model = train_loop(model, tr_ld, va_ld, device, class_weights, epochs=args.epochs, lr=args.lr, use_emb=use_emb)

    # Evaluate on TEST
    macro_f1, y_true, y_pred = evaluate(model, te_ld, device, use_emb)
    print("\nTEST macro-F1:", round(macro_f1, 4))
    print("\nClassification report (TEST):\n")
    print(classification_report(y_true, y_pred, digits=4))

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "use_emb": use_emb,
        "sym2id": sym2id,
        "numeric_cols": numeric_cols,
        "mean": mean,
        "std": std,
        "args": vars(args),
    }, out_dir / "model.pt")
    print(f"Saved model to {out_dir / 'model.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold-dir", default="data/historical/index/gold/NSE/day/1/1/", help="Path to GOLD labels parquet dir (e.g., .../h=1/labels_v1)")
    p.add_argument("--val-start", default='2024-12-01')
    p.add_argument("--test-start", default='2025-01-01')
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-emb", dest="use_emb", action="store_true")
    p.add_argument("--emb-dim", type=int, default=16)
    p.add_argument("--h1", type=int, default=256)
    p.add_argument("--h2", type=int, default=128)
    p.add_argument("--h3", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out-dir", default="artifacts/tabular_h1")
    args = p.parse_args()
    handler(args)

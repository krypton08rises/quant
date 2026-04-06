"""
This script is for classifying time series data using a sequence classification model.
"""


from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data.create_labels import StreamConfig, stream_sequence_windows


def xgboost(X_train, y_train, X_test, y_test):
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    print("\nXGBoost Results:")
    print(classification_report(y_test, xgb.predict(X_test)))


def cnn():
    pass


def load_data(cfg: StreamConfig):
    # Collect some samples (adjust n_samples to taste)
    X, y = [], []
    it = stream_sequence_windows(cfg)
    n_samples = 5000  # you can make this larger, depends on memory

    for i, sample in zip(range(n_samples), it):
        X.append(sample["x"].flatten())  # [L, D] -> [L*D]
        y.append(sample["y"])

    X = np.stack(X)
    y = np.array(y)
    print("Dataset shape:", X.shape, y.shape)

    # --------------------------
    # Train/Test split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def main():
    cfg = StreamConfig(folder_path=Path("data/historical"), window_len=30, horizon=1, balance=True)
    X_train, X_test, y_train, y_test = load_data(cfg)
    xgboost(X_train, y_train, X_test, y_test)

import numpy as np
import pandas as pd
import xgboost as xgb
from quant.data._common import (
    SILVER_DIR,
    BronzeColumns,
    EntryPriceMode,
    SilverColumns,
    TripleBarrierSpec,
    VolMethod,
)
from quant.logs.logging import logger
from quant.ml.classification._common import log_experiment
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight


def apply_stationary_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw indicators into stationary, scale-invariant features.
    """

    out = df.copy()

    # 1. Trend Proxies (Percentage Distances)
    out["feat_ema_12_dist"] = (df["close"] - df["ema_12"]) / df["ema_12"]
    out["feat_ema_26_dist"] = (df["close"] - df["ema_26"]) / df["ema_26"]

    # 2. Mean Reversion Proxies (Normalized)
    # BB_PCT is (Close - Lower) / (Upper - Lower)
    out["feat_bb_pct"] = df["bb_pct"]

    # 3. Momentum (Already Stationary)
    out["feat_rsi"] = df["rsi"] / 100.0  # Scale to [0, 1]
    out["feat_macd_hist_norm"] = df["macd_hist"] / df["close"]  # Scale by price

    # 4. Volume Dynamics (Relative to moving average)
    # Assuming volume_sma_20 is pre-calculated
    out["feat_rel_volume"] = df["volume"] / df["volume"].rolling(20).mean()

    # 5. Volatility (Stationary)
    out["feat_atr_norm"] = df["atr"] / df["close"]

    # 6. Returns (The purest stationary feature)
    out["feat_log_ret_5d"] = np.log(df["close"] / df["close"].shift(5))

    return out


def main(interval: str):
    logger.info("--- Starting Global Baseline Model (XGBoost) ---")

    spec = TripleBarrierSpec(
        H=5, pt_k=1.0, sl_k=1.0, vol_method=VolMethod.ATR, entry=EntryPriceMode.NEXT_OPEN
    )
    # 1. Load Data
    df = pd.read_parquet(f"{SILVER_DIR}/{interval}_{spec.config_str()}.parquet")
    df = apply_stationary_transforms(df)

    # 2. Pre-processing & Sanitation
    # Replace Infinite values with NaN first
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows where target is NaN (cannot train without labels)
    df = df.dropna(axis=0)

    assert not np.isinf(
        df.select_dtypes(include=np.number)
    ).values.any(), "Data still contains INF!"
    assert not df.isna().values.any(), "Data still contains NaN!"

    logger.info(f"Data sanitized. Final shape: {df.shape}")

    # Encode Symbol (Global model needs to know which stock is which, or treating them )
    # Note: anonymouslyFor a pure technical model, we often DROP the symbol to force learning generalized price action.
    # Let's keep it as category for now.
    df[BronzeColumns.SYMBOL.value] = df[BronzeColumns.SYMBOL.value].astype("category")

    # 3. Feature Selection
    # define drop columns (Raw prices must go!)
    drop_cols = [
        BronzeColumns.DATE.value,
        BronzeColumns.SYMBOL.value,
        BronzeColumns.OPEN.value,
        BronzeColumns.HIGH.value,
        BronzeColumns.LOW.value,
        BronzeColumns.CLOSE.value,
        BronzeColumns.VOLUME.value,  # Volume is non-stationary, use Vol/AvgVol instead if you have it
        SilverColumns.LABEL_5DAY.value,
        SilverColumns.EMA_12.value,
        SilverColumns.EMA_26.value,
        SilverColumns.BB_UPPER.value,
        SilverColumns.BB_LOWER.value,
        SilverColumns.BB_MID.value,
        SilverColumns.BB_PCT.value,
        SilverColumns.MACD_HIST.value,
        SilverColumns.MACD_LINE.value,
        SilverColumns.MACD_SIGNAL.value,
    ]

    # Check if 'volume' is actually in columns before dropping to avoid errors
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=existing_drop_cols)
    y = df[SilverColumns.LABEL_5DAY.value]

    # Map labels if necessary (XGBoost expects [0, 1, 2])
    # Assuming your Triple Barrier is -1, 0, 1. We map to 0, 1, 2.
    y_mapped = y + 1

    # 4. The Research Split (Strict Chronological)
    cutoff_date = "2018-12-31"

    logger.info(f"Splitting data at cutoff: {cutoff_date}")
    train_mask = df[BronzeColumns.DATE.value] <= cutoff_date
    test_mask = (df[BronzeColumns.DATE.value] > cutoff_date) & (
        df[BronzeColumns.DATE.value] <= "2025-12-31"
    )

    X_train = X[train_mask]
    y_train = y_mapped[train_mask]

    X_test = X[test_mask]
    y_test = y_mapped[test_mask]

    logger.info(f"Train Size: {X_train.shape[0]} rows | Test Size: {X_test.shape[0]} rows")

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    logger.info(f"Sample weights calculated for imbalanced classes: {weights}")

    # 5. Initialize XGBoost
    # enable_categorical=True allows XGB to handle the Symbol column natively
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        enable_categorical=True,
        n_jobs=-1,
        random_state=42,
    )

    # 6. Training
    logger.info("Fitting Global Model...")
    model.fit(X_train, y_train, sample_weight=weights)

    # 7. Evaluation
    logger.info("Generating Predictions...")
    y_pred = model.predict(X_test)

    # Remap back to -1, 0, 1 for reporting clarity
    target_names = ["Short (-1)", "Neutral (0)", "Long (1)"]

    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    logger.info(
        f"\n--- Out-of-Sample Performance (2021-2025) ---\n{classification_report(y_test, y_pred, target_names=target_names)}"
    )

    # 8. Feature Importance (The "Why")
    # This helps us identify if the model is just looking at "Symbol" or actual indicators
    importance = (
        pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_})
        .sort_values(by="Importance", ascending=False)
        .head(10)
    )

    print("\nTop 10 Predictive Features:")
    print(importance)

    # 9. Log Experiment
    log_experiment(
        params={
            "H": spec.H,
            "pt_k": spec.pt_k,
            "sl_k": spec.sl_k,
            "vol_method": spec.vol_method.value,
            "entry": spec.entry.value,
            "barrier_mode": spec.barrier_mode.value,
            "tie_breaking": spec.tie_breaking.value,
        },
        metrics={
            "prec_long": report["Long (1)"]["precision"],
            "prec_short": report["Short (-1)"]["precision"],
            "recall_long": report["Long (1)"]["recall"],
            "recall_short": report["Short (-1)"]["recall"],
        },
        top_features=importance["Feature"].tolist(),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a global XGBoost model on the silver dataset."
    )
    # load interval
    parser.add_argument(
        "--interval",
        type=str,
        default="day",
        help="Interval of the data to use (e.g., day, 60minute)",
    )

    args = parser.parse_args()
    main(args.interval)

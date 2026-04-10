"""
Universe-level regime report: ADF stationarity + Hurst exponent.

Iterates over all stocks in a silver dataset (for a given index, interval,
and triple-barrier spec), runs per-stock ADF and Hurst tests on the
*training slice only*, and aggregates results into a summary report.

Outputs
-------
- Per-stock detail JSON  (``regime_detail_{index}_{interval}.json``)
- Universe summary JSON   (``regime_summary_{index}_{interval}.json``)

Run:
    python -m quant.data.audit.regime_report --index nifty_50 --interval day
    python -m quant.data.audit.regime_report --index nifty_50 --interval day --csv
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from quant.data._common import (
    AUDIT_DIR,
    CUTOFF_DATE,
    TRAIN_END_DATE,
    Indices,
    Interval,
    RawColumns,
    SilverColumns,
    TripleBarrierSpec,
)
from quant.data.audit.adf import run_adf
from quant.data.audit.hurst_exponent import rescaled_range_hurst

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDICES_DIR = Path(__file__).resolve().parents[1] / "indices"

# Features to audit, split by expected stationarity
STATIONARY_FEATURES: list[str] = [
    SilverColumns.RSI,
    SilverColumns.BB_PCT,
    SilverColumns.MACD_HIST,
    "log_ret",
]

NON_STATIONARY_FEATURES: list[str] = [
    RawColumns.CLOSE,
    RawColumns.OPEN,
    RawColumns.HIGH,
    RawColumns.LOW,
    SilverColumns.ATR,
    SilverColumns.EMA_12,
    SilverColumns.EMA_26,
    SilverColumns.MACD_LINE,
    SilverColumns.MACD_SIGNAL,
    SilverColumns.BB_UPPER,
    SilverColumns.BB_LOWER,
    SilverColumns.BB_MID,
]

ALL_AUDIT_FEATURES = STATIONARY_FEATURES + NON_STATIONARY_FEATURES


@dataclass
class RegimeReportConfig:
    """Knobs for the universe regime audit."""

    adf_alpha: float = 0.05
    hurst_min_obs: int = 100
    adf_min_obs: int = 30
    hurst_random_walk_lb: float = 0.45
    hurst_random_walk_ub: float = 0.55


# ---------------------------------------------------------------------------
# Index membership helpers
# ---------------------------------------------------------------------------


def load_index_symbols(index: Indices) -> list[str]:
    """
    Load the symbol list for a given index from the static CSVs under
    ``src/quant/data/indices/``.
    """
    csv_path = INDICES_DIR / f"{index.value}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # The CSVs have a "Symbol" column (sometimes with trailing spaces in header)
    col = [c for c in df.columns if c.strip().lower() == "symbol"][0]
    return df[col].str.strip().tolist()


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------


def _slice_training(df: pd.DataFrame, date_col: str = RawColumns.DATE) -> pd.DataFrame:
    """Return only rows in [CUTOFF_DATE, TRAIN_END_DATE)."""
    dates = pd.to_datetime(df[date_col])
    mask = (dates >= CUTOFF_DATE) & (dates < TRAIN_END_DATE)
    return df.loc[mask].copy()


def _ensure_log_ret(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``log_ret`` column if missing."""
    if "log_ret" not in df.columns:
        close = df[RawColumns.CLOSE].astype(float)
        df = df.copy()
        df["log_ret"] = np.log(close / close.shift(1))
    return df


def audit_single_stock(
    df: pd.DataFrame,
    symbol: str,
    features: list[str],
    config: RegimeReportConfig,
) -> tuple[list[dict], list[dict]]:
    """
    Run ADF and Hurst on every feature for one stock.

    Returns (adf_rows, hurst_rows) — lists of dicts ready for DataFrame construction.
    """
    adf_rows: list[dict] = []
    hurst_rows: list[dict] = []

    for feat in features:
        if feat not in df.columns:
            continue

        series = df[feat].astype(float)

        adf = run_adf(
            series,
            feature=feat,
            symbol=symbol,
            alpha=config.adf_alpha,
            min_obs=config.adf_min_obs,
        )
        if adf is not None:
            adf_rows.append(adf.to_dict())

        hurst = rescaled_range_hurst(
            series,
            feature=feat,
            symbol=symbol,
            min_obs=config.hurst_min_obs,
        )
        if hurst is not None:
            hurst_rows.append(hurst.to_dict())

    return adf_rows, hurst_rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class UniverseRegimeSummary:
    """Aggregate statistics across all stocks for a given feature set."""

    index: str
    interval: str
    spec_str: str
    n_stocks_audited: int
    n_stocks_in_index: int

    # ADF summary: feature -> {pct_stationary, mean_p_value, n_tested}
    adf_summary: dict[str, dict] = field(default_factory=dict)
    # Hurst summary: feature -> {mean_H, median_H, pct_mean_reverting, pct_trending, n_tested}
    hurst_summary: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "interval": self.interval,
            "spec": self.spec_str,
            "n_stocks_audited": self.n_stocks_audited,
            "n_stocks_in_index": self.n_stocks_in_index,
            "adf": self.adf_summary,
            "hurst": self.hurst_summary,
        }


def aggregate_adf(adf_df: pd.DataFrame) -> dict[str, dict]:
    """Group ADF results by feature and compute universe-level stats."""
    summary: dict[str, dict] = {}
    for feat, grp in adf_df.groupby("feature"):
        n = len(grp)
        n_stationary = int(grp["stationary"].sum())
        summary[feat] = {
            "n_tested": n,
            "n_stationary": n_stationary,
            "pct_stationary": round(n_stationary / n * 100, 1) if n else 0.0,
            "mean_p_value": round(float(grp["p_value"].mean()), 6),
            "median_p_value": round(float(grp["p_value"].median()), 6),
        }
    return summary


def aggregate_hurst(hurst_df: pd.DataFrame) -> dict[str, dict]:
    """Group Hurst results by feature and compute universe-level stats."""
    summary: dict[str, dict] = {}
    for feat, grp in hurst_df.groupby("feature"):
        n = len(grp)
        n_mr = int((grp["regime"] == "mean_reverting").sum())
        n_rw = int((grp["regime"] == "random_walk").sum())
        n_tr = int((grp["regime"] == "trending").sum())
        summary[feat] = {
            "n_tested": n,
            "mean_H": round(float(grp["H"].mean()), 4),
            "median_H": round(float(grp["H"].median()), 4),
            "std_H": round(float(grp["H"].std()), 4),
            "pct_mean_reverting": round(n_mr / n * 100, 1) if n else 0.0,
            "pct_random_walk": round(n_rw / n * 100, 1) if n else 0.0,
            "pct_trending": round(n_tr / n * 100, 1) if n else 0.0,
        }
    return summary


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_regime_report(
    index: Indices,
    interval: Interval,
    spec: TripleBarrierSpec | None = None,
    config: RegimeReportConfig | None = None,
    save: bool = True,
    csv: bool = False,
) -> UniverseRegimeSummary:
    """
    Run the full universe regime audit for one index/interval combination.

    Parameters
    ----------
    index : Indices
        Market index whose constituents will be audited.
    interval : Interval
        Data interval (daily, 60min, etc.).
    spec : TripleBarrierSpec or None
        Triple-barrier spec to locate the silver directory. Defaults to ``TripleBarrierSpec()``.
    config : RegimeReportConfig or None
        Audit thresholds. Defaults to ``RegimeReportConfig()``.
    save : bool
        Whether to persist JSON (and optionally CSV) to AUDIT_DIR.
    csv : bool
        If True and save is True, also write per-stock detail CSVs.

    Returns
    -------
    UniverseRegimeSummary
    """
    spec = spec or TripleBarrierSpec()
    config = config or RegimeReportConfig()

    silver_dir = spec.silver_dir(interval)
    train_path = silver_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Silver training data not found at {train_path}. " f"Run build_silver.py first."
        )

    logger.info("Loading silver training data from %s", train_path)
    df_all = pd.read_parquet(train_path)

    # Slice to training window
    df_all = _slice_training(df_all)
    df_all = _ensure_log_ret(df_all)

    # Filter to index constituents
    index_symbols = load_index_symbols(index)
    available_symbols = df_all[RawColumns.SYMBOL].unique().tolist()
    target_symbols = [s for s in index_symbols if s in available_symbols]

    if not target_symbols:
        raise ValueError(
            f"No symbols from {index.value} found in silver data. "
            f"Index has {len(index_symbols)} symbols, silver has {len(available_symbols)}."
        )

    logger.info(
        "Auditing %d/%d %s symbols (silver has %d total)",
        len(target_symbols),
        len(index_symbols),
        index.value,
        len(available_symbols),
    )

    all_adf: list[dict] = []
    all_hurst: list[dict] = []

    for symbol in target_symbols:
        sym_df = df_all[df_all[RawColumns.SYMBOL] == symbol].copy()
        if sym_df.empty:
            continue

        sym_df = sym_df.sort_values(RawColumns.DATE).reset_index(drop=True)
        adf_rows, hurst_rows = audit_single_stock(sym_df, symbol, ALL_AUDIT_FEATURES, config)
        all_adf.extend(adf_rows)
        all_hurst.extend(hurst_rows)

    adf_df = pd.DataFrame(all_adf) if all_adf else pd.DataFrame()
    hurst_df = pd.DataFrame(all_hurst) if all_hurst else pd.DataFrame()

    summary = UniverseRegimeSummary(
        index=index.value,
        interval=interval.value,
        spec_str=spec.config_str(),
        n_stocks_audited=len(target_symbols),
        n_stocks_in_index=len(index_symbols),
        adf_summary=aggregate_adf(adf_df) if not adf_df.empty else {},
        hurst_summary=aggregate_hurst(hurst_df) if not hurst_df.empty else {},
    )

    if save:
        _save_report(summary, adf_df, hurst_df, index, interval, csv=csv)

    return summary


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_report(
    summary: UniverseRegimeSummary,
    adf_df: pd.DataFrame,
    hurst_df: pd.DataFrame,
    index: Indices,
    interval: Interval,
    *,
    csv: bool = False,
) -> None:
    out_dir = AUDIT_DIR / "regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{index.value}_{interval.value}"

    # Summary JSON
    summary_path = out_dir / f"regime_summary_{tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    logger.info("Wrote regime summary to %s", summary_path)

    # Detail JSON (per-stock rows)
    detail: dict = {}
    if not adf_df.empty:
        detail["adf"] = adf_df.to_dict(orient="records")
    if not hurst_df.empty:
        detail["hurst"] = hurst_df.to_dict(orient="records")
    detail_path = out_dir / f"regime_detail_{tag}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)
    logger.info("Wrote regime detail to %s", detail_path)

    if csv:
        if not adf_df.empty:
            adf_csv = out_dir / f"adf_detail_{tag}.csv"
            adf_df.to_csv(adf_csv, index=False)
            logger.info("Wrote ADF detail CSV to %s", adf_csv)
        if not hurst_df.empty:
            hurst_csv = out_dir / f"hurst_detail_{tag}.csv"
            hurst_df.to_csv(hurst_csv, index=False)
            logger.info("Wrote Hurst detail CSV to %s", hurst_csv)


# ---------------------------------------------------------------------------
# Pretty-print for quick terminal inspection
# ---------------------------------------------------------------------------


def print_summary(summary: UniverseRegimeSummary) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*70}")
    print(f"REGIME REPORT: {summary.index} | {summary.interval} | {summary.spec_str}")
    print(f"Stocks audited: {summary.n_stocks_audited}/{summary.n_stocks_in_index}")
    print(f"{'='*70}")

    if summary.adf_summary:
        print(f"\n{'─'*35} ADF {'─'*35}")
        print(f"{'Feature':<20} {'%Stationary':>12} {'Mean p':>10} {'Median p':>10} {'N':>5}")
        for feat, s in sorted(summary.adf_summary.items()):
            print(
                f"{feat:<20} {s['pct_stationary']:>11.1f}% "
                f"{s['mean_p_value']:>10.4f} {s['median_p_value']:>10.4f} "
                f"{s['n_tested']:>5}"
            )

    if summary.hurst_summary:
        print(f"\n{'─'*33} HURST {'─'*33}")
        print(
            f"{'Feature':<20} {'Mean H':>8} {'Med H':>8} "
            f"{'%MR':>6} {'%RW':>6} {'%Trend':>7} {'N':>5}"
        )
        for feat, s in sorted(summary.hurst_summary.items()):
            print(
                f"{feat:<20} {s['mean_H']:>8.4f} {s['median_H']:>8.4f} "
                f"{s['pct_mean_reverting']:>5.1f}% {s['pct_random_walk']:>5.1f}% "
                f"{s['pct_trending']:>6.1f}% {s['n_tested']:>5}"
            )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Universe regime report (ADF + Hurst) on silver training data."
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        choices=[i.value for i in Indices],
        help="Market index to audit.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=Interval.DAY.value,
        choices=[i.value for i in Interval],
        help="Data interval.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="Also save per-stock detail as CSV.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Skip writing output files (print only).",
    )
    args = parser.parse_args()

    summary = run_regime_report(
        index=Indices(args.index),
        interval=Interval(args.interval),
        save=not args.no_save,
        csv=args.csv,
    )
    print_summary(summary)


if __name__ == "__main__":
    cli()

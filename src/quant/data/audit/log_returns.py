from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns 

from quant.data._common import AUDIT_DIR, BRONZE_DIR, BronzeColumns, Interval
from quant.data.kite.kite_handler import KiteDataHandler
from quant.logs.logging import logger

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

STATS_COLS = ["mean", "std_dev", "skew", "kurtosis", "max_draw"]


@dataclass(frozen=True)
class LogReturnsConfig:
    """Configuration for log-returns audit: thresholds and constants."""

    min_observations: int = 252
    mean_threshold: float = 0.005
    outlier_log_return_threshold: float = 0.20
    outlier_z_score_threshold: float = 4.0
    kurtosis_low: float = 3.0
    kurtosis_high: float = 10.0
    skew_threshold: float = 1.0
    max_draw_warning_threshold: float = -0.10  # warn if worst single-day return < -10%


# -----------------------------------------------------------------------------
# Computation
# -----------------------------------------------------------------------------


def compute_log_returns(
    df: pd.DataFrame,
    price_col: BronzeColumns = BronzeColumns.CLOSE,
) -> pd.DataFrame:
    """
    Compute log returns for a price series. Does not mutate the input.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a date column and the given price column.
    price_col : BronzeColumns
        Column used for the price series (typically CLOSE).

    Returns
    -------
    pd.DataFrame
        Copy of the DataFrame with a new 'log_ret' column; rows with NaN log_ret are kept.
    """
    out = df.sort_values(BronzeColumns.DATE.value).copy()
    out["log_ret"] = np.log(
        out[price_col.value] / out[price_col.value].shift(1)
    )
    return out


def compute_z_scores(log_ret: pd.Series) -> pd.Series:
    """
    Compute z-scores for a log-return series (sample mean and std).

    Parameters
    ----------
    log_ret : pd.Series
        Log returns; NaNs are ignored in the computation.

    Returns
    -------
    pd.Series
        Z-scores; may contain NaN where input was NaN or std was 0.
    """
    return pd.Series(stats.zscore(log_ret, nan_policy="omit"), index=log_ret.index)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def identify_outliers(
    df: pd.DataFrame,
    config: LogReturnsConfig,
) -> pd.Series:
    """
    Identify outlier rows by absolute log return and z-score thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'log_ret' and 'z_score'.
    config : LogReturnsConfig
        Uses outlier_log_return_threshold and outlier_z_score_threshold.

    Returns
    -------
    pd.Series
        Boolean mask, True where the row is an outlier.
    """
    return (
        (df["log_ret"].abs() > config.outlier_log_return_threshold)
        | (df["z_score"].abs() > config.outlier_z_score_threshold)
    )


def validate_data_quality(
    n_observations: int,
    symbol: str,
    config: LogReturnsConfig,
) -> bool:
    """
    Check that the series has enough observations for statistics.

    Parameters
    ----------
    n_observations : int
        Number of (non-NaN) log-return observations.
    symbol : str
        Symbol label for logging.
    config : LogReturnsConfig
        Uses min_observations.

    Returns
    -------
    bool
        True if n_observations >= config.min_observations.
    """
    if n_observations < config.min_observations:
        logger.warning(
            "Size of df: %s | symbol: %s",
            n_observations,
            symbol,
        )
        return False
    return True


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------


def compute_summary_statistics(
    log_ret: pd.Series,
    symbol: str,
) -> dict:
    """
    Compute summary statistics for a log-return series.

    Parameters
    ----------
    log_ret : pd.Series
        Log returns (NaN can be present; they are dropped for stats).
    symbol : str
        Symbol label included in the returned dict.

    Returns
    -------
    dict
        Keys: mean, std_dev, skew, kurtosis, max_draw, symbol.
        max_draw is the minimum single-day log return (worst day).
    """
    clean = log_ret.dropna()
    return {
        "mean": float(clean.mean()),
        "std_dev": float(clean.std()),
        "skew": float(stats.skew(clean, bias=False)),
        "kurtosis": float(stats.kurtosis(clean, bias=False)),
        "max_draw": float(clean.min()),
        "symbol": symbol,
    }


def analyze_global_statistics(
    stats_summary: dict,
    config: LogReturnsConfig,
) -> dict:
    """
    Validate global statistics and log warnings. Returns a small result dict.

    Parameters
    ----------
    stats_summary : dict
        Output of compute_summary_statistics (mean, std_dev, skew, kurtosis, max_draw, symbol).
    config : LogReturnsConfig
        Thresholds for mean, kurtosis, skew, and max_draw.

    Returns
    -------
    dict
        Validation result with keys like 'mean_ok', 'kurtosis_ok', 'skew_ok', 'max_draw_ok'.
    """
    symbol = stats_summary.get("symbol", "?")
    logger.info("Analyzing global statistics for %s", symbol)

    result: dict = {}

    # 1. Mean
    mean = stats_summary["mean"]
    result["mean_ok"] = -config.mean_threshold < mean < config.mean_threshold
    if not result["mean_ok"]:
        raise ValueError(
            f"Mean log return {mean} is not between "
            f"-{config.mean_threshold} and {config.mean_threshold}"
        )

    # 2. Kurtosis
    k = stats_summary["kurtosis"]
    result["kurtosis_ok"] = config.kurtosis_low <= k <= config.kurtosis_high
    if k < config.kurtosis_low:
        logger.warning("Global statistics: Kurtosis < %s", config.kurtosis_low)
    elif k > config.kurtosis_high:
        logger.warning("Global statistics: Kurtosis > %s", config.kurtosis_high)
    else:
        logger.info("Global statistics: Kurtosis is standard")

    # 3. Skew
    s = stats_summary["skew"]
    result["skew_ok"] = -config.skew_threshold <= s <= config.skew_threshold
    if s < -config.skew_threshold:
        logger.warning("Global statistics: Skew < %s", -config.skew_threshold)
    elif s > config.skew_threshold:
        logger.warning("Global statistics: Skew > %s", config.skew_threshold)
    else:
        logger.info("Global statistics: Skew is standard")

    # 4. Max draw (worst single-day return)
    max_draw = stats_summary["max_draw"]
    result["max_draw_ok"] = max_draw >= config.max_draw_warning_threshold
    if max_draw < config.max_draw_warning_threshold:
        logger.warning(
            "Global statistics: Max draw (min single-day return) %s below threshold %s",
            max_draw,
            config.max_draw_warning_threshold,
        )
    else:
        logger.info("Global statistics: Max draw within expected range")

    return result


def compute_rolling_statistics(
    df: pd.DataFrame,
    analysis_window: int,
) -> pd.DataFrame:
    """
    Add rolling statistics for log returns. Does not mutate the input.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'log_ret'.
    analysis_window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Copy of the DataFrame with window_mean, window_std_dev, window_skew, window_kurtosis.
    """
    if "log_ret" not in df.columns:
        raise ValueError(
            "Input DataFrame must have a 'log_ret' column containing log returns."
        )
    out = df.copy()
    r = out["log_ret"].rolling(
        window=analysis_window, min_periods=analysis_window
    )
    out["window_mean"] = r.mean()
    out["window_std_dev"] = r.std()
    out["window_skew"] = r.apply(lambda x: stats.skew(x, bias=False), raw=True)
    out["window_kurtosis"] = r.apply(
        lambda x: stats.kurtosis(x, bias=False), raw=True
    )
    return out

def volume_price_disparity(
    df: pd.DataFrame,
    symbol: str,
    *,
    logret_col: str = "log_ret",
    volume_col: BronzeColumns = BronzeColumns.VOLUME,
    logreturn_threshold: float = 0.1,
    volume_window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute the volatility-price disparity for a symbol.
    if |logreturn| > .1 check if 20 day avg volume > today's volume

    Returns a boolean mask indexed like `df` where the disparity condition holds.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the log returns and volume data.
    symbol : str
        The symbol of the stock.
    logret_col : str
        The column name of the log returns.
    volume_col : BronzeColumns
        The column name of the volume.
    logreturn_threshold : float
        The threshold for the log returns.
    volume_window : int
        The window size for the average volume.
    min_periods : int | None
        The minimum number of periods to compute the average volume.    
    Returns
    -------
    pd.Series
        A boolean mask indexed like `df` where the disparity condition holds.   
    Raises
    ------
    ValueError
        If the input DataFrame does not contain the log returns or volume columns.
    """
    if logret_col not in df.columns:
        raise ValueError(
            f"Input DataFrame must contain '{logret_col}' (computed log returns)."
        )
    if volume_col.value not in df.columns:
        raise ValueError(
            f"Input DataFrame must contain '{volume_col.value}' (volume)."
        )

    out = df.copy()
    if BronzeColumns.DATE.value in out.columns:
        out = out.sort_values(BronzeColumns.DATE.value)

    mp = volume_window if min_periods is None else min_periods

    # Use the average of *prior* volumes (shift(1)) so today's volume is compared
    # against the trailing 20-day mean without look-ahead bias.
    avg_volume = (
        out[volume_col.value]
        .shift(1)
        .rolling(window=volume_window, min_periods=mp)
        .mean()
    )

    disparity = (
        out[logret_col].abs() > logreturn_threshold
    ) & (avg_volume > out[volume_col.value])

    # Keep caller's original row order/index alignment.
    return disparity.reindex(df.index)
# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def plot_histogram(
    df: pd.DataFrame,
    symbol: str,
    audit_dir: Path | None = None,
) -> None:
    """
    Generate and save a histogram of log returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a `z_score` column (added by this module).
    symbol : str
        Symbol for title and filename.
    audit_dir : Path or None
        Directory to save the figure; defaults to AUDIT_DIR.
    """
    # make plots with the same scales for every symbol -> -.5 to .5
    z_scores = df["z_score"].dropna()
    audit_dir = Path(audit_dir or AUDIT_DIR) / "histograms"
    audit_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(z_scores, kde=True, stat="density", label="Observed (Z)")

    # Overlay Theoretical Normal for comparison
    x = np.linspace(-5, 5, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r--', label="Theoretical Normal")
    plt.xlim(-5, 5) # Fixed scale: 5 standard deviations
    plt.title(f"Standardized Return Distribution: {symbol}")
    plt.legend()
    plt.savefig(audit_dir / f"{symbol}_histogram.png")
    plt.close()


# -----------------------------------------------------------------------------
# I/O and orchestration
# -----------------------------------------------------------------------------


def load_bronze_data(pth: Path, interval: Interval) -> pd.DataFrame:
    """
    Load a bronze pickle and normalize date column.

    Parameters
    ----------
    pth : Path
        Path to an interval bronze pickle (e.g. `*_day.pkl`, `*_60minute.pkl`).
    interval : Interval
        Bronze interval; date normalization is only applied for `Interval.DAY`.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with date normalized.
    """
    df = KiteDataHandler.load(pth)
    if interval == Interval.DAY:
        df[BronzeColumns.DATE.value] = df[BronzeColumns.DATE.value].dt.normalize()
    return df


def save_audit_results(
    stats_summary: dict,
    audit_dir: Path,
) -> None:
    """
    Save all audits in a single json
    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame (with log_ret, status, etc.).
    stats_summary : dict or None
        If provided, can be persisted (e.g. as JSON) alongside the DataFrame.
    audit_dir : Path or None
        Directory to write to; defaults to AUDIT_DIR.
    """
    stats_path = audit_dir / f"log_returns_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    logger.info("Saved stats to %s", stats_path)

def process_symbol(
    df: pd.DataFrame,
    symbol: str,
    config: LogReturnsConfig | None = None,
    price_col: BronzeColumns = BronzeColumns.CLOSE,
    histogram: bool = False,
    audit_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run the full log-returns audit pipeline for one symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Bronze-style DataFrame (date, symbol, OHLC, etc.).
    symbol : str
        Symbol name for logging and outputs.
    config : LogReturnsConfig or None
        If None, uses default LogReturnsConfig().
    price_col : BronzeColumns
        Price column for log returns.
    histogram : bool
        Whether to save a histogram.
    audit_dir : Path or None
        Where to save audit outputs; defaults to AUDIT_DIR.

    Returns
    -------
    pd.DataFrame
        DataFrame with log_ret, z_score, status (Clean/Outlier), and optionally rolling stats.
    """
    config = config or LogReturnsConfig()
    audit_dir = Path(audit_dir or AUDIT_DIR)

    # Compute log returns (pure)
    df = compute_log_returns(df, price_col=price_col)
    df = df.dropna(subset=["log_ret"]).copy()

    if df.empty:
        logger.warning("No valid log returns for symbol %s", symbol)
        return df

    df["z_score"] = compute_z_scores(df["log_ret"])
    # Drop rows where z_score is NaN (e.g. constant series)
    df = df.dropna(subset=["z_score"]).copy()

    df["status"] = "Clean"
    outlier_mask = identify_outliers(df, config)
    if outlier_mask.any():
        n_out = outlier_mask.sum()
        logger.warning(
            "Symbol: %s has %s outliers based on log returns.",
            symbol,
            n_out,
        )
    df.loc[outlier_mask, "status"] = "Outlier"

    if histogram:
        plot_histogram(df, symbol, audit_dir=audit_dir)

    stats_summary = None
    if validate_data_quality(len(df), symbol, config):
        stats_summary = compute_summary_statistics(df["log_ret"], symbol)
        analyze_global_statistics(stats_summary, config)
    
    # Aggregate all symbol results in a dict and then save json with symbol as key 
    return stats_summary
    # save_audit_results(df, symbol, stats_summary, audit_dir=audit_dir)
    # return df


def main(
    plot_hist: bool,
    interval: Interval,
) -> None:
    """ 
    Compute log returns for all symbols for the given interval and save to the audit directory.
    """
    config = LogReturnsConfig()
    interval_audit_dir = Path(AUDIT_DIR) / interval.value
    interval_audit_dir.mkdir(parents=True, exist_ok=True)

    bronze_paths = list(Path(BRONZE_DIR).glob(f"*_{interval.value}.pkl"))
    if not bronze_paths:
        logger.warning(
            "No bronze files found for interval '%s' in %s",
            interval.value,
            BRONZE_DIR,
        )
        return

    stats_summary = {}
    for pth in bronze_paths:
        try:
            df = load_bronze_data(pth, interval=interval)
        except Exception as e:
            logger.exception("Failed to load %s: %s", pth, e)
            continue
        for name, group in df.groupby(BronzeColumns.SYMBOL.value):
            try:
                stats_summary[name] = process_symbol(
                    group,
                    name,
                    config=config,
                    histogram=plot_hist,
                    audit_dir=interval_audit_dir,
                )
            except Exception as e:
                logger.exception("Failed to process symbol %s: %s", name, e)
        
    # save stats_summary unified using save_audit_results 
    save_audit_results(
        stats_summary=stats_summary, 
        audit_dir=interval_audit_dir,
    )

if __name__ == "__main__":
    import argparse 
    # add argument for plotting histogram
    parser = argparse.ArgumentParser(description="Log returns audit")
    parser.add_argument(
        "--histogram",
        action="store_true",
        default=False,
        help="Plot histograms for each symbol",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=Interval.DAY.value,
        choices=[i.value for i in Interval],
        help="Bronze interval to audit",
    )
    args = parser.parse_args()
    main(plot_hist=args.histogram, interval=Interval(args.interval))

"""
Hurst exponent estimation via Rescaled Range (R/S) analysis.

Interpretation:
    H < 0.5  →  mean-reverting
    H ≈ 0.5  →  random walk (geometric Brownian motion)
    H > 0.5  →  trend-persistent (momentum)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HurstResult:
    """Structured result of a Hurst exponent estimate on a single series."""

    feature: str
    symbol: str
    H: float
    n_obs: int
    regime: str  # "mean_reverting" | "random_walk" | "trending"

    def to_dict(self) -> dict:
        """
        Serialise the result to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Keys: ``feature``, ``symbol``, ``H``, ``n_obs``, ``regime``.
        """
        return {
            "feature": self.feature,
            "symbol": self.symbol,
            "H": round(self.H, 4),
            "n_obs": self.n_obs,
            "regime": self.regime,
        }


def _classify_regime(H: float, lb: float = 0.45, ub: float = 0.55) -> str:
    """
    Map a Hurst exponent to a regime label.

    Arguments
    ---------
    H : float
        Estimated Hurst exponent.
    lb : float
        Lower bound of the random-walk band (exclusive).
    ub : float
        Upper bound of the random-walk band (exclusive).

    Returns
    -------
    str
        ``"mean_reverting"`` when H < lb, ``"trending"`` when H > ub,
        ``"random_walk"`` otherwise.
    """
    if H < lb:
        return "mean_reverting"
    elif H > ub:
        return "trending"
    return "random_walk"


def rescaled_range_hurst(
    series: pd.Series,
    *,
    feature: str,
    symbol: str,
    min_obs: int = 100,
    min_chunk: int = 8,
) -> HurstResult | None:
    """
    Estimate the Hurst exponent using the Rescaled Range (R/S) method.

    Splits the series into chunks of varying sizes, computes R/S for each,
    then regresses log(R/S) on log(chunk size). The slope of that regression is H.

    Arguments
    ---------
    series : pd.Series
        The time series to analyse (e.g. log returns or a price column).
    feature : str
        Column name label included in the result (for reporting).
    symbol : str
        Ticker symbol label included in the result (for reporting).
    min_obs : int
        Minimum number of finite observations required; returns ``None`` if not met.
    min_chunk : int
        Smallest chunk size (power-of-2 series starts here).

    Returns
    -------
    HurstResult | None
        Structured result with H, regime, and n_obs, or ``None`` when the series is
        too short or produces fewer than two valid chunk-size points for regression.
    """
    clean = series.dropna().values.astype(float)
    N = len(clean)
    if N < min_obs:
        return None

    # Chunk sizes: powers of 2 from min_chunk up to N//2
    max_k = N // 2
    chunk_sizes = []
    k = min_chunk
    while k <= max_k:
        chunk_sizes.append(k)
        k *= 2

    if len(chunk_sizes) < 2:
        return None

    log_rs_means: list[float] = []
    log_ns: list[float] = []

    for n in chunk_sizes:
        n_chunks = N // n
        if n_chunks == 0:
            continue

        rs_values: list[float] = []
        for i in range(n_chunks):
            chunk = clean[i * n : (i + 1) * n]
            mean_chunk = chunk.mean()
            deviations = chunk - mean_chunk
            cumdev = np.cumsum(deviations)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_values.append(R / S)

        if rs_values:
            log_rs_means.append(np.log(np.mean(rs_values)))
            log_ns.append(np.log(n))

    if len(log_ns) < 2:
        return None

    # OLS: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(log_ns, log_rs_means, 1)
    H = float(coeffs[0])

    return HurstResult(
        feature=feature,
        symbol=symbol,
        H=H,
        n_obs=N,
        regime=_classify_regime(H),
    )

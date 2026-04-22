"""
Augmented Dickey-Fuller stationarity test helpers.

Wraps ``statsmodels.tsa.stattools.adfuller`` with a thin result dataclass
so callers get structured output instead of raw tuples.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class ADFResult:
    """Structured result of an ADF test on a single series."""

    feature: str
    symbol: str
    adf_stat: float
    p_value: float
    n_obs: int
    stationary: bool  # True when p_value < alpha

    def to_dict(self) -> dict:
        """
        Serialise the result to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Keys: ``feature``, ``symbol``, ``adf_stat``, ``p_value``, ``n_obs``, ``stationary``.
        """
        return {
            "feature": self.feature,
            "symbol": self.symbol,
            "adf_stat": round(self.adf_stat, 6),
            "p_value": round(self.p_value, 6),
            "n_obs": self.n_obs,
            "stationary": self.stationary,
        }


def run_adf(
    series: pd.Series,
    *,
    feature: str,
    symbol: str,
    alpha: float = 0.05,
    autolag: str | None = "AIC",
    min_obs: int = 30,
) -> ADFResult | None:
    """
    Run an ADF test on *series* and return a structured result.

    Arguments
    ---------
    series : pd.Series
        The time series to test for a unit root.
    feature : str
        Column name label included in the result (for reporting).
    symbol : str
        Ticker symbol label included in the result (for reporting).
    alpha : float
        Significance level; the series is considered stationary when p-value < alpha.
    autolag : str or None
        Lag selection criterion passed to :func:`statsmodels.tsa.stattools.adfuller`.
        Pass ``None`` to skip the lag search and use the Schwert-rule default
        maxlag (``int(ceil(12 * (nobs/100)^0.25))``) — much faster and uses far
        less memory at the cost of slightly noisier p-values.
    min_obs : int
        Minimum number of finite observations required to run the test.

    Returns
    -------
    ADFResult | None
        Structured result, or ``None`` when the series has fewer than *min_obs* finite
        values or is constant (zero variance).
    """
    clean = series.dropna()
    if len(clean) < min_obs:
        return None
    if clean.nunique() == 1:
        return None

    result = adfuller(clean.values, autolag=autolag)
    # statsmodels returns a 6-tuple when autolag is set (last element is icbest)
    # and a 5-tuple when autolag is None.
    stat, p, _used_lag, nobs = result[:4]
    return ADFResult(
        feature=feature,
        symbol=symbol,
        adf_stat=float(stat),
        p_value=float(p),
        n_obs=int(nobs),
        stationary=p < alpha,
    )

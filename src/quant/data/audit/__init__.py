"""
Data quality audits on bronze and silver feeds.

- ``quant.data.audit.returns`` — log returns, outliers, cross-sectional regime
- ``quant.data.audit.dates_and_volumes`` — missing bars, null volume (all intervals)
- ``quant.data.audit.adf`` — Augmented Dickey-Fuller stationarity test helpers
- ``quant.data.audit.hurst_exponent`` — Hurst exponent estimation (R/S method)
- ``quant.data.audit.regime_report`` — universe-level ADF + Hurst regime audit

Run: ``python -m quant.data.audit`` (returns audit CLI).
Run: ``python -m quant.data.audit.regime_report --index nifty_50 --interval day``
"""

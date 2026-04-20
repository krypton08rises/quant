# quant.data.audit

Data-quality and regime diagnostics. Two jobs:

1. **Row-level audit** — per-row flags attached to the bronze layer
   (outlier returns, abnormal volumes, missing dates). These are consumed
   by `processing/build_bronze.py` and written as `flag_*` columns.
2. **Universe-level audit** — stationarity (ADF) and long-memory (Hurst)
   diagnostics computed on silver training data; used to justify which
   features are safe for ML.

Artifacts land under `AUDIT_DIR = src/quant/data/audit/analysis/`.

## Files

- `returns/log_returns.py` — log-return distribution audit: mean/std/skew/
  kurtosis per symbol, rolling z-score, volume-price-disparity flag.
  Exposes `process_symbol(...)` used by `build_bronze`, plus a CLI.
- `dates_and_volumes/missing_dates.py` — `filter_active_universe` (drops
  dead symbols), `remove_phantom_ticks` (trims pre-gap history).
- `dates_and_volumes/null_volume.py` — zero-volume bar inspection.
- `adf.py` — Augmented Dickey–Fuller stationarity test wrapper, returning
  a structured `AdfResult`.
- `hurst_exponent.py` — rescaled-range Hurst estimator with regime
  classification (mean-reverting / random-walk / trending).
- `regime_report.py` — orchestrator that runs ADF + Hurst across an entire
  index universe and aggregates to JSON / CSV.

## Row-level flags

These columns are added during `build_bronze` and are present in every
bronze and silver file (defined in `FlagColumns`):

| Column                        | Added by                               | Meaning                                                         |
|-------------------------------|----------------------------------------|-----------------------------------------------------------------|
| `flag_null_volume`            | `build_bronze`                         | Row has `volume == 0`.                                          |
| `flag_log_ret`                | `log_returns.process_symbol`           | Row's absolute log return exceeds the configured threshold.     |
| `flag_z_score`                | `log_returns.process_symbol`           | Rolling-window z-score of log return is an outlier.             |
| `flag_volume_price_disparity` | `log_returns.process_symbol`           | Big price move on abnormally low volume.                        |

Tunable thresholds live in `LogReturnsConfig` (`returns/log_returns.py`).
Flags are advisory — they never drop rows, only mark them for downstream
inspection.

## Log-returns audit CLI

```bash
python -m quant.data.audit
# or, equivalently:
python -m quant.data.audit.returns
```

Produces per-symbol histograms, rolling-stats plots, and an
`log_returns_stats.json` aggregate under `AUDIT_DIR`. Good hygiene check
after a large raw-data refresh.

## Regime report (universe-level)

`regime_report.py` walks a silver `train.parquet`, slices it to
`[CUTOFF_DATE, TRAIN_END_DATE)`, and runs ADF + Hurst per
symbol × feature. It groups features into two expected buckets:

- **Stationary** (expected `H≈0.5`, ADF rejects): `rsi`, `bb_pct`,
  `macd_hist`, `log_ret`.
- **Non-stationary** (expected `H>0.5`, ADF fails to reject): `open`,
  `high`, `low`, `close`, `atr`, `ema_12`, `ema_26`, `macd_line`,
  `macd_signal`, `bb_upper`, `bb_lower`, `bb_mid`.

These lists are the empirical backstop for the stationarity rule in
`processing/README.md`.

```bash
python -m quant.data.audit.regime_report --index nifty_50 --interval day
python -m quant.data.audit.regime_report --index nifty_500 --interval day --csv
```

Outputs to `AUDIT_DIR/regime/`:

- `regime_summary_<index>_<interval>.json` — universe aggregates
  (percent stationary, mean p-value, Hurst regime breakdown).
- `regime_detail_<index>_<interval>.json` — per-stock rows.
- `{adf,hurst}_detail_<index>_<interval>.csv` — same data in CSV when
  `--csv` is passed.

A pretty-printed terminal summary is always shown at the end of a run.

## Tuning knobs

- `LogReturnsConfig` (`returns/log_returns.py`) — outlier thresholds,
  rolling window, minimum observations.
- `RegimeReportConfig` (`regime_report.py`) — ADF alpha, minimum
  observations, Hurst random-walk band.

Both are frozen dataclasses; override by constructing a new instance and
passing it explicitly.

## Conventions

- Audit reads **silver training data only** (`train.parquet`). Test and
  embargoed data are never touched here.
- All paths write under `AUDIT_DIR`; never into `historical/`.
- The regime report reuses the same index CSVs as the rest of the
  pipeline (`data/indices/*.csv`), indexed through the `Indices` enum.

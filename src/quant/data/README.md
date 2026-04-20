# quant.data

Everything that turns raw market data into training-ready feature tables lives
here. The layout follows a **medallion model** — raw bytes from the API flow
through progressively cleaner and more feature-rich layers, with quality
checks between each step.

```
raw  ──►  bronze  ──►  silver  ──►  (gold)
 │         │             │
 │         └── audit flags added
 │                       └── indicators + triple-barrier labels
 └── untouched Kite OHLCV + candle features
```

## Layers

| Layer  | Path                       | Contents                                                              | Producer                           |
|--------|----------------------------|-----------------------------------------------------------------------|------------------------------------|
| Raw    | `historical/raw/`          | Kite OHLCV + candle-shape features (`upper_shadow`, `tick_body`, …)   | `kite/kite_data.py`                |
| Bronze | `historical/bronze/`       | Raw + per-row audit flags (`flag_log_ret`, `flag_z_score`, …)         | `processing/build_bronze.py`       |
| Silver | `historical/silver/<spec>/`| Bronze + technical indicators + triple-barrier labels, split-sliced   | `processing/build_silver.py`       |
| Gold   | (pending)                  | Feature-validated parquet + manifest (spec hash)                      | future                             |

Files in `raw/` and `bronze/` are **per-symbol** dill pickles named
`<SYMBOL>_<interval>.pkl`. Silver data lands under
`silver/<interval>__<spec_config>/` and is split into `train.parquet`,
`test.parquet`, `embargoed.parquet` — see `processing/README.md` for cutoffs.

## Single source of truth: `_common.py`

Column names, enum values, data-root paths, and cutoff dates live in
`_common.py` and **must not be duplicated**. When writing new code:

- Column names → `RawColumns`, `FlagColumns`, `SilverColumns` (all `StrEnum`).
- Enums → `Indices`, `Interval`, `VolMethod`, `EntryPriceMode`, `TieBreaking`,
  `BarrierMode`.
- Triple-barrier config → `TripleBarrierSpec` dataclass; its `config_str()`
  and `silver_dir(interval)` define where silver artifacts are written.
- Paths → `RAW_DIR`, `BRONZE_DIR`, `SILVER_DIR`, `AUDIT_DIR`; all anchored to
  the package directory so working directory never matters.
- Cutoffs → `CUTOFF_DATE` (earliest training boundary), `TRAIN_END_DATE`
  (val/test boundary), `EMBARGOED_DATE_START` (held-out live-eval data —
  never used in training).

## Submodules

- **`kite/`** — Kite Connect API wrapper + per-symbol raw fetcher.
  See `kite/README.md`.
- **`processing/`** — bronze and silver builders, indicators, triple-barrier
  labelling. See `processing/README.md`.
- **`audit/`** — data quality checks: log-returns outliers, missing dates,
  ADF stationarity, Hurst exponent, universe regime reports.
  See `audit/README.md`.
- **`indices/`** — static NSE index constituent CSVs (Nifty50, BankNifty,
  Midcap150, sectoral). Referenced by `Indices` enum values.
- **`yfin_extract.py`** — Yahoo Finance fallback (token-bucket rate-limited,
  multi-threaded). Used when Kite is unavailable or for cross-checks.

## Typical flow

```bash
# 1. Backfill raw data for an index universe (one symbol at a time, resumable).
python -m quant.data.kite.kite_data --indices nifty_500 --interval day

# 2. Build bronze: audit flags + filtering.
python -m quant.data.processing.build_bronze --interval day

# 3. Build silver: indicators + triple-barrier labels. Per-symbol caches make
#    it safe to re-run — cached pairs are skipped unless --force is passed.
python -m quant.data.processing.build_silver --interval day

# 4. (Optional) universe-level regime audit on silver training data.
python -m quant.data.audit.regime_report --index nifty_50 --interval day
```

## Conventions

- All train/test splits are **time-based**, never random shuffle.
- `EMBARGOED_DATE_START = 2025-01-01` is strictly held out; no training
  code may read rows past it.
- Non-stationary features (raw prices, EMAs, Bollinger levels, ATR in rupee
  terms) are dropped before ML; stationary features (`rsi`, `bb_pct`,
  `percent_change`, `macd_hist`) are kept. Rationale in
  `src/quant/ml/README.md`.
- Legacy code in `historical_v1/` and `wastebin/` is archival — do not
  import from it.

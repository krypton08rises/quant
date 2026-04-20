# quant.data.processing

Turns raw Kite pickles into labelled, feature-rich silver data. Two
pipelines live here — `build_bronze.py` and `build_silver.py` — plus the
indicator and triple-barrier primitives they compose.

## Files

- `build_bronze.py` — raw → bronze: audit flags + universe/phantom filtering.
- `build_silver.py` — bronze → silver: technical indicators + triple-barrier
  labels, with per-symbol caching.
- `indicators.py` — Wilder's RSI, MACD, Bollinger Bands, ATR, EMA.
- `advanced_features.py` — stationarity (ADF) and collinearity validation
  used before promoting features into the gold layer.

## Bronze layer — `build_bronze.py`

For each raw pickle `<SYMBOL>_<interval>.pkl` in `RAW_DIR`, it:

1. Normalises daily timestamps and drops `(symbol, date)` duplicates.
2. Drops rows with `close <= 0` (phantom/zero bars from the API).
3. `filter_active_universe` — drops symbols with no data in 2026
   (delisted / inactive).
4. `remove_phantom_ticks` — truncates each symbol's pre-gap history.
5. Adds the four flag columns defined in `FlagColumns`:
   - `flag_null_volume` — row has zero volume.
   - `flag_log_ret` — row's log return is a distributional outlier.
   - `flag_z_score` — rolling z-score outlier.
   - `flag_volume_price_disparity` — big price move on abnormally low volume.

Output goes to `BRONZE_DIR/<SYMBOL>_<interval>.pkl` (same dill payload format
as raw; downstream code loads it through `KiteDataHandler.load`).

```bash
python -m quant.data.processing.build_bronze --interval day
python -m quant.data.processing.build_bronze --interval all
```

## Silver layer — `build_silver.py`

For each bronze pickle, `build_silver` produces indicators + triple-barrier
labels and writes a **per-symbol cache** so runs are resumable:

```
SILVER_DIR/<interval>__<spec_config>/
├── _per_symbol/
│   ├── 360ONE.parquet
│   ├── AARTIIND.parquet
│   └── ...
├── train.parquet        # date <= CUTOFF_DATE
├── test.parquet         # CUTOFF_DATE < date < EMBARGOED_DATE_START
└── embargoed.parquet    # date >= EMBARGOED_DATE_START (held-out)
```

The `<spec_config>` segment encodes every triple-barrier knob
(`H`, `pt_k`, `sl_k`, vol method, entry mode, tie-breaking), so different
specs produce different directories and never collide.

### Indicators added

From `indicators.py`, using the symbol's own sorted close series:

- `ema_12`, `ema_26`
- `rsi` (Wilder's 14-period)
- `macd_line`, `macd_signal`, `macd_hist` (12/26/9)
- `bb_mid`, `bb_upper`, `bb_lower`, `bb_pct` (20-period, k=2)
- `atr` (14-period, Wilder EMA of true range)

### Idempotency

A symbol-interval pair is **skipped** if its per-symbol parquet already
exists. Pass `--force` to reprocess:

```bash
python -m quant.data.processing.build_silver --interval day           # resumable
python -m quant.data.processing.build_silver --interval day --force   # rebuild all
python -m quant.data.processing.build_silver --interval all           # every supported interval
```

Per-symbol exceptions are logged and swallowed so one bad bronze file can't
kill a 500-symbol run. After every symbol is processed, the caches are
concatenated and split into `train/test/embargoed.parquet`.

## Triple-barrier labelling

Labels encode whether price hits a profit target (`+1`), stop loss (`-1`),
or neither (`0`) within `H` trading days.

**Spec knobs** (`TripleBarrierSpec` in `_common.py`):

| Field         | Default                | Meaning                                                    |
|---------------|------------------------|------------------------------------------------------------|
| `H`           | `5`                    | Vertical barrier in trading days                           |
| `pt_k`, `sl_k`| `1.0`, `1.0`           | Profit-target / stop-loss multipliers of vol               |
| `vol_method`  | `VolMethod.ATR`        | Volatility measure used to size barriers                   |
| `atr_length`  | `14`                   | ATR lookback                                               |
| `entry`       | `EntryPriceMode.NEXT_OPEN` | Entry at `t+1` open vs `t` close                       |
| `use_high_low`| `True`                 | Daily hit detection from highs/lows; intraday uses close   |
| `tie_breaking`| `TieBreaking.CONSERVATIVE` | If both PT and SL hit same day, treat as SL (`-1`)     |
| `barrier_mode`| `BarrierMode.RETURNS_BARRIERS` | Barrier parameterisation                           |

**Label rules:**

- Hit PT first → `+1`
- Hit SL first → `-1`
- Neither by `H` → `0`
- Both same day → resolved per `tie_breaking`:
  - `CONSERVATIVE` → `-1` (pessimistic; assume SL triggered first)
  - `ZERO` → `0`

Sample count and first/last-valid-`t` per symbol are worth plotting
whenever you change a spec — the vertical barrier eats `H` rows off the
tail and the ATR warmup eats ~`atr_length` rows off the head.

## Splits

Applied **after** labelling, at assembly time in `build_silver`:

- `train` — `date <= CUTOFF_DATE` (2020-12-31)
- `test` — `CUTOFF_DATE < date < EMBARGOED_DATE_START` (2025-01-01)
- `embargoed` — `date >= EMBARGOED_DATE_START` — held out for live
  evaluation; **never used in training**.

`TRAIN_END_DATE` (2024-01-01) is the separate val/test boundary used by
`src/quant/ml/config.py`; it's applied at dataset-loading time, not here.

## Stationarity rule for downstream ML

Silver exposes every indicator, but the ML layer only consumes stationary
ones. Hard rule (see `src/quant/ml/README.md`):

- **Drop (non-stationary):** raw prices, EMAs, Bollinger band levels,
  ATR in rupee terms, raw shadow/body values.
- **Keep (stationary):** `rsi`, `bb_pct`, `percent_change`, `macd_hist`.

The per-universe regime report under `audit/regime_report.py` empirically
confirms this split (ADF p-values + Hurst exponents).

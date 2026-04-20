# quant.data.kite

Zerodha Kite Connect wrapper + per-symbol raw data fetcher. This is the
**raw-layer producer** for the medallion pipeline — everything downstream
(bronze, silver) reads from the pickles this module writes.

## Files

- `kite_handler.py` — `KiteDataHandler`: API session, instrument-token lookup,
  chunked historical fetches with retries, dill-based save/load.
- `kite_data.py` — CLI / async orchestrator that iterates an index universe
  and backfills per-symbol raw files into `historical/raw/`.

## Authentication

Kite access tokens expire daily. The OAuth flow is handled outside this
module by `src/quant/apis/zerodha.py` (a FastAPI callback server). After
login it writes `data/access_token.json`, which `KiteDataHandler` reads on
construction.

Environment variables required (via `.env`):

- `ZERODHA_API_KEY`
- (plus whatever `apis/zerodha.py` needs for the OAuth handshake)

If the access token is expired or missing, `KiteDataHandler` logs a warning
and continues with **limited functionality** — `load()` still works for
reading previously saved pickles, but `fetch_historical()` will fail.

## Data fetching

`KiteDataHandler.fetch_historical(symbols, interval, start, end)` is `async`
and respects Kite's hard limit of `MAX_DAYS_PER_CALL = 100` days per
historical request. It:

1. Maps each symbol to its `instrument_token` via the NSE instruments CSV.
2. Chunks the date range into ≤100-day windows.
3. Retries transient failures with exponential backoff (`max_retries=3`).
4. Calls `_generate_features` to add candle-shape columns:
   `upper_shadow`, `lower_shadow`, `tick_body`, `diff`, `shifted_close`,
   `percent_change`, `classification_marker`.
5. Returns one concatenated DataFrame, sorted by `(symbol, date)`.

The historical endpoint is synchronous on Kite's side, so the "async" here
only buys us cooperative scheduling, not true parallelism. Rate limiting and
retry live inside the handler — don't add a second layer on top.

## Persistence

`save(path)` and `load(path)` use `dill` (not pickle) so that pandas objects
with Arrow-backed string arrays serialise cleanly. The on-disk payload is a
dict `{"data": <DataFrame>}`; the `.pkl` suffix is enforced automatically.

Files are **per-symbol, per-interval**:

```
historical/raw/<SYMBOL>_<interval>.pkl
```

This layout was chosen deliberately (see commit `b52f6a8`): it means
overlapping index memberships (e.g., a stock in both Nifty50 and
Nifty500) never trigger duplicate API calls, and a crash mid-backfill
leaves previously fetched symbols intact.

## CLI: backfilling a universe

`kite_data.py` is the entry point for bulk fetches. For each symbol:

- If `raw/<SYMBOL>_<interval>.pkl` exists, resume from the day **after** its
  max date.
- Otherwise, cold-start from `--date` through today.

```bash
# One index, one interval.
python -m quant.data.kite.kite_data --index nifty_500 --interval day --date 2015-01-01

# Full universe, every supported intraday + daily interval.
python -m quant.data.kite.kite_data --index all --interval all --date 2015-01-01
```

`--index all` expands to every `Indices` enum value; `--interval all` expands
to `{5minute, 15minute, 60minute, day}`. Symbols are deduplicated across the
union of selected index CSVs, so overlapping memberships never double-fetch.
Missing index CSVs are warned and skipped so partial universes still work.

## Notes for callers

- **Do not import `src/quant/utils/fetch_data.py`** — it's deprecated; use
  `KiteDataHandler` instead.
- `KiteDataHandler.load()` returns `None` (not a raise) when the file is
  missing; callers must handle that branch.
- `instrument_map` is filtered to `instrument_type == "EQ"` and
  `exchange == "NSE"` — futures, options, and BSE listings are not
  addressable through this handler.

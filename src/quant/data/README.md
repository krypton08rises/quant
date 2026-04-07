Defines a triple barrier labelling script.
## Config / spec parameters:
- H = vertical barrier in trading days (start with 5)
- vol_method = "ATR" (recommended for daily)
- atr_length = 14 // standard trading practice 
- pt_k, sl_k = multipliers (start with 1.0 / 1.0)
- price_ref = which price you “enter” at (start with next day open OR close at t — pick one and stick to it)
- barrier_mode:
- “returns barriers” (recommended): compare future high/low to entry price via returns

## label rules:
- hit PT first → +1
hit SL first → -1
- neither by H → 0

tie-breaking (rare but must be defined):
if both hit same day: choose the one that’s hit first intraday is unknown on daily bars → decide rule:
conservative: treat as SL (pessimistic)
or: treat as 0
or: use open->high/low path assumption (I’d avoid)
or: load intraday data to resolve tie (more complex, but most accurate)

For sanity check:
set input_window, padding_length (so that we have enough data to compute indicators at t=0, and normalize properly)
Plot: histogram of sample counts, and show the first/last valid t

From start of availability of data to cutoff date, we're taking input_window len sliding window, then checking on the basis of prices 5 days in the future, 
1. if it hits our target pl first: label 1
2. if it hits our stop loss first: label -1 
3. if it hits neither: label 0 


Save data to parquet / zarr / npy +a small manifest json with spec hash

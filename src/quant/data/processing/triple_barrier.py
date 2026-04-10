import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from quant.data._common import (
    BRONZE_DIR,
    CUTOFF_DATE,
    EMBARGOED_DATE_START,
    SILVER_DIR,
    EntryPriceMode,
    Indices,
    Interval,
    RawColumns,
    SilverColumns,
    TieBreaking,
    TripleBarrierSpec,
    VolMethod,
)
from tqdm import tqdm

timestamp = date.today().isoformat()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_entry_price(df: pd.DataFrame, t: int, spec: TripleBarrierSpec) -> float:
    """
    Compute the entry price at time t based on the specification.
    Arguments
    ---------
    df: pd.DataFrame
        The DataFrame containing price data.
    t: int
        The index of the data point.
    spec: TripleBarrierSpec
        The specification for the triple barrier method.
    Returns
    -------
    float
        The entry price at time t.
    """
    assert (
        "open" in df.columns and "close" in df.columns
    ), f"DataFrame must contain 'open' and 'close' columns. Only have: {df.columns.tolist()}"
    if spec.entry == EntryPriceMode.TODAYS_CLOSE:
        return df["close"].iloc[t]
    elif spec.entry == EntryPriceMode.NEXT_OPEN:
        return df["open"].iloc[t + 1]
    raise ValueError(f"Unsupported entry price mode: {spec.entry}")


def compute_atr(df: pd.DataFrame, t: int, spec: TripleBarrierSpec) -> float:
    """
    Compute the volatility measure at time t based on the specification.
    Arguments
    ---------
    df: pd.DataFrame
        The DataFrame containing price data and indicators.
    t: int
        The index of the data point.
    spec: TripleBarrierSpec
        The specification for the triple barrier method.
    Returns
    -------
    float
        The volatility measure at time t.
    """
    if spec.vol_method == VolMethod.ATR:
        return df[SilverColumns.ATR].iloc[t]
    raise ValueError(f"Unsupported volatility method: {spec.vol_method}")


def label_at_t(df: pd.DataFrame, t: int, spec: TripleBarrierSpec) -> Optional[int]:
    """
    Label the data point at time t using the triple barrier method.
    Arguments
    ---------
    df: pd.DataFrame
        The DataFrame containing price data and indicators.
    t: int
        The index of the data point to label.
    spec: TripleBarrierSpec
        The specification for the triple barrier method.
    Returns
    -------
    Optional[int]
        The label for the data point at time t: +1 for profit target hit, -1 for stop loss hit, 0 for neither.
    """

    df = df.reset_index(drop=True)
    entry_price = compute_entry_price(df, t, spec)
    vol = compute_atr(df, t, spec)

    # logger.info(f"At index {t}, entry price: {entry_price}, volatility: {vol}")
    if vol is None or np.isnan(vol) or vol == 0:
        logger.warning(f"Volatility is invalid at index {t}, cannot compute label.")
        return None  # Cannot compute label without valid volatility

    pt_barrier = entry_price + (spec.pt_k * vol)
    sl_barrier = entry_price - (spec.sl_k * vol)

    # If t = Jan 1st
    start_idx = t + 1 if spec.entry == EntryPriceMode.TODAYS_CLOSE else t + 1

    # start checking our labels from Jan 2nd
    end_idx = min(t + spec.H, len(df))  # if H=5, check till Jan 6th (exclusive)
    if t + spec.H >= len(df):
        date_val = df[RawColumns.DATE].iloc[t] if RawColumns.DATE in df.columns else t
        logger.warning(
            f"Not enough future data to apply vertical barrier of H={spec.H} days at index {t}. Only have {len(df)-t-1} days ahead. Date is {date_val}."
        )

    # logger.info(f"Checking barriers from index {start_idx} to {end_idx-1} for entry at index {t}. PT: {pt_barrier}, SL: {sl_barrier}")
    for i in range(start_idx, end_idx):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # logger.info(f"Index {i}: High={high}, Low={low}")
        if spec.use_high_low:
            if high >= pt_barrier and low <= sl_barrier:
                # both hit same day, apply tie-breaking rule
                if spec.tie_breaking == TieBreaking.CONSERVATIVE:
                    return -1  # treat as stop loss hit
                elif spec.tie_breaking == TieBreaking.ZERO:
                    return 0  # treat as neither
                # alternatively, can load intraday data to resolve tie
            elif high >= pt_barrier:
                return +1  # profit target hit
            elif low <= sl_barrier:
                return -1  # stop loss hit
        else:
            close = df["close"].iloc[i]
            # logger.info(f"Index {i}: Close={close}")
            if close >= pt_barrier:
                return +1
            elif close <= sl_barrier:
                return -1
    return 0  # neither barrier hit within H days


def generate_static_dataset(interval: Interval, spec: TripleBarrierSpec):
    """
    Generates a static dataset for all stocks available in our file system.
    Initially only for daily interval.
    Arguments
    ---------
    interval: Interval
        The data interval (e.g., daily, 60minute).
    Returns
    -------
    pd.DataFrame
        The DataFrame containing the labelled data.
    """
    from quant.data.kite.kite_handler import KiteDataHandler
    from quant.data.processing.indicators import generate_indicators_from_df

    master = []
    for index in Indices:
        # Load all data
        data = KiteDataHandler.load(Path(f"{BRONZE_DIR}/{index.value}_{interval.value}.pkl"))

        master.append(data)
    df = pd.concat(master, ignore_index=True)
    # remove any duplicates
    df.drop_duplicates(subset=[RawColumns.SYMBOL, RawColumns.DATE], inplace=True)
    number_of_symbols = df[RawColumns.SYMBOL].nunique()
    logger.info(
        f"Loaded bronze data for {number_of_symbols} unique symbols across indices at interval {interval.value}."
    )

    # For each symbol, compute indicators and subsequently labelling
    def process_symbol(sym_df: pd.DataFrame) -> pd.DataFrame:
        symbol_val = sym_df.name if hasattr(sym_df, "name") else sym_df.index[0]
        sym_df[RawColumns.SYMBOL.value] = symbol_val
        # logger.info(f"Columns available for symbol : {sym_df.columns.tolist()}")
        sym_df = sym_df.sort_values(by=RawColumns.DATE).reset_index(drop=True)
        sym_df = generate_indicators_from_df(sym_df, tag=RawColumns.CLOSE)
        assert (
            SilverColumns.ATR in sym_df.columns
        ), "ATR indicator must be computed before labelling."

        for t in tqdm(
            range(len(sym_df) - spec.H - 1),
            desc=f"Labeling Symbol {sym_df[RawColumns.SYMBOL.value].iloc[0]} for H={spec.H}",
        ):
            label = label_at_t(sym_df, t, spec)
            sym_df.loc[t, f"label_{spec.H}"] = label
        return sym_df

    try:
        [TripleBarrierSpec(H=5), TripleBarrierSpec(H=7)]
        df_labelled = (
            df.groupby(RawColumns.SYMBOL, as_index=False)
            .apply(process_symbol)
            .reset_index(drop=True)
        )
        assert (
            RawColumns.SYMBOL.value in df_labelled.columns
        ), "After labelling, SYMBOL column must exist."
        # logger.info(f"Completed labelling for all symbols.")
        return df_labelled
    except Exception as e:
        logger.error(f"Error during labelling: {e}")
        raise e


def test_silver_data(df: pd.DataFrame) -> bool:
    """
    Tests the silver data for correctness.
    Arguments
    --------
    df: pd.DataFrame
        The DataFrame containing the labelled data.
    Returns
    -------
    bool
        True if the data passes all tests, False otherwise.
    """
    # assert RSI is between 0 and 100
    if SilverColumns.RSI in df.columns:
        if not df[SilverColumns.RSI].between(0, 100).all():
            logger.error("RSI values out of bounds [0, 100].")
            return False

    # assert


def main():
    """
    Main function to generate the static dataset.
    """
    spec = TripleBarrierSpec(H=5)
    df_static = generate_static_dataset(Interval.DAY, spec)

    train_path = Path(f"{SILVER_DIR}/{Interval.DAY.value}_{spec.H}_{spec.pt_k}_{spec.sl_k}.parquet")
    test_path = Path(
        f"{SILVER_DIR}/test_{Interval.DAY.value}_{spec.H}_{spec.pt_k}_{spec.sl_k}.parquet"
    )
    embargoed_path = Path(
        f"{SILVER_DIR}/embargoed_{Interval.DAY.value}_{spec.H}_{spec.pt_k}_{spec.sl_k}.parquet"
    )

    train_df = df_static[df_static[RawColumns.DATE.value] <= CUTOFF_DATE]
    test_df = df_static[
        (df_static[RawColumns.DATE.value] > CUTOFF_DATE)
        & (df_static[RawColumns.DATE.value] <= EMBARGOED_DATE_START)
    ]
    logger.info(f"Training set size: {len(train_df)}, Testing set size: {len(test_df)}")
    embargoed_df = test_df[test_df[RawColumns.DATE.value] >= EMBARGOED_DATE_START]
    train_df.to_parquet(train_path)
    logger.info(f"Static labelled dataset saved to {train_path}.")

    test_df.to_parquet(test_path)
    logger.info(f"Static labelled dataset saved to {test_path}.")

    embargoed_df.to_parquet(embargoed_path)
    logger.info(f"Static labelled dataset saved to {embargoed_path}.")


if __name__ == "__main__":
    main()

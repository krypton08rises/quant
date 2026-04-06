"""
Eventually to be made into a service ran everyday as a cronjob to backfill data for indices
and update data for symbols.
Saves data and metadata to pickle files in data/historical directory.
Can be used to keep data updated and model trained on a daily basis.
"""
import argparse
import asyncio
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from quant.data._common import Indices
from quant.data.kite.kite_handler import KiteDataHandler

# Configure logging
timestamp = date.today().isoformat()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

"""
TODO:
Remove normalization over global statistics here -- should be done rolling window during training/inference only.
Add tests, typehints, docstrings.
Add functionality to backfill data from a given date to today.
"""


async def backfill_index(
    index_name: str,
    interval: str,
    from_date: date | None,
    indices_dir: Path = Path("data/indices"),
    out_dir: Path = Path("data/historical/bronze/"),
):
    """
    Backfill historical data for a given index from the specified date.
    If the data already exists, it will load the existing data and append new data.
    If there is a from_date, it will backfill from that date.
    to_date defaults to today if not provided.
    This function fetches historical data for all symbols in the index and saves it to a pickle file.
    Arguments
    ----------
    index_name (str)
        Name of the index (filename under data/indices without .csv extension).
    interval (str)
        Data interval (e.g., 'day', '60minute', '30minute', etc.).
    from_date (date | None)
        Date to start backfilling from.
    to_date (date | None)
        Date to end backfilling to (optional, defaults to today).
    indices_dir (Path)
        Directory where index CSV files are stored.
    out_dir (Path)
        Directory where historical data will be saved.
    """

    handler = KiteDataHandler()
    idx_file = indices_dir / f"{index_name}.csv"

    if not idx_file.exists():
        print(f"Index file not found: {idx_file}")
        raise FileNotFoundError()

    symbols = pd.read_csv(idx_file)["Symbol"].unique().tolist()
    out_path = out_dir / f"{index_name}_{interval}.pkl"
    logger.info(f"Out path: {out_path}")

    if out_path.exists():
        logger.info(f"Loading existing data for {index_name}")
        handler.data = handler.load(out_path)
        last_date = pd.to_datetime(handler.data["date"]).min().date()
        end_date = last_date + timedelta(days=1)
    else:
        end_date = None

    # if last_date is not Today's date --> fetch data from last_date to today
    if not end_date or end_date < datetime.now().date():
        end_date = datetime.now().date()

    if not from_date:
        from_date = end_date

    df_new = await handler.fetch_historical(symbols, interval, from_date, end_date)
    if not df_new.empty:
        handler.save(out_path)
        logger.info(f"Backfill complete for {index_name} from {from_date} to {end_date}")


async def handler():
    """
    Main function to parse arguments and trigger backfill or update.
    1. Backfill historical data for specified index and interval from a given date.
    2. Update data for a single symbol if specified.
    3. Supports 'all' option for index and interval to process multiple indices/intervals.
    4. Saves data and metadata to pickle files in data/historical directory.
    Can be run as a cronjob on a daily basis to keep data updated and model trained.
    Example usage:
        python kite_data.py --index nifty_50 --interval day --date 2023-01-01

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", help="Index name (filename under data/indices)")
    parser.add_argument(
        "--interval", default="day", help="Data interval -- accompanied by lookback days"
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Backfill from this date (YYYY-MM-DD)",
        default=None,
    )
    args = parser.parse_args()

    if args.interval == "all":
        INTERVAL_LIST = ["5minute", "15minute", "60minute", "day"]
    else:
        INTERVAL_LIST = [args.interval]
    print(f"Arguments: {args}")

    if args.index == "all":
        index_list = [idx.value for idx in Indices]
    else:
        index_list = [Indices(args.index).value] if args.index else []

    for index in index_list:
        for interval in INTERVAL_LIST:
            logger.info(f"Processing index: {index} with interval: {interval}")
            if args.date:
                assert f"{index}.csv" in os.listdir(
                    "data/indices"
                ), f"Index file not found for {index}"
                Path(f"data/indices/{index}.csv")
                await backfill_index(index, interval, args.date)


if __name__ == "__main__":
    try:
        asyncio.run(handler())
    except KeyboardInterrupt:
        logger.warning("Process Interrupted by user...")

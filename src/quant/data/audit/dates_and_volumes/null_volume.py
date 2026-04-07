from pathlib import Path

import pandas as pd
from quant.data._common import BRONZE_DIR, BronzeColumns, Interval
from quant.data.kite.kite_handler import KiteDataHandler
from quant.logs.logging import logger


def find_null_volume_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify rows in the DataFrame where the volume is zero, which may indicate missing or erroneous data.
    Arguments
    ---------
    df: pd.DataFrame
        The input DataFrame containing a 'volume' column.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows where volume is zero.
    """
    null_volume_df = df[df[BronzeColumns.VOLUME.value] == 0]
    return null_volume_df


def main(interval: Interval = Interval.DAY) -> None:
    """
    Parse through all bronze data files for the given interval and report zero-volume bars.
    """
    for pth in Path(BRONZE_DIR).glob(f"*_{interval.value}.pkl"):
        logger.info(f"Auditing null volume in {pth.name} ...")
        df = KiteDataHandler.load(pth)
        assert not df.empty, f"DataFrame for {pth.stem} is empty. Please check the data source."
        null_volume_data = find_null_volume_data(df)
        index_name = pth.stem.rsplit(f"_{interval.value}", 1)[0]
        if not null_volume_data.empty:
            logger.warning(f"{index_name}: {len(null_volume_data)} bars with zero volume.")
        volume_min = df.groupby(BronzeColumns.SYMBOL.value)[BronzeColumns.VOLUME.value].min()
        num_stocks_with_zero_volume = (volume_min == 0).sum()
        logger.info(
            f"{index_name}: {num_stocks_with_zero_volume} symbols have at least one zero-volume bar."
        )
        # Optionally, log the specific dates with zero volume for further investigation
        # zero_volume_dates = df[df[BronzeColumns.VOLUME.value] == 0][BronzeColumns.DATE.value]
        # logger.info(f"Symbol: {pth.stem.split('_')[0]} zero volume dates: {zero_volume_dates.tolist()}")


if __name__ == "__main__":
    main()

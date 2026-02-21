import pandas as pd


from pathlib import Path
from quant.logs.logging import logger   
from quant.data._common import BRONZE_DIR, BronzeColumns
from quant.data.kite.kite_handler import KiteDataHandler


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


def main():
    """
    Parse through all bronze data files, identify missing dates for each symbol besides weekends and holidays, and log the results.

    """
    for pth in Path(BRONZE_DIR).glob("*day.pkl"):
        df = KiteDataHandler.load(pth)
        assert not df.empty, f"DataFrame for {pth.stem} is empty. Please check the data source."
        null_volume_data = find_null_volume_data(df)
        if not null_volume_data.empty:
            logger.warning(f"Symbol: {pth.stem.split('_')[0]} has {len(null_volume_data)} rows with zero volume.")
        #volume_max = df[BronzeColumns.VOLUME.value].max()
        volume_min = df.groupby(BronzeColumns.SYMBOL.value)[BronzeColumns.VOLUME.value].min()
        # Number of stocks with at least one zero volume day
        num_stocks_with_zero_volume = (volume_min == 0).sum()
        logger.info(f"Symbol: {pth.stem.split('_')[0]} has {num_stocks_with_zero_volume} stocks with at least one zero volume day.")
        # Optionally, log the specific dates with zero volume for further investigation
        # zero_volume_dates = df[df[BronzeColumns.VOLUME.value] == 0][BronzeColumns.DATE.value]
        # logger.info(f"Symbol: {pth.stem.split('_')[0]} zero volume dates: {zero_volume_dates.tolist()}")
        
if __name__ == "__main__":
    main()
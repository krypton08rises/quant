import pandas as pd

from quant.data._common import BRONZE_DIR 


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
    null_volume_df = df[df['volume'] == 0]
    return null_volume_df


def main():
    """
    Parse through all bronze data files, identify missing dates for each symbol besides weekends and holidays, and log the results.

    """
    for pth in Path(BRONZE_DIR).glob("*day.pkl"):
        df = KiteDataHandler.load(pth)
        missing_dates = find_missing_dates(df, date_col=BronzeColumns.DATE)
    
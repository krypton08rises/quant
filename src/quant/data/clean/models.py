import dill 
import pandas as pd 

from pydantic import BaseModel
from pathlib import Path
from enum import Enum




class ReasonEnum(str, Enum):
    MISSING_DATES = "missing_dates"
    INCONSISTENT_DATA = "inconsistent_data"
    OTHER = "other"


class UncleanSymbol(BaseModel):
    symbol: str
    reason: ReasonEnum


class MissingDates(BaseModel):
    dates: pd.DatetimeIndex

    @classmethod
    def from_dates(cls, dates: pd.DatetimeIndex) -> "MissingDates":
        return cls(dates=dates)
    
    # save and load methods for MissingDates
    def save(self, path: Path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "MissingDates":
        with open(path, 'rb') as f:
            return dill.load(f)

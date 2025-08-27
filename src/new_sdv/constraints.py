import pandas as pd

class Constraint:
    def ensure(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

class UniqueConstraint(Constraint):
    def __init__(self, column):
        self.column = column

    def ensure(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=[self.column]).reset_index(drop=True)

class RangeConstraint(Constraint):
    def __init__(self, column, min_val=None, max_val=None):
        self.column = column
        self.min_val = min_val
        self.max_val = max_val

    def ensure(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.min_val is not None:
            df.loc[df[self.column] < self.min_val, self.column] = self.min_val
        if self.max_val is not None:
            df.loc[df[self.column] > self.max_val, self.column] = self.max_val
        return df

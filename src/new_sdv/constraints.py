# src/new_sdv/constraints.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

class Constraint:
    """Base no-op constraint with two hooks."""
    def apply_on_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def apply_on_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


@dataclass
class RangeConstraint(Constraint):
    column: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def apply_on_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[self.column]
        if self.min_val is not None:
            s = s.clip(lower=self.min_val)
        if self.max_val is not None:
            s = s.clip(upper=self.max_val)
        df[self.column] = s
        return df


@dataclass
class UniqueConstraint(Constraint):
    column: str
    strategy: str = "sequential"  # or "uuid"

    def apply_on_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace with guaranteed-unique values post-sampling.
        # This avoids awkward re-sampling loops and preserves app stability.
        if self.strategy == "uuid":
            import uuid
            df[self.column] = [str(uuid.uuid4()) for _ in range(len(df))]
        else:
            if pd.api.types.is_integer_dtype(df[self.column].dtype):
                start = 1
                df[self.column] = np.arange(start, start + len(df), dtype=df[self.column].dtype)
            else:
                df[self.column] = [f"id_{i}" for i in range(1, len(df) + 1)]
        return df

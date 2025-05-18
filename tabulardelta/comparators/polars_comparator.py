# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from tabulardelta.comparators.tabulardelta_dataclasses import (
    ColumnPair,
    TabularDelta,
)

@dataclass(frozen=True)
class PolarsComparator:
    """Implements :class:`Comparator` protocol for comparing polars DataFrames.

    Methods
    -------
    compare(old: pl.DataFrame, new: pl.DataFrame) -> TabularDelta:
        Compare two polars DataFrames
    """

    join_columns: list[str]
    """Columns to join on"""

    name: str = ""
    """Name of the comparison/tables."""
    float_rtol: float = 1.0e-5
    """Relative tolerance for comparing floats."""
    float_atol: float = 0.0
    """Absolute tolerance for comparing floats."""
    check_row_order: bool = True
    """Check if row order changed.

    Slight performance hit.
    """

    def compare(self, old: pl.DataFrame, new: pl.DataFrame) -> TabularDelta:
        """Compare two polars DataFrames.

        Arguments:
            old :pl.DataFrame:
                The old table (first table to compare).
            new :pl.DataFrame:
                The new table (second table to compare).

        Returns :class:`TabularDelta`:
            Metadata and results of the comparison.
        """
        return compare_polars(
            old,
            new,
            old.dtypes.astype("string").to_dict(),
            new.dtypes.astype("string").to_dict(),
            self.join_columns,
            self.name,
            self.float_rtol,
            self.float_atol,
            self.check_row_order,
        )

def compare_polars(
    old: pl.DataFrame,
    new: pl.DataFrame,
    old_dtypes: dict[str, str],
    new_dtypes: dict[str, str],
    join_columns: list[str] | None = None,
    name: str = "",
    float_rtol: float = 1.0e-5,
    float_atol: float = 0,
    check_row_order: bool = True,
) -> TabularDelta:
    """Compare pandas Dataframes.

    If data was de-serialized imperfectly, the original dtypes can be specified.

    Arguments:
        old :class:`pd.DataFrame`:
            The old table (first table to compare).
        new :class:`pd.DataFrame`:
            The new table (second table to compare).
        old_dtypes :class:`dict[str, str]`:
            The dtypes of the old table (name -> type).
        new_dtypes :class:`dict[str, str]`:
            The dtypes of the new table (name -> type).
        join_columns :class:`list[str]` | :code:`None`:
            Columns to join on, uses index if unspecified.
        name :class:`str`:
            Name of the comparison/tables.
        float_rtol :class:`float`:
            Relative tolerance for comparing floats.
        float_atol :class:`float`:
            Absolute tolerance for comparing floats.
        check_row_order :class:`bool`:
            Check if row order changed. Slight performance hit.

    Returns :class:`TabularDelta`:
        Metadata and results of the comparison.
    """
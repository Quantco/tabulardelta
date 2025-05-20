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
            {col: str(dtype) for col, dtype in zip(old.columns, old.dtypes)},
            {col: str(dtype) for col, dtype in zip(new.columns, new.dtypes)},
            self.join_columns,
            self.name,
            self.float_rtol,
            self.float_atol,
            self.check_row_order,
        )


def _join(
    old: pl.DataFrame, new: pl.DataFrame, join_cols: list[str], suffixes
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Join two DataFrames on join_cols, failing for non-unique join columns.

    Use outer join to get added and removed rows. Use inner join to not change data
    types, since outer join adds Nones.
    """
    if old.select(join_cols).dtypes != old.select(join_cols).dtypes:
        raise Exception("Datatypes of join columns changed. Cannot join dataframes for comparison."
                        f" Old dtypes {old.select(join_cols).dtypes}, "
                        f"New dtypes {old.select(join_cols).dtypes}")
    cols_old_rename = {col: f"{col}{suffixes[0]}" for col in old.columns}
    cols_new_rename = {col: f"{col}{suffixes[1]}" for col in new.columns}
    old = old.rename(cols_old_rename)
    new = new.rename(cols_new_rename)
    outer = old.join(new, left_on=[f"{col}_old" for col in join_cols], right_on=[f"{col}_new" for col in join_cols], how="outer", suffix=suffixes[1])

    row_in_old = pl.lit(True)
    for col in join_cols:
        row_in_old &= pl.col(f"{col}{suffixes[0]}").is_not_null()
    row_in_new = pl.lit(True)
    for col in join_cols:
        row_in_new &= pl.col(f"{col}{suffixes[1]}").is_not_null()

    # Added rows are in new but not in old
    added_rows = outer.filter(~row_in_old & row_in_new)
    # Removed rows are in old but not in new
    removed_rows = outer.filter(row_in_old & ~row_in_new)
    # Joined rows are in both
    joined = outer.filter(row_in_old & row_in_new)
    return added_rows, removed_rows, joined

def compare_polars(
    old: pl.DataFrame,
    new: pl.DataFrame,
    old_dtypes: dict[str, str],
    new_dtypes: dict[str, str],
    join_columns: list[str],
    name: str = "",
    float_rtol: float = 1.0e-5,
    float_atol: float = 0,
    check_row_order: bool = True,
) -> TabularDelta:
    """Compare polars Dataframes.

    Arguments:
        old :class:`pl.DataFrame`:
            The old table (first table to compare).
        new :class:`pd.DataFrame`:
            The new table (second table to compare).
        old_dtypes :class:`dict[str, str]`:
            The dtypes of the old table (name -> type).
        new_dtypes :class:`dict[str, str]`:
            The dtypes of the new table (name -> type).
        join_columns :class:`list[str]`:
            Columns to join on
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

    warnings = []
    info: list[str] = []
    if check_row_order:
        old = old.with_columns(_old_row_number=range(old.shape[0]))
        new = new.with_columns(_new_row_number=range(new.shape[0]))

    # 1. Join dataframes on join_columns
    suffixes = ("_old", "_new")
    try:
        added_rows, removed_rows, joined = _join(old, new, join_columns, suffixes)
    except KeyError as e:  # Probably non-unique join columns
        warnings.append(
            f"Error when joining on columns {join_columns}: {e}\n"
            f"Using all common columns as join columns instead.\n"
            f"MODIFIED ROWS will be LISTED AS ADDED AND REMOVED"
        )
        common = [col for col in old.columns if col in new.columns]
        join_columns = [col for col in common if old_dtypes[col] == new_dtypes[col]]
        added_rows, removed_rows, joined = _join(old, new, join_columns, suffixes)
    if check_row_order:
        joined = joined.sort(f"_old_row_number{suffixes[0]}")
        if not _is_increasing(joined[f"_new_row_number{suffixes[1]}"]):
            info.append("Row Order Changed!")
        joined = joined.drop(f"_old_row_number{suffixes[0]}", f"_new_row_number{suffixes[1]}")
        added_rows = added_rows.drop(f"_old_row_number{suffixes[0]}", f"_new_row_number{suffixes[1]}")
        removed_rows = removed_rows.drop(f"_old_row_number{suffixes[0]}", f"_new_row_number{suffixes[1]}")

    # 2. Match columns (rename equal columns, drop unmatched)
    only_old = set(old.columns) & set(joined.columns) - set(join_columns)
    only_new = set(new.columns) & set(joined.columns) - set(join_columns)
    tmp = {n: o for o in only_old for n in only_new if joined[o].equals(joined[n])}
    renamed = {v: k for k, v in tmp.items()}  # Make both sides unique by reversing
    if joined.shape[0] == 0:
        renamed = {}  # Without rows, every column is equal
    mapping = {v + suffixes[1]: v for v in renamed.values()}
    mapping |= {v + suffixes[0]: k for k, v in renamed.items()}
    unmatched_cols = (only_old | only_new) - set(renamed.keys()) - set(renamed.values())
    joined.rename(columns={v: k for k, v in mapping.items()}, inplace=True)
    joined.drop(columns=list(unmatched_cols), inplace=True)

def _is_increasing(x: pl.Expr, strict: bool = False) -> pl.Expr:
    """
    Checks whether the column is monotonically increasing.
    strict: Whether the check should be strict
    """
    if strict:
        return (x.diff() > 0.0).all()
    else:
        return (x.diff() >= 0.0).all()
# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from __future__ import annotations

from dataclasses import dataclass
from logging import warn
from typing import Any

import numpy as np
import polars as pl
from polars.polars import SchemaError

from tabulardelta.comparators.tabulardelta_dataclasses import (
    ColumnPair,
    TabularDelta,
)

LOSSLESS_CONV = {
    "Int8": {"Int16", "Int32", "Int64", "String"},
    "Int16": {"Int32", "Int64", "String"},
    "Int32": {"Int64", "String"},
    "Int64": {"object"},
    "UInt8": {"UInt16", "UInt32", "UInt64", "String"},
    "UInt16": {"UInt32", "UInt64", "String"},
    "UInt32": {"UInt64", "String"},
    "UInt64": {"String"},
    "Float32": {"Float64", "String"},
    "Float64": {"String"},
}

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
    try:
        outer = old.join(new, on=join_cols, how="outer", suffix=suffixes[1])
    except SchemaError:
        raise ValueError("Datatype of join columns changed. Cannot join dataframes.")

    in_old_and_new = set(old.columns) & set(new.columns)
    outer = outer.rename({col: f"{col}{suffixes[0]}" for col in in_old_and_new})
    if outer.select(*[f"{col}{suf}" for col in join_cols for suf in suffixes]).is_duplicated().any():
        raise KeyError(f"Join columns {join_cols} are not unique.")

    row_in_old = pl.lit(True)
    for col in join_cols:
        row_in_old &= pl.col(f"{col}{suffixes[0]}").is_not_null()
    row_in_new = pl.lit(True)
    for col in join_cols:
        row_in_new &= pl.col(f"{col}{suffixes[1]}").is_not_null()

    # Added rows are in new but not in old
    added_rows = outer.filter(~row_in_old & row_in_new).select(*[f"{col}{suffixes[1]}" for col in join_cols]).rename({f"{col}{suffixes[1]}": col for col in join_cols})
    added_rows = new.join(added_rows, on=join_cols, how="inner")

    # Removed rows are in old but not in new
    removed_rows = outer.filter(row_in_old & ~row_in_new).select(*[f"{col}{suffixes[0]}" for col in join_cols]).rename({f"{col}{suffixes[0]}": col for col in join_cols})
    removed_rows = old.join(removed_rows, on=join_cols, how="inner")

    # Joined rows are in both
    joined = outer.filter(row_in_old & row_in_new).drop([f"{col}{suffixes[1]}" for col in join_cols]).rename({f"{col}{suffixes[0]}": col for col in join_cols})
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
        joined = joined.sort("_old_row_number")
        if not _is_increasing(joined["_new_row_number"]):
            info.append("Row Order Changed!")
        joined = joined.drop("_old_row_number", "_new_row_number")
        added_rows = added_rows.drop("_new_row_number")
        removed_rows = removed_rows.drop("_old_row_number")

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
    joined = joined.rename({v: k for k, v in mapping.items()})
    joined.drop(list(unmatched_cols))

    # 2.5 Check column order
    old_cols_renamed = [renamed.get(c, c) for c in old.columns]
    new_col_positions = [np.flatnonzero(new.columns == c) for c in old_cols_renamed]
    if any(len(pos) > 1 for pos in new_col_positions):
        errors = [f"No injective column mapping (renaming: {renamed})"]
        return TabularDelta.from_errors(errors)
    filtered_positions = [pos[0] for pos in new_col_positions if len(pos) == 1]
    if not all(i < j for i, j in zip(filtered_positions, filtered_positions[1:])):
        info.append("Column Order Changed!")

    # Quick access to important datastructures:
    cols = set(joined.columns)  # Get interesting (non-joined) columns without suffix
    cols = {col[: -len(suffixes[0])] for col in cols if col.endswith(suffixes[0])}
    old_names = {col: mapping.get(col + suffixes[0], col) for col in cols}
    new_names = {col: mapping.get(col + suffixes[1], col) for col in cols}
    old_dt = {col: old_dtypes[old_names[col]] for col in cols}  # Using new names
    new_dt = {col: new_dtypes[new_names[col]] for col in cols}  # Using new names

    # 3. Match dtypes (cast if possible, otherwise mark incomparable)
    dtype_changes: list[ColumnPair] = []
    changed = {col for col in cols if old_dt[col] != new_dt[col]}
    cast_old = {c for c in changed if old_dt[c] in LOSSLESS_CONV.get(new_dt[c], set())}
    cast_new = {c for c in changed if new_dt[c] in LOSSLESS_CONV.get(old_dt[c], set())}
    unsupported = changed - cast_old - cast_new
    for col in cast_old | cast_new:
        joined = _cast(
            joined, col + suffixes[0], new_dt[col] if col in cast_new else old_dt[col]
        )
        joined = _cast(
            joined, col + suffixes[1], new_dt[col] if col in cast_new else old_dt[col]
        )
    for col in unsupported:
        change = _value_change(
            joined, join_columns, col, suffixes, old_dt, new_dt, True
        )
        dtype_changes.append(change)
        joined = joined.drop([col + suffixes[0], col + suffixes[1]])
    cols -= unsupported

    # 4. Compare values
    dtypes_compare_values = {col : dtype for col, dtype in zip(joined.columns, joined.dtypes)}
    column_changes = []
    for col in cols:
        left, right = joined.get_column(col + suffixes[0]), joined.get_column(col + suffixes[1])
        if dtypes_compare_values[col + suffixes[0]] in [pl.Float32, pl.Float64]:
            joined = joined.with_columns(pl.Series(name=col + "_equal", values=np.isclose(
                left, right, float_rtol, float_atol, True
            )))
        else:
            joined = joined.with_columns(pl.Series(name=col + "_equal", values=(left==right).fill_null(False) | (left.is_null() & right.is_null())))
        unequal = joined.filter(~pl.col(col + "_equal"))
        change = _value_change(unequal, join_columns, col, suffixes, old_dt, new_dt)
        if len(change) > 0:
            column_changes.append(change)
    joined = joined.with_columns(_equal=pl.all_horizontal(*[col + "_equal" for col in cols]))
    equal_rows = joined.get_column("_equal").value_counts().filter(pl.col("_equal")).get_column("count")
    if len(equal_rows) == 0:
        equal_rows = 0
    else:
        equal_rows = equal_rows[0]

    # 5. Return DataFrameDiff
    ren_cols = [
        ColumnPair.from_str(old, old_dtypes[old], new, new_dtypes[new])
        for old, new in renamed.items()
    ]
    added = [
        (None, None, col, new_dtypes[col]) for col in only_new - set(renamed.values())
    ]
    removed = [(col, old_dtypes[col]) for col in only_old - set(renamed.keys())]
    joins = [(c, old_dtypes[c], c, new_dtypes[c], True) for c in join_columns]
    uncompared = [ColumnPair.from_str(*pair) for pair in added + removed + joins]
    return TabularDelta(
        name,
        info=info,
        warnings=warnings,
        _columns=column_changes + dtype_changes + ren_cols + uncompared,
        _old_rows=old.shape[0],
        _new_rows=new.shape[0],
        _added_rows=new.shape[0] - joined.shape[0],
        _removed_rows=old.shape[0] - joined.shape[0],
        _equal_rows=equal_rows,
        _unequal_rows=joined.shape[0] - equal_rows,
        _example_added_rows=added_rows.to_dicts(),
        _example_removed_rows=removed_rows.to_dicts(),
    )


def _is_increasing(x: pl.Expr, strict: bool = False) -> pl.Expr:
    """
    Checks whether the column is monotonically increasing.
    strict: Whether the check should be strict
    """
    if strict:
        return (x.diff() > 0.0).all()
    else:
        return (x.diff() >= 0.0).all()

def _cast(df: pl.DataFrame, col: str, dtype: str) -> pl.DataFrame:
    """Cast object into comparable dtype.
    """
    return df.cast({col: getattr(pl, dtype)})

def _value_change(
    df: pl.DataFrame, join_cols: list[str], col: str, suffixes: list[str], old_dt: dict[str, str], new_dt: dict[str, str], incomparable: bool = False
) -> ColumnPair:
    """Create ColumnChange object for one column in a given DataFrame."""
    diff = df.select(join_cols + [col + suffixes[0], col + suffixes[1]]).rename({col + suffixes[1]: col}).with_columns(_count=pl.lit(1))
    combined = col, old_dt[col], col, new_dt[col], False
    return ColumnPair.from_str(*combined, incomparable, diff)

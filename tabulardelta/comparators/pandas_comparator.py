# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tabulardelta.comparators.tabulardelta_dataclasses import (
    ColumnPair,
    TabularDelta,
)

LOSSLESS_CONV = {
    "int8": {"int16", "int32", "int64", "object"},
    "int16": {"int32", "int64", "object"},
    "int32": {"int64", "object"},
    "int64": {"object"},
    "uint8": {"uint16", "uint32", "uint64", "object"},
    "uint16": {"uint32", "uint64", "object"},
    "uint32": {"uint64", "object"},
    "uint64": {"object"},
    "float32": {"float64", "object"},
    "float64": {"object"},
    # String dtypes are currently (2024-05-10) experimental and hard to compare.
}


@dataclass(frozen=True)
class PandasComparator:
    """Implements :class:`Comparator` protocol for comparing pandas DataFrames.

    Methods
    -------
    compare(old: pd.DataFrame, new: pd.DataFrame) -> TabularDelta:
        Compare two pandas DataFrames
    """

    join_columns: list[str] | None = None
    """Columns to join on, uses index if unspecified."""

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

    def compare(self, old: pd.DataFrame, new: pd.DataFrame) -> TabularDelta:
        """Compare two pandas DataFrames.

        Arguments:
            old :pd.DataFrame:
                The old table (first table to compare).
            new :pd.DataFrame:
                The new table (second table to compare).

        Returns :class:`TabularDelta`:
            Metadata and results of the comparison.
        """
        return compare_pandas(
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


def _specific_str(obj: Any) -> str | float:
    """Get deterministic string representation for arbitrary objects."""
    if isinstance(obj, bool | str | int | float | type(None)):
        return repr(obj)
    if type(obj).__repr__ != object.__repr__:
        return repr(obj) + f" (<{type(obj).__name__}>)"
    if type(obj).__str__ != object.__str__:
        return str(obj) + f" (<{type(obj).__name__}>)"
    raise ValueError(f"Cannot convert {obj} to deterministic string representation")


def _cast(df: pd.DataFrame, col: str, dtype: str) -> None:
    """Cast object into comparable dtype.

    Use string representation for objects.
    """
    df[col] = df[col].map(_specific_str) if dtype == "object" else df[col].astype(dtype)  # type: ignore


def _remove_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Remove suffix from all column names in a DataFrame."""
    mapping = {col: col[: -len(suffix)] for col in df.columns if col.endswith(suffix)}
    return df.rename(columns=mapping)


def _join(
    old: pd.DataFrame, new: pd.DataFrame, join_cols: list[str], suffixes
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join two DataFrames on join_cols, failing for non-unique join columns.

    Use outer join to get added and removed rows. Use inner join to not change data
    types, since outer join adds Nones.
    """
    outer = pd.merge(
        old,
        new,
        how="outer",
        on=join_cols or None,
        left_index=not join_cols,
        right_index=not join_cols,
        suffixes=suffixes,
        indicator=True,
    )
    removed = _remove_suffix(outer[outer["_merge"] == "left_only"], suffixes[0])
    added = _remove_suffix(outer[outer["_merge"] == "right_only"], suffixes[1])
    joined = pd.merge(
        old,
        new,
        how="inner",
        on=join_cols or None,
        left_index=not join_cols,
        right_index=not join_cols,
        suffixes=suffixes,
        indicator=True,
    )
    if join_cols:
        duplicates = joined.duplicated(subset=join_cols, keep=False)
    else:
        duplicates = pd.Series(joined.index.duplicated(keep=False))
    if duplicates.any():
        raise KeyError(f"Join columns {join_cols} are not unique.")
    return added[new.columns], removed[old.columns], joined


def _value_change(
    df, join_cols, col, suffixes, old_dt, new_dt, incomparable: bool = False
) -> ColumnPair:
    """Create ColumnChange object for one column in a given DataFrame."""
    diff = df[join_cols + [col + suffixes[0], col + suffixes[1]]].copy()
    if not join_cols:
        diff.reset_index(inplace=True)
    diff.rename(columns={col + suffixes[1]: col}, inplace=True)
    diff["_count"] = 1
    combined = col, old_dt[col], col, new_dt[col], False
    return ColumnPair.from_str(*combined, incomparable, diff)


def compare_pandas(
    old: pd.DataFrame,
    new: pd.DataFrame,
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
    if not join_columns:
        join_columns = []
    old = old.copy()
    new = new.copy()
    warnings = []
    info: list[str] = []
    if check_row_order:
        old["_old_row_number"] = range(len(old))
        new["_new_row_number"] = range(len(new))

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
    if not join_columns:
        added_rows.reset_index(inplace=True)
        removed_rows.reset_index(inplace=True)
    if check_row_order:
        joined = joined.sort_values("_old_row_number")
        if not joined["_new_row_number"].is_monotonic_increasing:
            info.append("Row Order Changed!")
        joined.drop(columns=["_old_row_number", "_new_row_number"], inplace=True)
        added_rows.drop(columns=["_new_row_number"], inplace=True)
        removed_rows.drop(columns=["_old_row_number"], inplace=True)

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
        _cast(
            joined, col + suffixes[0], new_dt[col] if col in cast_new else old_dt[col]
        )
        _cast(
            joined, col + suffixes[1], new_dt[col] if col in cast_new else old_dt[col]
        )
    for col in unsupported:
        change = _value_change(
            joined, join_columns, col, suffixes, old_dt, new_dt, True
        )
        dtype_changes.append(change)
        joined.drop(columns=[col + suffixes[0], col + suffixes[1]], inplace=True)
    cols -= unsupported

    # 4. Compare values
    column_changes = []
    for col in cols:
        left, right = joined[col + suffixes[0]], joined[col + suffixes[1]]
        if joined.dtypes[col + suffixes[0]] in ["float32", "float64"]:
            joined[col + "_equal"] = np.isclose(
                left, right, float_rtol, float_atol, True
            )
        else:
            joined[col + "_equal"] = (left == right).fillna(False) | pd.isna(
                left
            ) & pd.isna(right)
        unequal = joined[~joined[col + "_equal"].astype("bool")]
        change = _value_change(unequal, join_columns, col, suffixes, old_dt, new_dt)
        if len(change) > 0:
            column_changes.append(change)
    joined["_equal"] = joined[[col + "_equal" for col in cols]].agg("all", axis=1)
    equal_rows = joined[["_equal"]].value_counts().get(True, 0)

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
        _example_added_rows=added_rows.to_dict(orient="records"),  # type: ignore
        _example_removed_rows=removed_rows.to_dict(orient="records"),  # type: ignore
    )

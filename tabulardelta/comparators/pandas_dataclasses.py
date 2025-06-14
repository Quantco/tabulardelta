# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import pandas as pd

try:
    import sqlalchemy as sa
except ImportError as e:
    sa = f"SQLAlchemy not installed: {e}"  # type: ignore


@dataclass(frozen=True)
class Column:
    """Implements the :class:`Column` protocol for TabularDelta.

    Methods
    -------

    from_sqlalchemy(column: sa.sql.ColumnElement[Any]) -> Column:
        Turns SQLAlchemy.Column into TabularDelta.Column
    """

    name: str
    """Name of the column."""
    type: str
    """Data type of the column."""

    @staticmethod
    def from_sqlalchemy(column: sa.sql.ColumnElement[Any]) -> Column:
        """Turns SQLAlchemy.Column into TabularDelta.Column.

        Arguments:
            column :class:`sa.sql.ColumnElement[Any]`:
                SQLAlchemy column to be converted.

        Results :class:`Column`:
            TabularDelta column with according name and type.
        """
        return Column(column.name, str(column.type))


@dataclass(frozen=True)
class ChangedValue:
    """Implements the :class:`ChangedValue` protocol for TabularDelta."""

    example_join_columns: dict[str, Any]
    """Mapping from column names to values represents an example row where this change
    occurred."""
    old: Any
    """Previous value before the change."""
    new: Any
    """New value after the change."""
    count: int
    """(Positive) number of rows containing this change."""


@dataclass
class ColumnPair:
    """Implements the :class:`MatchedColumn` and :class:`ChangedColumn` protocol for
    TabularDelta.

    Methods
    -------
    from_sqlalchemy(...) -> ColumnPair:
        Creates TabularDelta.ColumnPair using SQLAlchemy columns
    from_str(...) -> ColumnPair:
        Creates TabularDelta.ColumnPair using column names and types
    __len__() -> int
        Total number of changes
    __iter__() -> Iterator[ChangedValue]
        Examples of value changes
    """

    _old: Column | None = None
    """Metadata of column in the old table or None if column was added."""
    _new: Column | None = None
    """Metadata of column in the new table or None if column was removed."""
    join: bool = False
    """Whether column was used as join column for comparison."""
    incomparable: bool = False
    """Whether data types of column in old and new table are incomparable."""

    _values: pd.DataFrame | None = None

    @property
    def old(self) -> Column:
        """The column in the old table.

        Fails if column was added.
        """
        if self._old is None:
            raise ValueError("Old column does not exist.")
        return self._old

    @property
    def new(self) -> Column:
        """The column in the new table.

        Fails if column was removed.
        """
        if self._new is None:
            raise ValueError("New column does not exist.")
        return self._new

    def __iter__(self) -> Iterator[ChangedValue]:
        """Examples of value changes."""
        for idx, row in self._values.iterrows() if self._values is not None else []:
            index = {k: v for k, v in row.to_dict().items() if k not in self._required}
            yield ChangedValue(index, *[row[key] for key in self._required])

    def __len__(self) -> int:
        """Total number of changes."""
        return self._values["_count"].sum() if self._values is not None else 0

    @staticmethod
    def from_sqlalchemy(
        old: sa.sql.ColumnElement[Any] | None = None,
        new: sa.sql.ColumnElement[Any] | None = None,
        join: bool = False,
        incomparable: bool = False,
        _values: pd.DataFrame | None = None,
    ) -> ColumnPair:
        """Creates TabularDelta.ColumnPair using SQLAlchemy columns.

        Arguments:
            old :class:`sa.sql.ColumnElement[Any]` | :code:`None`:
                Column in the old table or None if column was added.
            new :class:`sa.sql.ColumnElement[Any]` | :code:`None`:
                Column in the new table or None if column was removed.
            join :class:`bool`:
                Whether column was used as join column for comparison.
            incomparable :class:`bool`:
                Whether data types of column in old and new table are incomparable.
            _values :class:`pd.DataFrame`:
                DataFrame with examples of value changes.
                Contains old column, new column, :code:`_count` column,
                and optionally other columns for identifying example rows.

        Returns :class:`ColumnPair`:
            TabularDelta.ColumnPair with according metadata and values.
        """
        return ColumnPair(
            Column.from_sqlalchemy(old) if old is not None else None,
            Column.from_sqlalchemy(new) if new is not None else None,
            join,
            incomparable,
            _values,
        )

    @staticmethod
    def from_str(
        old: str | None = None,
        old_type: str | None = None,
        new: str | None = None,
        new_type: str | None = None,
        join: bool = False,
        incomparable: bool = False,
        _values: pd.DataFrame | None = None,
    ) -> ColumnPair:
        """Creates TabularDelta.ColumnPair using column names and types.

        Arguments:
            old :class:`str` | :code:`None`:
                Column name in the old table or None if column was added.
            old_type :class:`str` | :code:`None`:
                Column type in the old table or None if column was added.
            new :class:`str` | :code:`None`:
                Column name in the new table or None if column was removed.
            new_type :class:`str` | :code:`None`:
                Column type in the new table or None if column was removed.
            join :class:`bool`:
                Whether column was used as join column for comparison.
            incomparable :class:`bool`:
                Whether data types of column in old and new table are incomparable.
            _values :class:`pd.DataFrame`:
                DataFrame with examples of value changes.
                Contains old column, new column, :code:`_count` column,
                and optionally other columns for identifying example rows.
                If the column wasn't renamed, the name of the old column in the DataFrame should
                be suffixed by :code:`_old`.

        Returns :class:`ColumnPair`:
            TabularDelta.ColumnPair with according metadata and values.
        """
        return ColumnPair(
            Column(old, old_type) if old and old_type else None,
            Column(new, new_type) if new and new_type else None,
            join,
            incomparable,
            _values,
        )

    @cached_property
    def _df_old_name(self) -> str:
        """Name of the old column in the DataFrame.

        Suffixed with :code:`_old` if column wasn't renamed.
        """
        return self.old.name + ("_old" if self.old.name == self.new.name else "")

    @cached_property
    def _required(self) -> tuple[str, str, str]:
        """Required columns in the DataFrame."""
        return self._df_old_name, self.new.name, "_count"

    def __post_init__(self):
        """Check if DataFrame contains required columns and compactifies it."""
        if self._values is None:
            return

        if missing := set(self._required) - set(self._values.columns):
            raise ValueError(f"Missing columns in comparison: {missing}")

        group_by_cols = [self._df_old_name, self.new.name]

        # DIRTY FIX FOR PANDAS BUG: Categorical in GroupBy leads to
        # ValueError: Length of values (5) does not match length of index (25)
        for col in group_by_cols:
            if self._values[col].dtype.name == "category":
                self._values[col] = self._values[col].astype("object")

        # Compactify dataframe by grouping equal changes together
        groupby = self._values.groupby(group_by_cols, as_index=False, dropna=False)
        join_cols = {c: (c, "first") for c in set(self._values) - set(self._required)}
        agg = groupby.agg(_count=("_count", "sum"), **join_cols)  # type: ignore
        sort = agg.sort_values(by="_count", ascending=False)
        actual_changes = ~sort[self._df_old_name].isna() | ~sort[self.new.name].isna()
        self._values = sort[actual_changes]


@dataclass(frozen=True)
class ColumnDelta:
    """Implements the :class:`ColumnDelta` protocol."""

    cols: list[ColumnPair]
    """All (potentially matched) columns of both tables.

    Unmatched columns have None on other side.
    """

    @cached_property
    def old(self) -> list[Column]:
        """Metadata of columns in the old table."""
        return [col.old for col in self.cols if col._old]

    @cached_property
    def new(self) -> list[Column]:
        """Metadata of columns in the new table."""
        return [col.new for col in self.cols if col._new]

    @cached_property
    def added(self) -> list[Column]:
        """Metadata of added columns (only in new table)."""
        return [col.new for col in self.cols if col._new and not col._old]

    @cached_property
    def removed(self) -> list[Column]:
        """Metadata of removed columns (only in old table)."""
        return [col.old for col in self.cols if col._old and not col._new]

    @cached_property
    def joined(self) -> list[ColumnPair]:
        """Columns used to join the old and new table."""
        return [col for col in self.cols if col.join]

    @cached_property
    def matched(self) -> list[ColumnPair]:
        """Matched columns (exist in both tables)."""
        return [col for col in self.cols if col._old and col._new]

    @cached_property
    def renamed(self) -> list[ColumnPair]:
        """Renamed columns (matched using values and types)."""
        return [col for col in self.matched if col.old.name != col.new.name]

    @cached_property
    def _type_changed(self) -> list[ColumnPair]:
        """Columns with changed data types."""
        return [col for col in self.matched if col.old.type != col.new.type]

    @cached_property
    def comparable_type_changed(self) -> list[ColumnPair]:
        """Columns with changed but comparable data types."""
        return [col for col in self._type_changed if not col.incomparable]

    @cached_property
    def incomparable_type_changed(self) -> list[ColumnPair]:
        """Columns with changed and incomparable data types."""
        return [col for col in self._type_changed if col.incomparable]

    @cached_property
    def differences(self) -> list[ColumnPair]:
        """Comparable column with changed values."""
        return [col for col in self.cols if not col.incomparable and len(col) > 0]


@dataclass
class Rows:
    """Implements the :class:`Rows` protocol."""

    count: int
    """Number of represented rows."""
    examples: list[dict[str, Any]] | None
    """Optional examples of represented rows."""

    def __len__(self) -> int:
        """Number of represented rows."""
        return self.count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Examples of represented rows."""
        return iter(self.examples or [])


@dataclass(frozen=True)
class RowDelta:
    """Implements the :class:`RowDelta` protocol."""

    old: Rows
    """Rows in the old table."""
    new: Rows
    """Rows in the new table."""
    added: Rows
    """Added rows (only in new table)."""
    removed: Rows
    """Removed rows (only in old table)."""
    equal: Rows
    """Equal rows (joined and all columns unchanged)."""
    unequal: Rows
    """Unequal rows (joined but at least one column changed)."""


@dataclass(frozen=True)
class TabularDelta:
    """Implements the :class:`TabularDelta` protocol.

    Methods
    -------

    from_errors(errors: list[str], name: str = "") -> TabularDelta:
        Default TabularDelta containing errors
    """

    name: str = ""
    """Name of the new table and therefore the comparison."""
    old_name: str | None = None
    """Optional name of the old table."""
    info: list[str] = field(default_factory=list)
    """Additional information collected during the comparison."""
    warnings: list[str] = field(default_factory=list)
    """Warnings collected during the comparison."""
    errors: list[str] = field(default_factory=list)
    """Errors collected during the comparison."""

    _columns: list[ColumnPair] = field(default_factory=list)
    """All (potentially matched) columns of both tables.

    Unmatched columns have None on other side.
    """

    _old_rows: int = 0
    """Number of rows in the old table."""
    _new_rows: int = 0
    """Number of rows in the new table."""
    _added_rows: int = 0
    """Number of added rows (only in new table)."""
    _removed_rows: int = 0
    """Number of removed rows (only in old table)."""
    _equal_rows: int = 0
    """Number of equal rows (joined and all columns unchanged)."""
    _unequal_rows: int = 0
    """Number of unequal rows (joined but at least one column changed)."""
    _example_added_rows: list[dict[str, Any]] = field(default_factory=list)
    """Optional examples of added rows (only in new table)."""
    _example_removed_rows: list[dict[str, Any]] = field(default_factory=list)
    """Optional examples of removed rows (only in old table)."""
    _example_equal_rows: list[dict[str, Any]] = field(default_factory=list)
    """Optional examples of equal rows (joined and all columns unchanged)."""
    _example_unequal_rows: list[dict[str, Any]] = field(default_factory=list)
    """Optional examples of unequal rows (joined but at least one column changed)."""

    @cached_property
    def cols(self) -> ColumnDelta:
        """Returns :class:`ColumnDelta` implementation."""
        return ColumnDelta(self._columns)

    @cached_property
    def rows(self) -> RowDelta:
        """Returns :class:`RowDelta` implementation."""
        return RowDelta(
            Rows(self._old_rows, None),
            Rows(self._new_rows, None),
            Rows(self._added_rows, self._example_added_rows),
            Rows(self._removed_rows, self._example_removed_rows),
            Rows(self._equal_rows, self._example_equal_rows),
            Rows(self._unequal_rows, self._example_unequal_rows),
        )

    @staticmethod
    def from_errors(errors: list[str], name: str = "") -> TabularDelta:
        """Creates default TabularDelta object with errors.

        Arguments:
            errors :class:`list[str]`:
                List of error messages.
            name :class:`str`:
                Name of the table.

        Returns :class:`TabularDelta`:
            TabularDelta object containing errors.
        """
        return TabularDelta(name=name, errors=errors)

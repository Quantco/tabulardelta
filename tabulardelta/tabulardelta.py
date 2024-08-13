# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol


class ChangedValue(Protocol):
    """Represents a change of a value."""

    @property
    def old(self) -> Any:
        """Previous value before the change."""
        ...

    @property
    def new(self) -> Any:
        """New value after the change."""
        ...

    @property
    def count(self) -> int:
        """(Positive) number of rows containing this change."""
        ...

    @property
    def example_join_columns(self) -> Mapping[str, Any]:
        """Mapping from column names to values represents an example row where this
        change occurred."""
        ...


class Column(Protocol):
    """Represents a column in a table."""

    @property
    def name(self) -> str:
        """Name of the column."""
        ...

    @property
    def type(self) -> str:
        """Data type of the column."""
        ...


class MatchedColumn(Protocol):
    """Metadata of the same column in the old and new table."""

    @property
    def old(self) -> Column:
        """The column in the old table."""
        ...

    @property
    def new(self) -> Column:
        """The column in the new table."""
        ...


class ChangedColumn(MatchedColumn, Protocol, Iterable[ChangedValue]):
    """Represents changes within the same column between the old and new table.

    Use :code:`len()` to get the total number of changed values.

    Since this is an :class:`Iterable[ChangedValue]`, it may contain examples of the changes:

        A :class:`ChangedValue` can represent multiple occurrences of the same values changing.

        There might be fewer examples than the total number of changes (returned by :code:`len()`).


    Methods
    -------
    __len__() -> int
        Total number of changes.
    __iter__() -> Iterator[ChangedValue]
        Examples of value changes.
    """

    def __len__(self): ...


class Rows(Protocol, Iterable[Mapping[str, Any]]):
    """Represents multiple rows. Use :code:`len()` to get the number of represented
    rows.

    Since this is an :class:`Iterable[Mapping[str, Any]]`, it may contain examples of the represented rows:

        A :class:`Mapping[str, Any]` maps column names to values of one row.

        There might be fewer examples than the number represented rows (returned by :code:`len()`).


    Methods
    -------
    __len__() -> int
        Number of represented rows.
    __iter__() -> Iterator[Mapping[str, Any]]
        Examples of represented rows.
    """

    def __len__(self): ...


class ColumnDelta(Protocol):
    """Generic collection of information about columns, types and their changes."""

    @property
    def old(self) -> Sequence[Column]:
        """Metadata of columns in the old table."""
        ...

    @property
    def new(self) -> Sequence[Column]:
        """Metadata of columns in the new table."""
        ...

    @property
    def added(self) -> Sequence[Column]:
        """Metadata of added columns (only in new table)."""
        ...

    @property
    def removed(self) -> Sequence[Column]:
        """Metadata of removed columns (only in old table)."""
        ...

    @property
    def joined(self) -> Sequence[MatchedColumn]:
        """Metadata of columns used to join the old and new table."""
        ...

    @property
    def renamed(self) -> Sequence[MatchedColumn]:
        """Metadata of renamed columns (matched using values and types)."""
        ...

    @property
    def comparable_type_changed(self) -> Sequence[MatchedColumn]:
        """Metadata of columns with changed but comparable data types."""
        ...

    @property
    def incomparable_type_changed(self) -> Sequence[ChangedColumn]:
        """Columns with changed and incomparable data types, including value changes.

        All values are assumed to be different.
        """
        ...

    @property
    def differences(self) -> Sequence[ChangedColumn]:
        """Changed values of comparable columns."""
        ...


class RowDelta(Protocol):
    """Generic collection of information about rows."""

    @property
    def old(self) -> Rows:
        """Rows in the old table."""
        ...

    @property
    def new(self) -> Rows:
        """Rows in the new table."""
        ...

    @property
    def added(self) -> Rows:
        """Added rows (only in new table)."""
        ...

    @property
    def removed(self) -> Rows:
        """Removed rows (only in old table)."""
        ...

    @property
    def equal(self) -> Rows:
        """Equal rows (joined and all columns unchanged)."""
        ...

    @property
    def unequal(self) -> Rows:
        """Unequal rows (joined but at least one column changed)."""


class TabularDelta(Protocol):
    """Generic collection of information gathered by one table comparison.

    Contains metadata about the comparison, :class:`RowDelta` and :class:`ColumnDelta`.
    """

    @property
    def name(self) -> str:
        """Name of the new table and therefore the comparison."""
        ...

    @property
    def old_name(self) -> str | None:
        """Optional name of the old table."""
        ...

    @property
    def info(self) -> Sequence[str]:
        """Additional information collected during the comparison."""
        ...

    @property
    def warnings(self) -> Sequence[str]:
        """Warnings collected during the comparison."""
        ...

    @property
    def errors(self) -> Sequence[str]:
        """Errors collected during the comparison."""
        ...

    @property
    def cols(self) -> ColumnDelta:
        """Column comparison containing information about columns, types, and value
        changes."""
        ...

    @property
    def rows(self) -> RowDelta:
        """Row comparison containing information about row changes and examples."""
        ...

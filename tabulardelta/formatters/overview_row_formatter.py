# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tabulardelta.formatters.tabulartext.cell import Cell
from tabulardelta.formatters.tabulartext.table import Border, Table
from tabulardelta.tabulardelta import TabularDelta


def _shrink(s: int) -> str:
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return "" if s == 0 else f"{s:,}".translate(sub)


def _shorten(s: str, max_len: int = 30) -> str:
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


class Row(Protocol):
    """Protocol defining how to retrieve data from a row.

    Methods
    -------
    get_cells() -> list[Cell | str]:
        Retrieve cells from the row
    """

    def get_cells(self) -> list[Cell | str]:
        """Retrieve cells from the row.

        Returns :class:`list[Cell | str]`:
            Returns cells such that they can be casted using :meth:`tabulartext.Table.from_lists`.
        """
        ...


@dataclass(frozen=True)
class Header(Row):
    """Header added by :meth:`OverviewRowFormatter.add_header`.

    Methods
    -------
    get_cells() -> list[Cell | str]:
        Retrieve cells from the row
    """

    title: str
    """Title of the header."""

    def get_cells(self) -> list[Cell | str]:
        title = Cell([_shorten(self.title)], colspan=2)
        return [title] + "ROWS = ≠ + - ? COLUMNS + - ≠ Δ t".split()


@dataclass(frozen=True)
class Single(Row):
    """Row with single cell added by :meth:`OverviewRowFormatter.add_str`.

    Methods
    -------
    get_cells() -> list[Cell | str]:
        Retrieve cells from the row
    """

    content: str = ""
    """Content of the row."""

    def get_cells(self) -> list[Cell | str]:
        return [Cell([self.content], colspan=14)]


@dataclass(frozen=True)
class Triple(Row):
    """Row with three cells added by :meth:`OverviewRowFormatter.add_legend`.

    Methods
    -------
    get_cells() -> list[Cell | str]:
        Retrieve cells from the row
    """

    name: str = ""
    """Content for char and name columns."""
    rows: str = ""
    """Content for row-changes columns."""
    cols: str = ""
    """Content for column-changes columns."""

    def get_cells(self) -> list[Cell | str]:
        return [
            Cell([self.name], colspan=2),
            Cell([self.rows], colspan=6),
            Cell([self.cols], colspan=6),
        ]


@dataclass(frozen=True)
class Content(Row):
    """Row with comparison data added by :meth:`OverviewRowFormatter.format` or
    :meth:`OverviewRowFormatter.add_row`.

    Methods
    -------
    get_cells() -> list[Cell | str]:
        Retrieve cells from the row
    """

    state_char: str = ""
    """One-char summary of comparison."""
    name: str = ""
    """Name of the table."""
    rows: int = 0
    """Number of rows in the new table."""
    equal_rows: int = 0
    """Number of matched and equal rows."""
    unequal_rows: int = 0
    """Number of matched but unequal rows."""
    added_rows: int = 0
    """Number of added rows (not in old table)."""
    removed_rows: int = 0
    """Number of removed rows (not in new table)."""
    unknown_rows: int = 0
    """Number of rows with unknown state."""
    cols: int = 0
    """Number of columns in the new table."""
    added_cols: int = 0
    """Number of added columns (not in old table)."""
    removed_cols: int = 0
    """Number of removed columns (not in new table)."""
    value_changed_cols: int = 0
    """Number of columns with changed values."""
    name_changed_cols: int = 0
    """Number of columns with changed names."""
    type_changed_cols: int = 0
    """Number of columns with changed types."""

    def get_cells(self) -> list[Cell | str]:
        strs = ["state_char", "name"]
        big = ["rows", "cols"]
        return [
            _shorten(v) if k in strs else f"{v:,}" if k in big else _shrink(v)
            for k, v in self.__dict__.items()
        ]


@dataclass
class OverviewRowFormatter:
    """Implements :class:`Formatter` protocol for combining multiple comparisons.

    This formatter is stateful to accumulate TabularDelta comparisons as rows.



    Additional rows can be added using add_str, add_row, add_header, and add_legend.
    Call table() to retrieve the table, which by default resets the state of the
    formatter.

    Methods
    -------
    format(delta: TabularDelta) -> None:
        Stores comparison as row in internal state
    add_header(title: str = "  TABLE") -> None:
        Add header row to internal state
    add_legend() -> None:
        Add legend as rows to internal state
    add_str(row: str) -> None:
        Add string as row to internal state
    add_row(char: str, name: str, rows: int, cols: int) -> None:
        Manually add table as row to internal state
    table(keep_state: bool = False) -> str:
        Retrieve report from internal state
    """

    warnings: bool = True
    """Whether to show warnings in the report."""
    errors: bool = True
    """Whether to show errors in the report."""

    state: list[Row] = field(default_factory=list)
    """Internal state of the formatter.

    May be used to do custom modifications/analyses. May change.
    """

    def format(self, delta: TabularDelta) -> None:
        """Stores comparison as row in internal state.

        Arguments:
            delta :class:`TabularDelta`:
                Metadata and result of a comparison.

        Returns :code:`None`:
            No return value. Use :meth:`table` to retrieve report.
        """
        if delta.errors:
            if self.errors:
                self.state.append(Content("⚠", delta.name))
                self.state.extend([Single(f"    ERROR {err}") for err in delta.errors])
            return

        values: list[Any] = [
            len(delta.rows.new),
            len(delta.rows.equal),
            len(delta.rows.unequal),
            len(delta.rows.added),
            len(delta.rows.removed),
            len(delta.cols.new),
            len(delta.cols.added),
            len(delta.cols.removed),
            len(delta.cols.differences),
            len(delta.cols.renamed),
            len(delta.cols.comparable_type_changed)
            + len(delta.cols.incomparable_type_changed),
        ]
        values.insert(5, values[0] - sum(values[1:4]))  # unknown rows
        char = "≠" if any(values[2:5] + values[7:12]) else ("?" if values[5] else "")
        self.state.append(Content(char, delta.name, *values))

        if self.warnings:
            self.state.extend(Single(f"    WARNING {warn}") for warn in delta.warnings)

    def add_header(self, title: str = "  TABLE") -> None:
        """Add header row to internal state.

        This should be called at least once in the beginning.

        Use this method with the optional title, to separate different chunks of comparisons.

        Arguments:
            title :class:`str`:
                Title describing the following tables.

        Returns :code:`None`:
            No return value. Use :meth:`table` to retrieve report.
        """
        self.state.append(Header(title))

    def add_legend(self) -> None:
        """Add legend as rows to internal state.

        This should be called in the end, if the users aren't familiar with the column meanings.

        Returns :code:`None`:
            No return value. Use :meth:`table` to retrieve report.
        """
        legend = [
            Single(),
            Single("LEGEND"),
            Triple("  equal", "ROW COUNT", "COLUMN COUNT"),
            Triple("+ added", "= joined and identical", "+ added"),
            Triple("- removed", "≠ joined but different", "- removed"),
            Triple("≠ unequal", "+ added", "≠ values differ"),
            Triple("? unknown", "- removed", "Δ renamed"),
            Triple("⚠ error occurred", "? uncompared", "t type changed"),
        ]
        self.state.extend(legend)

    def add_str(self, row: str) -> None:
        """Add string as row to internal state.

        This can be used to customize the report with additional information.

        Arguments:
            row :class:`str`:
                Content of the row.

        Returns :code:`None`:
            No return value. Use :meth:`table` to retrieve report.
        """
        self.state.append(Single(row))

    def add_row(self, char: str, name: str, rows: int, cols: int) -> None:
        """Manually add table as row to internal state.

        This is especially helpful for including added or removed tables.

        :code:`add_row("+", "table", 10, 5)` will result in

        .. code-block:: none

              TABLE  ROWS  COLUMNS
            + table  10    5

        Arguments:
            char :class:`str`:
                One-char summary of comparison, please refer to the legend.
            name :class:`str`:
                Name of the table.
            rows :class:`int`:
                Number of rows in the table.
            cols :class:`int`:
                Number of columns in the table.

        Returns :code:`None`:
            No return value. Use :meth:`table` to retrieve report.
        """
        self.state.append(Content(char, name, rows, cols=cols))

    def table(self, keep_state: bool = False) -> str:
        """Retrieve report from internal state.

        Arguments:
            keep_state :class:`bool`:
                Whether to keep the internal state or clean it up.

        Returns :class:`str`:
            Table combining comparisons as rows.
        """
        content = [row.get_cells() for row in self.state]
        if not keep_state:
            self.state = []
        return str(Table.from_lists(content, Border.VerticalGap))

# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabulardelta.formatters.tabulartext.cell import Align, Cell
from tabulardelta.formatters.tabulartext.table import Border, Table
from tabulardelta.tabulardelta import (
    ChangedColumn,
    Column,
    MatchedColumn,
    TabularDelta,
)

ColType = Column | MatchedColumn


def _descr_border(top: bool, bottom: bool) -> Border:
    """Creates horizontal hyphen-borders, top and bottom are optional."""
    t, b = "-" * top, "-" * bottom
    return Border(f" {t}, {t},, {t},, -,,,, -,,,, -,, {b},, {b}, {b}")


@dataclass(frozen=True)
class DetailedTextFormatter:
    """Implements :class:`Formatter` protocol for detailed report of single comparison.

    Tries to show as much information as possible, without overwhelming the user.

    Methods
    -------
    format(delta: TabularDelta) -> str:
        Formats comparison result
    """

    errors = True
    """Whether to show errors in the report."""
    warnings = True
    """Whether to show warnings in the report."""
    infos = True
    """Whether to show additional information in the report."""
    column_changes = True
    """Whether to show column and type changes in the report."""
    row_overview = True
    """Whether to show graphical row matching in the report."""
    value_changes = 5
    """How many value changes per column to show in the report.

    0 to disable.
    """
    row_examples = 3
    """How many examples per row category to show in the report.

    0 to disable.
    """

    def format(self, delta: TabularDelta) -> str:
        """Formats comparison result.

        Arguments:
            delta :class:`TabularDelta`:
                Metadata and result of a comparison.

        Returns :class:`str`:
            Detailed comparison report.
        """
        name = (f"{delta.old_name} -> " if delta.old_name else "") + delta.name
        result: list[list[Any]] = [
            [Cell([f" TabularDelta Report: {name} "], Align.MidCenter("-"))],
            *[[f"ERROR: {error}"] for error in delta.errors if self.errors],
            *[[f"WARNING: {warning}"] for warning in delta.warnings if self.warnings],
            *[[f"INFO: {info}"] for info in delta.info if self.infos],
            [f"Joined on {[c.new.name for c in delta.cols.joined] or 'index'}."],
        ]

        if self.column_changes:
            col_changes: list[tuple[str, Any]] = [
                ("Added columns", delta.cols.added),
                ("Removed columns", delta.cols.removed),
                ("Renamed columns", delta.cols.renamed),
                ("Comparable type changes", delta.cols.comparable_type_changed),
            ]
            result.extend(
                [_show_cols(name, cols)] for name, cols in col_changes if cols
            )

            incomparable = delta.cols.incomparable_type_changed
            for chg in incomparable:
                left = _show_cols("INCOMPARABLE type change", [chg])
                row = [left, Cell(["  "]), _show_values(chg, len(left.content))]
                result.append([Table([row], Border.Nothing)])

            if not any([cols for _, cols in col_changes] + [incomparable]):
                result.append(["All columns and types are identical."])

        if self.row_overview:
            result.append([_show_rows(delta)])

        if self.value_changes > 0:
            for chg in delta.cols.differences:
                result.append([_show_values(chg, self.value_changes + 3, True)])

        result.append(_example_rows("ADDED", delta.rows.added, self.row_examples))
        result.append(_example_rows("REMOVED", delta.rows.removed, self.row_examples))

        return str(Table.from_lists(result, Border.InnerGap))


def _example_rows(
    name: str, rows: Iterable[Mapping[str, Any]], limit: int
) -> list[str]:
    """Returns example rows as line-separated strings.

    > example_rows("ADDED", [{"id": 3, "data": 13}, {"id": 4, "data": 15}], 3)
    ADDED ROWS EXAMPLES:
    id|data
    3 | 13
    4 | 15
    """
    result: list[list[Any]] = []
    for idx, row in enumerate(rows):
        if idx == limit:
            break
        if not result:
            result.append(list(row.keys()))
        result.append(list(row.values()))
    if result:
        border = Border(",,,, │,,,,,,,,,,,,,,")
        header = f"{name} ROWS EXAMPLES:\n"
        return [header + Table.from_lists(result, border).to_string()]
    return []


def _standardize_col_data(col: ColType) -> list[str]:
    """Formats column[-changes] in readable way.

    list('old_name [→ new_name]', '(old_type[→ new_type])')
    """

    if hasattr(col, "name") and hasattr(col, "type"):
        return [col.name, f"({col.type})"]
    name_change = f" → {col.new.name}" if col.old.name != col.new.name else ""
    type_change = f" → {col.new.type}" if col.old.type != col.new.type else ""
    return [col.old.name + name_change, f"({col.old.type + type_change})"]


def _show_cols(name: str, cols: list[ColType], max_width: int = 80) -> Cell:
    """Shows columns nicely visualized, or listed in columns if too wide.

    ExampleName:
    ┏━━━━━━━┯━━━━━━━┓             ExampleName:
    ┃ col1  │ col2  ┃             VeryLongColumnName (VeryLongType)
    ┃ type1 │ type2 ┃             col1 (type1)     col2 (type2)
    ┣━━━━━━━┿━━━━━━━┫             col3 (type3)     col4 (type4)
    """
    if not cols:
        return Cell()
    strs = [_standardize_col_data(col) for col in cols]

    if sum(len(s) for x in strs for s in x) <= max_width:
        # Print detailed table if not too wide:
        border = Border(" ┏, ━, ┯, ┓, │,,,, ┃,, ┃,,,,, ┣, ┷, ━, ┫")
        table = Table.from_lists([strs], border)
    else:
        # Otherwise list cells: Long cells at the top, short cells in two columns below
        flat = [" ".join(col) for col in strs]
        long = [[Cell([col], colspan=3)] for col in flat if len(col) > max_width // 2]
        short = [Cell([col]) for col in flat if len(col) <= max_width // 2] + [Cell()]
        split = [[short[i], "   ", short[i + 1]] for i in range(0, len(short) - 1, 2)]
        table = Table.from_lists(long + split, Border.Nothing)
    return Cell([f"{name}:"] + table.lines())


def _show_values(chg: ChangedColumn, max_height=10, header: bool = False) -> Cell:
    """
    Show value changes of columns:
         id_old ─►  id_new example_name
    (5x)      0 ─►     0.0        Alice
    (2x)      1 ─►     1.0          Bob
              2 ─►     2.0      Charlie
    ...6 other differences
    """
    old, new, total = chg.old.name, chg.new.name, len(chg)
    result: list[list[Cell] | list[str]] = []
    indices = []
    for row, idx in zip(chg, range(1 + header, max_height)):
        if not result:
            indices = sorted(list(row.example_join_columns))
            if header:
                content = [f"Column {new} - {total} rows changed:"]
                result.append([Cell(content, colspan=5 + len(indices))])
            result.append(["", "", old, "→", new] + [f"example_{i}" for i in indices])
        if idx == max_height - 1 and total > row.count:
            break  # max_height will not suffice to print all changes
        content = ["    ", f"({row.count}x)" * (row.count > 1), row.old, "→", row.new]
        result.append(content + [row.example_join_columns[ex] for ex in indices])
        total -= row.count
    if total > 0:
        content = [f"...{total} other differences"]
        result.append([Cell(content, colspan=5 + len(indices))])
    return Cell(Table.from_lists(result, Border.VerticalGap).lines())


def _show_rows(delta: TabularDelta) -> Table:
    """Row overview is created as a table of tables (without outer borders):

    header_old_rows   header_new_rows
    ┏━━━━━━━━━━━━┯━━━━┯━━━━━━━━━━━━┯━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━┓
    ┃    Old     │    │    New     │             │               ┃
    ┃ 1557572484 │    │ 1557247374 │             │               ┃
    ┠───────────┬┴────┴───────────┬┴─────────────┼───────────────┨
    ┃ ┏━┯━┯━┯━┓ │    "spacer"     │  325116      │               ┃
    ┃ ┃-│-│-│-┃ │                 │  removed     │               ┃
    ┃ ┃-│-│-│-┃ ├─────┬───────────┤ ------------ ├───────────────┨
    ┃ ┠─┼─┼─┼─┨ │ ╌╌╌ │ ┏━┯━┯━┯━┓ │              │               ┃
    ┃ ┃ │ │ │ ┃ │  =  │ ┃ │ │ │ ┃ │  44605       │               ┃
    ┃ ┃ │ │ │ ┃ │  =  │ ┃ │ │ │ ┃ │  identical   │               ┃
    ┃ ┠─┼─┼─┼─┨ │ ╌╌╌ │ ┠─┼─┼─┼─┨ │ ------------ │ ╶╮            ┃
    ┃ ┃ │ │ │ ┃ │  ≠  │ ┃ │ │ │ ┃ │              │  │            ┃
    ┃ ┃ │ │ │ ┃ │  ≠  │ ┃ │ │ │ ┃ │  1557120961  │  │            ┃
    ┃ ┃ │ │ │ ┃ │  ≠  │ ┃ │ │ │ ┃ │  changed     │  │ 1557202763 ┃
    ┃ ┃ │ │ │ ┃ │  ≠  │ ┃ │ │ │ ┃ │              │  ├╴rows       ┃
    ┃ ┠─┼─┼─┼─┨ │ ╌╌╌ │ ┠─┼─┼─┼─┨ │ ------------ │  │ joined     ┃
    ┃ ┃?│?│?│?┃ │  ?  │ ┃?│?│?│?┃ │  81802       │  │            ┃
    ┃ ┃?│?│?│?┃ │  ?  │ ┃?│?│?│?┃ │  unknown     │  │            ┃
    ┃ ┗━┷━┷━┷━┛ │ ╌╌╌ │ ┠─┼─┼─┼─┨ │ ------------ │ ╶╯            ┃
    ┃           │     │ ┃+│+│+│+┃ │  6 added     │               ┃
    ┃           │     │ ┗━┷━┷━┷━┛ │              │               ┃
    ┗━━━━━━━━━━━┷━━━━━┷━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━┛
        left    middle    right        text          join
    """
    # rows: [removed, equal, unequal, unknown, added]
    rows = [
        len(delta.rows.removed),
        len(delta.rows.equal),
        len(delta.rows.unequal),
        len(delta.rows.added),
    ]
    rows.insert(3, len(delta.rows.old) - sum(rows[0:3]))  # unknown rows
    heights = [int(np.log10(1000 * count + 1)) // 3 for count in rows]

    inner_cells = [[[c] * h] * 4 * bool(h) for h, c in zip(heights, "-  ?+")]
    left = Table.from_lists(inner_cells[:-1]).lines()
    right = Table.from_lists(inner_cells[1:]).lines()

    middle_cells = [[[c] * h] for h, c in zip(heights[1:], "=≠? ") if h or c == " "]
    middle_border = Border(" ╌, ╌,, ╌,, ╌,,,, ╌,,,, ╌,,,,,")
    middle = Table.from_lists(middle_cells, middle_border).lines()

    text_data = zip(rows, "removed identical changed unknown added".split())
    texts = [[str(n), t] if n > 999 else [f"{n} {t}"] for n, t in text_data]
    text_cells = [[[""] * h, t] for h, t in zip(heights, texts) if h]
    text_table = Table.from_lists(text_cells, _descr_border(not rows[0], not rows[4]))
    text = text_table.lines()

    join_height = heights[1] + heights[2] + bool(heights[1]) + bool(heights[2]) + 1
    bracket = [" │"] * join_height
    bracket[0], bracket[(join_height - 1) // 2], bracket[-1] = "╶╮", " ├╴", "╶╯"
    join_descr = Cell([str(sum(rows[1:3])), "rows", "joined"], Align.MidLeft())
    join_segs = [[bracket, join_descr]] if join_height > 1 else []
    join = Table.from_lists(join_segs, Border.Nothing).lines()

    header_old_rows = Cell(["Old", str(len(delta.rows.old))], Align.MidCenter(), 2)
    header_new_rows = Cell(["New", str(len(delta.rows.new))], Align.MidCenter(), 2)
    spacer = Cell(["\n" * heights[0]] if heights[0] else [], colspan=3)
    content: list[list[Cell]] = [
        [header_old_rows, Cell([" "]), header_new_rows],
        [Cell(left, rowspan=2), spacer, Cell(text, rowspan=2, colspan=2)],
        [Cell(middle, colspan=2), Cell(right), Cell(join)],
    ]

    return Table(content, Border.Nothing)

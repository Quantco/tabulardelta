# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol

from .cell import Align, Cell


class BorderStyle(Protocol):
    """BorderStyle is a protocol for adding border-cells in-between existing cells of a
    table.

    Arguments:
        table :class:`Table`:
            Table to add borders to.

    Returns :class:`list[list[Cell]]`:
        Table content with added border cells.

    Methods
    -------

    __call__(self, table: Table) -> list[list[Cell]]:
        Adds border to given table
    """

    def __call__(self, table: Table) -> list[list[Cell]]: ...


class Border:
    """Implementation of :class:`BorderStyle`, for adding a border to an existing table.

    Offers out-of-the-box border styles as class attributes:
        :attr:`Nothing`, :attr:`InnerGap`, :attr:`Basic`, and :attr:`VerticalGap`.

    Use :meth:`__init__` with :attr:`style_specifier` for specialized border styles:
        A style specifier are comma-separated border descriptions, for all 19 border positions.

        Border descriptions start with a space or a dot, followed by the content of the border cell.

        A dot means that the border cell is allowed to have size 0, a space means it is not.

        The style specifier for :attr:`Basic` therefore looks like this:

        .. code-block::

             ┏,.━, ┯, ┓,.│,
             ┠, ┬, ┤,.┃,.─,
            .┃, ├, ┴, ┨, ┼,
             ┗, ┷,.━, ┛

    Methods
    -------
    __init__(self, style_specifier: str):
        Initializes a :class:`Border` based on style specifier
    __call__(self, table: Table) -> list[list[Cell]]:
        Implements :class:`BorderStyle`
    """

    # Out-of-the-box Borders
    Nothing: ClassVar[BorderStyle]
    """Empty border cells.

    Can be used as syntactic sugar to not always specify `add_border=False`.
    """
    InnerGap: ClassVar[BorderStyle]
    """Adds empty lines between rows and two spaces between columns."""
    Basic: ClassVar[BorderStyle]
    """Uses box-drawing characters to draw a beautiful border."""
    VerticalGap: ClassVar[BorderStyle]
    """Adds two spaces between columns."""

    def __init__(self, style_specifier: str):
        style = style_specifier.replace("\n", "").split(",")
        if len(style) != 19:
            raise ValueError("Style must be 19 comma-separated strings")
        order = [10, 5, 0, 2, 1, 3, 16, 15, 17, 8, 18, 13, 4, 11, 7, 14, 12, 6, 9]
        contents = [(style[i][1:], (style[i][:1] not in (".", ""))) for i in order]
        self._style = [Cell([val] * fix, Align.Repeat(val)) for val, fix in contents]

    @staticmethod
    def _prune(arr: list[list[Any]], x: int, y: int) -> None:
        """
        style e.g.: ┃, ┠, ┏, ┯, ━, ┓, ┷, ┗, ━, ┃, ┛, ┨, │, ├, ┤, ┼, ┴, ┬, ─
        index:      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18
        pruneTop:     ┼ -> ┬,  ┷ -> ━,  ┴ -> ─   (index += 2)
        pruneBottom:  ┼ -> ┴,  ┯ -> ━,  ┬ -> ─   (index += 1)
        pruneRight:   ┼ -> ┤,  ┠ -> ┃,  ├ -> │   (index -= 1)
        pruneLeft:    ┼ -> ├,  ┨ -> ┃,  ┤ -> │   (index -=2)
        """
        arr[y][x] = None
        arr[y + 1][x] = arr[y + 1][x] + 2 if isinstance(arr[y + 1][x], int) else None
        arr[y - 1][x] = arr[y - 1][x] + 1 if isinstance(arr[y - 1][x], int) else None
        arr[y][x - 1] = arr[y][x - 1] - 1 if isinstance(arr[y][x - 1], int) else None
        arr[y][x + 1] = arr[y][x + 1] - 2 if isinstance(arr[y][x + 1], int) else None

    @staticmethod
    def _get_border_table(x: int, y: int) -> list[list[int | Cell | None]]:
        """Shows columns nicely visualized, or listed in columns if too wide
        Example for 2x2:
        │-----x=2-----│
        ┏━━━━━━┯━━━━━━┓  ┬
        ┃Cell()│Cell()┃  ╎
        ┠──────┼──────┨  y=2
        ┃Cell()│Cell()┃  ╎
        ┗━━━━━━┷━━━━━━┛  ┴
        """

        def _row(cells: tuple[Any, Any, Any, Any]) -> list[Any]:
            """(A,B,C,D) => [A, B, C, B, C, B, ..., C, B, D] (len=x)"""
            repeat_cells = [cell for _ in range(x - 1) for cell in cells[2:0:-1]]
            return [cells[0], cells[1], *repeat_cells, cells[3]]

        top = (2, 4, 3, 5)  # ......┏ ━ ┯ ┓
        mid = (0, Cell(), 12, 9)  # ┃ X │ ┃
        sep = (1, 18, 15, 11)  # ...┠ ─ ┼ ┨
        bot = (7, 8, 6, 10)  # .....┗ ━ ┷ ┛
        repeat_rows = [row for _ in range(y - 1) for row in (_row(sep), _row(mid))]
        return [_row(top), _row(mid), *repeat_rows, _row(bot)]

    def __call__(self, table: Table) -> list[list[Cell]]:
        dimensions = table._dimensions()
        if not any(dimensions):
            return table.content
        content: list[list[int | Cell | None]] = self._get_border_table(*dimensions)

        def fill(cell_x: int, cell_y: int, cell: Cell | None) -> None:
            for x_offset in range(getattr(cell, "colspan", 0) * 2 - 1):
                for y_offset in range(getattr(cell, "rowspan", 0) * 2 - 1):
                    x = 2 * cell_x + 1 + x_offset
                    y = 2 * cell_y + 1 + y_offset
                    # Iterate over borders within spanned cell:
                    if x % 2 == 0 or y % 2 == 0:
                        self._prune(content, x, y)
            content[2 * cell_y + 1][2 * cell_x + 1] = cell

        table._traverse(fill)

        result: list[list[Cell]] = []
        for row in content:
            result.append([])
            for cell in row:
                if isinstance(cell, int):
                    result[-1].append(self._style[cell])
                elif isinstance(cell, Cell):
                    colspan, rowspan = cell.colspan * 2 - 1, cell.rowspan * 2 - 1
                    result[-1].append(replace(cell, colspan=colspan, rowspan=rowspan))

        return result


Border.Basic = Border(
    ""  # linebreak
    + " ┏,.━, ┯, ┓,.│,"
    + " ┠, ┬, ┤,.┃,.─,"
    + ".┃, ├, ┴, ┨, ┼,"
    + " ┗, ┷,.━, ┛"
)
Border.Nothing = Border(",,,,,,,,,,,,,,,,,,")
Border.InnerGap = Border(",,,,   ,,,,,   ,,,,,,,,,")
Border.VerticalGap = Border(",,,,   ,,,,,,,,,,,,,,")


@dataclass
class Table:
    """Represents a table containing cells with a border style.

    Note:
        If you don't have a :class:`list[list[Cell]]`, try :meth:`from_lists` to create a table.

        This will cast the content accordingly (see :meth:`from_lists` documentation).

    Methods
    -------

    __getitem__(self, item: int) -> list[Cell]:
        Retrieves cells using :code:`table[row_nr][col_nr]`.
    __setitem__(self, key: int, value: list[Cell]) -> None:
        Setting cells using :code:`table[row_nr][col_nr] = cell`.
    from_lists(strs: list[list[Any]], border: BorderStyle = Border.Basic) -> Table:
        Casts content to create table (see :meth:`from_lists` documentation).
    lines(self, add_border: bool = True) -> list[str]:
        Renders table as line-separated list of strings.
    to_string(self, add_border: bool = True) -> str:
        Renders table as string.
    __str__(self) -> str:
        Renders table as string.
    """

    content: list[list[Cell]]
    """Table content as a list of rows, each containing a list of cells."""

    border: BorderStyle = Border.Basic
    """Border style to be added."""

    _border_added: bool = False

    def __getitem__(self, item: int) -> list[Cell]:
        return self.content[item]

    def __setitem__(self, key: int, value: list[Cell]) -> None:
        self.content[key] = value

    def _add_border(self, override: bool = False) -> bool:
        """Adds border to table if not already added or override is True."""
        if not self._border_added or override:
            self.content = self.border(self)
            self._border_added = True
            return True
        return False

    @staticmethod
    def from_lists(strs: list[list[Any]], border: BorderStyle = Border.Basic) -> Table:
        """Creates a table from list of lists of objects, where each object is cast to a
        cell:

            A :class:`Cell` won't be casted.

            A :class:`list` is interpreted as containing lines, with each line being casted to :class:`str`.

            Other objects are casted to strings.

        Arguments:
            strs :class:`list[list[Any]]`:
                List of lists of cell objects.
            border :class:`BorderStyle`:
                Border style of table.

        Returns :class:`Table`:
            Table containing casted cells and border style.
        """

        def _cast(cell: Cell | list[Any] | Any) -> Cell:
            if isinstance(cell, Cell):
                return cell
            elif isinstance(cell, list):
                return Cell([str(row) for row in cell])
            return Cell([str(cell)])

        return Table([[_cast(cell) for cell in row] for row in strs], border=border)

    def _traverse(self, func: Callable[[int, int, Cell | None], None]) -> None:
        """Calls func for each cell with x_pos, y_pos, and cell as arguments.

        The cell is None if position is occupied by other cell with rowspan/colspan > 1.
        """
        # Implicit cells due to rowspan: Dict row -> set(column)
        rowspan_cells: dict[int, set[int]] = defaultdict(set)

        y_pos = 0
        for row in self.content:
            row = [cell for cell in row if cell]
            x_pos = 0
            for cell in row:
                while x_pos in rowspan_cells[y_pos]:
                    func(x_pos, y_pos, None)
                    x_pos += 1
                for spanned_row in range(y_pos + 1, y_pos + cell.rowspan):
                    spanned_cols = range(x_pos, x_pos + cell.colspan)
                    rowspan_cells[spanned_row].update(spanned_cols)
                func(x_pos, y_pos, cell)
                for spanned_col in range(x_pos + 1, x_pos + cell.colspan):
                    func(spanned_col, y_pos, None)
                x_pos += cell.colspan
            if row:
                y_pos += 1
        while y_pos in rowspan_cells:
            for x_pos in rowspan_cells[y_pos]:
                func(x_pos, y_pos, None)
            y_pos += 1

    def _dimensions(self) -> tuple[int, int]:
        """Get width and height of cell grid, including rowspan/colspan."""
        width = _MaxValue()
        height = _MaxValue()

        def measure(x: int, y: int, _: Cell | None) -> None:
            width(x + 1)
            height(y + 1)

        self._traverse(measure)
        return width(), height()

    def lines(self, add_border: bool = True) -> list[str]:
        """Returns rendered table as line-separated list of strings.

        This method will never add a border if it already exists!

        Arguments:
            add_border :class:`bool`:
                Whether to add a border to the table.

        Returns :class:`list[str]`:
            Line-separated rendered table.
        """
        if add_border:
            self._add_border()

        # Cell size requirements: end_of_merge -> start_of_merge -> max_size
        horizontal_sizes: dict[int, dict[int, _MaxValue]] = defaultdict(
            lambda: defaultdict(_MaxValue)
        )
        vertical_sizes: dict[int, dict[int, _MaxValue]] = defaultdict(
            lambda: defaultdict(_MaxValue)
        )

        def update_size_requirements(x: int, y: int, cell: Cell | None) -> None:
            if cell is not None:
                height = len(cell.content)
                width = max((len(line) for line in cell.content), default=0)
                vertical_sizes[y + cell.rowspan - 1][y](height)
                horizontal_sizes[x + cell.colspan - 1][x](width)

        self._traverse(update_size_requirements)
        widths = _get_min_sizes(horizontal_sizes)
        heights = _get_min_sizes(vertical_sizes)

        result = [[""] * sum(widths) for _ in range(sum(heights))]

        def collect_result(x: int, y: int, cell: Cell | None) -> None:
            if cell is not None:
                width = sum(widths[x : x + cell.colspan])
                height = sum(heights[y : y + cell.rowspan])
                filled = cell.fill(width, height)
                current_row = sum(heights[:y])
                current_col = sum(widths[:x])
                for y_pos, line in enumerate(filled):
                    if line:
                        result[current_row + y_pos][current_col] = line

        self._traverse(collect_result)
        return ["".join(row) for row in result]

    def to_string(self, add_border: bool = True) -> str:
        """Returns rendered table as string by concatenating :meth:`lines`.

        This method will never add a border if it already exists!

        :meth:`__str__` calls this method with default arguments.

        Arguments:
            add_border :class:`bool`:
                Whether to add a border to the table.

        Returns :class:`str`:
            Rendered table.
        """
        return "\n".join(self.lines(add_border))

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class _MaxValue:
    value: int = 0

    def __call__(self, new: int | None = None) -> int:
        self.value = max(self.value, new or 0)
        return self.value


def _get_min_sizes(ranges: dict[int, dict[int, _MaxValue]]) -> list[int]:
    """
    ranges: Size requirements (end_of_range -> start_of_range -> required_size)
    Gets the minimum sizes for multiple overlapping ranges

    │A│B│CDEFGHI│J  │  <-> {   0: {0: 1}, 1: {1: 1}, 2: {2: 7}, 3: {3: 1}
    │ABCDEF     │GHI│          3: {0: 6}, 4: {4: 3}                         }
    ╎ ╎ ╎       ╎   ╎
    ╎1╎1╎   7   ╎ 3 ╎  --> [1, 1, 7, 3]
    """
    sizes: list[int] = []
    for end in range(max(ranges.keys(), default=-1) + 1):
        # Set sizes[end] to fit all ranges ending there
        gen = (
            min_range_size() - sum(sizes[start:end])
            for start, min_range_size in ranges[end].items()
        )
        sizes.append(max(0, max(gen, default=0)))
    return sizes

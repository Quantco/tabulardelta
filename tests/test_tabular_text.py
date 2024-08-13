# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from __future__ import annotations

from tabulardelta.formatters.tabulartext.cell import Align, Cell
from tabulardelta.formatters.tabulartext.table import Border, Table


def test_empty() -> None:
    table = Table.from_lists([])
    assert table.to_string(add_border=False) == ""
    assert str(table) == ""

    table = Table.from_lists([[]])
    assert table.to_string(add_border=False) == ""
    assert str(table) == ""

    table = Table.from_lists([[""]])
    assert table.to_string(add_border=False) == ""
    assert str(table) == "┏┓\n┃┃\n┗┛"

    table = Table.from_lists([[], []])
    assert table.to_string(add_border=False) == ""
    assert str(table) == ""

    table = Table.from_lists([[], [], [Cell()]])
    assert table.to_string(add_border=False) == ""
    assert str(table) == "┏┓\n┗┛"

    table = Table.from_lists([[""], [], [], [""]])
    assert table.to_string(add_border=False) == "\n"
    assert str(table) == "┏┓\n┃┃\n┠┨\n┃┃\n┗┛"


def test_table() -> None:
    content: list[list[str | Cell]] = [
        [Cell(["This"], colspan=2)],
        ["", Cell(["is"], colspan=2)],
        ["", "", Cell(["a"], colspan=2)],
        [
            "m\nu\nl\nt\ni\nline",
            "",
            Cell(["\n\n\ncolspan\nis\n3"], rowspan=3),
            Cell(["test"], colspan=2),
        ],
        ["..."],
        ["colspan\nends\nhere", "->\n->\n->"],
        ["A", "B", "C", "D", "E"],
        [
            Cell([], rowspan=3),
            "",
            Cell(["."], align=Align.Repeat(), rowspan=3, colspan=3),
        ],
        [],
        [Cell([])],
        ["The end.", ""],
    ]
    table = Table.from_lists(content)
    assert str(table) == (
        "┏━━━━━━━━━━━━━━━━┯━━━━━━━┯━┯━━┯┓\n"
        "┃This            │       │ │  │┃\n"
        "┠───────┬────────┴───────┼─┼──┼┨\n"
        "┃       │is              │ │  │┃\n"
        "┠───────┼────────┬───────┴─┼──┼┨\n"
        "┃       │        │a        │  │┃\n"
        "┠───────┼────────┼───────┬─┴──┼┨\n"
        "┃m      │        │       │test│┃\n"
        "┃u      │        │       │    │┃\n"
        "┃l      │        │       │    │┃\n"
        "┃t      │        │colspan│    │┃\n"
        "┃i      │        │is     │    │┃\n"
        "┃line   │        │3      │    │┃\n"
        "┠───────┼────────┤       ├─┬──┼┨\n"
        "┃...    │        │       │ │  │┃\n"
        "┠───────┼────────┤       ├─┼──┼┨\n"
        "┃colspan│->      │       │ │  │┃\n"
        "┃ends   │->      │       │ │  │┃\n"
        "┃here   │->      │       │ │  │┃\n"
        "┠───────┼────────┼───────┼─┼──┼┨\n"
        "┃A      │B       │C      │D│E │┃\n"
        "┠───────┼────────┼───────┴─┴──┼┨\n"
        "┃       │        │............│┃\n"
        "┃       ├────────┤............├┨\n"
        "┃       ├────────┤............├┨\n"
        "┃       │The end.│............│┃\n"
        "┗━━━━━━━┷━━━━━━━━┷━━━━━━━━━━━━┷┛"
    )


def test_only_vertical_borders() -> None:
    content: list[list[str | Cell]] = [
        [Cell(["This"], colspan=2)],
        ["", Cell(["is"], colspan=2)],
        ["", "", Cell(["a"], colspan=2)],
        [
            "m\nu\nl\nt\ni\nline",
            "",
            Cell(["\n\n\ncolspan\nis\n3"], rowspan=3),
            Cell(["test"], colspan=2),
        ],
        ["..."],
        ["colspan\nends\nhere", "->\n->\n->"],
        ["A", "B", "C", "D", "E"],
        [
            Cell([], rowspan=3),
            "",
            Cell(["."], align=Align.Repeat(), rowspan=3, colspan=3),
        ],
        [],
        ["The end.", ""],
    ]
    border = Border(
        """
 ┏,.━, ┯, ┓,.│,
.┠,.┬,.┤,.┃,.─,
.┃,.├,.┴,.┨,.┼,
 ┗, ┷,.━, ┛
"""
    )
    table = Table.from_lists(content, border)
    assert str(table) == (
        "┏━━━━━━━━━━━━━━━━┯━━━━━━━┯━┯━━┯┓\n"
        "┃This            │       │ │  │┃\n"
        "┃       │is              │ │  │┃\n"
        "┃       │        │a        │  │┃\n"
        "┃m      │        │       │test│┃\n"
        "┃u      │        │       │    │┃\n"
        "┃l      │        │       │    │┃\n"
        "┃t      │        │colspan│    │┃\n"
        "┃i      │        │is     │    │┃\n"
        "┃line   │        │3      │    │┃\n"
        "┃...    │        │       │ │  │┃\n"
        "┃colspan│->      │       │ │  │┃\n"
        "┃ends   │->      │       │ │  │┃\n"
        "┃here   │->      │       │ │  │┃\n"
        "┃A      │B       │C      │D│E │┃\n"
        "┃       │        │............│┃\n"
        "┃       │The end.│............│┃\n"
        "┗━━━━━━━┷━━━━━━━━┷━━━━━━━━━━━━┷┛"
    )

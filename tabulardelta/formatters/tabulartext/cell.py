# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import cycle, islice
from typing import ClassVar, Protocol


class Aligner(Protocol):
    """Aligner is a protocol for aligning existing cell content within a larger target
    size.

    Arguments:
        content :class:`list[str]`:
            Original cell content to be aligned.
        width :class:`int`:
            Target width of cell content, not smaller than length of any line in the original content.
        height :class:`int`:
            Target height of cell content, not smaller than number of lines of the original content.

    Returns :class:`list[str]`:
        Cell content with `height` number of lines, each with length `width`.

    Methods
    -------

    __call__(self, content: list[str], width: int, height: int) -> list[str]:
        Resize cell content to a target size.
    """

    def __call__(self, content: list[str], width: int, height: int) -> list[str]: ...


@dataclass(frozen=True)
class Align:
    """Implementation of :class:`Aligner`, for aligning cell content given a target
    size.

    Offers out-of-the-box alignment styles as class attributes:
        :attr:`TopLeft`, :attr:`MidLeft`, :attr:`MidCenter`, and :attr:`Repeat`.

        Each can be customized with an optional fill character (see :class:`AlignOptionalChar`).

    Implement :attr:`vertical` and :attr:`horizontal` for specialized alignment styles.

    Methods
    -------
    __call__(self, content: list[str], width: int, height: int) -> list[str]:
        Implements :class:`Aligner`
    """

    TopLeft: ClassVar[AlignOptionalChar]
    """:class:`AlignOptionalChar` that aligns content towards the top left."""
    MidLeft: ClassVar[AlignOptionalChar]
    """:class:`AlignOptionalChar` that aligns content to the left, but vertically
    centered."""
    MidCenter: ClassVar[AlignOptionalChar]
    """:class:`AlignOptionalChar` that aligns content in the middle."""
    Repeat: ClassVar[AlignOptionalChar]
    """:class:`AlignOptionalChar` that repeats the content through the cells."""

    vertical: Callable[[list[str], int], list[str]]
    """Callable adding lines to the cell content to reach the given height."""
    horizontal: Callable[[str, int], str]
    """Callable adding characters to a line to reach the given width."""

    class AlignOptionalChar(Protocol):
        """Protocol for customizing out-of-the-box alignments with an optional fill
        character.

        Arguments:
            char :class:`str`:
                Optional fill character for the alignment. Default is a space.

        Returns :class:`Align`:
            An Aligner with the specified fill character.

        Methods
        -------

        __call__(self, char: str = " ") -> Align:
            Gets the :class:`Align` instance
        """

        def __call__(self, char: str = " ") -> Align: ...

    def __call__(self, content: list[str], width: int, height: int) -> list[str]:
        content = self.vertical(content, height)
        return [self.horizontal(line, width) for line in content]


Align.TopLeft = lambda char=" ": Align(
    lambda lines, size: (lines + [""] * (size - len(lines))),
    lambda line, size: line.ljust(size, char),
)

Align.MidLeft = lambda char=" ": Align(
    lambda lines, size: Align.TopLeft(char).vertical(
        [""] * ((size - len(lines)) // 2) + lines, size
    ),
    lambda line, size: line.ljust(size, char),
)

Align.MidCenter = lambda char=" ": Align(
    lambda lines, size: Align.TopLeft(char).vertical(
        [""] * ((size - len(lines)) // 2) + lines, size
    ),
    lambda line, size: line.center(size, char),
)

Align.Repeat = lambda char=" ": Align(
    lambda lines, size: list(islice(cycle(lines or [""]), size)),
    lambda line, size: "".join(islice(cycle(line or char or " "), size)),
)


@dataclass
class Cell:
    """Represents a table cell with alignable content, which might span multiple columns
    and rows.

    Methods
    -------
    fill(self, width: int, height: int) -> list[str]:
        Increase size of table content using :attr:`align`
    """

    content: list[str] = field(default_factory=list)
    """Line-separated cell content."""

    align: Aligner = Align.TopLeft()
    """Alignment strategy (see :class:`Aligner`)."""

    colspan: int = 1
    """Number of columns to span."""

    rowspan: int = 1
    """Number of rows to span."""

    def __post_init__(self) -> None:
        self.content = [split for line in self.content for split in line.split("\n")]
        if self.rowspan < 1 or self.colspan < 1:
            raise ValueError("Rowspan and colspan must be positive")

    def fill(self, width: int, height: int) -> list[str]:
        """Return cell content resized to a target size by using :attr:`align` strategy.

        Arguments:
            width :class:`int`:
                The target width for the cell.
            height :class:`int`:
                The target height for the cell.

        Returns :class:`list[str]`:
            The resized line-separated content.
        """
        return self.align(self.content, width, height)

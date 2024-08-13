# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from typing import Protocol, TypeVar

from tabulardelta.tabulardelta import TabularDelta

T = TypeVar("T", contravariant=True)


class Comparator(Protocol[T]):
    """Protocol for offering table comparison logic for TabularDelta.

    Ensures compatibility with all :class:`Formatter` implementations.

    Methods
    -------
    compare(old: T, new: T) -> TabularDelta:
        Compare two tables
    """

    def compare(self, old: T, new: T) -> TabularDelta:
        """Compare two tables.

        Arguments:
            old :class:`T`:
                The old table (first table to compare).
            new :class:`T`:
                The new table (second table to compare).

        Returns :class:`TabularDelta`:
            Metadata and results of the comparison.
        """
        ...

# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from typing import Protocol, TypeVar

from tabulardelta.tabulardelta import TabularDelta

T = TypeVar("T", covariant=True)


class Formatter(Protocol[T]):
    """Protocol for offering visualizations or further analyses for TabularDelta.

    Ensures compatibility with all :class:`Comparator` implementations.

    Methods
    -------
    format(delta: TabularDelta) -> T:
        Formats comparison result
    """

    def format(self, delta: TabularDelta) -> T:
        """Formats comparison result.

        Arguments:
            delta :class:`TabularDelta`:
                Metadata and result of a comparison.

        Returns :class:`T`:
            Arbitrary output. Usually strings visualizing the comparison.
        """
        ...

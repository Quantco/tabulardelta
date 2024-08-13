# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import importlib.metadata
import warnings
from typing import Any

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as e:  # pragma: no cover
    warnings.warn(f"Could not determine version of {__name__}\n{e!s}", stacklevel=2)
    __version__ = "unknown"

from tabulardelta.comparators.comparator import Comparator
from tabulardelta.comparators.pandas_comparator import PandasComparator
from tabulardelta.formatters.detailed_text_formatter import DetailedTextFormatter
from tabulardelta.formatters.formatter import Formatter
from tabulardelta.formatters.overview_row_formatter import OverviewRowFormatter
from tabulardelta.tabulardelta import TabularDelta

try:
    from tabulardelta.comparators.sqlcompyre_comparator import SqlCompyreComparator
except ImportError:
    SqlCompyreComparator = Any  # type: ignore

try:
    from tabulardelta.comparators.sqlmetadata_comparator import SqlMetadataComparator
except ImportError:
    SqlMetadataComparator = Any  # type: ignore

__all__ = [
    "TabularDelta",
    "Formatter",
    "DetailedTextFormatter",
    "OverviewRowFormatter",
    "Comparator",
    "SqlCompyreComparator",
    "SqlMetadataComparator",
    "PandasComparator",
]

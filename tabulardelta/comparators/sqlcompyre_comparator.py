# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar
from warnings import warn

import pandas as pd
import sqlalchemy as sa
import sqlcompyre as sc
from sqlcompyre.analysis import TableComparison

from tabulardelta.comparators.tabulardelta_dataclasses import (
    ColumnPair,
    TabularDelta,
)


@dataclass(frozen=True)
class SqlCompyreComparator:
    """Implements :class:`Comparator` protocol for comparing SQL Tables.

    This is mostly a wrapper for SQLCompyre's table comparison.

    Methods
    -------
    compare(self, old: sa.Table | str, new: sa.Table | str) -> TabularDelta:
        Compare two SQL tables
    """

    engine: sa.Engine
    """SQLAlchemy engine to connect to the database."""
    join_columns: list[str] | None = None
    """Columns to join the two tables on.

    If None, SQLCompyre will try to infer them.
    """
    row_samples: int = 10
    """Number of example rows to sample."""
    value_samples: int = 50
    """Number of example value changes to sample."""
    float_precision: float = 1e-6
    """Relative precision for comparing floats."""
    warning_threshold: int = 10000
    """Minimal number of rows to warn about when comparing tables."""

    def compare(self, old: sa.Table | str, new: sa.Table | str) -> TabularDelta:
        """Compare two SQL tables.

        Arguments:
            old :class:`sa.Table` | :class:`str`:
                The old table (first table to compare).
            new :class:`sa.Table` | :class:`str`:
                The new table (second table to compare).

        Returns :class:`TabularDelta`:
            Metadata and results of the comparison.
        """
        try:
            old, new = _sanitize_input(
                self.engine, [old, new], self.join_columns or [], self.warning_threshold
            )
        except ValueError as e:
            return TabularDelta.from_errors([str(e)])

        def table_comparison(ignore_columns=None, renaming=None) -> TableComparison:
            config = ignore_columns, renaming, self.float_precision, None, False, True
            return sc.compare_tables(self.engine, old, new, self.join_columns, *config)

        unfiltered_comp = table_comparison()
        incomparable_cols = _incomparable_columns(unfiltered_comp)
        mapping = _get_renames(unfiltered_comp)
        comp = table_comparison(incomparable_cols, mapping)

        comparable_cols = comp.column_matches.mismatch_selects.keys()
        cols, rows = comp.column_names, comp.row_matches
        left, right = comp.left_table, comp.right_table

        removed = [(left.c[c], None) for c in cols.missing_right if c not in mapping]
        added = [
            (None, right.c[c]) for c in cols.missing_left if c not in mapping.values()
        ]
        join = [(left.c[c], right.c[c], True) for c in comp.join_columns]
        uncompared = [ColumnPair.from_sqlalchemy(*c) for c in removed + added + join]
        incomparable = [
            _get_value_change(unfiltered_comp, name, False, self.value_samples)
            for name in incomparable_cols
        ]
        comparable = [
            _get_value_change(comp, n, True, self.value_samples)
            for n in comparable_cols
        ]
        added_rows = _get_sample(self.engine, rows.unjoined_right, self.row_samples)
        removed_rows = _get_sample(self.engine, rows.unjoined_left, self.row_samples)

        return TabularDelta(
            new if isinstance(new, str) else getattr(right.original, "name", ""),
            old if isinstance(old, str) else getattr(left.original, "name", ""),
            _columns=incomparable + comparable + uncompared,
            _old_rows=comp.row_counts.left,
            _new_rows=comp.row_counts.right,
            _added_rows=comp.row_counts.right - rows.n_joined_total,
            _removed_rows=comp.row_counts.left - rows.n_joined_total,
            _equal_rows=rows.n_joined_equal,
            _unequal_rows=rows.n_joined_total - rows.n_joined_equal,
            _example_added_rows=added_rows,
            _example_removed_rows=removed_rows,
        )


def _split_name(name: str) -> tuple[str | None, str]:
    """Splits a name into (optional) schema and table name."""
    split = [part[1:-1] if part[0] == "[" else part for part in name.rsplit(".")]
    return (".".join(split[:-1]), ".".join(split)) if len(split) > 1 else (None, name)


def _sanitize_input(
    engine: sa.Engine,
    tables: list[sa.Table | str],
    cols: list[str],
    warning_rows: int,
) -> list[sa.Table]:
    """Checks whether tables and join columns are present in the database. Warns if
    tables are large.

    Return SqlAlchemy Tables and supports table names in square brackets. Otherwise,
    SqlCompyre would fail for table names in square brackets.
    """
    result = []
    for table in tables:
        if isinstance(table, str):
            schema, table = _split_name(table)
            meta = sa.MetaData()
            try:
                meta.reflect(bind=engine, schema=schema, views=True)
            except sa.exc.OperationalError:
                raise ValueError(f"Schema '{schema}' not found in database.")
            if table not in meta.tables:
                raise ValueError(f"Table '{table}' not found in database.")
            table = meta.tables[table]

        limited = sa.select(sa.literal(1)).select_from(table).limit(warning_rows)
        with engine.connect() as conn:
            count_query = sa.select(sa.func.count()).select_from(limited.subquery())
            if conn.execute(count_query).scalar() == warning_rows:
                print(f"Comparing over {warning_rows} rows. This may take a while.")

            for col in cols:
                try:
                    conn.execute(sa.select(sa.column(col)).select_from(table).limit(0))
                except sa.exc.ProgrammingError:
                    raise ValueError(f"Column '{col}' missing in table '{table.name}'.")
        result.append(table)
    return result


def _get_sample(engine: sa.Engine, query: sa.Select, limit=10) -> list[dict[str, Any]]:
    """Get first rows from a query, each as mapping from column name to value."""
    with engine.connect() as conn:
        result = conn.execute(query.limit(limit))
        return [dict(zip(result.keys(), row)) for row in result.all()]


def _get_value_change(
    tab_comp: TableComparison, name: str, comparable: bool = False, limit: int = 50
) -> ColumnPair:
    """Compare columns and ValueChange object with sample of changes.

    Uses SqlCompyre Mismatch-Query if columns are comparable (handles floats correctly).
    Uses direct join if incomparable, otherwise SqlCompyre would crash.
    """
    new_name = tab_comp.column_name_mapping.get(name, name)
    old_name = name + ("_old" if new_name == name else "")
    original_new = tab_comp.right_table.c[new_name]
    original_old = tab_comp.left_table.c[name]
    if comparable:
        diffs: sa.FromClauseRole = tab_comp.column_matches.mismatch_selects[name]
        cols = diffs.c
        new_col = diffs.c[new_name + ("_1" if new_name == name else "")].label(new_name)
    else:
        cols = tab_comp.left_table.c
        _join = [cols[c] == tab_comp.right_table.c[c] for c in tab_comp.join_columns]
        diffs = tab_comp.left_table.join(tab_comp.right_table, sa.and_(*_join))
        new_col = tab_comp.right_table.c[new_name]

    count = sa.func.count("*").label("_count")
    old_col = cols[name].label(old_name)
    join_cols = [sa.func.min(cols[c]).label(c) for c in tab_comp.join_columns]
    select = sa.select(new_col, old_col, count, *join_cols).select_from(diffs)
    query = select.group_by(new_col, old_col).order_by(count.desc()).limit(limit)

    total = min(tab_comp.row_counts.left, tab_comp.row_counts.right)
    try:
        with tab_comp.engine.connect() as conn:
            total = conn.execute(sa.select(count).select_from(diffs)).scalar() or total
        result = pd.read_sql(query, tab_comp.engine).astype("object")
        result.loc[result.shape[0], "_count"] = total - result["_count"].sum()
    except (sa.exc.ProgrammingError, sa.exc.DataError):
        warn(f"Couldn't get value change for {name}.")
        result = pd.DataFrame({old_name: None, new_name: None, "_count": [total]})
    return ColumnPair.from_sqlalchemy(
        original_old, original_new, False, not comparable, result
    )


T = TypeVar("T")


def _sa_raises(func: Callable[..., T], *args, **kwargs) -> tuple[bool, T | None]:
    """Check if a SqlAlchemy function raises an error, return result if not."""
    try:
        return False, func(*args, **kwargs)
    except (sa.exc.ProgrammingError, sa.exc.DataError):
        return True, None


def _incomparable_columns(tab_comp: TableComparison) -> list[str]:
    """Return column type changes and list of incomparable columns to ignore."""
    in_common, get_changes = tab_comp.column_names.in_common, tab_comp.get_top_changes
    return [n for n in in_common if _sa_raises(get_changes, n, n=1)[0]]


def _columns_equal(
    tab_comp: TableComparison, old_col_name: str, new_col_name: str
) -> bool:
    """Returns whether two columns are exactly equal."""
    left, right = tab_comp.left_table, tab_comp.right_table
    if type(left.c[old_col_name].type) is not type(right.c[new_col_name].type):  # noqa
        return False
    join = [left.c[col] == right.c[col] for col in tab_comp.join_columns]
    join.append(left.c[old_col_name] != right.c[new_col_name])
    query = sa.select("*").select_from(left.join(right, sa.and_(*join)))
    with tab_comp.engine.connect() as conn:
        error, result = _sa_raises(lambda: conn.execute(query).first())
        return not error and result is None


def _get_renames(tab_comp: TableComparison) -> dict[str, str]:
    """Returns Renamed columns and one-to-one mapping for SqlCompyre.

    Columns are only considered renamed if they are exactly equal.
    """
    only_old = set(tab_comp.column_names.missing_right)
    only_new = set(tab_comp.column_names.missing_left)
    renamed = {}
    for l_col in only_old:
        for r_col in only_new:
            if _columns_equal(tab_comp, l_col, r_col):
                renamed[l_col] = r_col
                break
        only_new.discard(renamed.get(l_col, ""))
    return renamed

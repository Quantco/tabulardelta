# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo


from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import sqlalchemy as sa

from tabulardelta.comparators.tabulardelta_dataclasses import (
    Column,
    ColumnPair,
    TabularDelta,
)

RowSelect = sa.Select[tuple[int, str, str]]
ColumnSelect = sa.Select[tuple[str, str, str, str]]


def get_mssql_metadata(db_name: str | None) -> tuple[RowSelect, ColumnSelect]:
    """Implementation for retrieving metadata from a Microsoft SQL Server database.

    Arguments:
        db_name :class:`str` | :code:`None`:
            Name of the database to retrieve metadata from.

    Returns :class:`tuple[RowSelect, ColumnSelect]`:
        SQL queries to retrieve row count and column information.
    """
    db_prefix = f"[{db_name}]." if db_name else ""
    rows = sa.select(
        sa.column("rows", sa.Integer),
        sa.column("table", sa.String),
        sa.column("schema", sa.String),
    ).select_from(
        sa.text(
            f"(SELECT max(i.rowcnt) as rows, o.name as [table], s.name as [schema] FROM "
            f"{db_prefix}sys.sysindexes i, {db_prefix}sys.objects o, {db_prefix}sys.schemas s "
            "WHERE i.id = o.object_id AND o.schema_id = s.schema_id GROUP BY o.name, s.name) tmp"
        )
    )
    columns = sa.select(
        sa.column("COLUMN_NAME", sa.String).label("column"),
        sa.column("DATA_TYPE", sa.String).label("type"),
        sa.column("TABLE_NAME", sa.String).label("table"),
        sa.column("TABLE_SCHEMA", sa.String).label("schema"),
    ).select_from(sa.text(f"{db_prefix}INFORMATION_SCHEMA.COLUMNS"))
    return rows, columns


@dataclass(frozen=True)
class _TableDescr:
    """Represents a table in a database using database, schema and table name."""

    db: str | None
    schema: str | None
    table: str

    @staticmethod
    def from_table(engine: sa.engine.base.Engine, table: sa.Table | str) -> _TableDescr:
        # Get string from sa.Table
        if not isinstance(table, str):
            table = getattr(table, "original", table)
            if not hasattr(table, "schema") or not hasattr(table, "name"):
                raise ValueError("SqlAlchemy object does not have schema/table name.")
            table = ".".join([part for part in [table.schema, table.name] if part])

        # Get database, schema and table name from string
        parts = [part[1:-1] if part[0] == "[" else part for part in table.split(".")]
        if len(parts) == 1:
            return _TableDescr(engine.url.database, None, table)
        if len(parts) == 2:
            return _TableDescr(engine.url.database, parts[0], parts[1])
        if len(parts) == 3:
            return _TableDescr(parts[0], parts[1], parts[2])
        raise ValueError(
            f"Table must be specified as '[[<db>.]<schema>.]<table>', not {table}."
        )

    def differing_names(self, other: _TableDescr) -> tuple[str, str]:
        """Returns the names of both tables, disregards equal prefixes."""
        self_parts = [self.db, self.schema, self.table]
        other_parts = [other.db, other.schema, other.table]
        for i, (old_part, new_part) in enumerate(zip(self_parts, other_parts)):
            if old_part != new_part:
                names = filter(None, self_parts[i:]), filter(None, other_parts[i:])
                return ".".join(names[0]), ".".join(names[1])
        return "", ""


@dataclass
class _TableMetadata:
    """Stores row count and column information for a table."""

    rows: int = 0
    cols: dict[str, Column] = field(default_factory=dict)


@dataclass
class SqlMetadataComparator:
    """Implements :class:`Comparator` protocol for comparing SQL Tables.

    This will only use metadata from the database, to be as fast as possible.

    Methods
    -------
    compare(self, old: sa.Table | str, new: sa.Table | str) -> TabularDelta:
        Compare two SQL tables
    """

    engine: sa.Engine
    """SQLAlchemy engine to connect to the database."""
    cache_db_metadata: bool = False
    """Whether to cache metadata for whole database to speed up comparisons."""
    dialects = {"mssql": get_mssql_metadata}
    """Dictionary of supported dialects and their metadata retrieval functions."""

    _cache: dict[_TableDescr, _TableMetadata] = field(
        default_factory=lambda: defaultdict(_TableMetadata)
    )

    def __post_init__(self):
        """Check that the dialect is supported."""
        if self.engine.dialect.name not in self.dialects:
            raise ValueError(f"Unsupported dialect: {self.engine.dialect.name}")

    def _cached_metadata(self, params: _TableDescr) -> _TableMetadata:
        """Return metadata if cached, otherwise fetches metadata for whole database."""
        if params in self._cache:
            return self._cache[params]
        Rows, Columns = self.dialects[self.engine.dialect.name](params.db)
        with self.engine.connect() as conn:
            for rows, table, schema in conn.execute(Rows).fetchall():
                self._cache[_TableDescr(params.db, schema, table)].rows = rows
            for name, dtype, table, schema in conn.execute(Columns).fetchall():
                descr = _TableDescr(params.db, schema, table)
                self._cache[descr].cols[name] = Column(name, dtype)
            return self._cache[params]

    @staticmethod
    def _filter(query, table: str, schema: str | None):
        """Where clause for queries to filter by table and optional schema."""
        if schema is None:
            return query.c["table"] == table
        return (query.c["table"] == table) & (query.c["schema"] == schema)

    def _uncached_metadata(self, params: _TableDescr) -> _TableMetadata:
        """Fetches metadata for specific TableDescription."""
        Rows, Columns = self.dialects[self.engine.dialect.name](params.db)
        with self.engine.connect() as conn:
            row_filter = self._filter(Rows, params.table, params.schema)
            rows = conn.execute(sa.select(Rows.c["rows"]).where(row_filter)).scalar()
            col_filter = self._filter(Columns, params.table, params.schema)
            col_result = conn.execute(
                sa.select(Columns.c["column"], Columns.c["type"]).where(col_filter)
            ).fetchall()
            if not isinstance(rows, int):
                rows = 0
            cols = {name: Column(name, dtype) for name, dtype in col_result}
            return _TableMetadata(rows, cols)

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
        old_descr = _TableDescr.from_table(self.engine, old)
        new_descr = _TableDescr.from_table(self.engine, new)

        get_metadata = self._uncached_metadata
        if self.cache_db_metadata:
            get_metadata = self._cached_metadata
        old_meta, new_meta = get_metadata(old_descr), get_metadata(new_descr)

        cols: list[tuple[Column, Column | None] | tuple[None, Column]]
        cols = [(old_meta.cols[n], new_meta.cols.get(n, None)) for n in old_meta.cols]
        cols += [(None, c) for n, c in new_meta.cols.items() if n not in old_meta.cols]

        return TabularDelta(
            *old_descr.differing_names(new_descr),
            warnings=["No value comparison, just metadata analysis."],
            _columns=[ColumnPair(*col_pair) for col_pair in cols],
            _old_rows=old_meta.rows,
            _new_rows=new_meta.rows,
            _added_rows=max(0, new_meta.rows - old_meta.rows),
            _removed_rows=max(0, old_meta.rows - new_meta.rows),
        )

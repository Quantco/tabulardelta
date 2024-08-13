# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import click
import pandas as pd
import polars as pl
import pytest
import sqlalchemy
from click.testing import CliRunner
from pydiverse import pipedag  # type: ignore
from pydiverse.pipedag import (  # type: ignore
    Flow,
    Stage,
    input_stage_versions,
    materialize,
)
from pydiverse.pipedag.core.config import create_basic_pipedag_config  # type: ignore
from pydiverse.pipedag.util.structlog import setup_logging  # type: ignore

from tabulardelta import (
    DetailedTextFormatter,
    OverviewRowFormatter,
    PandasComparator,
    SqlCompyreComparator,
    SqlMetadataComparator,
    TabularDelta,
)
from tabulardelta.formatters.tabulartext.cell import Cell
from tabulardelta.formatters.tabulartext.table import Border, Table

sa = sqlalchemy


try:
    from tests.mssql_container.cached_mssql_container import (
        MsSqlContainer,
        cached_clean_mssql_container,
        mssql_engine,
    )
except ImportError:

    @pytest.fixture
    def mssql_engine() -> Any:
        return "SqlAlchemy or pyodbc not installed."


def test_df_example():
    df_old = df_new = pd.DataFrame()
    # df_old, df_new = ...

    delta = PandasComparator().compare(df_old, df_new)
    print(DetailedTextFormatter().format(delta))


def test_csv_example():
    assets_path = Path(__file__).parent / "test_assets"

    df_old = pd.read_csv(assets_path / "week24.csv", index_col=[0, 1])
    df_new = pd.read_csv(assets_path / "week25.csv", index_col=[0, 1])

    delta = PandasComparator().compare(df_old, df_new)
    print(DetailedTextFormatter().format(delta))


@pytest.mark.sql
def test_sql_cli_example(mssql_engine: sqlalchemy.Engine):
    @click.command()
    @click.argument("old_table")
    @click.argument("new_table")
    @click.argument("join_columns", nargs=-1)
    def compare_sql(old_table: str, new_table: str, join_columns: list[str]):
        engine = mssql_engine
        # engine: sqlalchemy.Engine = ...

        delta = SqlCompyreComparator(engine, join_columns).compare(old_table, new_table)
        print(DetailedTextFormatter().format(delta))

    ################################## TEST EXECUTION ##################################
    with mssql_engine.connect() as conn:
        with conn.begin():
            conn.execute(sqlalchemy.text("CREATE SCHEMA archive"))
            conn.execute(sqlalchemy.text("CREATE SCHEMA master"))
    first = pd.DataFrame(
        {
            "activity": range(123, 892),
            "registration": pd.to_datetime("2024-07-18"),
            "valid_from": pd.Timestamp(year=2000, month=1, day=1),
            "valid_until": pd.Timestamp(year=9999, month=12, day=31),
        }
    )
    second = pd.DataFrame(
        {
            "activity": range(123, 903),
            "registration_date": pd.to_datetime("2024-07-18"),
            "free_days_until_registered": 0,
            "valid_from": pd.Timestamp(year=2000, month=1, day=1),
            "valid_until": [pd.Timestamp(year=2024, month=6, day=28)] * 12
            + [pd.Timestamp(year=9999, month=12, day=31)] * 768,
        }
    )
    first.to_sql("registration", mssql_engine, index=False, schema="archive")
    second.to_sql("registration", mssql_engine, index=False, schema="master")

    result = CliRunner().invoke(
        compare_sql,
        ["archive.registration", "master.registration", "activity", "valid_from"],
    )
    assert result.stdout == (
        "---------------------- TabularDelta Report: registration -> registration ----------------------\n"
        + "                                                                                               \n"
        + "Joined on ['activity', 'valid_from'].                                                          \n"
        + "                                                                                               \n"
        + "Added columns:                                                                                 \n"
        + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                                   \n"
        + "┃free_days_until_registered┃                                                                   \n"
        + "┃(BIGINT)                  ┃                                                                   \n"
        + "┣━━━━━━━━━━━━━━━━━━━━━━━━━━┫                                                                   \n"
        + "                                                                                               \n"
        + "Renamed columns:                                                                               \n"
        + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                             \n"
        + "┃registration → registration_date┃                                                             \n"
        + "┃(DATETIME)                      ┃                                                             \n"
        + "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫                                                             \n"
        + "                                                                                               \n"
        + "   Old         New                                                                             \n"
        + "   769         780                                                                             \n"
        + "┏━┯━┯━┯━┓╌╌╌┏━┯━┯━┯━┓----------------╶╮                                                        \n"
        + "┃ │ │ │ ┃ = ┃ │ │ │ ┃  757 identical  │ 769                                                    \n"
        + "┠─┼─┼─┼─┨╌╌╌┠─┼─┼─┼─┨---------------- ├╴rows                                                   \n"
        + "┃ │ │ │ ┃ ≠ ┃ │ │ │ ┃  12 changed     │ joined                                                 \n"
        + "┗━┷━┷━┷━┛╌╌╌┠─┼─┼─┼─┨----------------╶╯                                                        \n"
        + "            ┃+│+│+│+┃  11 added                                                                \n"
        + "            ┗━┷━┷━┷━┛                                                                          \n"
        + "                                                                                               \n"
        + "Column valid_until - 12 rows changed:                                                          \n"
        + "             valid_until          →  valid_until          example_activity  example_valid_from \n"
        + "      (12x)  9999-12-31 00:00:00  →  2024-06-28 00:00:00  123               2000-01-01 00:00:00\n"
        + "                                                                                               \n"
        + "ADDED ROWS EXAMPLES:                                                                           \n"
        + "activity│valid_from         │registration_date  │valid_until                                   \n"
        + "892     │2000-01-01 00:00:00│2024-07-18 00:00:00│9999-12-31 00:00:00                           \n"
        + "893     │2000-01-01 00:00:00│2024-07-18 00:00:00│9999-12-31 00:00:00                           \n"
        + "894     │2000-01-01 00:00:00│2024-07-18 00:00:00│9999-12-31 00:00:00                           \n"
    )


ENABLE_BUG = False


@materialize
def data_load():
    data = {
        "id": [1, 2, 3],
        "codes": ["A01", "B02", "A01\t"],
        "dates": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-03"]),
    }
    return pipedag.Table(pd.DataFrame(data), name="data_load")


@materialize(input_type=pd.DataFrame)
def clean(data):
    # No cleaning necessary
    return pipedag.Table(data, name="clean")


@materialize(input_type=pd.DataFrame)
def weekday_feature(data):
    data["weekday"] = data["dates"].dt.weekday
    return pipedag.Table(data[["id", "weekday"]], name="weekday_feature")


@materialize(input_type=pd.DataFrame)
def code_features(data):
    data["code"] = data["codes"].str.strip(" " if ENABLE_BUG else None)
    for code in data["code"].unique():
        data[f"code_{code}"] = data["codes"].str.contains(code)
    result = data[["id"] + [f"code_{code}" for code in data["code"].unique()]]
    return pipedag.Table(result, name="code_features")


@materialize(input_type=pd.DataFrame)
def feature_merge(weekday, codes):
    return pipedag.Table(weekday.merge(codes, on="id"), name="feature_merge")


@materialize(input_type=pd.DataFrame)
def filter(data):
    # No cleaning necessary
    return pipedag.Table(data, name="filter")


@materialize(input_type=pd.DataFrame)
def export_dataframe(df):
    print(df)


def row_count(tbl: sa.Table):
    with MsSqlContainer().get_cache().sqlalchemy_engine().connect() as conn:
        return conn.execute(sa.select(sa.func.count()).select_from(tbl)).scalar()


def col_count(tbl: sa.Table):
    return len(tbl.columns)


@input_stage_versions(input_type=sqlalchemy.Table)
def validate_stage(
    tbls: dict[str, sqlalchemy.Table], other_tbls: dict[str, sqlalchemy.Table]
):
    comparator = SqlMetadataComparator(MsSqlContainer().get_cache().sqlalchemy_engine())
    formatter = OverviewRowFormatter(warnings=False)
    formatter.add_header()

    for common in set(tbls) & set(other_tbls):
        delta = comparator.compare(tbls[common], other_tbls[common])
        formatter.format(delta)

    for added in set(tbls) - set(other_tbls):
        delta = comparator.compare(tbls[added], tbls[added])
        formatter.add_row("+", added, len(delta.rows.new), len(delta.cols.new))

    formatter.add_legend()
    print(formatter.table())


@pytest.mark.sql
def test_pipedag_example():
    url = cached_clean_mssql_container().sqlalchemy_url()
    cfg = create_basic_pipedag_config(url).get("default")
    setup_logging()

    with Flow("flow") as flow:
        with Stage("Features"):
            data = data_load()
            cleaned = clean(data)
            weekday = weekday_feature(cleaned)
            codes = code_features(cleaned)
            merge = feature_merge(weekday, codes)
            filtered = filter(merge)
            export_dataframe(filtered)
            validate_stage()

    flow.run(config=cfg)
    global ENABLE_BUG
    ENABLE_BUG = True
    flow.run(config=cfg)


@dataclass
class PolarsComparator:
    join_columns: list[str] | None = None
    name: str = ""

    def compare(self, left: pl.DataFrame, right: pl.DataFrame) -> TabularDelta:
        comparator = PandasComparator(self.join_columns, self.name)
        return comparator.compare(left.to_pandas(), right.to_pandas())


def test_polars_comparator():
    expected = pl.DataFrame({"id": [1, 2, 3]})
    observed = pl.DataFrame({"id": [1, 2, 3, 4], "data_column": [1, 2, 3, 4]})

    delta = PolarsComparator(join_columns=["id"]).compare(expected, observed)
    assert len(delta.rows.added) == 1
    assert len(delta.cols.added) == 1
    assert len(delta.rows.equal) == 3


def test_tabular_text():
    row1 = ["Just", "first", "row", Cell(["Two", "Rows"], rowspan=2)]
    row2 = [Cell(["Second\nRow"], colspan=3), "BottomRight"]

    print("\nDefault border:")
    print(Table.from_lists([row1, row2]).to_string())

    print("\nDIY border:")
    inner_border = Border(",,,,  | ,,  + ,  + ,, ---,,  + ,  + ,,  + ,,,,")
    print(Table.from_lists([row1, row2], inner_border).to_string())

    assert Table.from_lists([row1, row2]).to_string() == (
        "┏━━━━┯━━━━━┯━━━┯━━━━┯━━━━━━━━━━━┓\n"
        + "┃Just│first│row│Two │           ┃\n"
        + "┠────┴─────┴───┤Rows├───────────┨\n"
        + "┃Second        │    │BottomRight┃\n"
        + "┃Row           │    │           ┃\n"
        + "┗━━━━━━━━━━━━━━┷━━━━┷━━━━━━━━━━━┛"
    )
    assert Table.from_lists([row1, row2], inner_border).to_string() == (
        "Just | first | row | Two  |            \n"
        + "---- + ----- + --- + Rows + -----------\n"
        + "Second             |      | BottomRight\n"
        + "Row                |      |            "
    )


class AssertingFormatter:
    @staticmethod
    def format(delta: TabularDelta) -> None:
        type_chgs = chain.from_iterable(
            (delta.cols.comparable_type_changed, delta.cols.incomparable_type_changed)
        )
        changes = [
            [f"Removed column {c.name}" for c in delta.cols.removed],
            [f"Added column {c.name}" for c in delta.cols.added],
            [f"Renamed {c.old.name} -> {c.new.name}" for c in delta.cols.renamed],
            [f"{c.new.name} changed: {c.old.type} -> {c.new.type}" for c in type_chgs],
            ["New rows"] * (len(delta.rows.added) > 0),
            ["Missing rows"] * (len(delta.rows.removed) > 0),
            ["Changed rows"] * (len(delta.rows.unequal) > 0),
        ]
        if flat_changes := list(chain.from_iterable(changes)):
            raise AssertionError("\n - ".join(flat_changes))


def test_example():
    expected = pd.DataFrame({"id": [1, 2, 3]})
    # observed = ...
    observed = pd.DataFrame({"key": [1, 2, 3, 4], "data": ["a", "b", "c", "d"]})

    delta = PandasComparator().compare(expected, expected)
    AssertingFormatter.format(delta)
    delta = PandasComparator().compare(observed, observed)
    AssertingFormatter.format(delta)

    try:
        delta = PandasComparator().compare(expected, observed)
        AssertingFormatter.format(delta)
        raise AssertionError("No Assertion Error Raised")
    except AssertionError as e:
        assert str(e) == "Added column data\n - Renamed id -> key\n - New rows"

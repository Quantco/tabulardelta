# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo
from typing import Any, NamedTuple

import pytest
from pandas.testing import assert_frame_equal

import tabulardelta.comparators.pandas_dataclasses as pd_dc
from tabulardelta import (
    DetailedTextFormatter,
    PandasComparator,
    SqlCompyreComparator,
    SqlMetadataComparator,
)
from tabulardelta.comparators.native_dataclasses import (
    Column,
    ColumnPair,
)

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pass

try:
    from tests.mssql_container.cached_mssql_container import mssql_engine
except ImportError:

    @pytest.fixture
    def mssql_engine() -> Any:
        return "SqlAlchemy or pyodbc not installed."


try:
    import sqlalchemy as sa
except ImportError:
    sa = NamedTuple("sa", [("Engine", Any)])  # type: ignore # noqa: UP014


def rec_approx(a: Any, b: Any):
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            rec_approx(x, y)
    elif isinstance(a, dict) and isinstance(b, dict):
        assert len(a) == len(b)
        for k in a:
            assert k in b
            rec_approx(a[k], b[k])
    else:
        assert a == pytest.approx(b)


def old_df():
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "equal": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "paid": [
                "yes",
                "no",
                "yes",
                "maybe",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
            ],
            "unnecessary": [0] * 10,
            "name": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "measurement": [[val] * 3 for val in np.arange(0.1, 1.1, 0.1)],
            "expectation": [0.12, 0.24, 0.36, 0.48, 0.5, 0.68, 0.76, 0.84, 0.92, 1.0],
        }
    )


def new_df():
    df = old_df()
    # Swap first two rows
    df.iloc[0], df.iloc[1] = (df.iloc[1], df.iloc[0].copy())
    # Remove unnecessary column
    df.drop(columns=["unnecessary"], inplace=True)
    # Uncomparable dtype change: str -> bool
    df["paid"] = df["paid"] == "yes"
    # Rename measurements
    df.rename(columns={"measurement": "renamedmeasurement"}, inplace=True)
    # Results are average of measurements
    df["results"] = df["renamedmeasurement"].apply(lambda x: sum(x) / len(x))
    # Rename last row
    df.loc[9, "name"] = "Jess"
    # Add new row
    df.loc[10] = [10, 0, True, "Karl", [1.0, 1.1, 1.2], 0.9, 1.1]
    # Change expectation value for row 4
    df.loc[4, "expectation"] = 0.55
    # Change expectations to float32
    df["expectation"] = df["expectation"].astype("float32")
    # Change id to float64
    df["id"] = df["id"].astype("float64")
    # Create a second results column as float32
    df["second_result"] = df["results"].astype("float32")
    return df


def old_df_rows():
    return [
        {
            "id": 0,
            "equal": 0,
            "paid": "yes",
            "unnecessary": 0,
            "name": "A",
            "measurement": "[0.1, 0.1, 0.1]",
            "expectation": 0.12,
        },
        {
            "id": 1,
            "equal": 0,
            "paid": "no",
            "unnecessary": 0,
            "name": "B",
            "measurement": "[0.2, 0.2, 0.2]",
            "expectation": 0.24,
        },
        {
            "id": 2,
            "equal": 0,
            "paid": "yes",
            "unnecessary": 0,
            "name": "C",
            "measurement": "[0.3, 0.3, 0.3]",
            "expectation": 0.36,
        },
        {
            "id": 3,
            "equal": 0,
            "paid": "maybe",
            "unnecessary": 0,
            "name": "D",
            "measurement": "[0.4, 0.4, 0.4]",
            "expectation": 0.48,
        },
        {
            "id": 4,
            "equal": 0,
            "paid": "yes",
            "unnecessary": 0,
            "name": "E",
            "measurement": "[0.5, 0.5, 0.5]",
            "expectation": 0.5,
        },
        {
            "id": 5,
            "equal": 0,
            "paid": "no",
            "unnecessary": 0,
            "name": "F",
            "measurement": "[0.6, 0.6, 0.6]",
            "expectation": 0.68,
        },
        {
            "id": 6,
            "equal": 0,
            "paid": "yes",
            "unnecessary": 0,
            "name": "G",
            "measurement": "[0.7, 0.7, 0.7]",
            "expectation": 0.76,
        },
        {
            "id": 7,
            "equal": 0,
            "paid": "no",
            "unnecessary": 0,
            "name": "H",
            "measurement": "[0.8, 0.8, 0.8]",
            "expectation": 0.84,
        },
        {
            "id": 8,
            "equal": 0,
            "paid": "yes",
            "unnecessary": 0,
            "name": "I",
            "measurement": "[0.9, 0.9, 0.9]",
            "expectation": 0.92,
        },
        {
            "id": 9,
            "equal": 0,
            "paid": "no",
            "unnecessary": 0,
            "name": "J",
            "measurement": "[1.0, 1.0, 1.0]",
            "expectation": 1.0,
        },
    ]


def new_df_rows():
    return [
        {
            "id": 1.0,
            "equal": 0,
            "paid": False,
            "name": "B",
            "renamedmeasurement": "[0.2, 0.2, 0.2]",
            "expectation": 0.24,
            "results": 0.2,
            "second_result": 0.2,
        },
        {
            "id": 0.0,
            "equal": 0,
            "paid": True,
            "name": "A",
            "renamedmeasurement": "[0.1, 0.1, 0.1]",
            "expectation": 0.12,
            "results": 0.1,
            "second_result": 0.1,
        },
        {
            "id": 2.0,
            "equal": 0,
            "paid": True,
            "name": "C",
            "renamedmeasurement": "[0.3, 0.3, 0.3]",
            "expectation": 0.36,
            "results": 0.3,
            "second_result": 0.3,
        },
        {
            "id": 3.0,
            "equal": 0,
            "paid": False,
            "name": "D",
            "renamedmeasurement": "[0.4, 0.4, 0.4]",
            "expectation": 0.48,
            "results": 0.4,
            "second_result": 0.4,
        },
        {
            "id": 4.0,
            "equal": 0,
            "paid": True,
            "name": "E",
            "renamedmeasurement": "[0.5, 0.5, 0.5]",
            "expectation": 0.55,
            "results": 0.5,
            "second_result": 0.5,
        },
        {
            "id": 5.0,
            "equal": 0,
            "paid": False,
            "name": "F",
            "renamedmeasurement": "[0.6, 0.6, 0.6]",
            "expectation": 0.68,
            "results": 0.6,
            "second_result": 0.6,
        },
        {
            "id": 6.0,
            "equal": 0,
            "paid": True,
            "name": "G",
            "renamedmeasurement": "[0.7, 0.7, 0.7]",
            "expectation": 0.76,
            "results": 0.7,
            "second_result": 0.7,
        },
        {
            "id": 7.0,
            "equal": 0,
            "paid": False,
            "name": "H",
            "renamedmeasurement": "[0.8, 0.8, 0.8]",
            "expectation": 0.84,
            "results": 0.8,
            "second_result": 0.8,
        },
        {
            "id": 8.0,
            "equal": 0,
            "paid": True,
            "name": "I",
            "renamedmeasurement": "[0.9, 0.9, 0.9]",
            "expectation": 0.92,
            "results": 0.9,
            "second_result": 0.9,
        },
        {
            "id": 9.0,
            "equal": 0,
            "paid": False,
            "name": "Jess",
            "renamedmeasurement": "[1.0, 1.0, 1.0]",
            "expectation": 1.0,
            "results": 1.0,
            "second_result": 1.0,
        },
        {
            "id": 10.0,
            "equal": 0,
            "paid": True,
            "name": "Karl",
            "renamedmeasurement": "[1.0, 1.1, 1.2]",
            "expectation": 0.9,
            "results": 1.1,
            "second_result": 1.1,
        },
    ]


@pytest.mark.pandas
def test_pandas_comparator():
    delta = PandasComparator(["name"]).compare(old_df(), new_df())

    print(delta)

    assert delta.warnings == []
    assert delta.errors == []
    assert delta.cols.joined == [
        pd_dc.ColumnPair(
            pd_dc.Column("name", "object"), pd_dc.Column("name", "object"), join=True
        )
    ]
    assert delta.cols.removed == [pd_dc.Column("unnecessary", "int64")]
    assert set(delta.cols.added) == {
        pd_dc.Column("results", "float64"),
        pd_dc.Column("second_result", "float32"),
    }
    assert delta.cols.renamed == [
        pd_dc.ColumnPair(
            pd_dc.Column("measurement", "object"),
            pd_dc.Column("renamedmeasurement", "object"),
        )
    ]
    comparable = [chg for chg in delta.cols.comparable_type_changed]
    assert len(comparable) == 1
    assert comparable[0].old == pd_dc.Column("expectation", "float64")
    assert comparable[0].new == pd_dc.Column("expectation", "float32")
    assert all(chg.old and chg.new for chg in delta.cols.incomparable_type_changed)
    incomparable_dtype_dict = {
        chg.new.name: (
            chg.old.name,
            chg.new.name,
            chg.old.type,
            chg.new.type,
            chg._values,
        )
        for chg in delta.cols.incomparable_type_changed
        if chg.old and chg.new
    }

    assert "paid" in incomparable_dtype_dict
    assert incomparable_dtype_dict["paid"][2] == "object"
    assert incomparable_dtype_dict["paid"][3] == "bool"
    assert isinstance(incomparable_dtype_dict["paid"][4], pd.DataFrame)
    cols = incomparable_dtype_dict["paid"][4].columns
    assert "name" in cols
    assert "_count" in cols
    assert incomparable_dtype_dict["paid"][0] in cols
    assert incomparable_dtype_dict["paid"][1] in cols

    assert "id" in incomparable_dtype_dict
    assert incomparable_dtype_dict["id"][2] == "int64"
    assert incomparable_dtype_dict["id"][3] == "float64"
    assert isinstance(incomparable_dtype_dict["id"][4], pd.DataFrame)
    cols = incomparable_dtype_dict["id"][4].columns
    assert "name" in cols
    assert "_count" in cols
    assert incomparable_dtype_dict["id"][0] in incomparable_dtype_dict["id"][4].columns
    assert incomparable_dtype_dict["id"][1] in incomparable_dtype_dict["id"][4].columns

    assert len(delta.rows.old) == 10
    assert len(delta.rows.new) == 11
    assert len(delta.rows.equal) == 8
    assert len(delta.rows.removed) == 1
    assert len(delta.rows.added) == 2
    assert len(delta.rows.unequal) == 1

    added = {row["id"]: row for row in delta.rows.added}
    rec_approx(
        added[9.0],
        {
            "id": 9.0,
            "equal": 0.0,
            "paid": False,
            "name": "Jess",
            "renamedmeasurement": [1.0, 1.0, 1.0],
            "expectation": 1.0,
            "results": 1.0,
            "second_result": 1.0,
        },
    )
    rec_approx(
        added[10.0],
        {
            "id": 10.0,
            "equal": 0.0,
            "paid": True,
            "name": "Karl",
            "renamedmeasurement": [1.0, 1.1, 1.2],
            "expectation": 0.9,
            "results": 1.1,
            "second_result": 1.1,
        },
    )
    rec_approx(
        list(delta.rows.removed),
        [
            {
                "id": 9.0,
                "equal": 0.0,
                "paid": "no",
                "unnecessary": 0.0,
                "name": "J",
                "measurement": [1.0, 1.0, 1.0],
                "expectation": 1.0,
            }
        ],
    )

    actual_differences = [diff for diff in delta.cols.differences if len(diff) > 0]
    assert len(actual_differences) == 1
    assert actual_differences[0].new and actual_differences[0].new.name == "expectation"
    df = actual_differences[0]._values
    assert df is not None
    assert actual_differences[0].old and actual_differences[0].old.name in df.columns
    assert actual_differences[0].new and actual_differences[0].new.name in df.columns
    expected_df = pd.DataFrame(
        {
            "expectation_old": [0.5],
            "expectation": [0.55],
            "_count": [1],
            "name": ["E"],
        }
    )
    assert_frame_equal(df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.pandas
def test_pandas_comparator_row_col_orders():
    test_dfs = [  # (DataFrame, Order <1=RowChanged, 2=ColChanged>)
        (pd.DataFrame({"id": [1, 2], "col": ["a", "b"]}), 0),
        (pd.DataFrame({"id": [1, 2], "col": ["a", "b"], "extra": ["x", "y"]}), 0),
        (pd.DataFrame({"id": [1, 2, 3], "col": ["a", "b", "c"]}), 0),
        (pd.DataFrame({"id": [2, 1], "col": ["b", "a"]}), 1),
        (pd.DataFrame({"col": ["a", "b"], "id": [1, 2]}), 2),
        (pd.DataFrame({"col": ["b", "a"], "id": [2, 1]}), 3),
    ]
    tests = [(o_df, n_df, o ^ n) for o_df, o in test_dfs for n_df, n in test_dfs]
    for old_df, new_df, expected_order in tests:
        delta = PandasComparator(["id"]).compare(old_df, new_df)
        assert ("Row Order Changed!" in delta.info) == expected_order % 2
        assert ("Column Order Changed!" in delta.info) == expected_order // 2
        print(DetailedTextFormatter().format(delta))


@pytest.mark.sql
@pytest.mark.parametrize(
    "input_type", ["table_str", "schema_table_str", "bracket_table_str", "sa_table"]
)
def test_sqlcompyre_comparator(mssql_engine: sa.Engine, input_type: str):
    metadata = sa.MetaData()
    first_table = sa.Table(
        "first",
        metadata,
        sa.Column("id", sa.BigInteger()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.String()),
        sa.Column("unnecessary", sa.BigInteger()),
        sa.Column("name", sa.String()),
        sa.Column("measurement", sa.String()),
        sa.Column("expectation", sa.Float()),
        schema="dbo",
    )
    second_table = sa.Table(
        "second",
        metadata,
        sa.Column("id", sa.Float()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.Boolean()),
        sa.Column("name", sa.String()),
        sa.Column("renamedmeasurement", sa.String()),
        sa.Column("expectation", sa.Float(24)),
        sa.Column("results", sa.Float()),
        sa.Column("second_result", sa.Float(24)),
        schema="dbo",
    )
    metadata.create_all(mssql_engine)
    with mssql_engine.begin() as conn:
        conn.execute(sa.insert(first_table), old_df_rows())
        conn.execute(sa.insert(second_table), new_df_rows())

    old: str | sa.Table
    new: str | sa.Table
    if input_type == "table_str":
        old, new = "first", "second"
    elif input_type == "schema_table_str":
        old, new = "dbo.first", "dbo.second"
    elif input_type == "bracket_table_str":
        old, new = "[dbo].[first]", "[dbo].[second]"
    elif input_type == "sa_table":
        meta = sa.MetaData()
        meta.reflect(mssql_engine)
        old, new = meta.tables["first"], meta.tables["second"]
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    delta = SqlCompyreComparator(mssql_engine, ["name"]).compare(old, new)

    print(delta)

    assert delta.warnings == []
    assert delta.errors == []
    assert delta.cols.joined == [
        ColumnPair(
            Column("name", 'VARCHAR COLLATE "SQL_Latin1_General_CP1_CI_AS"'),
            Column("name", 'VARCHAR COLLATE "SQL_Latin1_General_CP1_CI_AS"'),
            True,
        )
    ]
    assert delta.cols.removed == [Column("unnecessary", "BIGINT")]
    assert set(delta.cols.added) == {
        Column("results", "FLOAT"),
        Column("second_result", "REAL"),
    }
    renamed = delta.cols.renamed
    assert len(renamed) == 1
    assert len(renamed[0]) == 0
    assert renamed[0].old == Column(
        "measurement", 'VARCHAR COLLATE "SQL_Latin1_General_CP1_CI_AS"'
    )
    assert renamed[0].new == Column(
        "renamedmeasurement", 'VARCHAR COLLATE "SQL_Latin1_General_CP1_CI_AS"'
    )
    assert all(chg.old for chg in delta.cols.comparable_type_changed)
    changes = {
        chg.old.name: chg for chg in delta.cols.comparable_type_changed if chg.old
    }
    assert changes.keys() == {"id", "expectation"}
    assert changes["id"].old == Column("id", "BIGINT")
    assert changes["id"].new == Column("id", "FLOAT")
    assert changes["expectation"].old == Column("expectation", "FLOAT")
    assert changes["expectation"].new == Column("expectation", "REAL")
    assert all(chg.old and chg.new for chg in delta.cols.incomparable_type_changed)
    incomparable_dtype_dict = {
        chg.new.name: (
            chg.old.name,
            chg.new.name,
            chg.old.type,
            chg.new.type,
            chg._values,
        )
        for chg in delta.cols.incomparable_type_changed
        if chg.old and chg.new
    }

    assert "paid" in incomparable_dtype_dict
    assert (
        incomparable_dtype_dict["paid"][2]
        == 'VARCHAR COLLATE "SQL_Latin1_General_CP1_CI_AS"'
    )
    assert incomparable_dtype_dict["paid"][3] == "BIT"
    assert incomparable_dtype_dict["paid"][4] is not None
    assert len(incomparable_dtype_dict["paid"][4]) > 0
    cols = incomparable_dtype_dict["paid"][4][0].keys()
    assert "name" in cols
    assert "_count" in cols
    assert incomparable_dtype_dict["paid"][0] in cols
    assert incomparable_dtype_dict["paid"][1] in cols

    assert len(delta.rows.old) == 10
    assert len(delta.rows.new) == 11

    assert len(delta.rows.equal) == 8
    assert len(delta.rows.removed) == 1
    assert len(delta.rows.added) == 2
    assert len(delta.rows.unequal) == 1

    added = {row["id"]: row for row in delta.rows.added}
    rec_approx(
        added[9.0],
        {
            "id": 9.0,
            "equal": 0.0,
            "name": "Jess",
            "renamedmeasurement": "[1.0, 1.0, 1.0]",
            "expectation": 1.0,
        },
    )
    rec_approx(
        added[10.0],
        {
            "id": 10.0,
            "equal": 0.0,
            "name": "Karl",
            "renamedmeasurement": "[1.0, 1.1, 1.2]",
            "expectation": 0.9,
        },
    )
    rec_approx(
        list(delta.rows.removed),
        [
            {
                "id": 9.0,
                "equal": 0.0,
                "name": "J",
                "measurement": "[1.0, 1.0, 1.0]",
                "expectation": 1.0,
            }
        ],
    )

    actual_differences = [diff for diff in delta.cols.differences if len(diff) > 0]
    assert len(actual_differences) == 1
    assert actual_differences[0].new
    assert actual_differences[0].old
    assert actual_differences[0].new.name == "expectation"
    diff_values = actual_differences[0]._values
    assert diff_values is not None
    assert len(diff_values) > 0
    assert actual_differences[0].old.name in diff_values[0].keys()
    assert actual_differences[0].new.name in diff_values[0].keys()
    expected_diff = [
        {
            "expectation": 0.55,
            "expectation_old": 0.5,
            "_count": 1,
            "name": "E",
        }
    ]
    rec_approx(diff_values, expected_diff)


@pytest.mark.sql
@pytest.mark.parametrize(
    "input_type", ["table_str", "schema_table_str", "bracket_table_str", "sa_table"]
)
def test_sqlmetadata_comparator(mssql_engine: sa.Engine, input_type: str):
    metadata = sa.MetaData()
    first_table = sa.Table(
        "first",
        metadata,
        sa.Column("id", sa.BigInteger()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.String()),
        sa.Column("unnecessary", sa.BigInteger()),
        sa.Column("name", sa.String()),
        sa.Column("measurement", sa.String()),
        sa.Column("expectation", sa.Float()),
        schema="dbo",
    )
    second_table = sa.Table(
        "second",
        metadata,
        sa.Column("id", sa.Float()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.Boolean()),
        sa.Column("name", sa.String()),
        sa.Column("renamedmeasurement", sa.String()),
        sa.Column("expectation", sa.Float(24)),
        sa.Column("results", sa.Float()),
        sa.Column("second_result", sa.Float(24)),
        schema="dbo",
    )
    metadata.create_all(mssql_engine)
    with mssql_engine.begin() as conn:
        conn.execute(sa.insert(first_table), old_df_rows())
        conn.execute(sa.insert(second_table), new_df_rows())

    old: str | sa.Table
    new: str | sa.Table
    if input_type == "table_str":
        old, new = "first", "second"
    elif input_type == "schema_table_str":
        old, new = "dbo.first", "dbo.second"
    elif input_type == "bracket_table_str":
        old, new = "[dbo].[first]", "[dbo].[second]"
    elif input_type == "sa_table":
        meta = sa.MetaData()
        meta.reflect(mssql_engine)
        old, new = meta.tables["first"], meta.tables["second"]
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    delta = SqlMetadataComparator(mssql_engine).compare(old, new)

    print(delta)

    assert delta.warnings == ["No value comparison, just metadata analysis."]
    assert delta.errors == []
    assert delta.cols.joined == []
    assert set(delta.cols.removed) == {
        Column("unnecessary", "bigint"),
        Column("measurement", "varchar"),
    }
    assert set(delta.cols.added) == {
        Column("results", "float"),
        Column("second_result", "real"),
        Column("renamedmeasurement", "varchar"),
    }
    assert delta.cols.renamed == []
    comparable = delta.cols.comparable_type_changed
    assert len(comparable) == 3
    assert comparable[0].old == Column("id", "bigint")
    assert comparable[0].new == Column("id", "float")
    assert comparable[1].old == Column("paid", "varchar")
    assert comparable[1].new == Column("paid", "bit")
    assert comparable[2].old == Column("expectation", "float")
    assert comparable[2].new == Column("expectation", "real")

    assert [chg for chg in delta.cols.incomparable_type_changed] == []

    assert len(delta.rows.old) == 10
    assert len(delta.rows.new) == 11

    assert len(delta.rows.equal) == 0
    assert len(delta.rows.removed) == 0
    assert len(delta.rows.added) == 1
    assert len(delta.rows.unequal) == 0

    assert delta.cols.differences == []


@pytest.mark.sql
def test_sqlmetadata_comparator_cache(mssql_engine: sa.Engine):
    metadata = sa.MetaData()
    first_table = sa.Table(
        "first",
        metadata,
        sa.Column("id", sa.BigInteger()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.String()),
        sa.Column("unnecessary", sa.BigInteger()),
        sa.Column("name", sa.String()),
        sa.Column("measurement", sa.String()),
        sa.Column("expectation", sa.Float()),
        schema="dbo",
    )
    second_table = sa.Table(
        "second",
        metadata,
        sa.Column("id", sa.Float()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.Boolean()),
        sa.Column("name", sa.String()),
        sa.Column("renamedmeasurement", sa.String()),
        sa.Column("expectation", sa.Float(24)),
        sa.Column("results", sa.Float()),
        sa.Column("second_result", sa.Float(24)),
        schema="dbo",
    )
    metadata.create_all(mssql_engine)
    with mssql_engine.begin() as conn:
        conn.execute(sa.insert(first_table), old_df_rows())
        conn.execute(sa.insert(second_table), new_df_rows())

    # Create a comparison with activated caching
    cache_comp = SqlMetadataComparator(mssql_engine, cache_db_metadata=True)
    original_delta = cache_comp.compare("first", "second")

    # Change (switch) the tables in the database
    first_table.drop(bind=mssql_engine, checkfirst=True)
    second_table.drop(bind=mssql_engine, checkfirst=True)
    first_renamed_table = sa.Table(
        "second",
        metadata,
        sa.Column("id", sa.BigInteger()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.String()),
        sa.Column("unnecessary", sa.BigInteger()),
        sa.Column("name", sa.String()),
        sa.Column("measurement", sa.String()),
        sa.Column("expectation", sa.Float()),
        schema="dbo",
        extend_existing=True,
    )
    second_renamed_table = sa.Table(
        "first",
        metadata,
        sa.Column("id", sa.Float()),
        sa.Column("equal", sa.Integer()),
        sa.Column("paid", sa.Boolean()),
        sa.Column("name", sa.String()),
        sa.Column("renamedmeasurement", sa.String()),
        sa.Column("expectation", sa.Float(24)),
        sa.Column("results", sa.Float()),
        sa.Column("second_result", sa.Float(24)),
        schema="dbo",
        extend_existing=True,
    )
    metadata.create_all(mssql_engine)
    with mssql_engine.begin() as conn:
        conn.execute(sa.insert(first_renamed_table), old_df_rows())
        conn.execute(sa.insert(second_renamed_table), new_df_rows())

    # New comparison should be different
    changed_delta = SqlMetadataComparator(mssql_engine).compare("first", "second")
    assert original_delta != changed_delta

    # Cached comparison should be equal to original
    cached_delta = cache_comp.compare("first", "second")
    assert cached_delta == original_delta

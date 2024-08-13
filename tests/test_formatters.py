# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy import (
    dtype,
    ndarray,
)

from tabulardelta import (
    DetailedTextFormatter,
    OverviewRowFormatter,
)
from tabulardelta.comparators.tabulardelta_dataclasses import (
    Column,
    ColumnPair,
    TabularDelta,
)


def get_random_tabulardelta(gen: np.random.Generator):
    def gen_str() -> str:
        return "".join(gen.choice(list("abcdefghijklmnopqrstuvwxyz"), size=6))

    def gen_dtype(exclude: str | None = None) -> str:
        dtypes = ["int64", "float64", "object", "bool"]
        if exclude:
            dtypes = [dtype for dtype in dtypes if dtype != exclude]
        return gen.choice(dtypes)

    def gen_col_meta() -> Column:
        return Column(f"col_{gen_str()}", gen_dtype())

    def gen_col_values(dtype: str, size: int) -> ndarray[Any, dtype[Any]]:
        if dtype == "int64":
            return gen.normal(0, 10 ** gen.integers(0, 10), size).astype("int")
        if dtype == "float64":
            return gen.normal(0, 10 ** gen.integers(0, 10), size)
        if dtype == "object":
            return np.array([gen_str() for _ in range(size)])
        if dtype == "bool":
            return gen.choice([True, False], size)
        if dtype == "uint64":
            return gen.integers(0, 10 ** gen.integers(0, 8), size) + 1
        raise NotImplementedError(f"Random data for {dtype}")

    def gen_rows(cols: int, rows: int) -> list[dict[str, Any]]:
        columns = [gen_str() for _ in range(cols)]
        types = [gen_dtype() for _ in range(cols)]
        values = [[gen_col_values(t, rows)[0] for t in types] for _ in range(rows)]
        return [dict(zip(columns, value)) for value in values]

    def gen_chg_values(
        chg: ColumnPair, join_columns: list[Column], incomparable: bool = False
    ) -> ColumnPair:
        if chg.old is None or chg.new is None:
            raise ValueError("Can't generate values for non-matched columns.")
        size = gen.integers(0, 2 ** gen.integers(0, 10))
        old_renamed = Column(
            chg.old.name + ("_old" if chg.old.name == chg.new.name else ""),
            chg.old.type,
        )
        indexes = {col.name: gen_col_values(col.type, size) for col in join_columns}
        df = pd.DataFrame(
            {
                old_renamed.name: gen_col_values(chg.old.type, size),
                chg.new.name: gen_col_values(chg.new.type, size),
                **indexes,
                "_count": gen_col_values("uint64", size).astype("int"),
            }
        )
        if gen.random() < 0.5:
            additional = gen.integers(0, 10 ** gen.integers(0, 8))
            df.loc[len(df)] = [None] * (2 + len(join_columns)) + [additional]
            df["_count"] = df["_count"].astype("int")
        return ColumnPair(chg.old, chg.new, incomparable=incomparable, _values=df)

    def gen_change() -> ColumnPair:
        old = gen_col_meta()
        new = old
        if gen.random() < 0.1:  # RENAMED COLUMN
            new = Column(f"col_{gen_str()}", new.type)
        if gen.random() < 0.1:  # CHANGED DTYPE
            new = Column(new.name, gen_dtype(new.type))
        return ColumnPair(old, new)

    strings = ["", "short", "LongRepeatingString" * 50]
    name = gen.choice(strings)
    warnings = list(gen.choice(strings, int(gen.exponential(3))))

    common = [gen_change() for _ in range(int(gen.exponential(10)))]
    added_cols = [gen_col_meta() for _ in range(int(gen.exponential(5)))]
    removed_cols = [gen_col_meta() for _ in range(int(gen.exponential(5)))]
    join_len = gen.integers(0, len(common) + 1)
    join_cols: list[Column] = list(
        gen.choice(np.array([chg.new for chg in common]), join_len, False)
    )
    dtype_changes_raw = [
        chg for chg in common if chg.old and chg.new and chg.old.type != chg.new.type
    ]
    uncomp_candidates = [chg for chg in dtype_changes_raw if chg.new not in join_cols]
    uncomp_len = gen.integers(0, len(uncomp_candidates) + 1)
    uncomp = gen.choice(np.array(uncomp_candidates), uncomp_len, False)
    type_changes: list[ColumnPair] = [
        (
            gen_chg_values(chg, join_cols, True)
            if (len(uncomp) > 0 and chg in uncomp)
            else chg
        )
        for chg in dtype_changes_raw
    ]

    diff_candidates: list[ColumnPair] = [
        chg
        for chg in common
        if (len(uncomp) == 0 or chg not in uncomp) and chg.new not in join_cols
    ]
    diff_len = gen.integers(0, len(diff_candidates) + 1)
    diff_cols = gen.choice(np.array(diff_candidates), diff_len, False)
    col_differences = [gen_chg_values(chg, join_cols) for chg in diff_cols]

    diff_counts = [len(chg) for chg in col_differences]
    unequal_rows = gen.integers(max(diff_counts, default=0), sum(diff_counts) + 1)

    added_rows = int(gen.exponential(10 ** gen.integers(10)))
    removed_rows = int(gen.exponential(10 ** gen.integers(10)))
    equal_rows = int(gen.exponential(10 ** gen.integers(10)))
    unknown_rows = int(gen.exponential(10 ** gen.integers(10)))
    old_rows = removed_rows + equal_rows + unequal_rows + unknown_rows
    new_rows = added_rows + equal_rows + unequal_rows + unknown_rows

    added = [ColumnPair(None, col) for col in added_cols]
    removed = [ColumnPair(col, None) for col in removed_cols]
    join = [ColumnPair(col, col, True) for col in join_cols]

    return TabularDelta(
        name,
        f"{name}_old" if gen.random() < 0.5 else None,
        [],
        warnings,
        [],
        added + removed + join + type_changes + col_differences,
        old_rows,
        new_rows,
        added_rows,
        removed_rows,
        equal_rows,
        unequal_rows,
        _example_added_rows=gen_rows(
            gen.choice([1, 5, 15]), gen.choice([0, 10, min(added_rows, 100)])
        ),
        _example_removed_rows=gen_rows(
            gen.choice([1, 5, 15]), gen.choice([0, 10, min(removed_rows, 100)])
        ),
    )


def test_random_tabulardelta_smoke():
    print(len(get_random_tabulardelta(np.random.default_rng(42)).rows.unequal))


def test_detailed_text_formatter_smoke(sample_size: int = 100):
    for idx in range(sample_size):
        delta = get_random_tabulardelta(np.random.default_rng(idx))
        try:
            print(DetailedTextFormatter().format(delta))
        except Exception as e:
            print(f"Failed on index {idx}/{sample_size}:")
            print(delta)
            raise e
        if idx % 100 == 0:
            print(f"Completed {idx}/{sample_size}")


def test_row_formatter_smoke(sample_size: int = 10):
    for idx in range(sample_size):
        gen = np.random.default_rng(idx)
        size = gen.integers(0, 20)
        deltas = [get_random_tabulardelta(gen) for _ in range(size)]
        try:
            rf = OverviewRowFormatter(
                gen.choice([True, False]), gen.choice([True, False])
            )
            if gen.choice([True, False]):
                rf.add_header()
            for delta in deltas:
                rf.format(delta)
            if gen.choice([True, False]):
                rf.add_legend()
            print(rf.table())
        except Exception as e:
            print(f"Failed on index {idx}/{sample_size}:")
            print(deltas)
            raise e
        if idx % 10 == 0:
            print(f"Completed {idx}/{sample_size}")

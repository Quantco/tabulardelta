# TabularDelta

[![CI](https://img.shields.io/github/actions/workflow/status/quantco/tabulardelta/ci.yml?style=flat-square&branch=main)](https://github.com/quantco/tabulardelta/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-success?branch=main&style=flat-square)](https://tabulardelta.readthedocs.io/en/latest/)
[![pypi-version](https://img.shields.io/pypi/v/tabulardelta.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/tabulardelta)
[![python-version](https://img.shields.io/pypi/pyversions/tabulardelta?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/tabulardelta)

TabularDelta helps to automate and simplify the often tedious and manual process of comparing relational data.

The so-called TabularDelta protocol defines a representation of the differences between two tables.
"Comparators" are used to generate such a representation from two table objects. The exchangeability of the comparators allows for varying table input formats like SQL tables or Pandas DataFrames.
"Formatters" allow to present the differences in different output formats depending on the desired usecase.
The flexibility in the output format allows to find small deviations in largely similar tables or provide an overview of more structural changes.

## Usage example

This snippet will report the differences of two CSV files.
You can execute it directly in [test_docs_examples.py](tests/test_docs_examples.py).

```python
import pandas as pd
from tabulardelta import PandasComparator, DetailedTextFormatter

df_old = pd.read_csv("week24.csv", index_col=[0, 1])
df_new = pd.read_csv("week25.csv", index_col=[0, 1])

delta = PandasComparator().compare(df_old, df_new)
print(DetailedTextFormatter().format(delta))
```

To compare two tables, first select a comparator that supports the table format. Now select any formatter that best suits your use case to obtain a visualization of the result.

To find more examples and get started, please visit the [documentation](https://tabulardelta.readthedocs.io/en/latest/).

## Development

This project is managed by [pixi](https://pixi.sh).
You can install the package in development mode using:

```bash
git clone https://github.com/quantco/tabulardelta
cd tabulardelta

pixi run pre-commit-install
pixi run postinstall
```

## Testing

- Make sure docker is installed
- Make sure `ODBC Driver 17 for SQL Server` is installed
  - See [Download ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16)
  - This may require setting the `ODBCSYSINI` environment variable to the path of msodbcsql17
- Run `pixi run test`

Setting up the MsSql docker container may take a while, but it will be cached for future runs as long as you keep it running.

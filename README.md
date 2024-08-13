# TabularDelta

[![CI](https://img.shields.io/github/actions/workflow/status/quantco/tabulardelta/ci.yml?style=flat-square&branch=main)](https://github.com/quantco/tabulardelta/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-success?branch=main&style=flat-square)](https://tabulardelta.readthedocs.io/en/latest/)
[![pypi-version](https://img.shields.io/pypi/v/tabulardelta.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/tabulardelta)
[![python-version](https://img.shields.io/pypi/pyversions/tabulardelta?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/tabulardelta)

Simplify table comparisons.

## Development

This project is managed by [pixi](https://pixi.sh).
You can install the package in development mode using:

```bash
git clone https://github.com/quantco/tabulardelta
cd tabulardelta

pixi run pre-commit-install
pixi run postinstall
pixi run test
```

## Testing

- Make sure docker is installed
- Make sure `ODBC Driver 17 for SQL Server` is installed
  - See [Download ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16)
  - This may require setting the `ODBCSYSINI` environment variable to the path of msodbcsql17
- Run `pixi run test`

Setting up the MsSql docker container may take a while, but it will be cached for future runs as long as you keep it running.

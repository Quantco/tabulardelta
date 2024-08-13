# Copyright (c) QuantCo 2022-2024
# SPDX-License-Identifier: LicenseRef-QuantCo
from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pyodbc  # type: ignore
import pytest
import sqlalchemy as sa
from pydantic import BaseModel, ValidationError

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class UnresponsiveMsSqlError(RuntimeError):
    pass


def free_port() -> int:
    """Get a (most likely) free port by closing socket on OS-assigned port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    sock.listen(1)
    _, port = sock.getsockname()
    sock.close()
    return port


class MsSqlContainer(BaseModel):
    """Serializable class representing a mssql docker container."""

    project_name: str = (
        f"tabulardelta_cached-mssql_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    port: int | None = None
    docker_compose_file: Path = Path(__file__).parent / "docker-compose.yml"
    cache_file: Path = Path(__file__).parent / ".mssql_container_cache.json"

    def start(self) -> MsSqlContainer:
        """Start the mssql container in detached mode."""
        self.port = self.port or free_port()
        assert (
            subprocess.Popen(
                f'docker compose -p "{self.project_name}" -f {self.docker_compose_file} up --detach',
                env=os.environ | {"MSSQL_PORT": str(self.port)},
                shell=True,
            ).wait()
            == 0
        )
        return self

    def store_cache(self) -> MsSqlContainer:
        """Store the current MsSqlContainer in the cache file."""
        self.cache_file.write_text(self.model_dump_json())
        return self

    def get_cache(self) -> MsSqlContainer:
        """Get the MsSqlContainer from the cache file."""
        return MsSqlContainer.model_validate_json(self.cache_file.read_text())

    def sqlalchemy_url(self) -> str:
        # If no port is set, try using CI windows connection string:
        host = f"localhost:{self.port}" if self.port else r"(localdb)\.\MSSQLLocalDB"
        return f"mssql+pyodbc://sa:Passw0rd@{host}/master?Encrypt=no&TrustServerCertificate=yes&driver=ODBC+Driver+17+for+SQL+Server"

    def sqlalchemy_engine(self) -> sa.Engine:
        return sa.create_engine(self.sqlalchemy_url())

    def pyodbc_connect(self) -> pyodbc.Connection:
        # If no port is set, try using CI windows connection string:
        host = f"localhost,{self.port}" if self.port else r"(localdb)\.\MSSQLLocalDB"
        pyodbc_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host};DATABASE=master;UID=sa;PWD=Passw0rd;TrustServerCertificate=yes;Encrypt=No;DATABASE=master"
        return pyodbc.connect(pyodbc_str)

    def healthcheck(
        self, max_attempts: int = 3, sleep_interval: float = 3
    ) -> MsSqlContainer:
        t = time.time()
        for attempt in range(max_attempts):
            try:
                logging.debug(f"Connecting to docker mssql, attempt {attempt}")
                cnn = self.pyodbc_connect()
                cursor = cnn.cursor()
                cursor.execute("SELECT 1")
                cnn.commit()
                logging.info(
                    f"Connecting to docker mssql, attempt {attempt} success after {time.time()-t:.2f}s"
                )
                return self
            except Exception:
                logging.debug(f"Connecting to docker db, attempt {attempt} failed")
                time.sleep(sleep_interval)
        logging.error(">" * 20 + " DOCKER COMPOSE LOGS BELOW " + "<" * 20)
        logging.error(
            subprocess.check_output(
                ["docker", "compose", "-p", self.project_name, "logs"]
            ).decode("utf-8")
        )
        logging.error(">" * 20 + " END DOCKER COMPOSE LOGS " + "<" * 20)
        raise UnresponsiveMsSqlError(
            f"Could not connect to temporary db on port :{self.port}\n"
        )

    def reset(self) -> MsSqlContainer:
        conn = self.pyodbc_connect()
        conn.autocommit = True
        cur = conn.cursor()

        # Security check
        if self.port == 1433:
            raise RuntimeError(
                "MISHAP PROTECTION: Let's not reset mssql on default ports."
            )

        # Drop databases
        all_dbs = cur.execute("SELECT name FROM sys.databases").fetchall()
        system_dbs = ["master", "tempdb", "model", "msdb"]
        user_dbs = [row[0] for row in all_dbs if row[0] not in system_dbs]
        if len(user_dbs) > 5:  # Security check
            raise RuntimeError(
                "MISHAP PROTECTION: Let's not drop more than 5 databases."
            )
        for user_db in user_dbs:
            cur.execute(f"DROP DATABASE {user_db}")

        # Drop foreign keys in master
        constraints_query = "SELECT * FROM master.INFORMATION_SCHEMA.TABLE_CONSTRAINTS"
        constraints = cur.execute(constraints_query).fetchall()
        for _, _, constraint, _, schema, table, category, _, _ in constraints:
            if category == "FOREIGN KEY":
                cur.execute(f"ALTER TABLE [{schema}].[{table}] DROP [{constraint}]")

        # Drop tables and views in master
        master_objs = cur.execute(
            "SELECT * FROM master.INFORMATION_SCHEMA.TABLES"
        ).fetchall()
        for _, schema, table, category in master_objs:
            cur.execute(f"DROP {category.split()[-1]} [{schema}].[{table}]")

        # Drop schemas
        system_schemas = {
            "dbo",
            "guest",
            "INFORMATION_SCHEMA",
            "sys",
            "db_owner",
            "db_accessadmin",
            "db_securityadmin",
            "db_ddladmin",
            "db_backupoperator",
            "db_datareader",
            "db_datawriter",
            "db_denydatareader",
            "db_denydatawriter",
        }
        master_schemas = cur.execute(
            "SELECT * FROM master.INFORMATION_SCHEMA.SCHEMATA"
        ).fetchall()
        user_schemas = {s[1] for s in master_schemas if s[1] not in system_schemas}
        for schema in user_schemas:
            cur.execute(f"DROP SCHEMA [{schema}]")

        cur.close()
        conn.close()
        return self


def cached_clean_mssql_container(
    attempts: int = 2,
    cache_file: Path = Path(__file__).parent / ".mssql_container_cache.json",
) -> MsSqlContainer:
    """Get a cached MsSqlContainer from the cache file, or start this one if missing."""
    new_container = MsSqlContainer(cache_file=cache_file)
    for _ in range(attempts):
        try:
            print(new_container.get_cache())
            return new_container.get_cache().healthcheck().reset()
        except (FileNotFoundError, ValidationError, UnresponsiveMsSqlError) as e:
            logging.debug(f"Couldn't get cached MsSql container: {e}")
            new_container.start().store_cache()
    raise UnresponsiveMsSqlError("Failed to start mssql container.")


@pytest.fixture(scope="function")
def mssql_engine() -> Iterator[sa.Engine]:
    yield cached_clean_mssql_container().sqlalchemy_engine()

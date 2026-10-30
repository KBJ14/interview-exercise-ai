"""PostgreSQL connection settings.

DATABASE_URL wins if set. Otherwise the URL is built from POSTGRES_* variables, taken from the
environment or, for local runs, from the repo's .env (the same file docker-compose reads).
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]
_SETTINGS = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_PORT")


def _read_env_file(path: Path) -> dict[str, str]:
    # Minimal KEY=VALUE parser for our own .env; avoids adding python-dotenv for four values.
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def database_url() -> str:
    if url := os.environ.get("DATABASE_URL"):
        return url
    file_values = _read_env_file(REPO_ROOT / ".env")
    settings = {key: os.environ.get(key) or file_values.get(key) for key in _SETTINGS}
    missing = [key for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB") if not settings[key]]
    if missing:
        raise RuntimeError(f"database settings missing: {', '.join(missing)} (set DATABASE_URL or fill .env)")
    return psycopg.conninfo.make_conninfo(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=settings["POSTGRES_PORT"] or "5432",
        user=settings["POSTGRES_USER"],
        password=settings["POSTGRES_PASSWORD"],
        dbname=settings["POSTGRES_DB"],
    )


def connect(**kwargs) -> psycopg.Connection:
    return psycopg.connect(database_url(), **kwargs)

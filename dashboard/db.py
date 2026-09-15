"""Dashboard data access. Reads persisted run state from Postgres/SQLite —
never touches the live event bus.
"""
from __future__ import annotations


def get_connection():
    # TODO: connect to Postgres (infra/docker-compose.yml service) or
    # SQLite, depending on env config.
    raise NotImplementedError


def list_runs() -> list[dict]:
    # TODO: SELECT run_id, symbols, status, started_at, completed_at
    # FROM runs ORDER BY started_at DESC.
    raise NotImplementedError


def get_run(run_id: str) -> dict:
    # TODO: fetch one run's full record, including each agent's output
    # events, for detail view.
    raise NotImplementedError

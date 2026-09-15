"""Postgres persistence for RunStatus and each stage's terminal payload.

Polled by the Streamlit dashboard, not a high-throughput path, so this
stays deliberately simple: two tables, plain SQLAlchemy declarative
models, select-then-write upserts (no dialect-specific ON CONFLICT) so
the same code also runs against SQLite for local testing.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from shared.schemas.events import AgentStatus, RunStatus

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://stocks_agent:stocks_agent@localhost:5432/stocks_agent",
)

Base = declarative_base()
_engine = create_engine(DATABASE_URL, future=True)
_SessionLocal = sessionmaker(bind=_engine, future=True)


class RunStatusRow(Base):
    __tablename__ = "run_status"

    run_id = Column(String, primary_key=True)
    symbols = Column(JSON, nullable=False, default=list)
    started_at = Column(DateTime(timezone=True), nullable=False)
    data_status = Column(JSON, nullable=False)  # {"market": "completed", ...}


class StageResultRow(Base):
    __tablename__ = "stage_results"

    run_id = Column(String, primary_key=True)
    stage = Column(String, primary_key=True)  # research | strategy_proposed | strategy_evaluated
    payload = Column(JSON, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def ping() -> bool:
    try:
        with _engine.connect():
            return True
    except Exception:
        return False


def upsert_run_status(status: RunStatus) -> None:
    data_status = {source: state.value for source, state in status.data_status.items()}
    with _SessionLocal() as session:
        row = session.get(RunStatusRow, status.run_id)
        if row is None:
            session.add(
                RunStatusRow(
                    run_id=status.run_id,
                    symbols=status.symbols,
                    started_at=status.started_at,
                    data_status=data_status,
                )
            )
        else:
            row.symbols = status.symbols
            row.data_status = data_status
        session.commit()


def get_run_status(run_id: str) -> RunStatus | None:
    with _SessionLocal() as session:
        row = session.get(RunStatusRow, run_id)
        if row is None:
            return None
        return RunStatus(
            run_id=row.run_id,
            symbols=row.symbols,
            started_at=row.started_at,
            data_status={source: AgentStatus(state) for source, state in row.data_status.items()},
        )


def save_stage_result(run_id: str, stage: str, payload: dict) -> None:
    with _SessionLocal() as session:
        row = session.get(StageResultRow, (run_id, stage))
        now = datetime.now(timezone.utc)
        if row is None:
            session.add(StageResultRow(run_id=run_id, stage=stage, payload=payload, recorded_at=now))
        else:
            row.payload = payload
            row.recorded_at = now
        session.commit()


def get_stage_result(run_id: str, stage: str) -> dict | None:
    with _SessionLocal() as session:
        row = session.get(StageResultRow, (run_id, stage))
        return row.payload if row else None

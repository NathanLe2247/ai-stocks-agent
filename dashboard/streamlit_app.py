"""Streamlit dashboard. Reads from Postgres/SQLite via db.py, not the live
event bus — the orchestrator's runs are already persisted by the time this
renders them.
"""
from __future__ import annotations

import streamlit as st

from dashboard.db import get_run, list_runs


def main() -> None:
    st.title("Stocks Agent — Runs")

    # TODO: render list_runs() as a table, let the user pick a run_id.
    # TODO: render get_run(run_id) — per-agent outputs, aggregated dataset,
    # final risk-assessed strategy.
    raise NotImplementedError


if __name__ == "__main__":
    main()

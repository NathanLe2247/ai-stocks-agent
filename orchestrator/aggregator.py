"""Fan-in join for the four data-agent completion events.

Plain orchestration logic only — no LLM calls, no agent reasoning of any
kind. It watches four Redis topics, tracks per-source completion for each
run_id in Postgres (via infra.db), and publishes AggregatedDataEvent once
every source has reported in or the join times out.

Join semantics: wait for ALL of market/social/news/fundamentals for a
run_id, or until `timeout_seconds` (default 90) elapses — whichever comes
first. On timeout, any source still PENDING is marked TIMED_OUT and left
out of the aggregated payload rather than blocking the run forever.
"""
from __future__ import annotations

import logging
import threading

from infra import db
from infra.event_bus import publish, subscribe
from shared.schemas.events import AgentStatus, AggregatedDataEvent, BaseEvent, RunStatus, Topic

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 90

SOURCE_TOPICS: dict[str, Topic] = {
    "market": Topic.MARKET_COMPLETED,
    "social": Topic.SOCIAL_COMPLETED,
    "news": Topic.NEWS_COMPLETED,
    "fundamentals": Topic.FUNDAMENTALS_COMPLETED,
}


class Aggregator:
    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds
        self._payloads: dict[str, dict[str, dict]] = {}  # run_id -> source -> payload
        self._watchdogs: dict[str, threading.Timer] = {}  # run_id -> timeout timer
        self._finalized: set[str] = set()  # run_ids already published to AGGREGATED_READY
        self._lock = threading.Lock()

    def start(self) -> None:
        """Subscribe to all four data.*.completed topics."""
        for source, topic in SOURCE_TOPICS.items():
            subscribe(topic.value, self._make_handler(source))

    def _make_handler(self, source: str) -> "callable[[BaseEvent], None]":
        def handler(event: BaseEvent) -> None:
            self._on_data_event(source, event)

        return handler

    def _on_data_event(self, source: str, event: BaseEvent) -> None:
        should_finalize = False
        with self._lock:
            status = db.get_run_status(event.run_id) or RunStatus(run_id=event.run_id)
            status.data_status[source] = AgentStatus.COMPLETED
            if not status.symbols and getattr(event, "symbols", None):
                status.symbols = event.symbols
            db.upsert_run_status(status)
            self._payloads.setdefault(event.run_id, {})[source] = event.payload  # type: ignore[attr-defined]

            if event.run_id not in self._watchdogs and event.run_id not in self._finalized:
                timer = threading.Timer(self.timeout_seconds, self._on_timeout, args=(event.run_id,))
                timer.daemon = True
                self._watchdogs[event.run_id] = timer
                timer.start()

            if status.data_ready and event.run_id not in self._finalized:
                self._finalized.add(event.run_id)
                should_finalize = True

        logger.info("run_id=%s source=%s data_status=%s", event.run_id, source, status.data_status)

        if should_finalize:
            self._finalize(event.run_id, status)

    def _on_timeout(self, run_id: str) -> None:
        with self._lock:
            status = db.get_run_status(run_id)
            if status is None or run_id in self._finalized:
                return
            for source, state in status.data_status.items():
                if state == AgentStatus.PENDING:
                    status.data_status[source] = AgentStatus.TIMED_OUT
            db.upsert_run_status(status)
            self._finalized.add(run_id)

        timed_out = [s for s, st in status.data_status.items() if st == AgentStatus.TIMED_OUT]
        logger.warning("run_id=%s timed out after %ss, missing=%s", run_id, self.timeout_seconds, timed_out)
        self._finalize(run_id, status)

    def _finalize(self, run_id: str, status: RunStatus) -> None:
        with self._lock:
            timer = self._watchdogs.pop(run_id, None)
            payloads = self._payloads.pop(run_id, {})
        if timer:
            timer.cancel()

        timed_out = [s for s, st in status.data_status.items() if st == AgentStatus.TIMED_OUT]
        aggregated = AggregatedDataEvent(
            run_id=run_id,
            symbols=status.symbols,
            market=payloads.get("market"),
            social=payloads.get("social"),
            news=payloads.get("news"),
            fundamentals=payloads.get("fundamentals"),
            timed_out_sources=timed_out,
        )
        publish(Topic.AGGREGATED_READY.value, aggregated)
        logger.info(
            "run_id=%s -> %s (timed_out_sources=%s)", run_id, Topic.AGGREGATED_READY.value, timed_out
        )

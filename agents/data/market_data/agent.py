"""market_data data agent: the bus wrapper around scripts.analyze_ticker.

This module owns only the plumbing — subscribing to analyze.requested,
running the collection with a timeout, persisting the result, and
publishing it. All the actual yfinance/SEC EDGAR/Alpha Vantage/FRED/Gemini
collection logic lives in scripts.analyze_ticker and is imported here, not
duplicated.

Scope note: scripts.analyze_ticker.analyze_ticker() produces the full
six-section report (company analysis, financials, technicals, market
context, future prospects, competitor comparison) — broader than the
market_data-only scope described in skills.md (yfinance + Alpha Vantage
news + FRED). For now this agent's payload is that entire report; the
other three data agents and quant_researcher may need less new work than
skills.md currently implies once they're built. Worth reconciling skills.md
against this once the rest of the pipeline exists.
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Optional

from infra import db
from infra.event_bus import publish, subscribe
from scripts.analyze_ticker import analyze_ticker, get_ticker
from shared.schemas.events import AnalyzeRequestEvent, MarketDataEvent, Topic

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 180.0  # analyze_ticker makes several sequential network calls


class MarketDataAgent:
    AGENT_NAME = "market"

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, max_workers: int = 4) -> None:
        self.timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="market_data")

    def register(self):
        """Subscribe to Topic.ANALYZE_REQUESTED. Call once at pipeline startup."""
        return subscribe(Topic.ANALYZE_REQUESTED.value, self.handle_request)

    # --- request handling ---------------------------------------------------

    def handle_request(self, request: AnalyzeRequestEvent) -> None:
        start = time.monotonic()
        logger.info(
            "run_id=%s agent=%s symbols=%s status=started",
            request.run_id, self.AGENT_NAME, request.symbols,
        )

        event = self._build_result(request.run_id, request.symbols, start)

        self._safe_persist(event)
        self._safe_publish(event)

        logger.info(
            "run_id=%s agent=%s symbols=%s status=%s elapsed=%.1fs error=%s",
            request.run_id, self.AGENT_NAME, request.symbols, event.status,
            event.elapsed_seconds, event.error,
        )

    def _build_result(self, run_id: str, symbols: list[str], start: float) -> MarketDataEvent:
        """Collect every symbol (each in its own thread, each bounded by
        timeout_seconds) and fold the results into one event. No exception
        from any individual symbol's collection escapes this method."""
        payload: dict = {}
        failures: list[str] = []
        data_quality: list[dict] = []

        futures = {sym: self._executor.submit(self._collect_one, sym) for sym in symbols}
        for sym, future in futures.items():
            try:
                result = future.result(timeout=self.timeout_seconds)
                payload[sym] = result
                # Surface analyze_ticker's own per-symbol data_quality onto the
                # event envelope too, so the aggregator/dashboard sees caveats
                # (e.g. a Gemini or SEC EDGAR failure) without digging into payload.
                for issue in result.get("data_quality", []):
                    data_quality.append({**issue, "field": f"{sym}.{issue.get('field')}"})
            except FuturesTimeoutError:
                msg = f"{sym}: timed out after {self.timeout_seconds:.0f}s"
                failures.append(msg)
                data_quality.append({"section": "market_data", "field": sym, "issue": msg, "severity": "missing"})
            except (SystemExit, Exception) as exc:
                msg = f"{sym}: {exc}"
                failures.append(msg)
                data_quality.append({"section": "market_data", "field": sym, "issue": msg, "severity": "missing"})

        elapsed = time.monotonic() - start
        status = "complete" if payload else "failed"
        error: Optional[str] = "; ".join(failures) if failures else None

        return MarketDataEvent(
            run_id=run_id,
            symbols=symbols,
            status=status,
            error=error,
            elapsed_seconds=elapsed,
            data_quality=data_quality,
            payload=payload,
        )

    @staticmethod
    def _collect_one(symbol: str) -> dict:
        ticker, company_name = get_ticker([symbol])  # validates via yfinance; raises SystemExit on a bad symbol
        return analyze_ticker(ticker, company_name)

    # --- persistence / publish, each independently best-effort --------------

    def _safe_persist(self, event: MarketDataEvent) -> None:
        try:
            db.save_agent_result(self.AGENT_NAME, event)
        except Exception:
            logger.exception("run_id=%s agent=%s failed to persist result", event.run_id, self.AGENT_NAME)

    def _safe_publish(self, event: MarketDataEvent) -> None:
        try:
            publish(Topic.MARKET_COMPLETED.value, event)
        except Exception:
            logger.exception("run_id=%s agent=%s failed to publish result", event.run_id, self.AGENT_NAME)


def main() -> None:
    """Standalone test entrypoint — runs this agent's logic for one ticker
    without the orchestrator, the aggregator, or any bus traffic in or out.
    Ticker comes from the CLI arg or an interactive prompt, same as
    scripts.analyze_ticker, via the same get_ticker() call."""
    import uuid

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ticker, company_name = get_ticker()
    run_id = str(uuid.uuid4())
    print(f"[standalone] run_id={run_id} ticker={ticker} ({company_name})", file=sys.stderr)

    agent = MarketDataAgent()
    event = agent._build_result(run_id, [ticker], time.monotonic())

    # Best-effort: exercises the real persist/publish path if Redis/Postgres
    # are up, but a standalone run works fine without them too.
    agent._safe_persist(event)
    agent._safe_publish(event)

    print(event.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

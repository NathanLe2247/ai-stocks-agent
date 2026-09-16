"""Starts the aggregator and every implemented data agent, then waits.

market_data is the first real (non-stub) data agent; social_sentiment,
news, and fundamentals still need to be started here once they exist.
"""
from __future__ import annotations

import logging
import threading

from agents.data.market_data.agent import MarketDataAgent
from infra import db
from infra.event_bus import REDIS_URL
from infra.event_bus import ping as redis_ping
from orchestrator.aggregator import SOURCE_TOPICS, Aggregator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("orchestrator.main")


def main() -> None:
    logger.info("checking Redis connection (%s)...", REDIS_URL)
    redis_ok = redis_ping()
    logger.info("redis: %s", "OK" if redis_ok else "UNREACHABLE")

    logger.info("checking Postgres connection (%s)...", db.DATABASE_URL)
    db_ok = db.ping()
    logger.info("postgres: %s", "OK" if db_ok else "UNREACHABLE")

    if not (redis_ok and db_ok):
        logger.error("cannot start aggregator without both Redis and Postgres reachable")
        raise SystemExit(1)

    db.init_db()
    logger.info("db schema ready (run_status, stage_results, agent_results)")

    aggregator = Aggregator()
    aggregator.start()
    logger.info("aggregator listening on: %s", [topic.value for topic in SOURCE_TOPICS.values()])
    logger.info("will publish to: %s", "data.aggregated.ready")

    market_data_agent = MarketDataAgent()
    market_data_agent.register()
    logger.info("market_data agent listening on: analyze.requested")

    logger.info("orchestrator up — waiting for events (Ctrl+C to stop)")

    threading.Event().wait()


if __name__ == "__main__":
    main()

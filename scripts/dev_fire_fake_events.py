"""Dev helper: fire fake data.*.completed events for one run_id so you can
watch the aggregator join them (or time out) with no real agents running.

Run the full join (all 4 sources -> data.aggregated.ready fires immediately):
    python -m scripts.dev_fire_fake_events

Run the timeout path (only 3 sources -> data.aggregated.ready fires after
the aggregator's timeout, default 90s, with fundamentals in timed_out_sources):
    python -m scripts.dev_fire_fake_events --sources market,social,news
"""
from __future__ import annotations

import argparse
import time
import uuid

from infra.event_bus import publish
from shared.schemas.events import (
    DATA_SOURCES,
    FundamentalsDataEvent,
    MarketDataEvent,
    NewsDataEvent,
    SocialDataEvent,
    Topic,
)

EVENT_TYPES = {
    "market": (MarketDataEvent, Topic.MARKET_COMPLETED),
    "social": (SocialDataEvent, Topic.SOCIAL_COMPLETED),
    "news": (NewsDataEvent, Topic.NEWS_COMPLETED),
    "fundamentals": (FundamentalsDataEvent, Topic.FUNDAMENTALS_COMPLETED),
}

FAKE_PAYLOADS = {
    "market": {"prices": {"AAPL": 227.15}, "note": "dev fake"},
    "social": {"sentiment_score": 0.62, "note": "dev fake"},
    "news": {"headlines": ["dev fake headline"], "note": "dev fake"},
    "fundamentals": {"pe_ratio": 31.4, "note": "dev fake"},
}


def fire(run_id: str, symbols: list[str], sources: list[str], delay: float) -> None:
    print(f"run_id={run_id} sources={sources} delay={delay}s")
    for source in sources:
        event_cls, topic = EVENT_TYPES[source]
        event = event_cls(run_id=run_id, symbols=symbols, payload=FAKE_PAYLOADS[source])
        publish(topic.value, event)
        print(f"  published {topic.value}")
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=str(uuid.uuid4()), help="defaults to a fresh uuid")
    parser.add_argument("--symbols", default="AAPL,MSFT")
    parser.add_argument(
        "--sources",
        default=",".join(DATA_SOURCES),
        help="comma-separated subset of market,social,news,fundamentals "
        "(send fewer than 4 to exercise the aggregator's timeout path)",
    )
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between each publish")
    args = parser.parse_args()

    fire(
        run_id=args.run_id,
        symbols=args.symbols.split(","),
        sources=args.sources.split(","),
        delay=args.delay,
    )


if __name__ == "__main__":
    main()

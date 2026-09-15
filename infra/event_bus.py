"""Thin wrapper around Redis Pub/Sub for the event bus.

Pub/Sub vs Streams, briefly: Streams buy durability and replay — a
subscriber that starts late (or reconnects) can still read everything it
missed, via consumer groups and XACK. Pub/Sub has none of that: a
subscriber only sees messages published *after* it subscribes, and a
dropped connection silently loses whatever was in flight. For this pass —
one long-lived aggregator listening while short-lived agent/dev-script
processes publish — that limitation is acceptable, and Pub/Sub keeps this
wrapper to two functions instead of stream IDs and consumer-group
bookkeeping. Start the aggregator (orchestrator.main) before firing any
events. Revisit Streams if agents ever need to publish before the
aggregator is guaranteed to be running, or if replay/debugging history
becomes important.

Events always serialize/deserialize through the Pydantic models in
shared.schemas.events — publish() takes a model instance, and subscribe()
hands the handler a parsed model instance, never a raw dict.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import redis
from pydantic import BaseModel

from shared.schemas.events import TOPIC_EVENT_TYPES

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_redis_client: "redis.Redis | None" = None


def _client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def publish(topic: str, event: BaseModel) -> None:
    """Publish a Pydantic event as JSON on `topic`."""
    _client().publish(topic, event.model_dump_json())


def subscribe(topic: str, handler: Callable[[BaseModel], None]) -> threading.Thread:
    """Run `handler(event)` for every message on `topic`, in a background
    daemon thread; returns that thread so callers can track/join it.

    The raw JSON is parsed into whichever Pydantic model
    shared.schemas.events.TOPIC_EVENT_TYPES registers for `topic` before
    the handler ever sees it.
    """
    event_type = TOPIC_EVENT_TYPES[topic]
    pubsub = _client().pubsub()
    pubsub.subscribe(topic)

    def _listen() -> None:
        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                handler(event_type.model_validate_json(message["data"]))
            except Exception:
                logger.exception("handler failed for topic=%s", topic)

    thread = threading.Thread(target=_listen, daemon=True, name=f"subscriber:{topic}")
    thread.start()
    return thread


def ping() -> bool:
    try:
        return bool(_client().ping())
    except redis.RedisError:
        return False

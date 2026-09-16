"""Pydantic event models and topic names for the trading pipeline event bus.

    data.market.completed        \
    data.social.completed         |-- aggregator joins all 4 (or times out)
    data.news.completed            |        |
    data.fundamentals.completed   /         v
                                       data.aggregated.ready
                                              |
                                              v  quant_researcher (reads ONLY this)
                                       research.completed
                                              |
                                              v  trading_strategist
                                       strategy.proposed
                                              |
                                              v  risk_manager
                                       strategy.evaluated

Every event carries a run_id so the aggregator (and the dashboard) can
join/look up everything belonging to the same orchestration run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Topic(str, Enum):
    """Redis pub/sub topic for each event type."""

    ANALYZE_REQUESTED = "analyze.requested"
    MARKET_COMPLETED = "data.market.completed"
    SOCIAL_COMPLETED = "data.social.completed"
    NEWS_COMPLETED = "data.news.completed"
    FUNDAMENTALS_COMPLETED = "data.fundamentals.completed"
    AGGREGATED_READY = "data.aggregated.ready"
    RESEARCH_COMPLETED = "research.completed"
    STRATEGY_PROPOSED = "strategy.proposed"
    STRATEGY_EVALUATED = "strategy.evaluated"


class BaseEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    run_id: str
    timestamp: datetime = Field(default_factory=_now)
    source: str


# --- Trigger event ---------------------------------------------------------

class AnalyzeRequestEvent(BaseEvent):
    """Published once per run to kick off the four parallel data agents.
    Always carries an explicit symbol list supplied by a human (CLI/prompt/
    dashboard) upstream of this event — no agent picks its own tickers."""

    source: str = "orchestrator"
    symbols: list[str]


# --- Data-agent completion events ---------------------------------------
#
# Common envelope every data agent reports through, regardless of source:
# whether it succeeded, how long it took, and any data-quality caveats on
# its payload. This is what the aggregator's join and the dashboard's trace
# view both read, so all four data-agent events share it via DataAgentEvent
# rather than redefining these fields four times.

class DataAgentEvent(BaseEvent):
    symbols: list[str]
    status: str = "complete"  # "complete" | "failed"
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    data_quality: list[dict] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)


class MarketDataEvent(DataAgentEvent):
    # payload shape: whatever scripts.analyze_ticker.analyze_ticker() returns,
    # keyed by symbol — see that module for the six-section schema.
    source: str = "market"


class SocialDataEvent(DataAgentEvent):
    source: str = "social"
    # payload: dict  # TODO: shape once X/Reddit/EODHD fields are settled


class NewsDataEvent(DataAgentEvent):
    source: str = "news"
    # payload: dict  # TODO: shape once Bloomberg/Finnhub/Reddit/Reuters fields are settled


class FundamentalsDataEvent(DataAgentEvent):
    source: str = "fundamentals"
    # payload: dict  # TODO: shape once profile/financials/insider fields are settled


# --- Aggregator output ---------------------------------------------------

class AggregatedDataEvent(BaseEvent):
    """Emitted once the data-agent join for run_id resolves: either all
    four sources completed, or the timeout elapsed and any still-pending
    sources were marked timed out (their field is left None here)."""

    source: str = "aggregator"
    symbols: list[str]
    market: Optional[dict] = None
    social: Optional[dict] = None
    news: Optional[dict] = None
    fundamentals: Optional[dict] = None
    timed_out_sources: list[str] = Field(default_factory=list)


# --- Downstream agent output ---------------------------------------------

class ResearchEvent(BaseEvent):
    source: str = "quant_researcher"
    payload: dict  # TODO: define quant research output schema


class StrategyProposedEvent(BaseEvent):
    source: str = "trading_strategist"
    payload: dict  # TODO: define strategy/signal output schema


class StrategyEvaluatedEvent(BaseEvent):
    source: str = "risk_manager"
    payload: dict  # TODO: define risk assessment output schema


# --- Fan-in tracking (RunStatus) ------------------------------------------

class AgentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"


DATA_SOURCES: tuple[str, ...] = ("market", "social", "news", "fundamentals")


class RunStatus(BaseModel):
    """Per-run_id fan-in state for the four data agents. Owned by the
    aggregator, persisted via infra.db, read by the dashboard."""

    run_id: str
    symbols: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)
    data_status: dict[str, AgentStatus] = Field(
        default_factory=lambda: {source: AgentStatus.PENDING for source in DATA_SOURCES}
    )

    @property
    def data_ready(self) -> bool:
        """True once every source has COMPLETED. False if any source is
        still PENDING or has TIMED_OUT."""
        return all(status == AgentStatus.COMPLETED for status in self.data_status.values())


# --- Topic <-> event type maps --------------------------------------------

EVENT_TOPICS: dict[type[BaseEvent], Topic] = {
    AnalyzeRequestEvent: Topic.ANALYZE_REQUESTED,
    MarketDataEvent: Topic.MARKET_COMPLETED,
    SocialDataEvent: Topic.SOCIAL_COMPLETED,
    NewsDataEvent: Topic.NEWS_COMPLETED,
    FundamentalsDataEvent: Topic.FUNDAMENTALS_COMPLETED,
    AggregatedDataEvent: Topic.AGGREGATED_READY,
    ResearchEvent: Topic.RESEARCH_COMPLETED,
    StrategyProposedEvent: Topic.STRATEGY_PROPOSED,
    StrategyEvaluatedEvent: Topic.STRATEGY_EVALUATED,
}

TOPIC_EVENT_TYPES: dict[str, type[BaseEvent]] = {
    topic.value: event_type for event_type, topic in EVENT_TOPICS.items()
}

# The four data-agent event types the aggregator joins per run_id, keyed
# in DATA_SOURCES order.
DATA_EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    MarketDataEvent,
    SocialDataEvent,
    NewsDataEvent,
    FundamentalsDataEvent,
)

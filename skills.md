# Agent skills reference

One entry per agent in the pipeline: what it's responsible for, what it
reads and emits, and what it's allowed to touch. All seven are currently
`# TODO: implement` stubs — this is the contract each implementation must
satisfy, not a description of working code yet.

## Data agents (parallel, fan in to the aggregator)

### `agents/data/market_data`
- **Reads**: nothing from the bus — pulls from external sources directly
- **Sources**: yfinance (price history, no key needed), Alpha Vantage
  (news), FRED (macro series)
- **Emits**: `MarketDataEvent` on `data.market.completed`

### `agents/data/social_sentiment`
- **Reads**: nothing from the bus
- **Sources**: X, Reddit, EODHD
- **Emits**: `SocialDataEvent` on `data.social.completed`

### `agents/data/news`
- **Reads**: nothing from the bus
- **Sources**: Bloomberg, Finnhub, Reddit, Reuters
- **Emits**: `NewsDataEvent` on `data.news.completed`

### `agents/data/fundamentals`
- **Reads**: nothing from the bus
- **Sources**: company profile, financials, insider transactions (no
  dedicated vendor picked yet — Finnhub/Alpha Vantage cover all three)
- **Emits**: `FundamentalsDataEvent` on `data.fundamentals.completed`

## Aggregator (infrastructure, not an "agent")

`orchestrator/aggregator.py` joins the four data events above per
`run_id` — either all four complete, or the join times out (default 90s)
and any still-pending source is marked `TIMED_OUT` and left out of the
payload. Plain orchestration logic, no LLM calls. Emits
`AggregatedDataEvent` on `data.aggregated.ready`.

## Downstream agents (sequential)

### `agents/quant_researcher`
- **Reads**: `AggregatedDataEvent` from `data.aggregated.ready` — **only**
  this. No other tools, no external API calls of its own.
- **Emits**: `ResearchEvent` on `research.completed`

### `agents/trading_strategist`
- **Reads**: `ResearchEvent` from `research.completed`
- **Emits**: `StrategyProposedEvent` on `strategy.proposed`

### `agents/risk_manager`
- **Reads**: `StrategyProposedEvent` from `strategy.proposed`
- **Emits**: `StrategyEvaluatedEvent` on `strategy.evaluated`

## Runtime notes

- Agent framework: LangGraph/CrewAI-style, not yet wired in — plain
  Python + Redis + Postgres until the plumbing is proven.
- LLM provider: see `.env.local` (`ANTHROPIC_API_KEY` / `GEMINI_API_KEY`)
  — not yet called from any agent.
- Every event type and topic name lives in `shared/schemas/events.py`;
  it's the single source of truth — don't redefine a shape or a topic
  string anywhere else.

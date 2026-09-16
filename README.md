# stocks-Agent

Event-driven multi-agent pipeline for trading research. Four data agents run
in parallel and fan in to a quant researcher, which feeds a trading
strategist, which feeds a risk manager. A Streamlit dashboard reads
persisted results.

```
data.market.completed        (yfinance price history, Alpha Vantage news, FRED macro)
data.social.completed        (X, Reddit, EODHD)
data.news.completed          (Bloomberg, Finnhub, Reddit, Reuters)
data.fundamentals.completed  (company profile, financials, insider transactions)
        |
        v  aggregator joins all 4, or times out (default 90s) and marks stragglers
data.aggregated.ready
        |
        v  quant_researcher  (reads ONLY this event — no other tools, no external calls)
research.completed
        |
        v  trading_strategist
strategy.proposed
        |
        v  risk_manager
strategy.evaluated
```

**Status**: infrastructure only. The event bus, DB persistence, and the
aggregator's join/timeout logic are real and tested. All seven agents
(`agents/data/*`, `quant_researcher`, `trading_strategist`, `risk_manager`)
are `# TODO: implement` placeholders — no LLM calls, no external API calls
anywhere yet.

## Project structure

- `orchestrator/` — `main.py` (entrypoint, starts the aggregator) and
  `aggregator.py` (the fan-in join — plain orchestration logic, no LLM calls)
- `infra/` — `docker-compose.yml` (Redis + Postgres), `event_bus.py` (Redis
  pub/sub wrapper), `db.py` (run status + stage-result persistence)
- `agents/` — one folder per agent; each `agent.py` is currently a stub
- `shared/schemas/events.py` — the Pydantic event models and topic names;
  single source of truth for every event shape and Redis topic
- `dashboard/` — Streamlit app that reads persisted runs from Postgres,
  never touches the live event bus
- `scripts/dev_fire_fake_events.py` — fires fake data-agent events so you
  can exercise the aggregator without any real agents
- `data/cache/`, `data/runs/` — local cache and per-run output storage
- `workflows/` — plain-English recipe files the agent follows for
  recurring tasks
- `output/` — finished deliverables (reports, drafts, analysis)
- `resources/` — reference docs and templates

## Setup

1. Copy API keys and connection strings into `.env.local` (git-ignored —
   see that file for the full list and which agent/MCP server uses each).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Bring up Redis + Postgres:
   ```bash
   docker compose -f infra/docker-compose.yml up -d
   ```

## Running it

```bash
# terminal 1 — start the orchestrator (Redis/Postgres check, then listens)
python -m orchestrator.main

# terminal 2 — fire fake events to prove the plumbing works
python -m scripts.dev_fire_fake_events                       # full join
python -m scripts.dev_fire_fake_events --sources market,social,news  # timeout path

# dashboard
streamlit run dashboard/streamlit_app.py
```

## MCP servers

`.mcp.json` configures GitHub, Figma, and Supabase. GitHub uses the
hosted `api.githubcopilot.com/mcp/` endpoint with an OAuth handshake on
first connect. Figma uses the local Dev Mode MCP Server exposed by the
Figma desktop app (enable it under Dev Mode) — nothing to configure
beyond having Figma open. Supabase needs `SUPABASE_ACCESS_TOKEN` and
`SUPABASE_PROJECT_REF` set in your shell environment (see `.env.local`).

## Rules

See [CLAUDE.md](CLAUDE.md) for project rules and conventions.

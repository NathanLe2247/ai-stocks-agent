"""Six-section market & fundamental analysis for a single ticker.

Ticker acquisition (`get_ticker`) is kept separate from the analysis
(`analyze_ticker`) so a future Streamlit widget can call straight into the
same analysis function regardless of where the ticker string came from.

Source routing (see workflows/analyze_ticker.md for the full rationale):
    price history / OHLCV        -> yfinance
    technical indicators         -> computed locally (ta) from that OHLCV
    financial statements         -> yfinance, cross-checked against SEC EDGAR
    exact figures + filing text  -> SEC EDGAR (XBRL company facts, 10-K text)
    news + sentiment             -> Alpha Vantage NEWS_SENTIMENT (1 call)
    extra valuation ratios       -> Alpha Vantage OVERVIEW (1 optional call)
    macro context                -> FRED
    narrative/open-ended context -> Gemini with Google Search grounding

Alpha Vantage is capped at 2 calls per run (NEWS_SENTIMENT + optional
OVERVIEW) to respect the 25/day free-tier budget.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env.local")

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SEC_EDGAR_CONTACT_EMAIL = os.environ.get("SEC_EDGAR_CONTACT_EMAIL")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"

SECTOR_ETF = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

FUNDAMENTAL_FIELDS = (
    "Total Revenue",
    "Gross Profit",
    "Operating Income",
    "Net Income",
)
BALANCE_SHEET_FIELDS = (
    "Total Assets",
    "Total Liabilities Net Minority Interest",
    "Stockholders Equity",
    "Current Assets",
    "Current Liabilities",
    "Inventory",
    "Total Debt",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag(value: Any, source: str, as_of: Optional[str] = None) -> dict:
    return {"value": _jsonable(value), "source": source, "as_of": as_of or _now_iso()}


def _jsonable(value: Any) -> Any:
    """Recursively convert pandas/numpy types into plain JSON-safe Python
    values. Needed because this data flows two ways: through json.dumps
    (which tolerates numpy floats but silently stringifies numpy ints) and
    through pydantic's model_dump_json in agents/data/market_data/agent.py
    (which rejects unknown numpy types outright)."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        try:
            return value.item()
        except Exception:
            return value
    return value


class DataQuality:
    def __init__(self) -> None:
        self.issues: list[dict] = []

    def flag(self, section: str, field: str, issue: str, severity: str = "fallback") -> None:
        self.issues.append({"section": section, "field": field, "issue": issue, "severity": severity})

    def as_list(self) -> list[dict]:
        return self.issues


# --------------------------------------------------------------------------
# Ticker acquisition
# --------------------------------------------------------------------------

def get_ticker(cli_args: Optional[list[str]] = None) -> tuple[str, str]:
    """Resolve and validate a ticker from CLI args or an interactive prompt.

    Returns (ticker, company_name). Fails fast (SystemExit) on an unresolved
    symbol rather than spending any paid API calls on a typo.
    """
    args = sys.argv[1:] if cli_args is None else cli_args
    raw = args[0] if args else input("Enter ticker symbol: ")
    ticker = raw.strip().upper()
    if not ticker:
        raise SystemExit("No ticker provided.")

    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception as exc:
        raise SystemExit(f"'{ticker}' failed to resolve via yfinance: {exc}")

    has_identity = bool(info.get("longName") or info.get("shortName"))
    has_price = info.get("regularMarketPrice") is not None or info.get("currentPrice") is not None
    if not (has_identity and has_price):
        recent = t.history(period="5d")
        if recent.empty:
            raise SystemExit(f"'{ticker}' does not resolve to a real, actively traded symbol.")

    company_name = info.get("longName") or info.get("shortName") or ticker
    return ticker, company_name


# --------------------------------------------------------------------------
# Section 3: price history + technicals
# --------------------------------------------------------------------------

def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    return df


def _trend_state(price: float, ma: float, ma_prev: float) -> str:
    direction = "rising" if ma > ma_prev else "falling" if ma < ma_prev else "flat"
    position = "above" if price > ma else "below" if price < ma else "at"
    if position == "above" and direction != "falling":
        return "uptrend"
    if position == "below" and direction != "rising":
        return "downtrend"
    return "mixed"


def _find_pivots(df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
    highs, lows = [], []
    h, l = df["High"], df["Low"]
    n = len(df)
    for i in range(window, n - window):
        seg_h = h.iloc[i - window : i + window + 1]
        seg_l = l.iloc[i - window : i + window + 1]
        if h.iloc[i] == seg_h.max():
            highs.append(float(h.iloc[i]))
        if l.iloc[i] == seg_l.min():
            lows.append(float(l.iloc[i]))
    return highs, lows


def _cluster_levels(levels: list[float], pct_tol: float = 0.02, top_n: int = 4) -> list[float]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters: list[list[float]] = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - clusters[-1][-1]) / clusters[-1][-1] <= pct_tol:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    clusters.sort(key=len, reverse=True)
    return sorted(sum(c) / len(c) for c in clusters[:top_n])


def compute_technicals(df: pd.DataFrame, dq: DataQuality, as_of: str) -> dict:
    if df.empty:
        dq.flag("3_technical_analysis", "all", "empty price history from yfinance", "missing")
        return {}

    close = df["Close"]
    volume = df["Volume"]
    price = float(close.iloc[-1])

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    trend = {
        "short_term_20d": tag(_trend_state(price, float(sma20.iloc[-1]), float(sma20.iloc[-2])), "yfinance (computed)", as_of)
        if len(sma20.dropna()) >= 2 else None,
        "medium_term_50d": tag(_trend_state(price, float(sma50.iloc[-1]), float(sma50.iloc[-2])), "yfinance (computed)", as_of)
        if len(sma50.dropna()) >= 2 else None,
        "long_term_200d": tag(_trend_state(price, float(sma200.iloc[-1]), float(sma200.iloc[-2])), "yfinance (computed)", as_of)
        if len(sma200.dropna()) >= 2 else None,
    }
    if trend["long_term_200d"] is None:
        dq.flag("3_technical_analysis", "trend.long_term_200d", "fewer than 200+1 trading days of history", "missing")

    recent = df.tail(252) if len(df) >= 252 else df
    swing_highs, swing_lows = _find_pivots(recent, window=5)
    resistance = _cluster_levels(swing_highs)
    support = _cluster_levels(swing_lows)

    rsi = RSIIndicator(close=close, window=14).rsi()
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    bb = BollingerBands(close=close, window=20, window_dev=2)

    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    vol_avg50 = float(volume.rolling(50).mean().iloc[-1])
    vol_current = float(volume.iloc[-1])
    price_change_20d = price - float(close.iloc[-21]) if len(close) > 21 else None
    if vol_current > vol_avg20 and price_change_20d is not None:
        volume_confirmation = "confirming" if price_change_20d > 0 else "diverging"
    else:
        volume_confirmation = "below-average volume, no strong signal"

    return {
        "trend": trend,
        "support_resistance": {
            "support": tag(support, "yfinance (computed, 12mo swing lows)", as_of),
            "resistance": tag(resistance, "yfinance (computed, 12mo swing highs)", as_of),
        },
        "indicators": {
            "rsi_14": tag(round(float(rsi.iloc[-1]), 2), "ta (computed from yfinance)", as_of),
            "macd": {
                "macd_line": tag(round(float(macd_ind.macd().iloc[-1]), 4), "ta (computed from yfinance)", as_of),
                "signal_line": tag(round(float(macd_ind.macd_signal().iloc[-1]), 4), "ta (computed from yfinance)", as_of),
                "histogram": tag(round(float(macd_ind.macd_diff().iloc[-1]), 4), "ta (computed from yfinance)", as_of),
            },
            "bollinger_bands": {
                "upper": tag(round(float(bb.bollinger_hband().iloc[-1]), 2), "ta (computed from yfinance)", as_of),
                "middle": tag(round(float(bb.bollinger_mavg().iloc[-1]), 2), "ta (computed from yfinance)", as_of),
                "lower": tag(round(float(bb.bollinger_lband().iloc[-1]), 2), "ta (computed from yfinance)", as_of),
            },
            "volume_trend": {
                "current_volume": tag(vol_current, "yfinance", as_of),
                "avg_volume_20d": tag(round(vol_avg20, 0), "yfinance (computed)", as_of),
                "avg_volume_50d": tag(round(vol_avg50, 0), "yfinance (computed)", as_of),
                "signal": tag(volume_confirmation, "yfinance (computed)", as_of),
            },
        },
    }


# --------------------------------------------------------------------------
# Section 2: fundamentals
# --------------------------------------------------------------------------

def _row(df: pd.DataFrame, label: str) -> Optional[pd.Series]:
    return df.loc[label] if label in df.index else None


def _yoy_growth(series: pd.Series) -> dict[str, Optional[float]]:
    cols = list(series.index)  # most-recent first, per yfinance convention
    out = {}
    for i in range(len(cols) - 1):
        cur, prev = series.iloc[i], series.iloc[i + 1]
        year = cols[i].year if hasattr(cols[i], "year") else str(cols[i])
        if pd.isna(cur) or pd.isna(prev) or prev == 0:
            out[str(year)] = None
        else:
            out[str(year)] = round(float((cur - prev) / abs(prev)) * 100, 2)
    return out


def _cagr(series: pd.Series) -> Optional[float]:
    vals = series.dropna()
    if len(vals) < 2:
        return None
    end, start = vals.iloc[0], vals.iloc[-1]
    n_years = len(vals) - 1
    if start <= 0 or end <= 0 or n_years <= 0:
        return None
    return round(((end / start) ** (1 / n_years) - 1) * 100, 2)


def sec_cik_lookup(ticker: str, dq: DataQuality) -> Optional[str]:
    if not SEC_EDGAR_CONTACT_EMAIL:
        dq.flag("2_financial_metrics", "sec_cross_check", "SEC_EDGAR_CONTACT_EMAIL not set; skipped SEC EDGAR entirely", "missing")
        return None
    headers = {"User-Agent": f"stocks-Agent research script ({SEC_EDGAR_CONTACT_EMAIL})"}
    try:
        r = requests.get(SEC_TICKERS_URL, headers=headers, timeout=15)
        r.raise_for_status()
        for row in r.json().values():
            if row.get("ticker", "").upper() == ticker.upper():
                return str(row["cik_str"]).zfill(10)
    except Exception as exc:
        dq.flag("2_financial_metrics", "sec_cik_lookup", f"SEC CIK lookup failed: {exc}", "missing")
    return None


def sec_get_json(url: str) -> Optional[dict]:
    headers = {"User-Agent": f"stocks-Agent research script ({SEC_EDGAR_CONTACT_EMAIL})"}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    return r.json()


def sec_latest_10k(cik: str, dq: DataQuality) -> Optional[dict]:
    try:
        sub = sec_get_json(SEC_SUBMISSIONS_URL.format(cik=cik))
        if not sub:
            return None
        recent = sub["filings"]["recent"]
        for i, form in enumerate(recent["form"]):
            if form == "10-K":
                return {
                    "accession": recent["accessionNumber"][i],
                    "primary_doc": recent["primaryDocument"][i],
                    "filing_date": recent["filingDate"][i],
                    "cik": cik,
                }
    except Exception as exc:
        dq.flag("1_company_analysis", "sec_10k_lookup", f"SEC 10-K lookup failed: {exc}", "missing")
    return None


def sec_fetch_filing_text(filing: dict, dq: DataQuality) -> Optional[str]:
    try:
        import warnings

        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

        cik_int = int(filing["cik"])
        accession_nodash = filing["accession"].replace("-", "")
        url = SEC_ARCHIVES_BASE.format(cik_int=cik_int, accession_nodash=accession_nodash, doc=filing["primary_doc"])
        headers = {"User-Agent": f"stocks-Agent research script ({SEC_EDGAR_CONTACT_EMAIL})"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        # 10-K primary docs are inline-XBRL (.htm with embedded XML namespaces),
        # which trips bs4's XML-vs-HTML heuristic even though HTML parsing is
        # exactly what we want here (plain get_text, not namespace-aware XML).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(r.text, "lxml")
        return soup.get_text("\n")
    except Exception as exc:
        dq.flag("1_company_analysis", "sec_filing_text", f"SEC filing text fetch failed: {exc}", "missing")
        return None


def sec_extract_items(full_text: str, dq: DataQuality) -> dict[str, Optional[str]]:
    """Best-effort Item 1 / Item 1A extraction from 10-K plain text."""
    pattern = re.compile(
        r"^\s*item\s+(1|1a|1b|2)\.?\s", re.IGNORECASE | re.MULTILINE
    )
    matches = list(pattern.finditer(full_text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        item = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end].strip()
        if item in ("1", "1a") and len(chunk) > len(sections.get(item, "")):
            sections[item] = chunk

    item1 = sections.get("1")
    item1a = sections.get("1a")
    if not item1:
        dq.flag("1_company_analysis", "sec_item1", "could not isolate Item 1 (Business) text from filing", "missing")
    if not item1a:
        dq.flag("5_future_prospects", "sec_item1a", "could not isolate Item 1A (Risk Factors) text from filing", "missing")

    return {
        "item1": (item1[:40000] if item1 else None),
        "item1a": (item1a[:40000] if item1a else None),
    }


def sec_company_facts(cik: str, dq: DataQuality) -> dict:
    try:
        facts = sec_get_json(SEC_COMPANYFACTS_URL.format(cik=cik))
        if not facts:
            return {}
        gaap = facts.get("facts", {}).get("us-gaap", {})
        out = {}
        for tag_name in ("Assets", "Liabilities", "StockholdersEquity", "Revenues", "NetIncomeLoss"):
            node = gaap.get(tag_name)
            if not node:
                continue
            units = node.get("units", {}).get("USD", [])
            annual = [u for u in units if u.get("form") == "10-K" and u.get("fp") == "FY"]
            if annual:
                annual.sort(key=lambda u: u["end"], reverse=True)
                out[tag_name] = {"value": annual[0]["val"], "as_of": annual[0]["end"]}
        return out
    except Exception as exc:
        dq.flag("2_financial_metrics", "sec_company_facts", f"SEC company facts fetch failed: {exc}", "missing")
        return {}


def compute_fundamentals(ticker: str, t: yf.Ticker, dq: DataQuality, as_of: str) -> dict:
    fin = t.financials
    bs = t.balance_sheet
    cf = t.cashflow
    info = t.info

    result: dict[str, Any] = {}

    revenue = _row(fin, "Total Revenue")
    net_income = _row(fin, "Net Income")
    gross_profit = _row(fin, "Gross Profit")
    operating_income = _row(fin, "Operating Income")

    if revenue is None or net_income is None:
        dq.flag("2_financial_metrics", "income_statement", "yfinance missing Total Revenue/Net Income rows", "missing")

    result["revenue_growth_yoy_pct"] = tag(_yoy_growth(revenue), "yfinance (computed)", as_of) if revenue is not None else None
    result["revenue_cagr_pct"] = tag(_cagr(revenue), "yfinance (computed)", as_of) if revenue is not None else None
    result["net_income_growth_yoy_pct"] = tag(_yoy_growth(net_income), "yfinance (computed)", as_of) if net_income is not None else None
    result["net_income_cagr_pct"] = tag(_cagr(net_income), "yfinance (computed)", as_of) if net_income is not None else None

    margins: dict[str, dict] = {}
    if revenue is not None:
        for col in revenue.index:
            year = str(col.year if hasattr(col, "year") else col)
            rev_v = revenue.get(col)
            gp_v = gross_profit.get(col) if gross_profit is not None else None
            oi_v = operating_income.get(col) if operating_income is not None else None
            ni_v = net_income.get(col) if net_income is not None else None
            if not rev_v or pd.isna(rev_v) or rev_v == 0:
                continue
            margins[year] = {
                "gross_margin_pct": round(float(gp_v / rev_v) * 100, 2) if gp_v is not None and not pd.isna(gp_v) else None,
                "operating_margin_pct": round(float(oi_v / rev_v) * 100, 2) if oi_v is not None and not pd.isna(oi_v) else None,
                "net_margin_pct": round(float(ni_v / rev_v) * 100, 2) if ni_v is not None and not pd.isna(ni_v) else None,
            }
    years_sorted = sorted(margins.keys())
    net_trend = None
    if len(years_sorted) >= 2:
        first, last = margins[years_sorted[0]]["net_margin_pct"], margins[years_sorted[-1]]["net_margin_pct"]
        if first is not None and last is not None:
            net_trend = "improving" if last > first else "declining" if last < first else "flat"
    result["margins_by_year"] = tag(margins, "yfinance (computed)", as_of)
    result["net_margin_trend"] = tag(net_trend, "yfinance (computed)", as_of)

    total_debt = _row(bs, "Total Debt")
    equity = _row(bs, "Stockholders Equity")
    current_assets = _row(bs, "Current Assets")
    current_liabilities = _row(bs, "Current Liabilities")
    inventory = _row(bs, "Inventory")

    def _ratio_series(numer: Optional[pd.Series], denom: Optional[pd.Series]) -> dict[str, Optional[float]]:
        if numer is None or denom is None:
            return {}
        out = {}
        for col in numer.index:
            n_v, d_v = numer.get(col), denom.get(col)
            year = str(col.year if hasattr(col, "year") else col)
            out[year] = round(float(n_v / d_v), 2) if d_v and not pd.isna(d_v) and d_v != 0 and not pd.isna(n_v) else None
        return out

    d_to_e = _ratio_series(total_debt, equity)
    current_ratio = _ratio_series(current_assets, current_liabilities)
    quick_assets = (current_assets - inventory) if (current_assets is not None and inventory is not None) else current_assets
    quick_ratio = _ratio_series(quick_assets, current_liabilities)

    if not d_to_e:
        dq.flag("2_financial_metrics", "debt_to_equity", "missing Total Debt or Stockholders Equity in yfinance balance sheet", "missing")

    result["debt_to_equity_by_year"] = tag(d_to_e, "yfinance (computed)", as_of)
    result["current_ratio_by_year"] = tag(current_ratio, "yfinance (computed)", as_of)
    result["quick_ratio_by_year"] = tag(quick_ratio, "yfinance (computed)", as_of)

    peg = info.get("trailingPegRatio")
    ev_ebitda = info.get("enterpriseToEbitda")
    result["valuation"] = {
        "trailing_pe": tag(info.get("trailingPE"), "yfinance", as_of),
        "forward_pe": tag(info.get("forwardPE"), "yfinance", as_of),
        "price_to_book": tag(info.get("priceToBook"), "yfinance", as_of),
        "dividend_yield_pct": tag(info.get("dividendYield"), "yfinance", as_of),
        "peg_ratio": tag(peg, "yfinance", as_of),
        "ev_to_ebitda": tag(ev_ebitda, "yfinance", as_of),
    }
    result["_needs_overview_fallback"] = peg is None or ev_ebitda is None

    return result


def historical_valuation_ranges(t: yf.Ticker, dq: DataQuality, as_of: str) -> dict:
    """Approximate 5yr P/E, P/B, and dividend-yield ranges by combining daily
    close prices with the nearest trailing annual EPS/book-value/dividend
    figures. This is a derived approximation, not a vendor-provided series."""
    try:
        prices = t.history(period="5y", auto_adjust=True)["Close"]
        if prices.empty:
            return {}
        fin = t.financials
        bs = t.balance_sheet
        net_income = _row(fin, "Net Income")
        equity = _row(bs, "Stockholders Equity")
        shares = t.info.get("sharesOutstanding")
        dividends = t.dividends

        pe_vals, pb_vals, yield_vals = [], [], []
        if net_income is not None and shares:
            for col, ni in net_income.items():
                year_prices = prices[prices.index.year == col.year]
                if year_prices.empty or pd.isna(ni) or ni <= 0:
                    continue
                eps = ni / shares
                pe_vals.extend((year_prices / eps).tolist())
        if equity is not None and shares:
            for col, eq in equity.items():
                year_prices = prices[prices.index.year == col.year]
                if year_prices.empty or pd.isna(eq) or eq <= 0:
                    continue
                bvps = eq / shares
                pb_vals.extend((year_prices / bvps).tolist())
        if dividends is not None and not dividends.empty:
            annual_div = dividends.resample("YE").sum()
            for ts, div in annual_div.items():
                year_prices = prices[prices.index.year == ts.year]
                if year_prices.empty or div <= 0:
                    continue
                yield_vals.extend(((div / year_prices) * 100).tolist())

        out = {}
        if pe_vals:
            out["trailing_pe_5yr_range"] = tag([round(min(pe_vals), 2), round(max(pe_vals), 2)], "yfinance (derived approximation)", as_of)
        else:
            dq.flag("2_financial_metrics", "valuation.trailing_pe_5yr_range", "insufficient data to derive historical P/E range", "missing")
        if pb_vals:
            out["price_to_book_5yr_range"] = tag([round(min(pb_vals), 2), round(max(pb_vals), 2)], "yfinance (derived approximation)", as_of)
        else:
            dq.flag("2_financial_metrics", "valuation.price_to_book_5yr_range", "insufficient data to derive historical P/B range", "missing")
        if yield_vals:
            out["dividend_yield_5yr_range_pct"] = tag([round(min(yield_vals), 2), round(max(yield_vals), 2)], "yfinance (derived approximation)", as_of)
        else:
            out["dividend_yield_5yr_range_pct"] = tag([0.0, 0.0], "yfinance (derived approximation)", as_of)
        return out
    except Exception as exc:
        dq.flag("2_financial_metrics", "valuation.historical_ranges", f"failed to derive historical valuation ranges: {exc}", "missing")
        return {}


# --------------------------------------------------------------------------
# Alpha Vantage (budget: max 2 calls/run)
# --------------------------------------------------------------------------

def _av_request(params: dict, dq: DataQuality, field: str) -> Optional[dict]:
    if not ALPHA_VANTAGE_API_KEY:
        dq.flag(field.split(".")[0], field, "ALPHA_VANTAGE_API_KEY not set", "missing")
        return None
    params = {**params, "apikey": ALPHA_VANTAGE_API_KEY}
    try:
        r = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "Note" in data or "Information" in data:
            dq.flag(field.split(".")[0], field, f"Alpha Vantage limit/notice: {data.get('Note') or data.get('Information')}", "missing")
            return None
        return data
    except Exception as exc:
        dq.flag(field.split(".")[0], field, f"Alpha Vantage request failed: {exc}", "missing")
        return None


def fetch_news_sentiment(ticker: str, dq: DataQuality, as_of: str) -> Optional[dict]:
    data = _av_request({"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": 50}, dq, "1_company_analysis.news_sentiment")
    if not data or "feed" not in data:
        return None
    articles = []
    topic_relevance: dict[str, list[float]] = {}
    scores = []
    for item in data["feed"]:
        ticker_match = next((ts for ts in item.get("ticker_sentiment", []) if ts.get("ticker") == ticker), None)
        relevance = float(ticker_match["relevance_score"]) if ticker_match else 0.0
        if relevance < 0.1:
            continue
        sentiment_score = float(ticker_match["ticker_sentiment_score"]) if ticker_match else None
        if sentiment_score is not None:
            scores.append(sentiment_score)
        articles.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "source": item.get("source"),
            "time_published": item.get("time_published"),
            "sentiment_label": ticker_match.get("ticker_sentiment_label") if ticker_match else None,
            "sentiment_score": sentiment_score,
            "relevance": relevance,
        })
        for topic in item.get("topics", []):
            topic_relevance.setdefault(topic["topic"], []).append(float(topic["relevance_score"]))

    articles.sort(key=lambda a: a["relevance"], reverse=True)
    top_topics = sorted(
        ((k, sum(v) / len(v)) for k, v in topic_relevance.items()),
        key=lambda kv: kv[1], reverse=True,
    )[:8]

    return {
        "as_of": as_of,
        "source": "Alpha Vantage NEWS_SENTIMENT",
        "avg_sentiment_score": round(sum(scores) / len(scores), 4) if scores else None,
        "article_count": len(articles),
        "top_articles": articles[:20],
        "top_topics": [{"topic": t, "avg_relevance": round(r, 3)} for t, r in top_topics],
    }


def fetch_overview(ticker: str, dq: DataQuality, as_of: str) -> Optional[dict]:
    data = _av_request({"function": "OVERVIEW", "symbol": ticker}, dq, "2_financial_metrics.overview")
    if not data or "Symbol" not in data:
        return None

    def _num(key: str) -> Optional[float]:
        v = data.get(key)
        try:
            return float(v) if v not in (None, "None", "-") else None
        except ValueError:
            return None

    return {
        "peg_ratio": tag(_num("PEGRatio"), "Alpha Vantage OVERVIEW", as_of),
        "ev_to_ebitda": tag(_num("EVToEBITDA"), "Alpha Vantage OVERVIEW", as_of),
        "analyst_target_price": tag(_num("AnalystTargetPrice"), "Alpha Vantage OVERVIEW", as_of),
    }


# --------------------------------------------------------------------------
# FRED macro
# --------------------------------------------------------------------------

def fred_series(series_id: str, limit: int = 14) -> list[dict]:
    r = requests.get(FRED_BASE, params={
        "series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json",
        "sort_order": "desc", "limit": limit,
    }, timeout=15)
    r.raise_for_status()
    return [o for o in r.json()["observations"] if o["value"] != "."]


def fetch_macro_snapshot(dq: DataQuality) -> dict:
    if not FRED_API_KEY:
        dq.flag("4_market_analysis", "macro_backdrop", "FRED_API_KEY not set", "missing")
        return {}
    out = {}
    try:
        for label, series_id in (
            ("fed_funds_rate_pct", "FEDFUNDS"),
            ("treasury_10y_yield_pct", "DGS10"),
            ("spread_2s10s_pct", "T10Y2Y"),
            ("unemployment_rate_pct", "UNRATE"),
        ):
            obs = fred_series(series_id, limit=2)
            if not obs:
                dq.flag("4_market_analysis", f"macro_backdrop.{label}", f"no observations for {series_id}", "missing")
                continue
            latest, prev = float(obs[0]["value"]), (float(obs[1]["value"]) if len(obs) > 1 else None)
            direction = None if prev is None else ("rising" if latest > prev else "falling" if latest < prev else "flat")
            out[label] = {**tag(latest, f"FRED ({series_id})", obs[0]["date"]), "direction": direction}

        cpi_obs = fred_series("CPIAUCSL", limit=18)  # buffer: '.' placeholder rows get filtered out
        if len(cpi_obs) >= 13:
            latest, year_ago = float(cpi_obs[0]["value"]), float(cpi_obs[12]["value"])
            cpi_yoy = round((latest - year_ago) / year_ago * 100, 2)
            out["cpi_yoy_pct"] = tag(cpi_yoy, "FRED (CPIAUCSL, computed)", cpi_obs[0]["date"])
        else:
            dq.flag("4_market_analysis", "macro_backdrop.cpi_yoy_pct", "insufficient CPIAUCSL history returned", "missing")
    except Exception as exc:
        dq.flag("4_market_analysis", "macro_backdrop", f"FRED fetch failed: {exc}", "missing")
    return out


# --------------------------------------------------------------------------
# Section 4: relative performance
# --------------------------------------------------------------------------

def _total_return(df: pd.DataFrame, months: int) -> Optional[float]:
    if df.empty:
        return None
    cutoff = df.index.max() - pd.DateOffset(months=months)
    window = df[df.index >= cutoff]
    if len(window) < 2:
        return None
    start, end = window["Close"].iloc[0], window["Close"].iloc[-1]
    return round(float((end - start) / start) * 100, 2)


def fetch_relative_performance(ticker: str, sector: Optional[str], dq: DataQuality, as_of: str) -> dict:
    sector_etf = SECTOR_ETF.get(sector) if sector else None
    symbols = [ticker, "SPY"] + ([sector_etf] if sector_etf else [])
    histories = {}
    for sym in symbols:
        try:
            histories[sym] = yf.Ticker(sym).history(period="5y", auto_adjust=True)
        except Exception as exc:
            dq.flag("4_market_analysis", f"relative_performance.{sym}", f"history fetch failed: {exc}", "missing")
            histories[sym] = pd.DataFrame()

    result: dict[str, Any] = {"sector_etf": sector_etf}
    for months, label in ((12, "1y"), (36, "3y"), (60, "5y")):
        ticker_ret = _total_return(histories[ticker], months)
        spy_ret = _total_return(histories["SPY"], months)
        sector_ret = _total_return(histories[sector_etf], months) if sector_etf else None
        result[label] = {
            "ticker_return_pct": tag(ticker_ret, "yfinance (computed)", as_of),
            "spy_return_pct": tag(spy_ret, "yfinance (computed)", as_of),
            "sector_etf_return_pct": tag(sector_ret, "yfinance (computed)", as_of) if sector_etf else None,
            "vs_spy_pct": round(ticker_ret - spy_ret, 2) if ticker_ret is not None and spy_ret is not None else None,
            "vs_sector_pct": round(ticker_ret - sector_ret, 2) if ticker_ret is not None and sector_ret is not None else None,
        }
    if not sector_etf:
        dq.flag("4_market_analysis", "relative_performance.sector_etf", f"no sector ETF mapping for sector={sector!r}", "missing")
    return result


# --------------------------------------------------------------------------
# Section 5: analyst estimates
# --------------------------------------------------------------------------

def fetch_analyst_estimates(t: yf.Ticker, dq: DataQuality, as_of: str) -> dict:
    out: dict[str, Any] = {}
    try:
        targets = t.analyst_price_targets
        out["price_targets"] = tag(targets, "yfinance", as_of) if targets else None
    except Exception as exc:
        dq.flag("5_future_prospects", "analyst_price_targets", f"fetch failed: {exc}", "missing")

    try:
        ee = t.earnings_estimate
        out["eps_estimates"] = tag(ee.reset_index().to_dict(orient="records"), "yfinance", as_of) if ee is not None and not ee.empty else None
    except Exception as exc:
        dq.flag("5_future_prospects", "earnings_estimate", f"fetch failed: {exc}", "missing")

    try:
        re_ = t.revenue_estimate
        out["revenue_estimates"] = tag(re_.reset_index().to_dict(orient="records"), "yfinance", as_of) if re_ is not None and not re_.empty else None
    except Exception as exc:
        dq.flag("5_future_prospects", "revenue_estimate", f"fetch failed: {exc}", "missing")

    return out


# --------------------------------------------------------------------------
# Gemini narrative synthesis (Google Search grounded)
# --------------------------------------------------------------------------

GEMINI_PROMPT_TEMPLATE = """You are a financial research analyst. Use Google Search to find current, \
accurate information. Research {ticker} ({company_name}) and return ONLY a single JSON object \
(no markdown fences, no commentary) with exactly these keys:

- "business_and_market_position": 3-5 sentences on what the company does, its market share/position \
  in its category, and how that has changed recently.
- "competitive_advantages": 3-5 sentences on moat, IP/technology, network effects, brand, switching costs.
- "management_and_leadership": 3-5 sentences on CEO/CFO tenure and background, notable strategic wins \
  or failures under current leadership, and any recent executive turnover.
- "industry_trends_and_risks": 3-5 sentences on sector-specific trends, analyst outlooks, and regulatory \
  developments relevant to this company.
- "planned_innovations_and_partnerships": 3-5 sentences on recent/upcoming product launches, expansions, \
  or partnerships, drawing on the last 1-2 earnings calls if available.
- "recent_risk_developments": 2-4 sentences on any new regulatory action, litigation, or macro shift \
  affecting the company since its most recent 10-K.
- "competitors": a list of 3-5 objects {{"ticker": "...", "name": "...", "rationale": "..."}} for direct competitors.
- "other_interesting_alternatives": a list of 1-3 objects {{"ticker": "...", "name": "...", "rationale": "..."}} \
  for other stocks in the same sector worth a look.

Context already gathered from SEC filings and news feeds (use to ground your answer, don't just repeat it):

--- SEC 10-K Item 1 (Business), filed {filing_date} ---
{item1}

--- SEC 10-K Item 1A (Risk Factors) ---
{item1a}

--- Recent news headlines (Alpha Vantage NEWS_SENTIMENT) ---
{news_headlines}
"""


def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def gemini_synthesize(
    ticker: str,
    company_name: str,
    filing_date: Optional[str],
    item1: Optional[str],
    item1a: Optional[str],
    news_headlines: list[str],
    dq: DataQuality,
) -> tuple[Optional[dict], list[str]]:
    if not GEMINI_API_KEY:
        dq.flag("1_company_analysis", "gemini_narrative", "GEMINI_API_KEY not set", "missing")
        return None, []

    from google import genai
    from google.genai import types

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=company_name,
        filing_date=filing_date or "unknown",
        item1=(item1 or "not available")[:20000],
        item1a=(item1a or "not available")[:20000],
        news_headlines="\n".join(f"- {h}" for h in news_headlines[:20]) or "not available",
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
            )
            parsed = _extract_json_object(resp.text or "")
            if parsed is None:
                dq.flag("1_company_analysis", "gemini_narrative", "Gemini response was not valid JSON", "fallback")
                return None, []

            sources = []
            gm = resp.candidates[0].grounding_metadata if resp.candidates else None
            if gm and gm.grounding_chunks:
                sources = [c.web.uri for c in gm.grounding_chunks if c.web and c.web.uri]
            return parsed, sources
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                time.sleep(5 * (attempt + 1))
                continue
            break

    dq.flag("1_company_analysis", "gemini_narrative", f"Gemini synthesis failed after retries: {last_exc}", "missing")
    return None, []


# --------------------------------------------------------------------------
# Section 6: competitors
# --------------------------------------------------------------------------

def fetch_competitor_metrics(tickers: list[str], dq: DataQuality, as_of: str) -> dict:
    table = {}
    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            if not (t.info.get("longName") or t.info.get("shortName")):
                dq.flag("6_competitor_comparison", sym, f"'{sym}' did not resolve via yfinance, skipped", "missing")
                continue
            fundamentals = compute_fundamentals(sym, t, dq, as_of)
            years = sorted(fundamentals.get("margins_by_year", {}).get("value", {}).keys())
            latest_year = years[-1] if years else None
            latest_margins = fundamentals["margins_by_year"]["value"].get(latest_year, {}) if latest_year else {}
            d_to_e_years = sorted(fundamentals.get("debt_to_equity_by_year", {}).get("value", {}).keys())
            latest_d_to_e = fundamentals["debt_to_equity_by_year"]["value"].get(d_to_e_years[-1]) if d_to_e_years else None
            table[sym] = {
                "name": t.info.get("longName") or t.info.get("shortName"),
                "gross_margin_pct": latest_margins.get("gross_margin_pct"),
                "net_margin_pct": latest_margins.get("net_margin_pct"),
                "debt_to_equity": latest_d_to_e,
                "trailing_pe": fundamentals["valuation"]["trailing_pe"]["value"],
                "price_to_book": fundamentals["valuation"]["price_to_book"]["value"],
                "dividend_yield_pct": fundamentals["valuation"]["dividend_yield_pct"]["value"],
                "revenue_cagr_pct": fundamentals["revenue_cagr_pct"]["value"] if fundamentals.get("revenue_cagr_pct") else None,
            }
        except Exception as exc:
            dq.flag("6_competitor_comparison", sym, f"failed to fetch competitor metrics: {exc}", "missing")
    return table


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def analyze_ticker(ticker: str, company_name: str) -> dict:
    """Core analysis entrypoint. Ticker-source-agnostic: called the same way
    whether the ticker came from the CLI, an interactive prompt, or (later)
    a Streamlit text input."""
    dq = DataQuality()
    as_of = _now_iso()
    t = yf.Ticker(ticker)
    sector = t.info.get("sector")

    print(f"[{ticker}] fetching price history + computing technicals...", file=sys.stderr)
    price_df = fetch_price_history(ticker)
    technicals = compute_technicals(price_df, dq, as_of)

    print(f"[{ticker}] pulling financial statements from yfinance...", file=sys.stderr)
    fundamentals = compute_fundamentals(ticker, t, dq, as_of)
    fundamentals["valuation_historical_ranges"] = historical_valuation_ranges(t, dq, as_of)

    needs_overview = fundamentals.pop("_needs_overview_fallback", False)
    overview = None
    if needs_overview:
        print(f"[{ticker}] PEG/EV-EBITDA missing from yfinance, spending 1 Alpha Vantage OVERVIEW call...", file=sys.stderr)
        overview = fetch_overview(ticker, dq, as_of)
        if overview:
            for k in ("peg_ratio", "ev_to_ebitda"):
                if fundamentals["valuation"][k]["value"] is None and overview.get(k) and overview[k]["value"] is not None:
                    fundamentals["valuation"][k] = overview[k]

    print(f"[{ticker}] looking up SEC EDGAR filing...", file=sys.stderr)
    cik = sec_cik_lookup(ticker, dq)
    filing, item1, item1a, sec_facts = None, None, None, {}
    if cik:
        filing = sec_latest_10k(cik, dq)
        sec_facts = sec_company_facts(cik, dq)
        if filing:
            full_text = sec_fetch_filing_text(filing, dq)
            if full_text:
                items = sec_extract_items(full_text, dq)
                item1, item1a = items["item1"], items["item1a"]

    print(f"[{ticker}] pulling Alpha Vantage NEWS_SENTIMENT (1 of 2 budgeted calls)...", file=sys.stderr)
    news = fetch_news_sentiment(ticker, dq, as_of)
    news_headlines = [a["title"] for a in news["top_articles"]] if news else []

    print(f"[{ticker}] pulling FRED macro snapshot...", file=sys.stderr)
    macro = fetch_macro_snapshot(dq)

    print(f"[{ticker}] computing relative performance vs SPY/sector...", file=sys.stderr)
    relative_perf = fetch_relative_performance(ticker, sector, dq, as_of)

    print(f"[{ticker}] pulling analyst estimates...", file=sys.stderr)
    estimates = fetch_analyst_estimates(t, dq, as_of)

    print(f"[{ticker}] synthesizing narrative sections via Gemini (Google Search grounded)...", file=sys.stderr)
    narrative, gemini_sources = gemini_synthesize(
        ticker, company_name,
        filing.get("filing_date") if filing else None,
        item1, item1a, news_headlines, dq,
    )
    narrative = narrative or {}

    competitor_specs = narrative.get("competitors") or []
    competitor_tickers = [c["ticker"] for c in competitor_specs if c.get("ticker")]
    print(f"[{ticker}] pulling competitor metrics for {competitor_tickers or 'none identified'}...", file=sys.stderr)
    comparison_table = fetch_competitor_metrics(competitor_tickers, dq, as_of) if competitor_tickers else {}
    comparison_table[ticker] = fetch_competitor_metrics([ticker], dq, as_of).get(ticker, {})

    sections = {
        "1_company_analysis": {
            "business_and_market_position": narrative.get("business_and_market_position"),
            "sec_item1_excerpt": tag(item1[:2000] if item1 else None, "SEC EDGAR 10-K Item 1", filing.get("filing_date") if filing else None),
            "competitive_advantages": narrative.get("competitive_advantages"),
            "management_and_leadership": narrative.get("management_and_leadership"),
            "news_sentiment": news,
            "gemini_sources": gemini_sources,
        },
        "2_financial_metrics": {**fundamentals, "sec_cross_check": sec_facts, "overview_fallback_used": overview is not None},
        "3_technical_analysis": technicals,
        "4_market_analysis": {
            "relative_performance": relative_perf,
            "macro_backdrop": macro,
            "industry_trends_and_risks": narrative.get("industry_trends_and_risks"),
        },
        "5_future_prospects": {
            "growth_forecasts": estimates,
            "planned_innovations_and_partnerships": narrative.get("planned_innovations_and_partnerships"),
            "risks": {
                "sec_risk_factors_excerpt": tag(item1a[:3000] if item1a else None, "SEC EDGAR 10-K Item 1A", filing.get("filing_date") if filing else None),
                "recent_risk_developments": narrative.get("recent_risk_developments"),
            },
        },
        "6_competitor_comparison": {
            "competitors": competitor_specs,
            "comparison_table": comparison_table,
            "other_interesting_alternatives": narrative.get("other_interesting_alternatives") or [],
        },
    }

    return _jsonable({
        "ticker": ticker,
        "company_name": company_name,
        "generated_at": as_of,
        "sections": sections,
        "data_quality": dq.as_list(),
    })


def build_report_markdown(data: dict) -> str:
    s = data["sections"]
    ticker, name = data["ticker"], data["company_name"]
    lines = [f"# {name} ({ticker}) — Market & Fundamental Analysis", "", f"_Generated {data['generated_at']}_", ""]

    lines += ["## 1. Company Analysis", ""]
    lines.append(s["1_company_analysis"].get("business_and_market_position") or "_Not available._")
    lines += ["", "**Competitive advantages**", "", s["1_company_analysis"].get("competitive_advantages") or "_Not available._"]
    lines += ["", "**Management and leadership**", "", s["1_company_analysis"].get("management_and_leadership") or "_Not available._"]

    lines += ["", "## 2. Key Financial Metrics and Fundamental Analysis", ""]
    fm = s["2_financial_metrics"]
    val = fm.get("valuation", {})
    lines.append(f"- Trailing P/E: {val.get('trailing_pe', {}).get('value')} | Forward P/E: {val.get('forward_pe', {}).get('value')}")
    lines.append(f"- P/B: {val.get('price_to_book', {}).get('value')} | Dividend yield: {val.get('dividend_yield_pct', {}).get('value')}%")
    lines.append(f"- PEG: {val.get('peg_ratio', {}).get('value')} | EV/EBITDA: {val.get('ev_to_ebitda', {}).get('value')}")
    ranges = fm.get("valuation_historical_ranges", {})
    if ranges.get("trailing_pe_5yr_range"):
        lines.append(f"- 5yr P/E range (derived): {ranges['trailing_pe_5yr_range']['value']}")
    rev_growth = fm.get("revenue_growth_yoy_pct", {}).get("value") if fm.get("revenue_growth_yoy_pct") else {}
    lines.append(f"- Revenue YoY growth by year: {rev_growth}")
    lines.append(f"- Revenue CAGR: {fm.get('revenue_cagr_pct', {}).get('value') if fm.get('revenue_cagr_pct') else 'N/A'}%")
    lines.append(f"- Debt/Equity by year: {fm.get('debt_to_equity_by_year', {}).get('value')}")
    lines.append(f"- Current ratio by year: {fm.get('current_ratio_by_year', {}).get('value')}")

    lines += ["", "## 3. Technical Analysis", ""]
    ta_data = s["3_technical_analysis"]
    trend = ta_data.get("trend", {})
    for k in ("short_term_20d", "medium_term_50d", "long_term_200d"):
        v = trend.get(k)
        lines.append(f"- {k.replace('_', ' ')}: {v['value'] if v else 'N/A'}")
    ind = ta_data.get("indicators", {})
    if ind:
        lines.append(f"- RSI(14): {ind.get('rsi_14', {}).get('value')}")
        macd = ind.get("macd", {})
        lines.append(f"- MACD: line {macd.get('macd_line', {}).get('value')}, signal {macd.get('signal_line', {}).get('value')}, hist {macd.get('histogram', {}).get('value')}")
        bb = ind.get("bollinger_bands", {})
        lines.append(f"- Bollinger Bands: upper {bb.get('upper', {}).get('value')}, mid {bb.get('middle', {}).get('value')}, lower {bb.get('lower', {}).get('value')}")
        lines.append(f"- Volume trend: {ind.get('volume_trend', {}).get('signal', {}).get('value')}")
    sr = ta_data.get("support_resistance", {})
    if sr:
        lines.append(f"- Support levels: {sr.get('support', {}).get('value')}")
        lines.append(f"- Resistance levels: {sr.get('resistance', {}).get('value')}")

    lines += ["", "## 4. Market Analysis", ""]
    rp = s["4_market_analysis"].get("relative_performance", {})
    for horizon in ("1y", "3y", "5y"):
        h = rp.get(horizon)
        if h:
            lines.append(f"- {horizon}: {ticker} {h['ticker_return_pct']['value']}% vs SPY {h['spy_return_pct']['value']}% vs sector ETF ({rp.get('sector_etf')}) {h.get('sector_etf_return_pct', {}).get('value') if h.get('sector_etf_return_pct') else 'N/A'}%")
    macro = s["4_market_analysis"].get("macro_backdrop", {})
    for k, v in macro.items():
        lines.append(f"- {k}: {v.get('value')} ({v.get('direction', 'n/a')})")
    lines += ["", s["4_market_analysis"].get("industry_trends_and_risks") or "_Industry trends: not available._"]

    lines += ["", "## 5. Future Prospects", ""]
    lines.append(s["5_future_prospects"].get("planned_innovations_and_partnerships") or "_Not available._")
    lines += ["", "**Risks**", "", s["5_future_prospects"]["risks"].get("recent_risk_developments") or "_Not available._"]

    lines += ["", "## 6. Comparison to Competitors", ""]
    comp = s["6_competitor_comparison"]
    if comp.get("comparison_table"):
        header = ["Ticker", "Gross Margin %", "Net Margin %", "D/E", "P/E", "P/B", "Div Yield %", "Rev CAGR %"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for sym, row in comp["comparison_table"].items():
            lines.append(
                f"| {sym} | {row.get('gross_margin_pct')} | {row.get('net_margin_pct')} | "
                f"{row.get('debt_to_equity')} | {row.get('trailing_pe')} | {row.get('price_to_book')} | "
                f"{row.get('dividend_yield_pct')} | {row.get('revenue_cagr_pct')} |"
            )
    for c in comp.get("other_interesting_alternatives", []):
        lines.append(f"- **{c.get('ticker')}** ({c.get('name')}): {c.get('rationale')}")

    gemini_sources = s["1_company_analysis"].get("gemini_sources") or []
    if gemini_sources:
        lines += ["", "## Sources", ""]
        lines += [f"- [{u}]({u})" for u in gemini_sources]

    if data["data_quality"]:
        lines += ["", "## Data Quality Notes", ""]
        for issue in data["data_quality"]:
            lines.append(f"- **[{issue['severity']}]** `{issue['section']} / {issue['field']}`: {issue['issue']}")

    return "\n".join(lines)


def main() -> None:
    ticker, company_name = get_ticker()
    print(f"Resolved {ticker} -> {company_name}", file=sys.stderr)

    data = analyze_ticker(ticker, company_name)

    out_dir = REPO_ROOT / "output" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{ticker}_analysis.json"
    md_path = out_dir / f"{ticker}_analysis_report.md"

    json_path.write_text(json.dumps(data, indent=2, default=str))
    md_path.write_text(build_report_markdown(data))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if data["data_quality"]:
        print(f"{len(data['data_quality'])} data-quality issue(s) flagged — see the report or JSON.", file=sys.stderr)


if __name__ == "__main__":
    main()

"""The live pieces: prices from Yahoo, the graph through Claude Code."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.vendors.yahoo.market import get_open_close
from tradingagents.default_config import build_default_config
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .runner import Analysis
from .scoring import HORIZONS, Bar, score_rating
from .store import Store

logger = logging.getLogger(__name__)
HOME = Path.home() / ".local/share/tradingagents-shadow"
BENCHMARK = "SPY"
_NEWS_LOOKBACK_DAYS = 7
_PRICE_LOOKBACK_DAYS = 6
_SCORED_STATUSES = ("ok", "degraded")


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def load_bars(symbol: str, start: str, end: str | None = None) -> list[Bar]:
    return [Bar(ts.strftime("%Y-%m-%d"), float(row["Open"]), float(row["Close"]))
            for ts, row in get_open_close(symbol, start, end).iterrows()]


def build_config(home: Path = HOME) -> dict:
    base = build_default_config()
    return {**base, "llm_provider": "claude_code", "quick_think_llm": "claude-sonnet-5-5",
            "deep_think_llm": "claude-opus-5-5", "output_language": "Korean",
            "data_vendors": {**base["data_vendors"], "news_data": "finnhub,yfinance"},
            "results_dir": str(home / "logs"), "data_cache_dir": str(home / "cache"),
            "memory_log_path": str(home / "memory.md")}


def preflight(ticker: str, trade_date: str, config: dict | None = None) -> str | None:
    """Why a run should not start. A rating made without prices or news is not worth its tokens."""
    try:
        days = [b.day for b in load_bars(ticker, _shift(trade_date, -_PRICE_LOOKBACK_DAYS), _shift(trade_date, 1))]
    except Exception as exc:
        return f"price: {type(exc).__name__}"
    if trade_date not in days:
        return f"price: no bar for {trade_date}"
    try:
        with run_config(config or build_config()):
            news = route_to_vendor("get_news", ticker, _shift(trade_date, -_NEWS_LOOKBACK_DAYS), trade_date)
    except Exception as exc:
        return f"news: {type(exc).__name__}"
    return None if str(news).startswith("## ") else "news: no articles"


class _Usage(BaseCallbackHandler):
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_equiv_usd = 0.0

    def on_llm_end(self, response, **kwargs):
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        self.calls += 1
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.api_equiv_usd += message.response_metadata.get("estimated_cost_usd") or 0.0


_KEPT_STATE = ("market_report", "sentiment_report", "news_report", "fundamentals_report", "investment_plan",
               "trader_investment_plan", "final_trade_decision", "final_rating", "memory_note")


def analyze(ticker: str, trade_date: str) -> Analysis:
    usage = _Usage()
    started = time.monotonic()
    graph = TradingAgentsGraph(debug=False, config=build_config(), callbacks=[usage])
    state, rating = graph.propagate(ticker, trade_date)
    stats = {"seconds": round(time.monotonic() - started), "llm_calls": usage.calls,
             "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
             "api_equiv_usd": round(usage.api_equiv_usd, 3)}
    return Analysis({k: state.get(k) for k in _KEPT_STATE}, rating, stats)


def score_all(store: Store) -> int:
    """Score every horizon that has finished trading. Returns how many scores were added."""
    pending = [(r, h) for r in store.ratings() if r["status"] in _SCORED_STATUSES
               for h in HORIZONS if h not in store.scored_horizons(r["ticker"], r["trade_date"])]
    if not pending:
        return 0
    start = min(r["trade_date"] for r, _ in pending)
    bench = load_bars(BENCHMARK, start)
    bars = {t: load_bars(t, start) for t in {r["ticker"] for r, _ in pending}}
    added = 0
    for row, horizon in pending:
        score = score_rating(bars[row["ticker"]], bench, row["trade_date"], horizon, row["rating"])
        if score:
            store.save_score(row["ticker"], row["trade_date"], horizon, entry_day=score.entry_day, ret=score.ret,
                             bench_ret=score.bench_ret, excess=score.excess, hit=score.hit)
            added += 1
    return added

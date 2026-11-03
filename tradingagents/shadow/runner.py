"""One shadow pass: rate each ticker once per session, outside the trading bot's hours."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from tradingagents.agents.rating import RATINGS_5_TIER
from tradingagents.llm_clients.claude_code_client import ClaudeCodeLimitError

from .store import Store

logger = logging.getLogger(__name__)
SEOUL = ZoneInfo("Asia/Seoul")
# The bot shares this subscription and decides from the US open to its forced exit. Its
# session starts 22:30 KST in US summer time and ends 06:00 KST in winter.
_BLOCK_START, _BLOCK_END = time(22, 0), time(6, 5)
_MIN_REPORT_CHARS = 500
_REQUIRED_REPORTS = ("market_report", "sentiment_report", "news_report", "fundamentals_report",
                     "final_trade_decision")
_FINAL_STATUSES = ("ok", "degraded")
_MAX_ERROR_CHARS = 500
_TOOL_FAILURE = "No such tool available"


@dataclass(frozen=True)
class Analysis:
    state: dict
    rating: str
    stats: dict


def in_blocked_window(now: datetime) -> bool:
    local = now.astimezone(SEOUL).time()
    return local >= _BLOCK_START or local < _BLOCK_END


def _quality_problems(analysis: Analysis) -> list[str]:
    thin = [k for k in _REQUIRED_REPORTS if len(analysis.state.get(k) or "") < _MIN_REPORT_CHARS]
    unreadable = [] if analysis.rating in RATINGS_5_TIER else [f"rating={analysis.rating}"]
    broken = [k for k in _REQUIRED_REPORTS if _TOOL_FAILURE in (analysis.state.get(k) or "")]
    return thin + unreadable + [f"{k}: tools failed" for k in broken]


def _is_limit(exc: BaseException | None) -> bool:
    while exc is not None:
        if isinstance(exc, ClaudeCodeLimitError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def run_once(store: Store, tickers: Iterable[str], trade_date: str, *,
             analyze: Callable[[str, str], Analysis],
             preflight: Callable[[str, str], str | None],
             now: datetime | None = None) -> dict[str, str]:
    clock = (lambda: now) if now else (lambda: datetime.now(UTC))
    results: dict[str, str] = {}
    stopped = False

    def record(ticker, status, *, rating=None, state=None, stats=None, error=None):
        store.save_rating(ticker, trade_date, status=status, rating=rating, state=state or {}, stats=stats or {},
                          error=error, finished_at=clock().isoformat())
        results[ticker] = status

    for ticker in tickers:
        existing = store.rating(ticker, trade_date)
        if stopped:
            results[ticker] = "not_started"
        elif existing and existing["status"] in _FINAL_STATUSES:
            results[ticker] = "already_recorded"
        elif in_blocked_window(clock()):
            results[ticker] = "blocked_window"
        elif problem := preflight(ticker, trade_date):
            record(ticker, "skipped_data", error=problem)
        else:
            try:
                analysis = analyze(ticker, trade_date)
            except Exception as exc:
                logger.exception("Shadow run failed for %s", ticker)
                stopped = _is_limit(exc)
                record(ticker, "limit" if stopped else "failed", error=f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS])
                continue
            problems = _quality_problems(analysis)
            record(ticker, "degraded" if problems else "ok", rating=analysis.rating, state=analysis.state,
                   stats=analysis.stats, error=", ".join(problems) or None)
    return results

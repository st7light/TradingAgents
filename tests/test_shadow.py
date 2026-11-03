"""Shadow record: ratings are stored before the session they are scored on, and never traded."""
import os
from datetime import UTC, datetime

import pytest

from tradingagents.llm_clients.claude_code_client import ClaudeCodeLimitError
from tradingagents.shadow.report import build_report
from tradingagents.shadow.runner import Analysis, in_blocked_window, run_once
from tradingagents.shadow.scoring import (
    Bar,
    direction,
    entry_open_utc,
    latest_completed_session,
    score_rating,
)
from tradingagents.shadow.store import Store

FULL = "x" * 600
NOW = datetime(2026, 11, 4, 8, 0, tzinfo=UTC)  # 17:00 KST, 03:00 ET


def _bars(*rows):
    return [Bar(day, o, c) for day, o, c in rows]


def _state(**over):
    return {**dict.fromkeys(("market_report", "sentiment_report", "news_report", "fundamentals_report", "final_trade_decision"), FULL), **over}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "shadow.sqlite3")


def _run(store, tickers=("NVDA",), analyze=None, preflight=None, now=NOW):
    return run_once(store, tickers, "2026-11-03", now=now,
                    analyze=analyze or (lambda t, d: Analysis(_state(), "Buy", {"input_tokens": 5})),
                    preflight=preflight or (lambda t, d: None))


@pytest.mark.parametrize("rating,expected", [("Buy", 1), ("Overweight", 1), ("Hold", 0), ("Underweight", -1),
                                             ("Sell", -1), ("REVIEW", 0)])
def test_direction_maps_each_rating(rating, expected):
    assert direction(rating) == expected


def test_score_enters_at_next_session_open_and_ignores_the_rating_day():
    stock = _bars(("2026-11-03", 50, 999), ("2026-11-04", 100, 110))
    bench = _bars(("2026-11-03", 50, 999), ("2026-11-04", 200, 202))
    score = score_rating(stock, bench, "2026-11-03", 1, "Buy")
    assert score.entry_day == "2026-11-04"
    assert score.ret == pytest.approx(0.10)
    assert score.bench_ret == pytest.approx(0.01)
    assert score.excess == pytest.approx(0.09)
    assert score.hit is True


def test_a_sell_call_hits_when_the_stock_lags_the_benchmark():
    stock = _bars(("2026-11-04", 100, 99))
    bench = _bars(("2026-11-04", 100, 101))
    assert score_rating(stock, bench, "2026-11-03", 1, "Sell").hit is True


def test_hold_has_no_hit():
    stock = _bars(("2026-11-04", 100, 99))
    assert score_rating(stock, stock, "2026-11-03", 1, "Hold").hit is None


def test_score_waits_until_the_whole_horizon_has_traded():
    stock = _bars(("2026-11-04", 100, 110), ("2026-11-05", 110, 120))
    assert score_rating(stock, stock, "2026-11-03", 5, "Buy") is None


def test_score_refuses_a_benchmark_on_another_day():
    stock = _bars(("2026-11-04", 100, 110))
    bench = _bars(("2026-11-05", 100, 110))
    assert score_rating(stock, bench, "2026-11-03", 1, "Buy") is None


def test_latest_session_is_yesterday_until_today_has_closed():
    days = ["2026-11-02", "2026-11-03"]
    during = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)  # 13:00 ET
    after = datetime(2026, 11, 3, 21, 30, tzinfo=UTC)  # 16:30 ET
    assert latest_completed_session(days, during) == "2026-11-02"
    assert latest_completed_session(days, after) == "2026-11-03"


def test_entry_open_is_0930_new_york_in_both_offsets():
    assert entry_open_utc("2026-11-04") == datetime(2026, 11, 4, 14, 30, tzinfo=UTC)
    assert entry_open_utc("2026-07-01") == datetime(2026, 7, 1, 13, 30, tzinfo=UTC)


def test_blocked_window_covers_the_trading_bots_session():
    assert in_blocked_window(datetime(2026, 11, 4, 13, 30, tzinfo=UTC)) is True   # 22:30 KST
    assert in_blocked_window(datetime(2026, 11, 4, 20, 0, tzinfo=UTC)) is True    # 05:00 KST
    assert in_blocked_window(datetime(2026, 11, 4, 8, 0, tzinfo=UTC)) is False    # 17:00 KST


def test_nothing_starts_inside_the_blocked_window(store):
    calls = []
    late = datetime(2026, 11, 4, 14, 0, tzinfo=UTC)
    result = _run(store, analyze=lambda t, d: calls.append(t), now=late)
    assert calls == [] and result == {"NVDA": "blocked_window"}
    assert store.rating("NVDA", "2026-11-03") is None


def test_complete_run_is_recorded_ok(store):
    assert _run(store) == {"NVDA": "ok"}
    row = store.rating("NVDA", "2026-11-03")
    assert (row["status"], row["rating"]) == ("ok", "Buy")
    assert row["stats"]["input_tokens"] == 5
    assert row["state"]["market_report"] == FULL


def test_a_recorded_session_is_not_analysed_twice(store):
    _run(store)
    calls = []
    assert _run(store, analyze=lambda t, d: calls.append(t)) == {"NVDA": "already_recorded"}
    assert calls == []


def test_missing_data_spends_no_model_calls(store):
    calls = []
    result = _run(store, analyze=lambda t, d: calls.append(t), preflight=lambda t, d: "news: unavailable")
    assert calls == [] and result == {"NVDA": "skipped_data"}
    assert store.rating("NVDA", "2026-11-03")["error"] == "news: unavailable"


def test_a_stub_report_marks_the_rating_degraded(store):
    result = _run(store, analyze=lambda t, d: Analysis(_state(market_report="위 보고서 참조"), "Overweight", {}))
    assert result == {"NVDA": "degraded"}
    assert "market_report" in store.rating("NVDA", "2026-11-03")["error"]


def test_an_unreadable_rating_is_degraded(store):
    assert _run(store, analyze=lambda t, d: Analysis(_state(), "REVIEW", {})) == {"NVDA": "degraded"}


def test_usage_limit_stops_the_remaining_tickers(store):
    calls = []

    def analyze(ticker, day):
        calls.append(ticker)
        raise ClaudeCodeLimitError("limit reached")

    result = _run(store, ("NVDA", "AMD"), analyze=analyze)
    assert calls == ["NVDA"]
    assert result == {"NVDA": "limit", "AMD": "not_started"}


def test_one_failure_does_not_stop_the_next_ticker_and_can_be_retried(store):
    def analyze(ticker, day):
        if ticker == "NVDA":
            raise RuntimeError("boom")
        return Analysis(_state(), "Hold", {})

    assert _run(store, ("NVDA", "AMD"), analyze=analyze) == {"NVDA": "failed", "AMD": "ok"}
    assert _run(store, ("NVDA",)) == {"NVDA": "ok"}


def _scored(store, ticker, rating, excess, *, status="ok", finished=None, entry="2026-11-04", horizon=1):
    store.save_rating(ticker, "2026-11-03", status=status, rating=rating, state={}, stats={}, error=None,
                      finished_at=finished or "2026-11-04T08:00:00+00:00")
    bench = 0.0
    store.save_score(ticker, "2026-11-03", horizon, entry_day=entry, ret=excess + bench, bench_ret=bench,
                     excess=excess, hit=None if direction(rating) == 0 else direction(rating) * excess > 0)


def test_report_counts_only_clean_ratings_made_before_the_open(store):
    _scored(store, "NVDA", "Buy", 0.02)
    _scored(store, "AMD", "Sell", -0.01)
    _scored(store, "TSLA", "Buy", -0.50, status="degraded")
    _scored(store, "MU", "Buy", 0.90, finished="2026-11-04T15:00:00+00:00")  # after the 14:30 UTC open
    text = build_report(store)
    assert "강세 | 1" in text and "약세 | 1" in text
    assert "전체(기준선) | 2" in text
    assert "장 시작 뒤 기록" in text
    assert "90.00%" not in text and "-50.00%" not in text


def test_report_says_when_the_sample_is_too_small(store):
    _scored(store, "NVDA", "Buy", 0.02)
    assert "표본 부족" in build_report(store)


def test_only_named_data_keys_are_loaded(tmp_path, monkeypatch):
    from tradingagents.shadow.keys import load_data_keys

    env_file = tmp_path / ".env"
    env_file.write_text('FINNHUB_API_KEY="abc"\nFRED_API_KEY=def\nTOSS_CLIENT_SECRET=no\nUPBIT_SECRET_KEY=no\n')
    for name in ("FINNHUB_API_KEY", "FRED_API_KEY", "TOSS_CLIENT_SECRET", "UPBIT_SECRET_KEY", "SEC_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    assert load_data_keys(env_file) == ["FINNHUB_API_KEY", "FRED_API_KEY"]
    assert os.environ["FINNHUB_API_KEY"] == "abc" and os.environ["FRED_API_KEY"] == "def"
    assert "TOSS_CLIENT_SECRET" not in os.environ and "UPBIT_SECRET_KEY" not in os.environ


def test_an_unreadable_key_file_loads_nothing(tmp_path):
    from tradingagents.shadow.keys import load_data_keys

    assert load_data_keys(tmp_path / "missing") == []


def test_a_report_that_says_its_tools_failed_is_degraded(store):
    broken = "오류 No such tool available " + FULL
    result = _run(store, analyze=lambda t, d: Analysis(_state(market_report=broken), "Buy", {}))
    assert result == {"NVDA": "degraded"}
    assert "market_report: tools failed" in store.rating("NVDA", "2026-11-03")["error"]

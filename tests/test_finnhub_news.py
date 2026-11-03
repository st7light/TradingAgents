"""Finnhub company news: a keyed fallback for when Yahoo's news endpoint is down."""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import requests

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorUnavailableError
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.vendors import finnhub


def _ts(day: int) -> int:
    return int(datetime(2026, 11, day, 12, tzinfo=UTC).timestamp())


def _article(headline, day=2, summary=""):
    return {"headline": headline, "summary": summary, "source": "Reuters", "url": "https://x/1", "datetime": _ts(day)}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    state = SimpleNamespace(news=[], profile={"name": "NVIDIA Corp"}, status=200, seen=[])

    def get(url, *, params, timeout, secret, passthrough=()):
        state.seen.append(SimpleNamespace(url=url, params=params, secret=secret))
        if state.status != 200:
            raise requests.HTTPError(f"HTTP {state.status}")
        body = state.profile if "profile" in url else state.news
        return SimpleNamespace(json=lambda: body)

    monkeypatch.setattr(finnhub, "get_scrubbed", get)
    return state


@pytest.mark.unit
def test_missing_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(VendorNotConfiguredError):
        finnhub.get_news("NVDA", "2026-11-01", "2026-11-03")


@pytest.mark.unit
def test_articles_naming_the_company_come_first_and_the_rest_are_dropped(api):
    api.news = [_article("Coca-Cola dividends"), _article("Nvidia wins deal"),
                _article("Chips rally", summary="NVDA rose")]
    out = finnhub.get_news("NVDA", "2026-11-01", "2026-11-03")
    assert "Nvidia wins deal" in out and "Chips rally" in out and "Coca-Cola" not in out
    assert api.seen[-1].params["token"] == api.seen[-1].secret == "k"


@pytest.mark.unit
def test_articles_after_the_analysis_date_never_leak(api):
    api.news = [_article("Nvidia old", day=2), _article("Nvidia future", day=5)]
    out = finnhub.get_news("NVDA", "2026-11-01", "2026-11-03")
    assert "Nvidia old" in out and "Nvidia future" not in out


@pytest.mark.unit
def test_failed_request_is_unavailable_not_no_news(api):
    api.status = 429
    with pytest.raises(VendorUnavailableError):
        finnhub.get_news("NVDA", "2026-11-01", "2026-11-03")


@pytest.mark.unit
def test_no_matching_article_says_so(api):
    out = finnhub.get_news("NVDA", "2026-11-01", "2026-11-03")
    assert "No Finnhub news" in out


@pytest.mark.unit
def test_router_serves_news_from_finnhub(api):
    api.news = [_article("Nvidia wins deal")]
    set_config({"data_vendors": {"news_data": "finnhub,yfinance"}})
    assert "Nvidia wins deal" in route_to_vendor("get_news", "NVDA", "2026-11-01", "2026-11-03")

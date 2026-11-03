"""Finnhub company news (free key: FINNHUB_API_KEY, https://finnhub.io/register)."""

import os
from datetime import UTC, datetime

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import in_window
from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorUnavailableError
from tradingagents.dataflows.net import get_scrubbed
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.news import _format_articles

_API = "https://finnhub.io/api/v1"
_TIMEOUT = 15


def _get(path: str, params: dict) -> dict | list:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise VendorNotConfiguredError("FINNHUB_API_KEY environment variable is not set")
    try:
        return get_scrubbed(f"{_API}/{path}", params={**params, "token": key}, timeout=_TIMEOUT, secret=key).json()
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"Finnhub {path} request failed: {exc}") from exc


def _names(symbol: str) -> tuple[str, ...]:
    """What an article about the company says: its ticker, and the first word of its name."""
    name = (_get("stock/profile2", {"symbol": symbol}) or {}).get("name") or ""
    return tuple(n.lower() for n in (symbol, *name.split()[:1]))


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """News naming ``ticker`` in the window. Finnhub tags much unrelated news with a symbol."""
    symbol = normalize_symbol(ticker)
    names = _names(symbol)
    start, end = (datetime.strptime(d, "%Y-%m-%d") for d in (start_date, end_date))
    articles = []
    for item in _get("company-news", {"symbol": symbol, "from": start_date, "to": end_date}):
        published = datetime.fromtimestamp(item.get("datetime") or 0, tz=UTC)
        text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()
        if in_window(published, start, end) and any(n in text for n in names):
            articles.append({"title": item.get("headline") or "No title", "summary": item.get("summary") or "",
                             "publisher": item.get("source") or "Unknown", "link": item.get("url") or "",
                             "pub_date": published})
    if not articles:
        return f"No Finnhub news naming {ticker} between {start_date} and {end_date}"
    articles.sort(key=lambda a: a["pub_date"], reverse=True)
    limit = get_config()["news_article_limit"]
    return f"## {ticker} News, from {start_date} to {end_date}:\n\n{_format_articles(articles[:limit])}"

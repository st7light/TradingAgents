"""python -m tradingagents.shadow {run,score,report} [--tickers NVDA,TSLA,AMD]"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime, timedelta

from . import engine
from .keys import KEY_FILE_ENV, load_data_keys
from .report import build_report
from .runner import run_once
from .scoring import latest_completed_session
from .store import Store

_DEFAULT_TICKERS = "NVDA,TSLA,AMD"
_SESSION_LOOKBACK_DAYS = 10


def _run(store: Store, tickers: list[str]) -> dict:
    now = datetime.now(UTC)
    start = (now - timedelta(days=_SESSION_LOOKBACK_DAYS)).date().isoformat()
    session = latest_completed_session([b.day for b in engine.load_bars(engine.BENCHMARK, start)], now)
    if session is None:
        raise SystemExit("no completed session found in recent benchmark prices")
    results = run_once(store, tickers, session, analyze=engine.analyze, preflight=engine.preflight)
    return {"trade_date": session, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and score ratings. Places no orders.")
    parser.add_argument("command", choices=("run", "score", "report"))
    parser.add_argument("--tickers", default=_DEFAULT_TICKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if key_file := os.environ.get(KEY_FILE_ENV):
        print(json.dumps({"data_keys_loaded": load_data_keys(key_file)}), flush=True)
    store = Store(engine.HOME / "shadow.sqlite3")

    if args.command in ("run", "score"):
        print(json.dumps({"at": datetime.now(UTC).isoformat(), "scores_added": engine.score_all(store)}), flush=True)
    if args.command == "run":
        print(json.dumps(_run(store, args.tickers.split(",")), ensure_ascii=False), flush=True)
    if args.command in ("report", "run"):
        print(build_report(store), flush=True)


if __name__ == "__main__":
    main()

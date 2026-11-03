"""SQLite record of ratings and their scores."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings (
  ticker TEXT NOT NULL, trade_date TEXT NOT NULL, status TEXT NOT NULL, rating TEXT, error TEXT,
  finished_at TEXT NOT NULL, stats TEXT NOT NULL, state TEXT NOT NULL,
  PRIMARY KEY (ticker, trade_date));
CREATE TABLE IF NOT EXISTS scores (
  ticker TEXT NOT NULL, trade_date TEXT NOT NULL, horizon INTEGER NOT NULL, entry_day TEXT NOT NULL,
  ret REAL NOT NULL, bench_ret REAL NOT NULL, excess REAL NOT NULL, hit INTEGER,
  PRIMARY KEY (ticker, trade_date, horizon));
"""


def _rating_row(row: sqlite3.Row) -> dict:
    return {**dict(row), "stats": json.loads(row["stats"]), "state": json.loads(row["state"])}


class Store:
    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    def save_rating(self, ticker, trade_date, *, status, rating, state, stats, error, finished_at) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?,?,?,?)",
                (ticker, trade_date, status, rating, error, finished_at,
                 json.dumps(stats, default=str), json.dumps(state, ensure_ascii=False, default=str)))

    def rating(self, ticker, trade_date) -> dict | None:
        row = self._db.execute("SELECT * FROM ratings WHERE ticker=? AND trade_date=?",
                               (ticker, trade_date)).fetchone()
        return _rating_row(row) if row else None

    def ratings(self) -> list[dict]:
        rows = self._db.execute("SELECT * FROM ratings ORDER BY trade_date, ticker").fetchall()
        return [_rating_row(r) for r in rows]

    def save_score(self, ticker, trade_date, horizon, *, entry_day, ret, bench_ret, excess, hit) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)",
                             (ticker, trade_date, horizon, entry_day, ret, bench_ret, excess, hit))

    def scored_horizons(self, ticker, trade_date) -> set[int]:
        rows = self._db.execute("SELECT horizon FROM scores WHERE ticker=? AND trade_date=?",
                                (ticker, trade_date)).fetchall()
        return {r["horizon"] for r in rows}

    def scores(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT s.*, r.rating, r.status, r.finished_at FROM scores s "
            "JOIN ratings r USING (ticker, trade_date) ORDER BY s.trade_date, s.ticker").fetchall()
        return [dict(r) for r in rows]

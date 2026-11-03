"""Text summary of the record so far."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean

from .scoring import HORIZONS, direction, entry_open_utc
from .store import Store

MIN_SAMPLE = 30  # below this a difference in means says little; one sector day moves every ticker together
_GROUPS = (("강세", lambda s: direction(s["rating"]) > 0), ("중립", lambda s: direction(s["rating"]) == 0),
           ("약세", lambda s: direction(s["rating"]) < 0), ("전체(기준선)", lambda s: True))


def _is_late(score: dict) -> bool:
    return datetime.fromisoformat(score["finished_at"]) >= entry_open_utc(score["entry_day"])


def _line(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"{name} | 0 | - | -"
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    hit_rate = f"{mean(hits):.0%}" if hits else "-"
    return f"{name} | {len(rows)} | {mean(r['excess'] for r in rows):+.2%} | {hit_rate}"


def build_report(store: Store) -> str:
    ratings = store.ratings()
    scores = store.scores()
    clean = [s for s in scores if s["status"] == "ok" and not _is_late(s)]
    late = {(s["ticker"], s["trade_date"]) for s in scores if s["status"] == "ok" and _is_late(s)}
    tokens = [r["stats"].get("input_tokens", 0) + r["stats"].get("output_tokens", 0)
              for r in ratings if r["stats"]]
    lines = [f"기록 {len(ratings)}건: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["status"] for r in ratings).items()))]
    if tokens:
        lines.append(f"실행당 평균 토큰 {mean(tokens):,.0f}")
    lines.append(f"장 시작 뒤 기록이라 제외 {len(late)}건, 품질 미달은 채점에서 제외")
    for horizon in HORIZONS:
        rows = [s for s in clean if s["horizon"] == horizon]
        verdict = " (표본 부족: 결론 금지)" if len(rows) < MIN_SAMPLE else ""
        lines += ["", f"[{horizon}거래일, 다음 시가 진입, SPY 대비]{verdict}", "구분 | 건수 | 평균 초과수익 | 적중률"]
        lines += [_line(name, [r for r in rows if keep(r)]) for name, keep in _GROUPS]
    return "\n".join(lines)

"""Pure scoring of a stored rating against the sessions that followed it."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
HORIZONS = (1, 5, 20)  # sessions held, counting the entry session
_SESSION_SETTLED = time(16, 15)  # the day's bar is final a little after the 16:00 close
_DIRECTIONS = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}


@dataclass(frozen=True)
class Bar:
    day: str
    open: float
    close: float


@dataclass(frozen=True)
class Score:
    entry_day: str
    ret: float
    bench_ret: float
    excess: float
    hit: bool | None


def direction(rating: str) -> int:
    return _DIRECTIONS.get(rating, 0)


def _holding_return(bars: list[Bar]) -> float:
    return bars[-1].close / bars[0].open - 1


def score_rating(stock: list[Bar], bench: list[Bar], trade_date: str, horizon: int, rating: str) -> Score | None:
    """Enter at the open of the session after ``trade_date``; None until ``horizon`` sessions have traded."""
    stock_after = [b for b in stock if b.day > trade_date][:horizon]
    bench_after = [b for b in bench if b.day > trade_date][:horizon]
    if len(stock_after) < horizon or [b.day for b in stock_after] != [b.day for b in bench_after]:
        return None
    ret, bench_ret = _holding_return(stock_after), _holding_return(bench_after)
    excess = ret - bench_ret
    sign = direction(rating)
    return Score(stock_after[0].day, ret, bench_ret, excess, None if sign == 0 else sign * excess > 0)


def latest_completed_session(days: list[str], now: datetime) -> str | None:
    """The newest session in ``days`` whose close has passed at ``now``."""
    local = now.astimezone(NEW_YORK)
    today = local.date().isoformat()
    done = [d for d in days if d < today or (d == today and local.time() >= _SESSION_SETTLED)]
    return max(done, default=None)


def entry_open_utc(day: str) -> datetime:
    opening = datetime.combine(datetime.fromisoformat(day).date(), time(9, 30), NEW_YORK)
    return opening.astimezone(UTC)

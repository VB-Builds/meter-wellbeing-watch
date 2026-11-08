"""Local-clock helpers: Europe/Zurich days (23/24/25 h), sunrise/sunset in elapsed minutes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

ZRH = ZoneInfo("Europe/Zurich")
LAT, LON = 47.37, 8.54  # Zurich; only used for the seasonal daylight curve


@dataclass(frozen=True)
class Day:
    date: date
    start_utc: datetime  # local midnight, in UTC
    n_min: int  # 1380 (spring forward), 1440, 1500 (fall back)

    @property
    def kind(self) -> str:
        return {1380: "spring", 1500: "fall"}.get(self.n_min, "normal")

    @property
    def doy(self) -> int:
        return self.date.timetuple().tm_yday


def local_midnight_utc(d: date) -> datetime:
    return datetime.combine(d, time(0), tzinfo=ZRH).astimezone(timezone.utc)


def day_table(start: date, n_days: int) -> list[Day]:
    out = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        s = local_midnight_utc(d)
        e = local_midnight_utc(d + timedelta(days=1))
        out.append(Day(d, s, int(round((e - s).total_seconds() / 60))))
    return out


def clock_minutes(day: Day) -> np.ndarray:
    """Local wall-clock minute-of-day for every elapsed minute of the day."""
    if day.kind == "spring":  # 02:00 -> 03:00
        return np.concatenate([np.arange(120), np.arange(180, 1440)])
    if day.kind == "fall":  # 03:00 -> 02:00 (02:00-03:00 occurs twice)
        return np.concatenate([np.arange(180), np.arange(120, 1440)])
    return np.arange(1440)


def clock_to_idx(day: Day, clock_min: float) -> int:
    """Elapsed-minute index of a wall-clock minute (people follow the wall clock)."""
    c = int(round(clock_min))
    if day.kind == "spring":
        idx = c - 60 if c >= 180 else min(c, 120)
    elif day.kind == "fall":
        idx = c + 60 if c >= 180 else c
    else:
        idx = c
    return int(min(max(idx, 0), day.n_min - 1))


def sun_idx(day: Day) -> tuple[int, int]:
    """Sunrise and sunset as elapsed-minute indices of the local day (approximate, +-10 min)."""
    n = day.doy
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + n) / 365)
    lat = math.radians(LAT)
    cosh = -math.tan(lat) * math.tan(decl)
    ha = math.degrees(math.acos(min(1.0, max(-1.0, cosh))))  # degrees
    b = 2 * math.pi * (n - 81) / 364
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    noon_utc = 720 - 4 * LON - eot  # minutes after UTC midnight of the date
    utc_midnight = datetime.combine(day.date, time(0), tzinfo=timezone.utc)
    off = (utc_midnight - day.start_utc).total_seconds() / 60  # minutes from local midnight to UTC midnight
    rise = noon_utc - 4 * ha + off
    sset = noon_utc + 4 * ha + off
    return int(rise), int(sset)


def to_utc(d: date, clock_min: float) -> datetime:
    """Wall-clock (date, minute-of-day, may exceed 1440) to a UTC datetime."""
    base = datetime.combine(d, time(0), tzinfo=ZRH)
    return (base + timedelta(minutes=float(clock_min))).astimezone(timezone.utc)


def fmt_clock(clock_min: float) -> str:
    c = int(round(clock_min)) % 1440
    return f"{c // 60:02d}:{c % 60:02d}"

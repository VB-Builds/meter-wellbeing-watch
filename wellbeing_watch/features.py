"""Edge-side feature extraction: 15-min (or finer) load -> 'household activity' flags and daily features.

Only the output of this module (activity slot times + a few daily numbers) is meant to leave the home.
A burst filter is used: a slot is *active* when its mean power exceeds the slots 30 and 45 min before and
after it by more than a learned noise threshold. Long plateaus (heat pump, boiler, TV, lights) cancel out;
short bursts (kettle, hob, microwave) survive. A bump from a heat-pump cycle looks like a burst, which is
the masking effect this repo measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, tzinfo

import numpy as np
import pandas as pd

from .clock import ZRH

MORNING_START = 240  # first-activity search starts 04:00 local


@dataclass(frozen=True)
class ExtractConfig:
    floor_w: float = 150.0  # never call a burst smaller than this (mean W over a 15-min slot)
    noise_q: float = 0.9
    noise_mult: float = 2.0  # threshold = noise_mult * quantile(night bursts of the last 14 nights)
    night_days: int = 14
    neighbour_min: float = 30.0  # k = neighbour_min / resolution slots


@dataclass(frozen=True)
class DayFeatures:
    date: date
    res_min: int
    active_clock: np.ndarray  # wall-clock minute at start of every active slot, in time order
    active_wh: float  # energy of the bursts (above neighbours) in Wh
    thr_w: float  # threshold used that day

    @property
    def first_activity(self) -> float | None:
        a = self.active_clock[self.active_clock >= MORNING_START]
        return float(a[0]) if a.size else None

    @property
    def has_activity(self) -> bool:
        return self.first_activity is not None


def slot_table(ts_utc: pd.DatetimeIndex, tz: tzinfo = ZRH) -> tuple[np.ndarray, np.ndarray]:
    """Local date (days since epoch) and wall-clock minute of each slot start."""
    loc = ts_utc.tz_convert(tz)
    days = loc.normalize().tz_localize(None).to_numpy().astype("datetime64[D]").astype(int)
    return np.asarray(days), (loc.hour * 60 + loc.minute).to_numpy()


def burst_score(w: np.ndarray, res_min: int, neighbour_min: float = 30.0) -> tuple[np.ndarray, int]:
    """w_t - max of the slots k and k+1 steps before/after (k = neighbour_min / resolution).

    Plateaus of about 4+ slots cancel out; the second neighbour absorbs half-filled edge slots.
    Returns (score, lookahead in slots).
    """
    k = max(1, int(round(neighbour_min / res_min)))
    n = w.size
    nb = np.full(n, -np.inf)
    for s in (k, k + 1):
        if n > s:
            nb[s:] = np.maximum(nb[s:], w[:-s])
            nb[:-s] = np.maximum(nb[:-s], w[s:])
    nb = np.where(np.isfinite(nb), nb, w)
    return w - nb, k + 1


def extract(ts_utc: pd.DatetimeIndex, w: np.ndarray, res_min: int, tz: tzinfo = ZRH,
            cfg: ExtractConfig | None = None) -> list[DayFeatures]:
    cfg = cfg or ExtractConfig()
    if len(ts_utc) != len(w):
        raise ValueError("timestamps and load must have the same length")
    if len(w) == 0:
        return []
    w = np.asarray(w, dtype=float)
    if np.isnan(w).any():
        w = pd.Series(w).interpolate(limit_direction="both").to_numpy()
    dayord, clock = slot_table(ts_utc, tz)
    a, _ = burst_score(w, res_min, cfg.neighbour_min)
    floor = cfg.floor_w * min(1.0, 15.0 / res_min)  # averaging over longer slots dilutes bursts and noise alike
    udays, first = np.unique(dayord, return_index=True)
    bounds = list(first) + [len(w)]
    night = [a[bounds[i]:bounds[i + 1]][clock[bounds[i]:bounds[i + 1]] < MORNING_START] for i in range(len(udays))]
    out = []
    for i, du in enumerate(udays):
        lo = max(0, i - cfg.night_days + 1)
        pool = np.concatenate(night[lo:i + 1])
        if pool.size >= 32:
            thr = max(floor, cfg.noise_mult * float(np.quantile(pool, cfg.noise_q)))
        else:
            thr = max(floor, 250.0 * min(1.0, 15.0 / res_min))
        sl = slice(bounds[i], bounds[i + 1])
        flag = a[sl] > thr
        out.append(DayFeatures(
            date=date.fromordinal(int(du) + date(1970, 1, 1).toordinal()),
            res_min=res_min,
            active_clock=clock[sl][flag].astype(float),
            active_wh=float(a[sl][flag].sum() * res_min / 60.0),
            thr_w=float(thr),
        ))
    return out

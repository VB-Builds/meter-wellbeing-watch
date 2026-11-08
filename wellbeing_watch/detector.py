"""Routine learning, inactivity scoring, away handling and the escalation ladder.

Morning detector: first-activity time ~ Normal with unknown mean/variance (Normal-Inverse-Gamma prior,
28-day rolling window) -> Student-t predictive. Alert time = predictive (1 - alpha) quantile.
Gap detector: a break in activity longer than 1.25x the longest break seen in the last 28 days.
Ladder: gentle check-in to the person -> wait -> relative. Nothing here calls emergency services.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Protocol

import numpy as np

from .clock import fmt_clock, to_utc
from .features import MORNING_START, DayFeatures
from .privacy import RETENTION_DAYS, Consent, ConsentError


@lru_cache(maxsize=None)
def _t_cdf_table(nu: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 3000.0, 600_001)
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    pdf = np.exp(c - (nu + 1) / 2 * np.log1p(x * x / nu))
    cdf = 0.5 + np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * (x[1] - x[0]))])
    return x, cdf


def t_ppf(p: float, nu: float) -> float:
    """Quantile of the Student t distribution (numerical, no scipy). Valid for 0.5 <= p < 0.9999."""
    if not 0.5 <= p < 0.9999:
        raise ValueError("p must be in [0.5, 0.9999)")
    x, cdf = _t_cdf_table(float(nu))
    return float(np.interp(p, cdf, x))


@dataclass(frozen=True)
class WatchConfig:
    res_min: int = 15
    lookahead_min: float = 30.0  # burst filter needs 30 min of future data
    data_delay_min: float = 15.0  # meter -> service latency (assumption, see README)
    alpha: float = 0.005  # predictive tail probability for 'no first activity yet'
    min_margin_min: float = 75.0  # never alert earlier than mean + this
    latest_morning_min: float = 14 * 60  # cap on the morning alert time
    prior_mean: float = 480.0
    prior_sd: float = 60.0
    prior_kappa: float = 1.0
    prior_nu: float = 3.0
    min_scale: float = 15.0
    gap_mult: float = 1.25
    gap_min: float = 150.0
    gap_max: float = 720.0
    gap_window_end: float = 22 * 60 + 30
    checkin_wait_min: float = 45.0
    away_cap_days: int = 21
    learning_days: int = 14  # no alerts until this many usable days were seen
    max_usual_by_min: float = 12 * 60  # if the 'usual by' time is later, the signal is too weak to promise anything
    window_days: int = RETENTION_DAYS

    @property
    def latency(self) -> float:
        return self.res_min + self.lookahead_min + self.data_delay_min


class Responder(Protocol):
    def responds(self, d: date, clock_min: float) -> str | None:
        """Reply to the check-in SMS: 'ok', 'away' or None (no reply)."""


class NoResponder:
    def responds(self, d: date, clock_min: float) -> str | None:
        return None


@dataclass
class Alert:
    day: date
    kind: str  # morning | gap
    trigger_clock: float
    checkin_clock: float
    outcome: str  # cancelled_activity | cancelled_reply | away_mode | relative_notified
    relative_clock: float | None
    text: str

    @property
    def checkin_utc(self):
        return to_utc(self.day, self.checkin_clock)

    @property
    def relative_utc(self):
        return None if self.relative_clock is None else to_utc(self.day, self.relative_clock)


@dataclass
class DaySummary:
    day: date
    first_activity: float | None
    morning_alert_at: float
    usual_by: float
    active: bool
    gap_limit: float
    active_wh: float
    active_wh_z: float | None


@dataclass
class WatchResult:
    alerts: list[Alert] = field(default_factory=list)
    summaries: list[DaySummary] = field(default_factory=list)
    stopped_on: date | None = None  # processing stopped because consent was withdrawn


class RoutineModel:
    """Rolling-window Bayesian routine model of one household."""

    def __init__(self, cfg: WatchConfig):
        self.cfg = cfg
        self.first: deque[float] = deque(maxlen=cfg.window_days)
        self.gaps: deque[float] = deque(maxlen=cfg.window_days)
        self.wh: deque[float] = deque(maxlen=cfg.window_days)

    def update(self, f: DayFeatures) -> None:
        fa = f.first_activity
        if fa is None:
            return
        self.first.append(fa)
        self.gaps.append(max_gap(f, self.cfg.gap_window_end))
        self.wh.append(f.active_wh)

    def predictive(self) -> tuple[float, float, float]:
        """(location, scale, nu) of the Student-t predictive for tomorrow's first activity."""
        c = self.cfg
        x = np.asarray(self.first, dtype=float)
        n = x.size
        k0, nu0 = c.prior_kappa, c.prior_nu
        a0 = nu0 / 2
        b0 = a0 * c.prior_sd**2
        if n == 0:
            mu_n, kn, an, bn = c.prior_mean, k0, a0, b0
        else:
            xb = x.mean()
            kn = k0 + n
            mu_n = (k0 * c.prior_mean + n * xb) / kn
            an = a0 + n / 2
            bn = b0 + 0.5 * ((x - xb) ** 2).sum() + k0 * n * (xb - c.prior_mean) ** 2 / (2 * kn)
        scale = math.sqrt(bn * (kn + 1) / (an * kn))
        return float(mu_n), max(scale, c.min_scale), 2 * an

    def morning_threshold(self) -> tuple[float, float]:
        """(alert time, 'usually by' time), both wall-clock minutes."""
        c = self.cfg
        mu, sc, nu = self.predictive()
        tau = mu + t_ppf(1 - c.alpha, nu) * sc
        tau = min(max(tau, mu + c.min_margin_min), c.latest_morning_min)
        return float(tau), float(mu + t_ppf(0.90, nu) * sc)

    def gap_limit(self) -> float:
        c = self.cfg
        if not self.gaps:
            return c.gap_max
        return float(min(max(c.gap_mult * max(self.gaps), c.gap_min), c.gap_max))

    def wh_z(self, wh: float) -> float | None:
        if len(self.wh) < 7:
            return None
        v = np.asarray(self.wh)
        sd = v.std(ddof=1)
        return None if sd < 1e-9 else float((wh - v.mean()) / sd)


def max_gap(f: DayFeatures, window_end: float = 22 * 60 + 30) -> float:
    """Longest break (minutes) between first activity and ``window_end``, including the quiet evening tail."""
    a = np.sort(f.active_clock[(f.active_clock >= MORNING_START)])
    if a.size == 0:
        return 0.0
    ends = a + f.res_min
    nxt = np.append(a[1:], max(window_end, ends[-1]))
    return float(np.max(nxt - ends))


def _morning_text(tau: float, usual: float, prev_last: float | None) -> str:
    s = f"no kettle/cooker-scale household activity by {fmt_clock(tau)}; usually by {fmt_clock(usual)}"
    if prev_last is not None:
        s += f" (last activity yesterday at {fmt_clock(prev_last)})"
    return s


def run_watch(days: list[DayFeatures], cfg: WatchConfig | None = None, *, consent: Consent,
              responder: Responder | None = None, declared_away: frozenset[date] = frozenset()) -> WatchResult:
    """Replay a sequence of consecutive daily feature records through the watch service."""
    cfg = cfg or WatchConfig()
    responder = responder or NoResponder()
    res = WatchResult()
    model = RoutineModel(cfg)
    suppressed = False
    away_days = 0
    prev_last: float | None = None
    lat = cfg.latency
    for f in days:
        if not consent.active_on(f.date):
            if consent.revoked_on is not None and f.date >= consent.revoked_on:
                res.stopped_on = f.date  # withdrawal: stop and drop the learned routine
                model = RoutineModel(cfg)
                break
            raise ConsentError(f"no consent on {f.date}")
        tau, usual = model.morning_threshold()
        glim = model.gap_limit()
        fa = f.first_activity
        act = np.sort(f.active_clock[f.active_clock >= MORNING_START])
        was_suppressed = suppressed
        if was_suppressed:
            away_days += 1
            if away_days > cfg.away_cap_days:
                suppressed, was_suppressed, away_days = False, False, 0
        enough = len(model.first) >= cfg.learning_days and usual <= cfg.max_usual_by_min
        skip = was_suppressed or f.date in declared_away or not enough
        triggers: list[tuple[float, str, str]] = []
        if not skip:
            if fa is None or fa > tau:
                triggers.append((tau, "morning", _morning_text(tau, usual, prev_last)))
            if fa is not None:
                ends = np.append(act + f.res_min, np.inf)
                starts = np.append(act[1:], np.inf)
                for e, s in zip(ends, starts, strict=True):
                    t_gap = e + glim
                    if t_gap > cfg.gap_window_end:
                        break
                    if s > t_gap:
                        triggers.append((t_gap, "gap",
                                         f"no household activity for {glim / 60:.1f} h since {fmt_clock(e)}; "
                                         f"longest usual break {glim / 60:.1f} h"))
                        break
        for trig, kind, text in sorted(triggers)[:3]:
            if suppressed and not was_suppressed:
                break
            check = trig + lat
            wait_end = trig + cfg.checkin_wait_min
            if bool(((act > trig) & (act <= wait_end)).any()):
                res.alerts.append(Alert(f.date, kind, trig, check, "cancelled_activity", None, text))
                continue
            reply = responder.responds(f.date, check)
            if reply == "ok":
                res.alerts.append(Alert(f.date, kind, trig, check, "cancelled_reply", None, text))
                continue
            if reply == "away":
                res.alerts.append(Alert(f.date, kind, trig, check, "away_mode", None, text))
                suppressed, away_days = True, 0
                continue
            res.alerts.append(Alert(f.date, kind, trig, check, "relative_notified", wait_end + lat, text))
            suppressed, away_days = True, 0
        res.summaries.append(DaySummary(f.date, fa, tau, usual, enough, glim, f.active_wh, model.wh_z(f.active_wh)))
        if fa is not None:
            if suppressed:
                suppressed, away_days = False, 0  # activity is back: lift away / alerted state
            if not was_suppressed and f.date not in declared_away:
                model.update(f)
        prev_last = float(act[-1] + f.res_min) if act.size else prev_last
    return res

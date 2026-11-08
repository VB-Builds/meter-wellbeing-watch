"""Measured experiments on simulated households: detection, time-to-alert, false alerts, ablations."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .clock import Day, to_utc
from .detector import WatchConfig, run_watch
from .features import extract
from .privacy import Consent
from .simulator import PERSONAS, Event, Sim, draw_event, load_window, make_household, simulate, with_heat_pump

SIM_START = date(2025, 1, 1)
FIXED_PLUS1 = timezone(timedelta(hours=1))  # a clock that ignores daylight saving
P_REPLY = 0.9  # assumed: a healthy person answers a check-in SMS this often
EVENT_KINDS = ["fall"] * 4 + ["night_fall"] * 3 + ["hospital"]
DETECT_WITHIN_H = 24.0
LEARNING_SKIP_DAYS = 35  # alerts in the first five weeks are not scored


def make_ts(days: list[Day], res: int) -> pd.DatetimeIndex:
    parts = []
    for d in days:
        t0 = np.datetime64(d.start_utc.replace(tzinfo=None), "s")
        parts.append(t0 + (np.arange(d.n_min // res) * res * 60).astype("timedelta64[s]"))
    return pd.DatetimeIndex(np.concatenate(parts)).tz_localize("UTC")


@dataclass
class SimResponder:
    """Assumed human behaviour: a healthy person replies with probability p; after an event, never."""

    sim: Sim
    rng: np.random.Generator
    p_reply: float
    onset_utc: datetime | None = None

    def responds(self, d: date, clock_min: float) -> str | None:
        if self.onset_utc is not None and to_utc(d, clock_min) >= self.onset_utc:
            return None
        if self.rng.random() < self.p_reply:
            return "ok" if self.sim.status[(d - SIM_START).days] == "home" else "away"
        return None


def _declared(sim: Sim, d0: int, d1: int) -> frozenset[date]:
    return frozenset(sim.days[i].date for i in range(d0, d1) if sim.declared[i])


def year_run(sim: Sim, res: int, cfg: WatchConfig, p_reply: float, seed: int, tz=None) -> dict:
    """Event-free run: every alert here is a false alert."""
    days, w = load_window(sim, 0, len(sim.days), res)
    kw = {} if tz is None else {"tz": tz}
    feats = extract(make_ts(days, res), w, res, **kw)
    resp = SimResponder(sim, np.random.default_rng(seed), p_reply)
    r = run_watch(feats, cfg, consent=Consent.simulated(), responder=resp,
                  declared_away=_declared(sim, 0, len(days)))
    scored = [a for a in r.alerts if (a.day - SIM_START).days >= LEARNING_SKIP_DAYS]
    active = [x.active for x in r.summaries[LEARNING_SKIP_DAYS:]]
    rel = [a for a in scored if a.outcome == "relative_notified"]
    return dict(
        years=(len(days) - LEARNING_SKIP_DAYS) / 365.0, active_days=int(sum(active)), total_days=len(active),
        checkins=len(scored), rel=len(rel),
        rel_home=sum(sim.status[(a.day - SIM_START).days] == "home" for a in rel),
        rel_morning=sum(a.kind == "morning" for a in rel), rel_gap=sum(a.kind == "gap" for a in rel),
    )


def event_run(sim: Sim, ev: Event, res: int, cfg: WatchConfig, seed: int, tz=None) -> dict:
    """Inject one event; replay 35 learning days + the event + 3 days; time to first alert."""
    d0, d1 = ev.day - 35, ev.day + 4
    days, w = load_window(sim, d0, d1, res, ev)
    kw = {} if tz is None else {"tz": tz}
    feats = extract(make_ts(days, res), w, res, **kw)
    onset = to_utc(sim.days[ev.day].date, ev.onset_clock)
    resp = SimResponder(sim, np.random.default_rng(seed), P_REPLY, onset_utc=onset)
    r = run_watch(feats, cfg, consent=Consent.simulated(), responder=resp, declared_away=_declared(sim, d0, d1))
    rel_h = chk_h = None
    for a in r.alerts:
        if chk_h is None and a.checkin_utc >= onset:
            chk_h = (a.checkin_utc - onset).total_seconds() / 3600
        if a.relative_utc is not None and a.relative_utc >= onset:
            rel_h = (a.relative_utc - onset).total_seconds() / 3600
            break
    return dict(kind=ev.kind, month=sim.days[ev.day].date.month, rel_h=rel_h, checkin_h=chk_h)


# (name, heat-pump kW, resolution min). 'base' = household without heat pump (electric boiler only)
CONFIGS = (
    [("base", 0.0, 15)]
    + [(f"hp{h}", h, 15) for h in (1.0, 2.0, 3.5)]
    + [(f"res{r}", 0.0, r) for r in (1, 5, 30, 60)]
    + [(f"hp2res{r}", 2.0, r) for r in (1, 5, 30, 60)]
)


def household_job(args: tuple) -> list[dict]:
    persona, seed, n_events, n_days = args
    sim0 = simulate(make_household(persona, seed, hp_kw=0.0), SIM_START, n_days)
    rng = np.random.default_rng(seed + 1)
    events = [draw_event(sim0, k, rng) for k in (EVENT_KINDS * ((n_events + 7) // 8))[:n_events]]
    rows = []
    cache: dict[float, Sim] = {0.0: sim0}
    for name, hp, res in CONFIGS:
        if hp not in cache:
            cache[hp] = with_heat_pump(sim0, hp)
        sim = cache[hp]
        cfg = WatchConfig(res_min=res)
        for pr in (P_REPLY, 0.0):
            rows.append(dict(table="year", config=name, persona=persona, p_reply=pr,
                             **year_run(sim, res, cfg, pr, seed + 5)))
        for i, ev in enumerate(events):
            rows.append(dict(table="event", config=name, persona=persona,
                             **event_run(sim, ev, res, cfg, seed + 10 + i)))
    # daylight-saving ablation: a service that ignores local clock changes (fixed UTC+1)
    rows.append(dict(table="year", config="fixed_clock", persona=persona, p_reply=P_REPLY,
                     **year_run(sim0, 15, WatchConfig(), P_REPLY, seed + 5, FIXED_PLUS1)))
    return rows


def run_all(n_hh: int = 8, n_events: int = 8, n_days: int = 365, workers: int | None = None, seed0: int = 1000):
    jobs = [(p, seed0 + 100 * pi + i, n_events, n_days) for pi, p in enumerate(PERSONAS) for i in range(n_hh)]
    if workers == 1:
        out = [household_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(household_job, jobs, chunksize=1))
    return pd.DataFrame([r for rows in out for r in rows])


def _ev_stats(e: pd.DataFrame) -> dict:
    det = e.rel_h.notna() & (e.rel_h <= DETECT_WITHIN_H)
    d = e.rel_h[det]
    return dict(
        n=int(len(e)), detection_rate=float(det.mean()) if len(e) else None,
        median_tta_h=float(d.median()) if len(d) else None,
        p90_tta_h=float(d.quantile(0.9)) if len(d) else None,
        median_checkin_h=float(e.checkin_h.dropna().median()) if e.checkin_h.notna().any() else None,
    )


def _yr_stats(y: pd.DataFrame, pr: float) -> dict:
    y = y[y.p_reply == pr]
    yrs = float(y.years.sum())
    if yrs <= 0:
        return dict(household_years=0.0, active_share=0.0, false_relative_per_hh_year=0.0,
                    false_relative_home_days_per_hh_year=0.0, checkins_per_hh_year=0.0, morning_share=0.0)
    return dict(
        household_years=round(yrs, 1), active_share=float(y.active_days.sum() / max(1, y.total_days.sum())),
        false_relative_per_hh_year=float(y.rel.sum() / yrs),
        false_relative_home_days_per_hh_year=float(y.rel_home.sum() / yrs),
        checkins_per_hh_year=float(y.checkins.sum() / yrs),
        morning_share=float(y.rel_morning.sum() / max(1, y.rel.sum())),
    )


def summarise(df: pd.DataFrame) -> dict:
    yr, ev = df[df.table == "year"], df[df.table == "event"]
    out: dict = {"assumptions": {"p_reply_healthy": P_REPLY, "detect_within_h": DETECT_WITHIN_H},
                 "base_definition": "household without heat pump (electric boiler), 15-min data"}
    base_e, base_y = ev[ev.config == "base"], yr[yr.config == "base"]
    out["base_all"] = {**_ev_stats(base_e), "reply0.9": _yr_stats(base_y, P_REPLY), "reply0.0": _yr_stats(base_y, 0.0)}
    out["base_by_persona"] = {
        p: {**_ev_stats(base_e[base_e.persona == p]),
            "reply0.9": _yr_stats(base_y[base_y.persona == p], P_REPLY),
            "reply0.0": _yr_stats(base_y[base_y.persona == p], 0.0)} for p in PERSONAS}
    out["base_by_event"] = {k: _ev_stats(base_e[base_e.kind == k]) for k in sorted(base_e.kind.unique())}
    out["heat_pump"] = {}
    for name in ("base", "hp1.0", "hp2.0", "hp3.5"):
        e, y = ev[ev.config == name], yr[yr.config == name]
        winter = e[e.month.isin([11, 12, 1, 2, 3])]
        out["heat_pump"][name] = {**_ev_stats(e), "winter": _ev_stats(winter), "reply0.9": _yr_stats(y, P_REPLY)}
    out["resolution"] = {}
    for tag, names in (("boiler_only", ("res1", "res5", "base", "res30", "res60")),
                       ("heat_pump_2kw", ("hp2res1", "hp2res5", "hp2.0", "hp2res30", "hp2res60"))):
        out["resolution"][tag] = {}
        for name, r in zip(names, (1, 5, 15, 30, 60), strict=True):
            e, y = ev[ev.config == name], yr[yr.config == name]
            out["resolution"][tag][str(r)] = {**_ev_stats(e), "reply0.9": _yr_stats(y, P_REPLY)}
    out["dst"] = {"local_clock": _yr_stats(base_y, P_REPLY),
                  "fixed_clock": _yr_stats(yr[yr.config == "fixed_clock"], P_REPLY)}
    return out


def save(df: pd.DataFrame, path_json: str, path_csv: str | None = None) -> dict:
    s = summarise(df)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    if path_csv:
        df.to_csv(path_csv, index=False)
    return s

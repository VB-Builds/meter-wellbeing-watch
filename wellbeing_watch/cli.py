"""Command line: demo (reproduces README numbers), simulate (write a 15-min CSV), watch (run on a CSV)."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .clock import fmt_clock
from .detector import WatchConfig, run_watch
from .features import extract
from .privacy import Consent
from .simulator import PERSONAS, load_window, make_household, simulate

BANNER = (
    "NOT a medical or emergency device. Simulated data only. Opt-in; nothing here calls emergency services."
)


def _tables(s: dict) -> str:
    b = s["base_all"]
    def fmt_tta(v):
        return "n/a" if v["median_tta_h"] is None else f"{v['median_tta_h']:.1f} h"

    lines = [BANNER, "", f"Base case: {s['base_definition']} (n={b['n']} simulated events)",
             f"  detection within 24 h : {b['detection_rate']:.0%}",
             f"  median time to relative alert : {b['median_tta_h']:.1f} h  (p90 {b['p90_tta_h']:.1f} h)",
             f"  false relative alerts / household-year : {b['reply0.9']['false_relative_per_hh_year']:.2f}"
             f" (healthy person answers 90% of check-ins), {b['reply0.0']['false_relative_per_hh_year']:.2f}"
             " (nobody ever answers)",
             f"  check-in SMS / household-year : {b['reply0.9']['checkins_per_hh_year']:.1f}", "", "By persona"]
    for p, v in s["base_by_persona"].items():
        lines.append(f"  {p:13s} detect {v['detection_rate']:.0%}  median {fmt_tta(v)}  "
                     f"false {v['reply0.9']['false_relative_per_hh_year']:.2f}/yr  "
                     f"worst-case {v['reply0.0']['false_relative_per_hh_year']:.2f}/yr")
    lines += ["", "Heat-pump size (electrical rating), 15-min data"]
    for k, v in s["heat_pump"].items():
        lines.append(f"  {k:6s} detect {v['detection_rate']:.0%}  winter {v['winter']['detection_rate']:.0%}  "
                     f"median {fmt_tta(v)}  false {v['reply0.9']['false_relative_per_hh_year']:.2f}/yr  "
                     f"service active {v['reply0.9']['active_share']:.0%} of days")
    for tag, rows in s["resolution"].items():
        lines += ["", f"Meter resolution ({tag})"]
        for k, v in rows.items():
            lines.append(f"  {k:>2s} min  detect {v['detection_rate']:.0%}  median {fmt_tta(v)}  "
                         f"false {v['reply0.9']['false_relative_per_hh_year']:.2f}/yr")
    d = s["dst"]
    lines += ["", f"Daylight saving: local clock {d['local_clock']['false_relative_per_hh_year']:.2f}/yr, "
                  f"fixed-offset clock {d['fixed_clock']['false_relative_per_hh_year']:.2f}/yr false relative alerts"]
    return "\n".join(lines)


def cmd_demo(a: argparse.Namespace) -> int:
    from . import evaluate, figures

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_hh, n_ev, n_days = (2, 4, 200) if a.quick else (a.households, a.events, 365)
    df = evaluate.run_all(n_hh=n_hh, n_events=n_ev, n_days=n_days, workers=a.workers)
    s = evaluate.save(df, str(out / "results.json"), None if a.quick else str(out / "results.csv"))
    print(_tables(s))
    if not a.no_figures:
        figures.make_all(df, s, Path(a.docs))
        print(f"\nfigures written to {a.docs}")
    return 0


def cmd_simulate(a: argparse.Namespace) -> int:
    sim = simulate(make_household(a.persona, a.seed, hp_kw=a.heat_pump_kw), date.fromisoformat(a.start), a.days)
    days, w = load_window(sim, 0, a.days, 15)
    from .evaluate import make_ts

    ts = make_ts(days, 15)
    pd.DataFrame({"timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "mean_w": np.round(w, 1)}).to_csv(a.out, index=False)
    print(f"wrote {len(w)} simulated 15-min values to {a.out} (SIMULATED, persona={a.persona})")
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    df = pd.read_csv(a.csv)
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"], utc=True))
    res = int(round((ts[1] - ts[0]).total_seconds() / 60))
    consent = Consent("CLI-USER", date.fromisoformat(a.consent_given), tuple(a.contact))
    feats = extract(ts, df["mean_w"].to_numpy(float), res)
    declared = frozenset(date.fromisoformat(x) for x in a.away) if a.away else frozenset()
    r = run_watch(feats, WatchConfig(res_min=res), consent=consent, declared_away=declared)
    print(BANNER)
    for al in r.alerts:
        who = "relative notified" if al.outcome == "relative_notified" else al.outcome
        print(f"{al.day} {fmt_clock(al.checkin_clock)} check-in -> {who}: {al.text}")
    if not r.alerts:
        print("no alerts")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wellbeing-watch", description=BANNER)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the experiments and write results + figures")
    d.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    d.add_argument("--households", type=int, default=10, help="households per persona")
    d.add_argument("--events", type=int, default=8, help="events per household")
    d.add_argument("--workers", type=int, default=None)
    d.add_argument("--out", default="results")
    d.add_argument("--docs", default="docs")
    d.add_argument("--no-figures", action="store_true")
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("simulate", help="write a simulated 15-min load CSV")
    s.add_argument("--persona", choices=sorted(PERSONAS), default="early_riser")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--start", default="2025-01-01")
    s.add_argument("--heat-pump-kw", type=float, default=0.0)
    s.add_argument("--out", default="simulated_load.csv")
    s.set_defaults(fn=cmd_simulate)
    w = sub.add_parser("watch", help="run the watch on a CSV (timestamp_utc, mean_w); needs explicit consent date")
    w.add_argument("csv")
    w.add_argument("--consent-given", required=True, help="date (YYYY-MM-DD) the person opted in")
    w.add_argument("--contact", action="append", default=["relative-1"])
    w.add_argument("--away", action="append", help="declared absence date, repeatable")
    w.set_defaults(fn=cmd_watch)
    a = p.parse_args(argv)
    return a.fn(a)




if __name__ == "__main__":
    sys.exit(main())

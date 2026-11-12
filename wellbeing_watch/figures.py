"""Figures and linkedin.json, all generated from measured results."""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .clock import clock_to_idx, fmt_clock, to_utc  # noqa: E402
from .detector import WatchConfig, run_watch  # noqa: E402
from .evaluate import SIM_START, SimResponder, make_ts  # noqa: E402
from .features import extract, slot_table  # noqa: E402
from .privacy import Consent  # noqa: E402
from .simulator import Event, load_window, make_household, simulate  # noqa: E402

C = dict(bg="#16213A", grid="#2A3750", text="#F8FAFC", muted="#A9B6CB", accent="#38BDF8", green="#34D399",
         amber="#FBBF24", red="#F87171", grey="#64748B", violet="#A78BFA")
REPO = "meter-wellbeing-watch"
PERSONA_LABEL = {"early_riser": "early riser", "late_sleeper": "late sleeper", "irregular": "irregular",
                 "holiday_goer": "holiday goer", "guest_visits": "guest visits"}


def _style() -> None:
    warnings.filterwarnings("ignore", message="findfont")
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": C["text"], "axes.labelcolor": C["muted"],
        "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.edgecolor": C["grid"],
        "axes.facecolor": C["bg"], "figure.facecolor": C["bg"], "savefig.facecolor": C["bg"],
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 11.5,
    })


def hook_case(seed: int = 1203, hp_kw: float = 0.0, event_day: int = 135) -> dict:
    """One simulated week: six normal days, an evening stop at 21:40, and the morning the alert fires."""
    sim = simulate(make_household("holiday_goer", seed, hp_kw=hp_kw), SIM_START, 150)
    d = event_day
    ev = Event("fall", d - 1, clock_to_idx(sim.days[d - 1], 21 * 60 + 40), 21 * 60 + 40)
    d0, d1 = d - 40, d + 2
    days, w = load_window(sim, d0, d1, 15, ev)
    _, w_norm = load_window(sim, d0, d1, 15, None)
    feats = extract(make_ts(days, 15), w, 15)

    onset = to_utc(sim.days[d - 1].date, ev.onset_clock)
    resp = SimResponder(sim, np.random.default_rng(1), 0.9, onset_utc=onset)
    res = run_watch(feats, WatchConfig(), consent=Consent.simulated(), responder=resp)
    alert = next(a for a in res.alerts if a.day == sim.days[d].date and a.outcome == "relative_notified")
    ts = make_ts(days, 15)
    _, clock = slot_table(ts)
    n = len(days)
    per_day = 96
    wn = w_norm.reshape(n, per_day)
    band_lo = np.quantile(wn[:34], 0.10, axis=0)
    band_hi = np.quantile(wn[:34], 0.90, axis=0)
    sl = slice(n - 2 - 6, n - 1)  # six normal days + the event day (the last window day is spare)
    return dict(w=(w.reshape(n, per_day)[sl]).ravel(), band_lo=band_lo, band_hi=band_hi, alert=alert,
                dates=[days[i].date for i in range(n)][sl], onset_clock=ev.onset_clock,
                summary=next(s for s in res.summaries if s.day == sim.days[d].date))


def draw_week(ax, h: dict, fs: float = 11.5, annotate_box: bool = True) -> None:
    x = np.arange(h["w"].size) * 0.25  # hours
    ndays = h["w"].size // 96
    lo, hi = np.tile(h["band_lo"], ndays) / 1000, np.tile(h["band_hi"], ndays) / 1000
    ax.fill_between(x, lo, hi, step="post", color=C["violet"], alpha=0.30, lw=0, label="usual range (10-90 %)")
    ax.step(x, h["w"] / 1000, where="post", color=C["accent"], lw=1.5, label="15-min mean load")
    a = h["alert"]
    last = ndays - 1
    onset_x = (last - 1) * 24 + h["onset_clock"] / 60
    chk_x = last * 24 + a.checkin_clock / 60
    rel_x = last * 24 + a.relative_clock / 60
    ax.axvline(onset_x, color=C["grey"], ls=":", lw=1.4)
    ax.axvline(chk_x, color=C["amber"], lw=1.8)
    ax.axvline(rel_x, color=C["red"], lw=2.2)
    ymax = max(float(np.max(h["w"])) / 1000, 1.0)
    ax.set_ylim(0, ymax * 1.38)
    ax.set_xlim(0, ndays * 24)
    ax.set_xticks([i * 24 + 12 for i in range(ndays)])
    ax.set_xticklabels([d.strftime("%a %d") for d in h["dates"]], fontsize=fs)
    ax.tick_params(axis="x", length=0)
    for i in range(1, ndays):
        ax.axvline(i * 24, color=C["grid"], lw=1)
    ax.set_ylabel("kW", fontsize=fs)
    ax.annotate("stops at " + fmt_clock(h["onset_clock"]), (onset_x, ymax * 0.80), xytext=(-8, 0),
                textcoords="offset points", ha="right", color=C["muted"], fontsize=fs - 1)
    ax.annotate("check-in SMS " + fmt_clock(a.checkin_clock), (chk_x, ymax * 1.30), xytext=(-8, 0),
                textcoords="offset points", ha="right", color=C["amber"], fontsize=fs - 1)
    ax.annotate("relative notified " + fmt_clock(a.relative_clock), (rel_x, ymax * 1.12), xytext=(-8, 0),
                textcoords="offset points", ha="right", color=C["red"], fontsize=fs, fontweight="bold")
    if annotate_box:
        s = h["summary"]
        ax.text(0.01, 0.97, f"no household activity by {fmt_clock(s.morning_alert_at)}; usually by "
                f"{fmt_clock(s.usual_by)}", transform=ax.transAxes, va="top", color=C["text"], fontsize=fs - 0.5)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=fs - 1.5, labelcolor=C["muted"],
              bbox_to_anchor=(0.45, 0.90))


def _title(fig, title: str, sub: str, size: float = 13.8) -> None:
    fig.text(0.04, 0.965, title, fontsize=size, fontweight="bold", color=C["text"], va="top")
    fig.text(0.04, 0.965 - size * 2.0 / 700, sub, fontsize=11, color=C["muted"], va="top")


def _bars(ax, labels, vals, colors, fmt, fs=11.5, ylim=None):
    b = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=fs)
    for r, v in zip(b, vals, strict=True):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt(v), ha="center", va="bottom", fontsize=fs,
                color=C["text"])
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def make_post_figure(h: dict, s: dict, path: Path) -> None:
    _style()
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=C["bg"])
    _title(fig, "Smart-meter watch-over: the morning the alert fires",
           "SIMULATED household, 15-min meter data only. Not a medical or emergency device.")
    ax = fig.add_axes([0.075, 0.43, 0.9, 0.44])
    draw_week(ax, h, fs=11.5)
    ax2 = fig.add_axes([0.075, 0.075, 0.9, 0.24])
    per = s["base_by_persona"]
    names = list(per)
    vals = [per[n]["median_tta_h"] or 0 for n in names]
    _bars(ax2, [PERSONA_LABEL[n] for n in names], vals, [C["green"]] * len(names), lambda v: f"{v:.1f} h",
          ylim=(0, max(vals) * 1.35))
    ax2.set_title("Median hours from stop to relative alert (simulated, 15-min)", loc="left", fontsize=12, color=C["muted"])
    ax2.set_yticks([])
    fig.savefig(path, dpi=100, facecolor=C["bg"])
    plt.close(fig)


def make_main_figure(h: dict, df: pd.DataFrame, s: dict, path: Path) -> None:
    _style()
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=C["bg"])
    _title(fig, "Opt-in inactivity watch from a 15-minute smart-meter profile",
           "SIMULATED households (5 personas), measured by this repo. Not a medical or emergency device.", size=17)
    ax = fig.add_axes([0.05, 0.56, 0.60, 0.30])
    draw_week(ax, h, fs=11)
    ax.set_title("a  one simulated week: usual range, the stop, and the alert", loc="left", fontsize=12, color=C["muted"])
    # b: detection curve
    axb = fig.add_axes([0.72, 0.56, 0.25, 0.30])
    ev = df[(df.table == "event") & (df.config == "base")]
    xs = np.linspace(0, 24, 97)
    for kind, col, lab in (("fall", C["accent"], "daytime stop"), ("night_fall", C["violet"], "night stop"),
                           ("hospital", C["amber"], "hospital")):
        e = ev[ev.kind == kind]
        y = [(e.rel_h <= x).mean() for x in xs]
        axb.plot(xs, y, color=col, lw=2, label=lab)
    axb.set_ylim(0, 1.02)
    axb.set_xlabel("hours since activity stopped")
    axb.set_ylabel("share alerted")
    axb.set_title("b  time to relative alert", loc="left", fontsize=12, color=C["muted"])
    axb.legend(frameon=False, fontsize=10, labelcolor=C["muted"], loc="lower right")
    # c: false alerts
    axc = fig.add_axes([0.05, 0.10, 0.28, 0.32])
    per = s["base_by_persona"]
    names = list(per)
    w = 0.38
    xs_ = np.arange(len(names))
    v9 = [per[n]["reply0.9"]["false_relative_per_hh_year"] for n in names]
    v0 = [per[n]["reply0.0"]["false_relative_per_hh_year"] for n in names]
    axc.bar(xs_ - w / 2, v9, w, color=C["green"], label="90 % of check-ins answered")
    axc.bar(xs_ + w / 2, v0, w, color=C["amber"], label="nobody answers (worst case)")
    axc.set_xticks(xs_)
    axc.set_xticklabels([PERSONA_LABEL[n].replace(" ", "\n") for n in names], fontsize=10.5)
    axc.set_ylabel("false relative alerts / household-year")
    axc.legend(frameon=False, fontsize=9.5, labelcolor=C["muted"])
    axc.grid(axis="x", visible=False)
    axc.set_title("c  false alerts, event-free years", loc="left", fontsize=12, color=C["muted"])
    # d: heat pump + resolution
    axd = fig.add_axes([0.42, 0.10, 0.55, 0.32])
    hp = s["heat_pump"]
    rs = s["resolution"]
    labs = ["no HP", "1 kW", "2 kW", "3.5 kW", "", "1 min", "5 min", "15 min", "30 min", "60 min"]
    vals = [hp[k]["detection_rate"] for k in ("base", "hp1.0", "hp2.0", "hp3.5")] + [np.nan] + \
           [rs["boiler_only"][k]["detection_rate"] for k in ("1", "5", "15", "30", "60")]
    cols = [C["accent"]] * 4 + [C["bg"]] + [C["green"]] * 5
    b = axd.bar(range(len(vals)), np.nan_to_num(vals), color=cols, width=0.65)
    for r, v in zip(b, vals, strict=True):
        if not np.isnan(v):
            axd.text(r.get_x() + r.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom", fontsize=10.5)
    axd.set_xticks(range(len(vals)))
    axd.set_xticklabels(labs, fontsize=10.5)
    axd.set_ylim(0, 1.12)
    axd.set_ylabel("detected within 24 h")
    axd.grid(axis="x", visible=False)
    axd.text(1.5, 1.02, "heat-pump rating (masking)", ha="center", color=C["accent"], fontsize=10.5)
    axd.text(7.0, 1.02, "meter resolution", ha="center", color=C["green"], fontsize=10.5)
    axd.set_title("d  what hurts detection", loc="left", fontsize=12, color=C["muted"])
    fig.savefig(path, dpi=100, facecolor=C["bg"])
    plt.close(fig)


def write_linkedin(s: dict, path: Path) -> dict:
    b = s["base_all"]
    hp, rs = s["heat_pump"], s["resolution"]
    det, tta = b["detection_rate"], b["median_tta_h"]
    fa = b["reply0.9"]["false_relative_per_hh_year"]
    fa0 = b["reply0.0"]["false_relative_per_hh_year"]
    d = {
        "claim": f"A watch-over service from the 15-min meter profile alerted a relative in {tta:.0f} h (simulated).",
        "data_label": "simulated data",
        "stats": [[f"{det:.0%}", "simulated stops alerted within 24 h", "accent"],
                  [f"{tta:.1f} h", "median time to relative alert", "green"],
                  [f"{fa:.1f}/yr", "false relative alerts per household", "amber"]],
        "limit": "Simulated only; no real household, no clinical or legal validation, not an emergency device.",
        "alt": ("A week of simulated 15-minute household load with a shaded usual range. On the last day nothing "
                "rises above the range after the evening stop, a check-in message goes out, and the relative is "
                "notified later that morning."),
        "hashtags": ["#IoT", "#SmartMeter", "#AgeTech", "#Datenschutz", "#OpenSource"],
    }
    d["post"] = (
        "Many older people live alone, and the first sign that something is wrong is often silence: no kettle, "
        "no cooker, no lights in the morning. Swiss utilities are rolling out smart meters that record the "
        "household load every 15 minutes, with a legal target of 80 percent of metering points by the end "
        "of 2027. Japanese electricity utilities already use such meter data for watch-over services.\n\n"
        "I built an open-source, opt-in version: it learns each household's routine from the 15-minute profile "
        "only (no camera, no wearable), notices unusual inactivity, sends a gentle check-in to the person first "
        "and then notifies a relative, with a plain-text reason such as 'no household activity by 11:05; usually "
        "by 08:10'. Only derived daily-routine features leave the home.\n\n"
        f"Measured on simulated households (5 personas, falls, night events, hospital stays): {det:.0%} of "
        f"stops alerted within 24 h, median {tta:.1f} h to the relative. False relative alerts: {fa:.1f} per "
        f"household-year if 9 in 10 check-ins are answered, {fa0:.1f} if nobody ever answers. A 2 kW heat pump "
        f"cut detection to {hp['hp2.0']['detection_rate']:.0%} (no heat pump: {hp['base']['detection_rate']:.0%}); "
        f"1-minute data reached {rs['boiler_only']['1']['detection_rate']:.0%} and 60-minute data "
        f"{rs['boiler_only']['60']['detection_rate']:.0%}.\n\n"
        "Not verified: any real household, real meter data latency, real replies to SMS, legal review. It is not "
        "a medical or emergency device.\n\n"
        "Would you opt in for a parent, and what would you need to see first?\n\n"
        f"Code: https://github.com/VB-Builds/{REPO}"
    )
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def make_all(df: pd.DataFrame, s: dict, docs: Path) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    h = hook_case()
    make_main_figure(h, df, s, docs / "figure.png")
    make_post_figure(h, s, docs / "post-figure.png")
    write_linkedin(s, docs / "linkedin.json")


_ = date  # imported for type clarity in notebooks

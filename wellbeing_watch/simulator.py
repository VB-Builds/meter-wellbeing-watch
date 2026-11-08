"""Appliance-level household simulator (SIMULATED data only).

Three components are kept separate so that events can be injected cheaply:
  * ``evt``   - person-driven loads with edges (kettle, cooker, oven, washing machine, guest use)
  * ``state`` - person-driven loads that stay on if the person stops (lights, TV)
  * ``bg``    - always-on background (fridge cycles, standby, heat pump / boiler)
Everything is generated at 1-minute resolution on the local wall-clock (DST days are 23/25 h) and
aggregated afterwards, so 15-min vs 1-min comparisons use identical physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .clock import Day, clock_minutes, clock_to_idx, day_table, sun_idx

PERSONAS: dict[str, dict] = {
    # wake / bed: mean and sd in minutes after local midnight
    "early_riser": dict(wake=(345, 20), bed=(1305, 30), p_out=0.30, p_breakfast_hob=0.30, trips=2, guest_p=0.05),
    "late_sleeper": dict(wake=(540, 40), bed=(1410, 30), p_out=0.25, p_breakfast_hob=0.20, trips=2, guest_p=0.05),
    "irregular": dict(wake=(450, 100), bed=(1380, 90), p_out=0.35, p_breakfast_hob=0.15, trips=3, guest_p=0.05),
    "holiday_goer": dict(wake=(435, 25), bed=(1350, 35), p_out=0.30, p_breakfast_hob=0.30, trips=2,
                         guest_p=0.05, holidays=2),
    "guest_visits": dict(wake=(470, 30), bed=(1360, 35), p_out=0.25, p_breakfast_hob=0.30, trips=2, guest_p=0.30),
}


@dataclass
class HouseholdSpec:
    persona: str
    seed: int
    wake: tuple[float, float]
    bed: tuple[float, float]
    p_out: float
    p_breakfast_hob: float
    guest_p: float
    trips: int
    holidays: int
    fridge_w: float
    standby_w: float
    light_w: float
    tv_w: float
    kettle_w: float
    hob_w: float
    dhw_night: bool
    hp_kw: float = 2.0  # electrical heat-pump rating; 0 = no heat pump (boiler only)


def make_household(persona: str, seed: int, hp_kw: float = 2.0) -> HouseholdSpec:
    p = PERSONAS[persona]
    r = np.random.default_rng(seed)
    return HouseholdSpec(
        persona=persona, seed=seed,
        wake=(p["wake"][0] + r.normal(0, 15), p["wake"][1] * r.uniform(0.8, 1.2)),
        bed=(p["bed"][0] + r.normal(0, 15), p["bed"][1] * r.uniform(0.8, 1.2)),
        p_out=p["p_out"], p_breakfast_hob=p["p_breakfast_hob"], guest_p=p["guest_p"],
        trips=p["trips"], holidays=p.get("holidays", 0),
        fridge_w=r.uniform(80, 140), standby_w=r.uniform(45, 95), light_w=r.uniform(60, 160),
        tv_w=r.uniform(60, 140), kettle_w=r.uniform(1800, 2200), hob_w=r.uniform(1400, 2200),
        dhw_night=bool(r.random() < 0.5), hp_kw=hp_kw,
    )


@dataclass
class Sim:
    spec: HouseholdSpec
    days: list[Day]
    evt: np.ndarray
    state: np.ndarray
    bg: np.ndarray
    off: np.ndarray  # start index of every day, plus the end
    status: list[str]  # per day: home | trip | holiday
    declared: list[bool]  # absence announced in advance
    wake_idx: list[int] = field(default_factory=list)  # global index of waking, -1 if none
    bed_idx: list[int] = field(default_factory=list)

    def day_slice(self, d: int) -> slice:
        return slice(int(self.off[d]), int(self.off[d + 1]))


def _add(a: np.ndarray, s: int, dur: float, w: float) -> None:
    s = max(int(s), 0)
    e = min(int(s + max(dur, 1)), a.size)
    if e > s:
        a[s:e] += w


def _add_cycle(a, s, dur, w, duty, period, rng) -> None:
    """Thermostat-style on/off inside [s, s+dur): ``duty`` of each ``period`` minutes."""
    t = int(s)
    end = min(int(s + dur), a.size)
    while t < end:
        on = max(1, int(period * duty * rng.uniform(0.8, 1.2)))
        _add(a, t, min(on, end - t), w)
        t += max(int(period * rng.uniform(0.8, 1.2)), on + 1)


def _wave(n: int, rng, on_mean: float, off_mean: float) -> np.ndarray:
    """Random on/off wave of length n (bool)."""
    k = int(n / (on_mean + off_mean)) * 2 + 8
    on = np.maximum(1, rng.normal(on_mean, on_mean * 0.15, k)).astype(int)
    off = np.maximum(1, rng.normal(off_mean, off_mean * 0.15, k)).astype(int)
    seg = np.empty(2 * k, dtype=int)
    seg[0::2], seg[1::2] = off, on
    w = np.repeat(np.tile([False, True], k), seg)
    start = int(rng.integers(0, int(on_mean + off_mean)))
    return w[start:start + n]


def outdoor_temp(days: list[Day], rng) -> np.ndarray:
    """Daily mean outdoor temperature (deg C): Swiss-Plateau-like seasonal cycle + AR(1) weather."""
    n = len(days)
    doy = np.array([d.doy for d in days])
    seas = 9.5 - 10.0 * np.cos(2 * np.pi * (doy - 20) / 365)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 2.0)
    return seas + ar


def _background(spec: HouseholdSpec, days: list[Day], off: np.ndarray, hp_kw: float, rng) -> np.ndarray:
    n = int(off[-1])
    bg = np.zeros(n, dtype=np.float32)
    drift = np.cumsum(rng.normal(0, 0.6, n // 30 + 2))
    drift = np.clip(drift - drift.mean(), -15, 15)
    bg += spec.standby_w + np.repeat(drift, 30)[:n] + rng.normal(0, 3, n)
    warm = np.concatenate([
        np.full(int(off[i + 1] - off[i]), 1.0 + 0.25 * max(0.0, np.sin(np.pi * (days[i].doy - 80) / 200)))
        for i in range(len(days))
    ])
    bg += _wave(n, rng, 14, 26) * spec.fridge_w * (0.93 + 0.1 * rng.random(n)) * warm
    temps = outdoor_temp(days, rng)
    for i, d in enumerate(days):
        s, e = int(off[i]), int(off[i + 1])
        m = e - s
        if hp_kw > 0:
            demand = float(np.clip((15.0 - temps[i]) / 25.0, 0.0, 1.0))
            duty = min(1.0, demand * 1.15)
            if duty > 0.04:
                p_on = hp_kw * 1000 * (0.30 + 0.60 * demand) * rng.uniform(0.95, 1.05)
                if duty > 0.97:
                    bg[s:e] += p_on
                else:
                    per = rng.uniform(100, 300)  # inverter unit with buffer tank: long cycles
                    on_m = max(per * duty, 50.0)
                    off_m = max(on_m * (1 - duty) / duty, 10.0)
                    bg[s:e] += _wave(m, rng, on_m, off_m) * p_on * (1 + rng.normal(0, 0.03, m))
        # domestic hot water: electric boiler (2 kW, 100 min) or heat-pump DHW (60 min)
        if spec.dhw_night:
            st = clock_to_idx(d, int(rng.integers(60, 5 * 60)))
        else:
            st = clock_to_idx(d, int(rng.integers(12 * 60, 14 * 60)))
        if hp_kw > 0:
            _add(bg[s:e], st, 60, hp_kw * 1000 * 1.1)
        elif rng.random() < 0.8:
            _add(bg[s:e], st, 100, 2000.0)
    return np.maximum(bg, 0)


def _trips(spec: HouseholdSpec, n_days: int, rng):
    """List of (start_day, n_days, kind, declared in advance)."""
    out: list[tuple[int, int, str, bool]] = []
    used = np.zeros(n_days, dtype=bool)

    def place(length: int, kind: str, declared: bool) -> None:
        lo = 30 if n_days >= 60 else 2
        if n_days - length - 3 <= lo:
            return
        for _ in range(50):
            s = int(rng.integers(lo, n_days - length - 3))
            if not used[max(0, s - 3): s + length + 3].any():
                used[s: s + length] = True
                out.append((s, length, kind, declared))
                return

    for _ in range(spec.holidays):
        place(int(rng.integers(7, 15)), "holiday", bool(rng.random() < 0.85))
    for _ in range(spec.trips):
        place(int(rng.integers(2, 5)), "trip", bool(rng.random() < 0.3))
    return out


def simulate(spec: HouseholdSpec, start: date = date(2025, 1, 1), n_days: int = 365,
             hp_kw: float | None = None) -> Sim:
    """Simulate ``n_days`` of one household. ``hp_kw`` overrides the heat pump without changing behaviour."""
    days = day_table(start, n_days)
    off = np.concatenate([[0], np.cumsum([d.n_min for d in days])]).astype(int)
    rng = np.random.default_rng(spec.seed)
    status = ["home"] * n_days
    declared = [False] * n_days
    leave: dict[int, float] = {}
    back: dict[int, float] = {}
    for s, ln, kind, dec in _trips(spec, n_days, rng):
        for k in range(s, s + ln):
            status[k] = kind
            declared[k] = dec
        leave[s] = rng.uniform(9 * 60, 11 * 60)
        back[s + ln - 1] = rng.uniform(15 * 60, 19 * 60)

    n = int(off[-1])
    evt = np.zeros(n, dtype=np.float32)
    state = np.zeros(n, dtype=np.float32)
    wake_idx, bed_idx = [], []
    for i, d in enumerate(days):
        s0, s1 = int(off[i]), int(off[i + 1])
        ev, stt = evt[s0:s1], state[s0:s1]
        if status[i] != "home" and i not in leave and i not in back:
            wake_idx.append(-1)
            bed_idx.append(-1)
            continue
        weekend = d.date.weekday() >= 5
        wake = float(np.clip(rng.normal(spec.wake[0] + (20 if weekend else 0), spec.wake[1]), 4 * 60, 12 * 60))
        bed = float(np.clip(rng.normal(spec.bed[0], spec.bed[1]), wake + 10 * 60, 1439))
        t_leave, t_back = leave.get(i), back.get(i)
        p_start = wake if t_back is None else t_back
        p_end = bed if t_leave is None else t_leave
        present = np.zeros(1440, dtype=bool)
        if p_end > p_start:
            present[int(p_start): int(p_end)] = True
        if status[i] == "home" and rng.random() < spec.p_out:  # shopping, appointment, visit
            o_s = rng.uniform(9.5 * 60, 15 * 60)
            present[int(o_s): int(o_s + rng.uniform(90, 210))] = False
        wake_idx.append(s0 + clock_to_idx(d, wake) if t_back is None else -1)
        bed_idx.append(s0 + clock_to_idx(d, bed) if t_leave is None else -1)

        def idx(c: float, d=d) -> int:
            return clock_to_idx(d, c)

        def at(c: float, dur: float, w: float, present=present, ev=ev, d=d) -> None:
            if 0 <= int(c) < 1440 and present[int(c)]:
                _add(ev, clock_to_idx(d, c), dur, w)

        if t_back is None and p_end > p_start:
            at(wake + rng.uniform(2, 10), rng.uniform(2, 3.5), spec.kettle_w * rng.uniform(0.93, 1.0))
            if rng.random() < 0.9:
                at(wake + rng.uniform(5, 25), rng.uniform(1.5, 3), 800 * rng.uniform(0.8, 1.1))  # toaster
            if rng.random() < spec.p_breakfast_hob:
                s_ = wake + rng.uniform(8, 25)
                if s_ < 1440 and present[int(s_)]:
                    _add_cycle(ev, idx(s_), rng.uniform(6, 12), spec.hob_w * 0.7, 0.7, 4, rng)
            if rng.random() < 0.6:
                at(wake + rng.normal(150, 40), rng.uniform(2, 3.5), spec.kettle_w)
        if rng.random() < 0.65:
            c = rng.normal(12 * 60 + 15, 30)
            if 0 <= c < 1440 and present[int(c)]:
                _add_cycle(ev, idx(c), rng.uniform(8, 20), spec.hob_w * 0.7, 0.7, 4, rng)
        else:
            at(rng.normal(12 * 60 + 20, 30), rng.uniform(3, 5), 1100)  # microwave
        if rng.random() < 0.75:
            at(rng.normal(15 * 60 + 30, 40), rng.uniform(2, 3.5), spec.kettle_w)
        dn = rng.normal(18 * 60 + 15, 40)
        if 0 <= dn < 1440 and present[int(dn)] and rng.random() < 0.92:
            _add_cycle(ev, idx(dn), rng.uniform(15, 35), spec.hob_w * 0.75, 0.7, 4, rng)
            if rng.random() < 0.3:
                _add_cycle(ev, idx(dn + rng.uniform(-5, 10)), rng.uniform(30, 55), 2200, 0.6, 12, rng)
        if rng.random() < 0.2:  # washing machine
            c = rng.uniform(wake + 30, 14 * 60)
            if 0 <= c < 1440 and present[int(c)]:
                _add(ev, idx(c), 18, 2000)
                _add(ev, idx(c) + 18, 60, 250)
        # small everyday loads (kettle, microwave, coffee, hair dryer, vacuum, iron) as a Poisson process
        pres_min = np.flatnonzero(present)
        if pres_min.size:
            for c in rng.choice(pres_min, size=min(rng.poisson(0.30 * pres_min.size / 60), pres_min.size)):
                w_, du_ = [(spec.kettle_w, 2.5), (1100, 3.0), (1300, 5.0), (1500, 4.0), (1200, 20.0), (1000, 15.0)][
                    int(rng.choice(6, p=[0.35, 0.25, 0.15, 0.08, 0.09, 0.08]))]
                _add(ev, idx(c), du_ * rng.uniform(0.7, 1.3), w_ * rng.uniform(0.9, 1.0))
        if status[i] == "home" and rng.random() < spec.guest_p:  # guest: extra kettle + cooking
            g = rng.uniform(10 * 60, 14 * 60)
            for k in range(2):
                _add(ev, idx(g + k * 100), rng.uniform(2, 3.5), spec.kettle_w)
            _add_cycle(ev, idx(g + 30), rng.uniform(20, 40), spec.hob_w * 0.7, 0.7, 4, rng)
        # state loads: lights follow presence + darkness (seasonal), TV in the evening
        rise, sset = sun_idx(d)
        minute = np.arange(d.n_min)
        dark = (minute < rise + 25) | (minute > sset - 25)
        if rng.random() < 0.35:  # dull day: lights on around midday
            a = idx(rng.uniform(10 * 60, 13 * 60))
            dark[a: a + int(rng.uniform(120, 300))] = True
        pres_e = present[clock_minutes(d)]
        lw = spec.light_w * rng.uniform(0.85, 1.15)
        stt += (pres_e & dark) * lw
        stt += (pres_e & dark & ((minute // 90) % 2 == 0)) * (lw * 0.4)  # second room, part of the time
        if rng.random() < 0.85 and t_leave is None:
            tv_s = rng.normal(19 * 60 + 30, 40)
            if 0 <= tv_s < 1440 and present[int(tv_s)]:
                a = idx(tv_s)
                b = min(a + int(max(30, rng.normal(170, 60))), idx(bed))
                stt[a:b] += spec.tv_w * rng.uniform(0.9, 1.1)
    bg = _background(spec, days, off, spec.hp_kw if hp_kw is None else hp_kw, np.random.default_rng(spec.seed + 7919))
    return Sim(spec, days, evt, state, bg, off, status, declared, wake_idx, bed_idx)


def with_heat_pump(sim: Sim, hp_kw: float) -> Sim:
    """Same behaviour, different heat-pump size (separate random stream for the background)."""
    bg = _background(sim.spec, sim.days, sim.off, hp_kw, np.random.default_rng(sim.spec.seed + 7919))
    return Sim(sim.spec, sim.days, sim.evt, sim.state, bg, sim.off, sim.status, sim.declared,
               sim.wake_idx, sim.bed_idx)


@dataclass(frozen=True)
class Event:
    kind: str  # fall | night_fall | hospital
    day: int
    idx_in_day: int  # elapsed minute of the day at which the person stops
    onset_clock: float


def draw_event(sim: Sim, kind: str, rng, lo: int = 40) -> Event:
    """Random event on a 'home' day away from any trip; night events hit between 01:00 and 04:00."""
    homes = [i for i in range(lo, len(sim.days) - 5) if all(s == "home" for s in sim.status[i - 2: i + 3])]
    d = int(rng.choice(homes))
    clock = rng.uniform(60, 240) if kind == "night_fall" else rng.uniform(8 * 60, 20 * 60)
    return Event(kind, d, clock_to_idx(sim.days[d], clock), float(clock))


def aggregate(w1: np.ndarray, day_lens: list[int], res: int, quantise_wh: bool = True) -> np.ndarray:
    """Mean power per ``res``-minute slot, day by day (slots never straddle local midnight)."""
    out, p = [], 0
    for n in day_lens:
        seg = w1[p:p + n]
        p += n
        m = (n // res) * res
        out.append(seg[:m].reshape(-1, res).mean(axis=1))
    x = np.concatenate(out)
    if quantise_wh:  # meter register resolution: 1 Wh per slot
        x = np.round(x * res / 60.0) * 60.0 / res
    return x


def load_window(sim: Sim, d0: int, d1: int, res: int, event: Event | None = None):
    """Total load for days [d0, d1) as (days, mean-W per slot). With ``event`` the person stops at onset."""
    a, b = int(sim.off[d0]), int(sim.off[d1])
    evt = sim.evt[a:b].copy()
    state = sim.state[a:b].copy()
    if event is not None:
        o = int(sim.off[event.day]) - a + event.idx_in_day
        hold = state[o - 1] if o > 0 else 0.0
        evt[o:] = 0.0
        state[o:] = hold  # lights / TV stay on as they were
    total = evt + state + sim.bg[a:b]
    return sim.days[d0:d1], aggregate(total, [d.n_min for d in sim.days[d0:d1]], res)


def slot_clock(days: list[Day], res: int) -> tuple[np.ndarray, np.ndarray]:
    """(day index, wall-clock minute at slot start) for every slot of ``days``."""
    di, cl = [], []
    for k, d in enumerate(days):
        c = clock_minutes(d)[: (d.n_min // res) * res: res]
        cl.append(c)
        di.append(np.full(c.size, k))
    return np.concatenate(di), np.concatenate(cl)

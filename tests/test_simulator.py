from datetime import date

import numpy as np
import pytest

from wellbeing_watch.simulator import (
    PERSONAS,
    aggregate,
    draw_event,
    load_window,
    make_household,
    simulate,
    slot_clock,
    with_heat_pump,
)


@pytest.fixture(scope="module")
def sim():
    return simulate(make_household("holiday_goer", 7, hp_kw=2.0), date(2025, 1, 1), 120)


def test_same_seed_same_series():
    a = simulate(make_household("early_riser", 3), date(2025, 1, 1), 20)
    b = simulate(make_household("early_riser", 3), date(2025, 1, 1), 20)
    assert np.array_equal(a.evt, b.evt) and np.array_equal(a.bg, b.bg)


def test_different_seed_differs():
    a = simulate(make_household("early_riser", 3), date(2025, 1, 1), 20)
    b = simulate(make_household("early_riser", 4), date(2025, 1, 1), 20)
    assert not np.array_equal(a.evt, b.evt)


def test_all_personas_run_and_are_plausible():
    for p in PERSONAS:
        s = simulate(make_household(p, 11, hp_kw=0.0), date(2025, 3, 1), 30)
        _, w = load_window(s, 0, 30, 15)
        assert (w >= 0).all()
        kwh_day = w.mean() * 24 / 1000
        assert 3 < kwh_day < 25, (p, kwh_day)


def test_array_sizes_follow_dst_days():
    s = simulate(make_household("early_riser", 1), date(2025, 3, 28), 5)
    assert s.off[-1] == sum(d.n_min for d in s.days)
    assert 1380 in [d.n_min for d in s.days]
    assert s.evt.size == s.state.size == s.bg.size == s.off[-1]


def test_background_is_nonnegative_and_has_standby(sim):
    assert sim.bg.min() >= 0
    assert np.median(sim.bg) > 40


def test_heat_pump_changes_only_background(sim):
    other = with_heat_pump(sim, 0.0)
    assert np.array_equal(other.evt, sim.evt) and np.array_equal(other.state, sim.state)
    winter = slice(0, 60 * 1440)
    assert sim.bg[winter].mean() > other.bg[winter].mean() + 300


def test_heat_pump_nearly_off_in_summer():
    s = simulate(make_household("early_riser", 5, hp_kw=2.0), date(2025, 7, 1), 20)
    s0 = with_heat_pump(s, 0.0)
    assert s.bg.mean() - s0.bg.mean() < 150  # only the hot-water pulse remains


def test_lighting_is_seasonal():
    w = simulate(make_household("early_riser", 5), date(2025, 12, 1), 20)
    su = simulate(make_household("early_riser", 5), date(2025, 6, 1), 20)
    assert w.state.mean() > 1.5 * su.state.mean()


def test_personas_differ_in_wake_time():
    def mean_wake(p):
        s = simulate(make_household(p, 21), date(2025, 4, 1), 40)
        return np.mean([i - s.off[k] for k, i in enumerate(s.wake_idx) if i >= 0])

    assert mean_wake("late_sleeper") > mean_wake("early_riser") + 120


def test_holiday_persona_has_absence_days():
    s = simulate(make_household("holiday_goer", 9), date(2025, 1, 1), 365)
    assert sum(x == "holiday" for x in s.status) >= 7
    assert any(x == "trip" for x in s.status)


def test_away_days_have_no_person_load():
    s = simulate(make_household("holiday_goer", 9), date(2025, 1, 1), 365)
    mid = [i for i in range(1, 364)
           if s.status[i] != "home" and s.status[i - 1] != "home" and s.status[i + 1] != "home"]
    assert mid
    i = mid[0]
    assert s.evt[s.day_slice(i)].sum() == 0 and s.state[s.day_slice(i)].sum() == 0


def test_aggregate_known_values():
    x = np.concatenate([np.full(15, 100.0), np.full(15, 300.0)])
    assert np.allclose(aggregate(x, [30], 15, quantise_wh=False), [100.0, 300.0])
    assert np.allclose(aggregate(x, [30], 30, quantise_wh=False), [200.0])


def test_aggregate_quantises_to_wh():
    out = aggregate(np.full(15, 101.3), [15], 15, quantise_wh=True)
    assert out[0] == pytest.approx(round(101.3 * 15 / 60) * 60 / 15)


def test_aggregate_dst_day_lengths_divisible():
    s = simulate(make_household("early_riser", 1), date(2025, 3, 29), 3)
    for res in (1, 5, 15, 30, 60):
        days, w = load_window(s, 0, 3, res)
        assert w.size == sum(d.n_min // res for d in days)


def test_slot_clock_matches_slot_count(sim):
    di, cl = slot_clock(sim.days[:3], 15)
    assert di.size == cl.size == 96 * 3 and cl[0] == 0 and cl[95] == 1425


def test_event_zeroes_person_events_and_holds_state(sim):
    rng = np.random.default_rng(1)
    ev = draw_event(sim, "fall", rng, lo=20)
    d0, d1 = ev.day - 2, ev.day + 2
    _, w_ev = load_window(sim, d0, d1, 1, ev)
    _, w_no = load_window(sim, d0, d1, 1, None)
    o = int(sim.off[ev.day] - sim.off[d0]) + ev.idx_in_day
    assert np.allclose(w_ev[:o], w_no[:o])
    after = w_ev[o:] - sim.bg[int(sim.off[d0]):int(sim.off[d1])][o:]
    assert after.max() <= sim.state.max() + 1  # only held lights/TV remain: no kettle or cooker bursts


def test_draw_event_picks_home_days(sim):
    rng = np.random.default_rng(2)
    for kind in ("fall", "night_fall", "hospital"):
        ev = draw_event(sim, kind, rng, lo=20)
        assert sim.status[ev.day] == "home"
    ev = draw_event(sim, "night_fall", rng, lo=20)
    assert 60 <= ev.onset_clock <= 240

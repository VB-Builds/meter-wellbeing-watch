from datetime import date, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from wellbeing_watch.features import MORNING_START, burst_score, extract, slot_table


def ts_for(days, res=15, start=date(2025, 5, 1)):
    idx = pd.date_range(pd.Timestamp(start, tz="Europe/Zurich"), periods=days * 24 * 60 // res, freq=f"{res}min")
    return idx.tz_convert("UTC")


def test_burst_isolated_spike_detected():
    w = np.full(40, 100.0)
    w[20] = 500.0
    a, k = burst_score(w, 15)
    assert a[20] == pytest.approx(400.0) and k == 3
    assert abs(a[10]) < 1e-9


def test_burst_plateau_cancels():
    w = np.full(60, 100.0)
    w[20:30] = 2000.0  # 150-min plateau
    a, _ = burst_score(w, 15)
    assert a[22:28].max() < 1


def test_burst_edges_do_not_crash():
    a, _ = burst_score(np.array([1.0, 2.0]), 15)
    assert a.shape == (2,)


def test_extract_detects_kettle_bursts():
    ts = ts_for(30)
    w = np.full(len(ts), 80.0)
    for d in range(30):
        w[d * 96 + 28] += 450.0  # 07:00 local
    f = extract(ts, w, 15)
    assert len(f) == 30
    assert all(x.first_activity == 420.0 for x in f)


def test_extract_ignores_night_activity_for_first_activity():
    ts = ts_for(3)
    w = np.full(len(ts), 80.0)
    w[1 * 96 + 8] += 600.0  # 02:00 on day 2
    f = extract(ts, w, 15)
    assert f[1].first_activity is None
    assert f[1].active_clock.size == 1 and f[1].active_clock[0] < MORNING_START


def test_extract_ignores_long_plateaus():
    ts = ts_for(20)
    w = np.full(len(ts), 80.0)
    for d in range(20):
        w[d * 96 + 30: d * 96 + 40] += 1500.0
    f = extract(ts, w, 15)
    assert all(x.first_activity is None for x in f[3:])


def test_extract_threshold_rises_with_night_noise():
    ts = ts_for(20)
    quiet = np.full(len(ts), 80.0)
    noisy = quiet.copy()
    for d in range(20):
        for s in (2, 7, 12):
            noisy[d * 96 + s] += 700.0  # isolated night bumps (like short heat-pump cycles)
    assert extract(ts, noisy, 15)[-1].thr_w > extract(ts, quiet, 15)[-1].thr_w + 500


def test_extract_active_energy():
    ts = ts_for(3)
    w = np.full(len(ts), 80.0)
    w[40] += 800.0
    f = extract(ts, w, 15)
    assert f[0].active_wh == pytest.approx(200.0)


def test_extract_length_mismatch_raises():
    with pytest.raises(ValueError):
        extract(ts_for(1), np.zeros(5), 15)


def test_extract_empty():
    assert extract(pd.DatetimeIndex([], tz="UTC"), np.array([]), 15) == []


def test_extract_handles_nan():
    ts = ts_for(2)
    w = np.full(len(ts), 80.0)
    w[10] = np.nan
    assert len(extract(ts, w, 15)) == 2


def test_slot_table_uses_local_clock_across_dst():
    ts = pd.date_range("2025-03-30 00:00", periods=96, freq="15min", tz="UTC")
    _, clock = slot_table(ts)
    assert clock[0] == 60  # 01:00 CET
    _, naive = slot_table(ts, timezone(timedelta(hours=1)))
    assert naive[0] == 60
    assert clock[-1] != naive[-1]  # after the switch the two clocks disagree


def test_dates_are_local_dates():
    ts = pd.DatetimeIndex(["2025-05-01T21:30:00", "2025-05-01T22:30:00"]).tz_localize("UTC")
    ordn, clock = slot_table(ts)
    assert ordn[1] == ordn[0] + 1 and clock[0] == 23 * 60 + 30 and clock[1] == 30  # CEST = UTC+2

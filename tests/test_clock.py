from datetime import date

import numpy as np

from wellbeing_watch.clock import clock_minutes, clock_to_idx, day_table, fmt_clock, sun_idx, to_utc


def test_dst_day_lengths():
    t = day_table(date(2025, 3, 29), 3)
    assert [d.n_min for d in t] == [1440, 1380, 1440]
    t = day_table(date(2025, 10, 25), 3)
    assert [d.n_min for d in t] == [1440, 1500, 1440]


def test_day_kinds():
    assert day_table(date(2025, 3, 30), 1)[0].kind == "spring"
    assert day_table(date(2025, 10, 26), 1)[0].kind == "fall"
    assert day_table(date(2025, 6, 1), 1)[0].kind == "normal"


def test_clock_minutes_lengths_match_day():
    for d in (date(2025, 3, 30), date(2025, 10, 26), date(2025, 7, 1)):
        day = day_table(d, 1)[0]
        assert clock_minutes(day).size == day.n_min


def test_spring_skips_02_to_03():
    day = day_table(date(2025, 3, 30), 1)[0]
    c = clock_minutes(day)
    assert c[119] == 119 and c[120] == 180
    assert not ((c >= 120) & (c < 180)).any()


def test_fall_repeats_02_to_03():
    day = day_table(date(2025, 10, 26), 1)[0]
    c = clock_minutes(day)
    assert ((c >= 120) & (c < 180)).sum() == 120


def test_clock_to_idx_roundtrip_normal_and_dst():
    for d in (date(2025, 6, 1), date(2025, 3, 30), date(2025, 10, 26)):
        day = day_table(d, 1)[0]
        cm = clock_minutes(day)
        for clock in (30, 400, 720, 1300):
            assert cm[clock_to_idx(day, clock)] == clock


def test_clock_to_idx_clipped():
    day = day_table(date(2025, 6, 1), 1)[0]
    assert clock_to_idx(day, 5000) == 1439
    assert clock_to_idx(day, -5) == 0


def test_sun_seasonal_order():
    summer = sun_idx(day_table(date(2025, 6, 21), 1)[0])
    winter = sun_idx(day_table(date(2025, 12, 21), 1)[0])
    assert summer[1] - summer[0] > 15 * 60
    assert winter[1] - winter[0] < 9.5 * 60
    assert 7 * 60 < winter[0] < 9 * 60  # sunrise around 08:10 local in Zurich
    assert 4 * 60 < summer[0] < 6 * 60


def test_to_utc_offsets():
    assert to_utc(date(2025, 1, 15), 0).hour == 23  # CET = UTC+1
    assert to_utc(date(2025, 7, 15), 0).hour == 22  # CEST = UTC+2
    assert to_utc(date(2025, 1, 15), 1500).day == 16  # minutes beyond 24 h roll over


def test_fmt_clock():
    assert fmt_clock(65) == "01:05"
    assert fmt_clock(1440 + 30) == "00:30"
    assert fmt_clock(np.float64(494.9)) == "08:15"

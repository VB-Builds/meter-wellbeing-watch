from datetime import date

import numpy as np
import pandas as pd
import pytest

from wellbeing_watch import evaluate
from wellbeing_watch.cli import main
from wellbeing_watch.features import DayFeatures
from wellbeing_watch.privacy import ALLOWED_PAYLOAD_KEYS, RETENTION_DAYS, Consent, assert_minimised, minimise


def feats():
    return DayFeatures(date(2025, 5, 1), 15, np.array([420.0, 735.0, 1100.0]), 1234.0, 150.0)


def test_minimise_has_only_allowed_scalar_fields():
    p = minimise(feats(), 300.0)
    assert set(p) == ALLOWED_PAYLOAD_KEYS
    assert_minimised(p)
    assert p["active_wh"] == 1230 and p["first_activity_min"] == 420


def test_minimise_never_contains_a_time_series():
    p = minimise(feats(), 300.0)
    assert all(not isinstance(v, (list, tuple, dict, np.ndarray)) for v in p.values())


def test_assert_minimised_rejects_extra_fields_and_lists():
    with pytest.raises(ValueError):
        assert_minimised({**minimise(feats(), 1.0), "load_w": 5})
    with pytest.raises(ValueError):
        assert_minimised({**minimise(feats(), 1.0), "date": [1, 2]})


def test_retention_window_is_bounded():
    assert RETENTION_DAYS <= 31


def test_consent_active_window():
    c = Consent("p", date(2025, 1, 10), ("a",), revoked_on=date(2025, 2, 1))
    assert not c.active_on(date(2025, 1, 9))
    assert c.active_on(date(2025, 1, 10)) and c.active_on(date(2025, 1, 31))
    assert not c.active_on(date(2025, 2, 1))


def test_make_ts_covers_dst_days():
    from wellbeing_watch.clock import day_table

    days = day_table(date(2025, 3, 29), 3)
    ts = evaluate.make_ts(days, 15)
    assert len(ts) == (1440 + 1380 + 1440) // 15
    assert ts.is_monotonic_increasing and ts.tz is not None


def test_year_run_and_event_run_smoke():
    from wellbeing_watch.detector import WatchConfig
    from wellbeing_watch.simulator import draw_event, make_household, simulate

    sim = simulate(make_household("early_riser", 31, hp_kw=0.0), evaluate.SIM_START, 120)
    y = evaluate.year_run(sim, 15, WatchConfig(), 0.9, 1)
    assert y["years"] == pytest.approx((120 - 35) / 365) and y["total_days"] == 85
    ev = draw_event(sim, "night_fall", np.random.default_rng(2), lo=45)
    r = evaluate.event_run(sim, ev, 15, WatchConfig(), 3)
    assert r["kind"] == "night_fall"
    assert r["rel_h"] is None or r["rel_h"] > 0


def test_household_job_rows(monkeypatch):
    monkeypatch.setattr(evaluate, "CONFIGS", [("base", 0.0, 15)])
    rows = evaluate.household_job(("early_riser", 41, 2, 100))
    tables = {r["table"] for r in rows}
    assert tables == {"year", "event"}
    assert sum(r["table"] == "event" for r in rows) == 2
    assert any(r["config"] == "fixed_clock" for r in rows)


def test_summarise_on_handmade_frame():
    rows = []
    for cfg in ("base", "hp1.0", "hp2.0", "hp3.5", "res1", "res5", "res30", "res60", "hp2res1", "hp2res5",
                "hp2res30", "hp2res60", "fixed_clock"):
        for pr in (0.9, 0.0):
            rows.append(dict(table="year", config=cfg, persona="early_riser", p_reply=pr, years=1.0, active_days=300,
                             total_days=330, checkins=10, rel=2 if pr == 0.9 else 5, rel_home=2, rel_morning=1,
                             rel_gap=1))
        if cfg != "fixed_clock":
            for i, h in enumerate([5.0, 30.0, None, 10.0]):
                rows.append(dict(table="event", config=cfg, persona="early_riser",
                                 kind="fall" if i % 2 else "night_fall", month=1, rel_h=h, checkin_h=h))
    s = evaluate.summarise(pd.DataFrame(rows))
    b = s["base_all"]
    assert b["n"] == 4 and b["detection_rate"] == 0.5  # 30 h is beyond 24 h, None is a miss
    assert b["median_tta_h"] == 7.5
    assert b["reply0.9"]["false_relative_per_hh_year"] == 2.0
    assert b["reply0.0"]["false_relative_per_hh_year"] == 5.0
    assert set(s["resolution"]) == {"boiler_only", "heat_pump_2kw"}


def test_cli_simulate_then_watch(tmp_path, capsys):
    out = tmp_path / "sim.csv"
    assert main(["simulate", "--persona", "early_riser", "--days", "60", "--seed", "5", "--out", str(out)]) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["timestamp_utc", "mean_w"] and len(df) == 60 * 96
    capsys.readouterr()
    assert main(["watch", str(out), "--consent-given", "2024-12-01"]) == 0
    text = capsys.readouterr().out
    assert "NOT a medical or emergency device" in text


def test_cli_watch_requires_consent_date(tmp_path):
    with pytest.raises(SystemExit):
        main(["watch", str(tmp_path / "x.csv")])

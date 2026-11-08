from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest
from conftest import blank_day, make_days

from wellbeing_watch.detector import RoutineModel, WatchConfig, max_gap, run_watch, t_ppf
from wellbeing_watch.features import DayFeatures
from wellbeing_watch.privacy import Consent, ConsentError


class Reply:
    def __init__(self, kind):
        self.kind = kind

    def responds(self, d, clock_min):
        return self.kind


def history(n=20, **kw):
    return make_days(n, **kw)


def with_blank(n=20, extra_days=1):
    days = history(n)
    last = days[-1].date
    return days + [blank_day(last + timedelta(days=i + 1)) for i in range(extra_days)]


def test_t_ppf_known_values():
    assert t_ppf(0.975, 10) == pytest.approx(2.228, abs=0.005)
    assert t_ppf(0.99, 30) == pytest.approx(2.457, abs=0.005)
    assert t_ppf(0.995, 5) == pytest.approx(4.032, abs=0.01)
    assert t_ppf(0.5, 7) == pytest.approx(0.0, abs=1e-6)


def test_t_ppf_tends_to_normal():
    assert t_ppf(0.99, 2000) == pytest.approx(2.326, abs=0.01)


def test_t_ppf_rejects_bad_p():
    with pytest.raises(ValueError):
        t_ppf(0.3, 5)
    with pytest.raises(ValueError):
        t_ppf(0.99999, 5)


def test_prior_only_model_is_wide():
    m = RoutineModel(WatchConfig())
    mu, sc, nu = m.predictive()
    assert mu == 480 and sc > 60 and nu == 3


def test_model_learns_location():
    m = RoutineModel(WatchConfig())
    for f in history(28):
        m.update(f)
    mu, sc, nu = m.predictive()
    assert mu == pytest.approx(422, abs=1.0)
    assert nu == pytest.approx(3 + 28)
    assert WatchConfig().min_scale <= sc < 30  # a perfectly regular household: small, prior-inflated scale


def test_noisier_routine_gives_later_alert_time():
    cfg = WatchConfig()
    a, b = RoutineModel(cfg), RoutineModel(cfg)
    for f in make_days(28, jitter=5):
        a.update(f)
    for f in make_days(28, jitter=45):
        b.update(f)
    assert b.morning_threshold()[0] > a.morning_threshold()[0] + 20


def test_alert_time_respects_margin_and_cap():
    m = RoutineModel(WatchConfig())
    for f in history(28):
        m.update(f)
    tau, usual = m.morning_threshold()
    assert tau >= 422 + 75 - 1 and usual < tau
    m2 = RoutineModel(WatchConfig())
    assert m2.morning_threshold()[0] == WatchConfig().latest_morning_min


def test_regular_household_raises_no_alerts(consent):
    r = run_watch(history(40), consent=consent)
    assert r.alerts == []
    assert all(s.active for s in r.summaries[14:])


def test_missing_morning_notifies_relative_with_explanation(consent):
    r = run_watch(with_blank(), consent=consent)
    assert len(r.alerts) == 1
    a = r.alerts[0]
    assert a.kind == "morning" and a.outcome == "relative_notified"
    assert "usually by" in a.text and "no kettle/cooker-scale household activity by 08:" in a.text
    cfg = WatchConfig()
    assert a.checkin_clock == pytest.approx(a.trigger_clock + cfg.latency)
    assert a.relative_clock == pytest.approx(a.trigger_clock + cfg.latency + cfg.checkin_wait_min)


def test_person_is_asked_before_relative(consent):
    a = run_watch(with_blank(), consent=consent).alerts[0]
    assert a.checkin_clock < a.relative_clock


def test_no_alert_during_learning_period(consent):
    days = history(5) + [blank_day(date(2025, 5, 6))]
    assert run_watch(days, consent=consent).alerts == []


def test_declared_away_suppresses(consent):
    days = with_blank(extra_days=3)
    away = frozenset(d.date for d in days[-3:])
    assert run_watch(days, consent=consent, declared_away=away).alerts == []


def test_reply_ok_cancels_escalation(consent):
    a = run_watch(with_blank(), consent=consent, responder=Reply("ok")).alerts[0]
    assert a.outcome == "cancelled_reply" and a.relative_clock is None


def test_reply_away_sets_away_mode_and_silences_following_days(consent):
    days = with_blank(extra_days=4)
    r = run_watch(days, consent=consent, responder=Reply("away"))
    assert [a.outcome for a in r.alerts] == ["away_mode"]


def test_activity_back_ends_away_mode(consent):
    days = with_blank(extra_days=3)
    start = days[-1].date + timedelta(days=1)
    back = make_days(20, start=start)
    r = run_watch(days + back + [blank_day(start + timedelta(days=20))], consent=consent, responder=Reply("away"))
    assert [a.outcome for a in r.alerts] == ["away_mode", "away_mode"]


def test_activity_within_wait_cancels(consent):
    days = history(20)
    cfg = WatchConfig()
    m = RoutineModel(cfg)
    for f in days:
        m.update(f)
    tau, _ = m.morning_threshold()
    late = DayFeatures(days[-1].date + timedelta(days=1), 15, np.array([tau + 15, 660.0, 780.0, 1080.0]), 100.0, 150.0)
    r = run_watch(days + [late], consent=consent)
    assert [a.outcome for a in r.alerts] == ["cancelled_activity"]


def test_alert_not_repeated_while_inactivity_continues(consent):
    r = run_watch(with_blank(extra_days=4), consent=consent)
    assert len([a for a in r.alerts if a.outcome == "relative_notified"]) == 1


def test_alert_again_after_activity_resumes(consent):
    days = with_blank(extra_days=2)
    nxt = days[-1].date + timedelta(days=1)
    days += make_days(3, start=nxt) + [blank_day(nxt + timedelta(days=3))]
    r = run_watch(days, consent=consent)
    assert len([a for a in r.alerts if a.outcome == "relative_notified"]) == 2


def test_gap_detector_fires_after_morning_activity(consent):
    days = history(20)
    last = days[-1].date + timedelta(days=1)
    days.append(DayFeatures(last, 15, np.array([420.0, 435.0]), 100.0, 150.0))
    r = run_watch(days, consent=consent)
    a = r.alerts[0]
    assert a.kind == "gap" and a.outcome == "relative_notified"
    assert "no household activity for" in a.text


def test_weak_signal_household_is_not_promised_anything(consent):
    late = make_days(30, first=14 * 60)
    last = late[-1].date + timedelta(days=1)
    r = run_watch(late + [blank_day(last)], consent=consent)
    assert r.alerts == [] and not r.summaries[-1].active


def test_consent_withdrawal_stops_processing_and_clears_model():
    days = history(30)
    c = replace(Consent.simulated(), revoked_on=days[20].date)
    r = run_watch(days, consent=c)
    assert r.stopped_on == days[20].date and len(r.summaries) == 20


def test_processing_before_consent_is_refused():
    days = history(5)
    c = Consent("X", date(2030, 1, 1), ("a",))
    with pytest.raises(ConsentError):
        run_watch(days, consent=c)


def test_max_gap_known_value():
    f = DayFeatures(date(2025, 1, 1), 15, np.array([420.0, 480.0, 900.0]), 0.0, 150.0)
    # breaks: 495->480 none (adjacent), 495->900 = 405, tail 915->1350 = 435
    assert max_gap(f, 1350.0) == 435.0
    assert max_gap(DayFeatures(date(2025, 1, 1), 15, np.array([]), 0.0, 150.0)) == 0.0


def test_wh_z_needs_history():
    m = RoutineModel(WatchConfig())
    assert m.wh_z(100.0) is None
    for i in range(10):
        m.update(DayFeatures(date(2025, 1, 1) + timedelta(days=i), 15, np.array([420.0]), 100.0 + i, 150.0))
    assert m.wh_z(300.0) > 3


def test_higher_resolution_config_has_same_latency_floor():
    assert WatchConfig(res_min=1).latency < WatchConfig(res_min=60).latency

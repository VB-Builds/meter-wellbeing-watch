from datetime import date, timedelta

import numpy as np
import pytest

from wellbeing_watch.features import DayFeatures
from wellbeing_watch.privacy import Consent


def make_days(n, first=420.0, start=date(2025, 5, 1), extra=(660.0, 780.0, 1080.0), jitter=None, res=15):
    """Synthetic daily features: first activity ~ ``first`` (+jitter), a few later activities."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        j = 0.0 if jitter is None else float(rng.normal(0, jitter))
        f = float(round((first + j) / 15) * 15)
        out.append(DayFeatures(start + timedelta(days=i), res, np.array([f, *extra]), 300.0, 150.0))
    return out


def blank_day(d, res=15):
    return DayFeatures(d, res, np.array([]), 0.0, 150.0)


@pytest.fixture
def consent():
    return Consent.simulated()

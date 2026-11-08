"""Opt-in consent and data minimisation.

Design principles (generic, not legal advice, no legal claims are made here):
  * purpose limitation: the data is used only for the inactivity watch the person agreed to;
  * data minimisation: only derived daily-routine features leave the home, never the load curve;
  * storage limitation: the learned routine uses a rolling window and is deleted on withdrawal;
  * the person can withdraw at any time and processing stops from that day on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .features import DayFeatures

RETENTION_DAYS = 28  # rolling window of daily features kept by the watch service
ALLOWED_PAYLOAD_KEYS = frozenset({"date", "first_activity_min", "n_active_slots", "active_wh", "max_gap_min"})


class ConsentError(PermissionError):
    pass


@dataclass(frozen=True)
class Consent:
    person_id: str  # pseudonymous id, never a name or meter number in this repo
    given_on: date
    contacts: tuple[str, ...]  # who may be notified (relative first)
    revoked_on: date | None = None

    def active_on(self, d: date) -> bool:
        return d >= self.given_on and (self.revoked_on is None or d < self.revoked_on)

    @staticmethod
    def simulated(given_on: date = date(2000, 1, 1)) -> Consent:
        return Consent("SIMULATED-0000", given_on, ("relative-1", "relative-2"))


def minimise(f: DayFeatures, max_gap_min: float) -> dict:
    """The only record about a day that may leave the home (no load values, no appliance names)."""
    return {
        "date": f.date.isoformat(),
        "first_activity_min": None if f.first_activity is None else int(f.first_activity),
        "n_active_slots": int(f.active_clock.size),
        "active_wh": round(f.active_wh, -1),
        "max_gap_min": int(max_gap_min),
    }


def assert_minimised(payload: dict) -> None:
    extra = set(payload) - ALLOWED_PAYLOAD_KEYS
    if extra:
        raise ValueError(f"payload carries non-minimised fields: {sorted(extra)}")
    for v in payload.values():
        if isinstance(v, (list, tuple, dict)):
            raise ValueError("payload must hold scalars only (no time series)")

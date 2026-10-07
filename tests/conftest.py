"""Shared fixtures. Tests use a seeded synthetic results table plus the committed
reference data, so the suite runs offline and deterministically."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from wcpred import config as C
from wcpred.data import ReferenceData, clean_results
from wcpred.match import MatchContext
from wcpred.simulation import TournamentFormat

COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def make_results(rows: Sequence[tuple]) -> pd.DataFrame:
    """Clean a list of (date, home, away, hs, as, tournament, city, country, neutral)."""
    return clean_results(pd.DataFrame(list(rows), columns=COLUMNS))


def team_strengths(teams: Sequence[str]) -> dict[str, float]:
    """Deterministic latent strengths; earlier teams in `teams` are stronger."""
    return {t: 0.6 - 1.2 * i / (len(teams) - 1) for i, t in enumerate(teams)}


def synthetic_results(fmt: TournamentFormat, seed: int = 7) -> pd.DataFrame:
    """~4,000 played matches (2014-2026) among the 48 finalists + unplayed WC2026 groups."""
    rng = np.random.default_rng(seed)
    teams = fmt.teams
    strength = team_strengths(teams)
    days = pd.date_range("2014-01-01", "2026-06-01", freq="D")
    tournaments = ["Friendly", "FIFA World Cup qualification", "UEFA Nations League"]
    rows = []
    busy: set[tuple[pd.Timestamp, str]] = set()  # no team plays twice on one day
    while len(rows) < 4000:
        h, a = rng.choice(len(teams), size=2, replace=False)
        home, away = teams[h], teams[a]
        day = days[rng.integers(len(days))]
        if (day, home) in busy or (day, away) in busy:
            continue
        busy |= {(day, home), (day, away)}
        neutral = bool(rng.random() < 0.2)
        adv = 0.0 if neutral else 0.25
        lh = np.exp(0.15 + strength[home] - strength[away] + adv)
        la = np.exp(0.15 + strength[away] - strength[home])
        rows.append(
            (
                day,
                home,
                away,
                rng.poisson(lh),
                rng.poisson(la),
                tournaments[rng.integers(3)],
                "Capital",
                home,
                neutral,
            )
        )
    venues = [(m.city, m.country) for m in fmt.matches]
    for g, members in fmt.groups.items():
        k = 0
        for i in range(4):
            for j in range(i + 1, 4):
                city, country = venues[(ord(g) + k) % len(venues)]
                date = C.WC2026_START + pd.Timedelta(days=(k // 2) * 6 + ord(g) % 4)
                rows.append(
                    (date, members[i], members[j], None, None, C.WORLD_CUP, city, country, True)
                )
                k += 1
    return make_results(rows)


class StubModel:
    """Expected goals from fixed strengths: lets simulation tests skip XGBoost."""

    rho = C.DIXON_COLES_RHO

    def __init__(self, strengths: dict[str, float]) -> None:
        self.strengths = strengths
        self.calls = 0

    def expected_goals(
        self, fixtures: Sequence[tuple[str, str, MatchContext]]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        diff = np.array([self.strengths[h] - self.strengths[a] for h, a, _ in fixtures])
        return np.exp(0.2 + diff), np.exp(0.2 - diff)


@pytest.fixture(scope="session")
def wc_format() -> TournamentFormat:
    return TournamentFormat.load()


@pytest.fixture(scope="session")
def reference() -> ReferenceData:
    return ReferenceData.load()


@pytest.fixture(scope="session")
def results(wc_format: TournamentFormat) -> pd.DataFrame:
    return synthetic_results(wc_format)

"""World Football Elo ratings computed in one chronological pass.

Each row records both teams' ratings *before* the match, so the feature never
contains that match's result. Only played matches update ratings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wcpred import config as C


@dataclass
class EloRatings:
    pre_home: np.ndarray  # home side's rating before each row's match
    pre_away: np.ndarray
    final: dict[str, float]  # rating after the last played match


def goal_diff_multiplier(goal_diff: float) -> float:
    """World Football Elo margin multiplier: 1, 1.5, then (11 + N) / 8."""
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def k_factor(importance: float) -> float:
    """K scales linearly with match importance (friendly 20, World Cup 80)."""
    return C.ELO_K * importance / C.DEFAULT_IMPORTANCE


def expected_score(rating: float, opp_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating - opp_rating) / 400.0))


def compute_elo(results: pd.DataFrame) -> EloRatings:
    """Run Elo over `results` (sorted by date) and return pre-match snapshots."""
    n = len(results)
    pre_home = np.empty(n)
    pre_away = np.empty(n)
    ratings: dict[str, float] = {}

    rows = zip(
        results["home_team"].to_numpy(),
        results["away_team"].to_numpy(),
        results["home_score"].to_numpy(dtype=float),
        results["away_score"].to_numpy(dtype=float),
        results["neutral"].to_numpy(dtype=bool),
        results["importance"].to_numpy(dtype=float),
        results["played"].to_numpy(dtype=bool),
        strict=True,
    )
    for i, (home, away, hs, as_, neutral, importance, played) in enumerate(rows):
        r_home = ratings.get(home, C.ELO_START)
        r_away = ratings.get(away, C.ELO_START)
        pre_home[i] = r_home
        pre_away[i] = r_away
        if not played:
            continue
        home_adv = 0.0 if neutral else C.ELO_HOME_ADV
        expected = expected_score(r_home + home_adv, r_away)
        actual = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        delta = k_factor(importance) * goal_diff_multiplier(hs - as_) * (actual - expected)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    return EloRatings(pre_home=pre_home, pre_away=pre_away, final=ratings)

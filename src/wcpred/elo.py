"""World-Football-style Elo ratings computed chronologically.

For every row we record the *pre-match* rating of each side (no leakage), and
only matches with a real score update the ratings. Because the WC2026 matches
played so far are in the data, the ratings going into the upcoming-fixture rows
already reflect current-tournament form.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def _goal_diff_multiplier(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _k_factor(importance: float) -> float:
    # friendly(1.0)->16, WCQ(2.5)->40, continental(3.5)->56, WC(4.0)->64
    return C.ELO_K * importance / C.DEFAULT_IMPORTANCE


def compute_elo(results: pd.DataFrame):
    """Return (pre_home, pre_away, final_ratings, rating_date).

    pre_home/pre_away: np.ndarray aligned to `results` rows = each side's Elo
    immediately before that match. final_ratings: dict team -> latest Elo.
    """
    ratings: dict[str, float] = {}
    rating_date: dict[str, pd.Timestamp] = {}
    n = len(results)
    pre_home = np.empty(n)
    pre_away = np.empty(n)

    ht = results["home_team"].to_numpy()
    at = results["away_team"].to_numpy()
    hs = results["home_score"].to_numpy()
    as_ = results["away_score"].to_numpy()
    neut = results["neutral"].to_numpy()
    imp = results["importance"].to_numpy()
    played = results["played"].to_numpy()
    dates = results["date"].to_numpy()

    for i in range(n):
        h, a = ht[i], at[i]
        rh = ratings.get(h, C.ELO_START)
        ra = ratings.get(a, C.ELO_START)
        pre_home[i] = rh
        pre_away[i] = ra
        if not played[i]:
            continue
        home_adv = 0.0 if neut[i] else C.ELO_HOME_ADV
        we_home = 1.0 / (1.0 + 10.0 ** (-((rh + home_adv) - ra) / 400.0))
        gd = hs[i] - as_[i]
        w_home = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k = _k_factor(imp[i]) * _goal_diff_multiplier(gd)
        delta = k * (w_home - we_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        rating_date[h] = dates[i]
        rating_date[a] = dates[i]

    return pre_home, pre_away, ratings, rating_date

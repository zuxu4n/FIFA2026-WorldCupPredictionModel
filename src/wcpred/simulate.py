"""Monte-Carlo tournament simulator.

Reconstructs the 12 WC2026 groups from the fixtures, completes the group stage
by sampling scorelines from the model's score matrices, picks the 32 qualifiers
(12 winners + 12 runners-up + 8 best third-placed), then plays a strength-seeded
single-elimination knockout many times to estimate advancement and title odds.

The knockout uses a strength-seeded bracket (seed 1 vs 32, ...). This is a
documented approximation of the official slot mapping; drop a real bracket
template in to replace it if you want exact paths (see README).
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .predict import score_matrix, knockout_advance, predict_lambdas


# --------------------------------------------------------------------------
# Group reconstruction
# --------------------------------------------------------------------------
KNOCKOUT_START = pd.Timestamp("2026-06-28")   # first Round-of-32 date


def _finals(results: pd.DataFrame) -> pd.DataFrame:
    f = results[(results.tournament == C.WC_TOURNAMENT_NAME)
                & (results.date >= pd.Timestamp("2026-06-01"))
                & (results.date <= pd.Timestamp("2026-07-31"))]
    return f.copy()


def _group_stage(results: pd.DataFrame) -> pd.DataFrame:
    """Finals matches from the group stage only (before the knockout rounds)."""
    f = _finals(results)
    return f[f.date < KNOCKOUT_START]


def reconstruct_groups(results: pd.DataFrame) -> dict[str, list[str]]:
    """Union-find over the group fixtures -> {group_label: [4 teams]}."""
    f = _group_stage(results)
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for r in f.itertuples(index=False):
        union(r.home_team, r.away_team)
    comps: dict[str, list[str]] = defaultdict(list)
    for t in {*f.home_team, *f.away_team}:
        comps[find(t)].append(t)
    groups = sorted(comps.values(), key=lambda g: sorted(g)[0])
    return {chr(ord("A") + i): sorted(g) for i, g in enumerate(groups)}


def current_standings(results: pd.DataFrame, groups: dict[str, list[str]]):
    """Return {team: dict(pts, gd, gf, ga, played)} from PLAYED group-stage matches
    (knockout results must not pollute group tables)."""
    f = _group_stage(results)
    st = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "played": 0}
          for g in groups.values() for t in g}
    for r in f[f.played].itertuples(index=False):
        h, a, hs, as_ = r.home_team, r.away_team, int(r.home_score), int(r.away_score)
        for t, gf, ga in ((h, hs, as_), (a, as_, hs)):
            s = st[t]
            s["gf"] += gf; s["ga"] += ga; s["gd"] += gf - ga; s["played"] += 1
            s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    return st


def remaining_group_fixtures(results: pd.DataFrame, groups: dict[str, list[str]],
                             booster, world):
    """List of dicts: {group, home, away, M(score matrix)} for unplayed games."""
    f = _group_stage(results)
    team_group = {t: g for g, ts in groups.items() for t in ts}
    out = []
    for r in f[~f.played].itertuples(index=False):
        lh, la = predict_lambdas(booster, world, r.home_team, r.away_team,
                                 city=r.city, country=r.country,
                                 neutral=bool(r.neutral), importance=4.0)
        out.append({"group": team_group[r.home_team], "home": r.home_team,
                    "away": r.away_team, "M": score_matrix(lh, la)})
    return out


# --------------------------------------------------------------------------
# Knockout helpers
# --------------------------------------------------------------------------
def _seed_positions(n: int) -> list[int]:
    """Classic single-elim seeding order (1-indexed) for a bracket of size n."""
    res = [1]
    while len(res) < n:
        m = len(res) * 2
        nxt = []
        for x in res:
            nxt.append(x)
            nxt.append(m + 1 - x)
        res = nxt
    return res


def _sample_scoreline(M: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = M.ravel()
    k = rng.choice(flat.size, p=flat / flat.sum())
    return divmod(int(k), M.shape[1])


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
ROUND_NAMES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def simulate(booster, world, results: pd.DataFrame, n_sims: int = 5000,
             seed: int = 0, knockout_city="East Rutherford",
             knockout_country="United States") -> dict:
    rng = np.random.default_rng(seed)
    groups = reconstruct_groups(results)
    base = current_standings(results, groups)
    rem = remaining_group_fixtures(results, groups, booster, world)
    teams = [t for g in groups.values() for t in g]

    # cache pairwise knockout advance prob (host-neutral venue)
    adv_cache: dict[tuple[str, str], float] = {}

    def p_adv(a: str, b: str) -> float:
        key = (a, b)
        if key not in adv_cache:
            lh, la = predict_lambdas(booster, world, a, b, city=knockout_city,
                                     country=knockout_country, neutral=True,
                                     importance=4.0)
            adv_cache[key] = knockout_advance(lh, la)[0]
        return adv_cache[key]

    elo = {t: world.state(t).elo for t in teams}
    counters = {t: dict(win_group=0, advance=0,
                        **{r: 0 for r in ROUND_NAMES}) for t in teams}

    def rank_key(team, s):
        return (s["pts"], s["gd"], s["gf"], rng.random())

    for _ in range(n_sims):
        st = {t: dict(base[t]) for t in teams}
        for fx in rem:
            gh, ga = _sample_scoreline(fx["M"], rng)
            for t, f_, a_ in ((fx["home"], gh, ga), (fx["away"], ga, gh)):
                s = st[t]
                s["gf"] += f_; s["ga"] += a_; s["gd"] += f_ - a_
                s["pts"] += 3 if f_ > a_ else (1 if f_ == a_ else 0)

        winners, runners, thirds = [], [], []
        for g, ts in groups.items():
            ordered = sorted(ts, key=lambda t: rank_key(t, st[t]), reverse=True)
            counters[ordered[0]]["win_group"] += 1
            winners.append(ordered[0]); runners.append(ordered[1])
            thirds.append(ordered[2])
        best_thirds = sorted(thirds, key=lambda t: rank_key(t, st[t]),
                             reverse=True)[:8]
        qualifiers = winners + runners + best_thirds
        for t in qualifiers:
            counters[t]["advance"] += 1

        # strength-seeded 32-team bracket
        seeded = sorted(qualifiers, key=lambda t: elo[t], reverse=True)
        order = _seed_positions(32)
        bracket = [seeded[i - 1] for i in order]
        for rname in ROUND_NAMES[:-1]:           # R32..Final
            for t in bracket:
                counters[t][rname] += 1
            nxt = []
            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i + 1]
                winner = a if rng.random() < p_adv(a, b) else b
                nxt.append(winner)
            bracket = nxt
        counters[bracket[0]]["Champion"] += 1     # last team standing

    rows = []
    for t in teams:
        c = counters[t]
        g = next(g for g, ts in groups.items() if t in ts)
        rows.append({
            "team": t, "group": g, "elo": round(elo[t]),
            "P_win_group": c["win_group"] / n_sims,
            "P_advance": c["advance"] / n_sims,
            "P_QF": c["QF"] / n_sims, "P_SF": c["SF"] / n_sims,
            "P_final": c["Final"] / n_sims, "P_champion": c["Champion"] / n_sims,
        })
    table = pd.DataFrame(rows).sort_values("P_champion", ascending=False)
    return {"groups": groups, "standings": base, "table": table,
            "n_sims": n_sims}

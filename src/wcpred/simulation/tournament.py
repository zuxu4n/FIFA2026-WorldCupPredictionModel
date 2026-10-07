"""Vectorised Monte Carlo simulation of the 2026 World Cup.

All runs are simulated together as NumPy arrays of shape (n_runs, ...):

1. Group stage: each unplayed group match gets a score matrix from the model;
   scorelines are sampled by inverse-CDF lookup. Played matches are fixed.
2. Group ranking: points, goal difference, goals scored, then a random draw
   (FIFA's head-to-head and fair-play criteria are not modelled).
3. The eight best third-placed teams are slotted into the Round of 32 with
   FIFA's official 495-row allocation table.
4. Knockout rounds: P(home side advances) - 90 minutes, extra time, penalties -
   is computed once per (match, pairing) in a batched model call and cached;
   knockout matches already played are fixed to their real winner.

Team strength is frozen at the simulation start: simulated results do not feed
back into Elo or form.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from wcpred import config as C
from wcpred.data.results import DataError
from wcpred.match import MatchContext
from wcpred.models.scoreline import advance_probability, score_matrices
from wcpred.simulation.bracket import GROUP_LETTERS, ROUND_ORDER, TournamentFormat

log = logging.getLogger(__name__)

Fixture = tuple[str, str, MatchContext]
STAGES = ("R32", "R16", "QF", "SF", "Final")


class MatchModel(Protocol):
    """What the simulator needs from a predictor (lets tests use a stub)."""

    @property
    def rho(self) -> float: ...

    def expected_goals(self, fixtures: Sequence[Fixture]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class SimulationResult:
    table: pd.DataFrame  # one row per team, probabilities of reaching each stage
    n_runs: int
    seed: int


def _world_cup_rows(results: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (results["tournament"] == C.WORLD_CUP)
        & (results["date"] >= C.WC2026_START)
        & (results["date"] <= C.WC2026_END)
    )
    return results[mask]


def _knockout_locks(
    ko_rows: pd.DataFrame, fmt: TournamentFormat, index: dict[str, int], shootouts: pd.DataFrame
) -> dict[int, tuple[int, int, int]]:
    """Played knockout matches: bracket match number -> (home, away, winner) team indices.

    Rows are matched to the bracket by (date, city); a row whose city is spelled
    differently (e.g. "Dallas" for the Arlington stadium) is matched to the only
    unclaimed bracket match on that date.
    """
    played = list(ko_rows[ko_rows["played"]].itertuples(index=False))
    by_venue = {(m.date, m.city): m.number for m in fmt.matches}
    numbers: dict[int, int] = {}  # row position -> bracket match number
    for i, r in enumerate(played):
        if (r.date, r.city) in by_venue:
            numbers[i] = by_venue[(r.date, r.city)]
    for i, r in enumerate(played):
        if i in numbers:
            continue
        free = [
            m.number for m in fmt.matches if m.date == r.date and m.number not in numbers.values()
        ]
        if len(free) != 1:
            raise DataError(
                f"cannot place knockout match {r.home_team} v {r.away_team} "
                f"({r.date.date()}, {r.city}) in the bracket"
            )
        numbers[i] = free[0]

    so = {
        (r.date, frozenset((r.home_team, r.away_team))): r.winner
        for r in shootouts.itertuples(index=False)
    }
    locks: dict[int, tuple[int, int, int]] = {}
    for i, r in enumerate(played):
        number = numbers[i]
        if r.home_score != r.away_score:
            winner = r.home_team if r.home_score > r.away_score else r.away_team
        else:
            winner = so.get((r.date, frozenset((r.home_team, r.away_team))))
            if winner is None:
                log.warning(
                    "no shootout winner for %s v %s; simulating it", r.home_team, r.away_team
                )
                continue
        locks[number] = (index[r.home_team], index[r.away_team], index[winner])
    return locks


def simulate_world_cup(
    results: pd.DataFrame,
    fmt: TournamentFormat,
    model: MatchModel,
    *,
    n_runs: int = C.DEFAULT_SIM_RUNS,
    seed: int = C.DEFAULT_SEED,
    shootouts: pd.DataFrame | None = None,
) -> SimulationResult:
    """Simulate the tournament from whatever has been played in `results`."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rng = np.random.default_rng(seed)
    teams = fmt.teams
    index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    groups = np.array([[index[t] for t in fmt.groups[g]] for g in GROUP_LETTERS])  # (12, 4)
    group_of = {t: g for g, ts in fmt.groups.items() for t in ts}

    wc = _world_cup_rows(results)
    group_rows = wc[wc["date"] < C.WC2026_KNOCKOUT_START]
    for r in group_rows.itertuples(index=False):
        if r.home_team not in index or r.away_team not in index:
            raise DataError(f"{r.home_team} v {r.away_team} involves a team not in the groups")
        if group_of[r.home_team] != group_of[r.away_team]:
            raise DataError(f"{r.home_team} v {r.away_team} is not a group-stage pairing")

    # ---- group stage -------------------------------------------------------
    pts = np.zeros((n_runs, n_teams), dtype=np.int32)
    gf = np.zeros((n_runs, n_teams), dtype=np.int32)
    ga = np.zeros((n_runs, n_teams), dtype=np.int32)

    def record(h: int, a: int, gh: np.ndarray | int, gaw: np.ndarray | int) -> None:
        pts[:, h] += np.where(gh > gaw, 3, np.where(gh == gaw, 1, 0))
        pts[:, a] += np.where(gaw > gh, 3, np.where(gh == gaw, 1, 0))
        gf[:, h] += gh
        ga[:, h] += gaw
        gf[:, a] += gaw
        ga[:, a] += gh

    for r in group_rows[group_rows["played"]].itertuples(index=False):
        record(index[r.home_team], index[r.away_team], int(r.home_score), int(r.away_score))

    pending = group_rows[~group_rows["played"]]
    if len(pending):
        fixtures = [
            (
                r.home_team,
                r.away_team,
                MatchContext(r.date, r.city, r.country, bool(r.neutral), C.WORLD_CUP, False),
            )
            for r in pending.itertuples(index=False)
        ]
        lh, la = model.expected_goals(fixtures)
        mats = score_matrices(lh, la, model.rho)
        size = mats.shape[1]
        cdf = np.cumsum(mats.reshape(len(fixtures), -1), axis=1)
        cdf[:, -1] = 1.0
        u = rng.random((n_runs, len(fixtures)))
        for f, (home, away, _) in enumerate(fixtures):
            k = np.minimum(np.searchsorted(cdf[f], u[:, f], side="right"), size * size - 1)
            record(index[home], index[away], k // size, k % size)

    key = (pts * 1_000_000 + (gf - ga + 500) * 1_000 + gf).astype(float)
    key += rng.random((n_runs, n_teams))  # random draw breaks exact ties
    gkey = key[:, groups]  # (n, 12, 4)
    order = np.argsort(-gkey, axis=2, kind="stable")
    ranked = groups[np.arange(len(groups))[None, :, None], order]  # team index by position
    third_key = np.take_along_axis(gkey, order[:, :, 2:3], axis=2)[:, :, 0]

    # ---- best third-placed teams -> Round of 32 slots ----------------------
    best8 = np.argsort(-third_key, axis=1, kind="stable")[:, :8]
    mask = (1 << best8).sum(axis=1)
    third_slots = fmt.third_place_slots()
    column = {winner_slot: j for j, (winner_slot, _) in enumerate(third_slots)}
    lookup = np.full((1 << len(GROUP_LETTERS), len(third_slots)), -1)
    for qualified, assignment in fmt.third_place_table.items():
        bits = sum(1 << GROUP_LETTERS.index(g) for g in qualified)
        for winner_slot, group in assignment.items():
            lookup[bits, column[winner_slot]] = GROUP_LETTERS.index(group)
    third_group = lookup[mask]  # (n, 8) group index of the third facing each winner slot

    runs = np.arange(n_runs)
    won: dict[int, np.ndarray] = {}
    lost: dict[int, np.ndarray] = {}

    def resolve(slot: str, opponent_slot: str) -> np.ndarray:
        if slot[0] in "12":
            return ranked[:, GROUP_LETTERS.index(slot[1]), int(slot[0]) - 1]
        if slot[0] == "3":
            return ranked[runs, third_group[:, column[opponent_slot]], 2]
        return won[int(slot[1:])] if slot[0] == "W" else lost[int(slot[1:])]

    # ---- knockout rounds ---------------------------------------------------
    locks = _knockout_locks(
        wc[wc["date"] >= C.WC2026_KNOCKOUT_START],
        fmt,
        index,
        shootouts
        if shootouts is not None
        else pd.DataFrame(columns=["date", "home_team", "away_team", "winner"]),
    )
    reached = {stage: np.zeros(n_teams) for stage in STAGES}
    n_model_calls = 0
    for stage in ROUND_ORDER:
        matches = [m for m in fmt.matches if m.round == stage]
        sides = {
            m.number: (resolve(m.home_slot, m.away_slot), resolve(m.away_slot, m.home_slot))
            for m in matches
        }
        # P(home side advances) for every pairing that occurs in this round
        p_adv = {m.number: np.full((n_teams, n_teams), np.nan) for m in matches}
        for number, (home_i, away_i, winner_i) in locks.items():
            if number in p_adv:  # either orientation of the real pairing is fixed
                p_adv[number][home_i, away_i] = 1.0 if winner_i == home_i else 0.0
                p_adv[number][away_i, home_i] = 1.0 - p_adv[number][home_i, away_i]
        ko_fixtures: list[Fixture] = []
        cells: list[tuple[int, int, int]] = []
        for m in matches:
            a, b = sides[m.number]
            ctx = MatchContext(m.date, m.city, m.country, True, C.WORLD_CUP, True)
            pairs = np.unique(np.stack([a, b], axis=1), axis=0)
            for h, w in pairs:
                if np.isnan(p_adv[m.number][h, w]):
                    ko_fixtures.append((teams[h], teams[w], ctx))
                    cells.append((m.number, int(h), int(w)))
        if ko_fixtures:
            lh, la = model.expected_goals(ko_fixtures)
            n_model_calls += 1
            for (number, h, w), p in zip(
                cells, advance_probability(lh, la, model.rho), strict=True
            ):
                p_adv[number][h, w] = p
        for m in matches:
            a, b = sides[m.number]
            home_wins = rng.random(n_runs) < p_adv[m.number][a, b]
            won[m.number] = np.where(home_wins, a, b)
            lost[m.number] = np.where(home_wins, b, a)
            if stage in reached:
                reached[stage] += np.bincount(np.concatenate([a, b]), minlength=n_teams)
    log.debug("knockout probabilities from %d batched model calls", n_model_calls)

    final = next(m.number for m in fmt.matches if m.round == "Final")
    table = pd.DataFrame(
        {
            "team": teams,
            "group": [group_of[t] for t in teams],
            "P_win_group": np.bincount(ranked[:, :, 0].ravel(), minlength=n_teams) / n_runs,
            **{f"P_{stage}": reached[stage] / n_runs for stage in STAGES},
            "P_champion": np.bincount(won[final], minlength=n_teams) / n_runs,
        }
    ).sort_values(
        ["P_champion", "P_Final", "P_SF", "team"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return SimulationResult(table=table.reset_index(drop=True), n_runs=n_runs, seed=seed)

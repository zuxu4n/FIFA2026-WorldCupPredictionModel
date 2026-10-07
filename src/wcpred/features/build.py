"""Leak-free feature engineering.

`build_features` makes one chronological pass over the results and returns a
long table with one row per (match, team perspective). Each row describes the
team, its opponent and the match context as they stood *before* kickoff; the
target is the number of goals that team scored.

`FeatureWorld` holds every team's state after the last played match and builds
rows for arbitrary fixtures (predictions, tournament simulation). Both paths go
through `perspective_features`, so training and prediction features cannot
drift apart; tests/test_features.py checks that they agree exactly.
"""

from __future__ import annotations

import difflib
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from wcpred import config as C
from wcpred.data.reference import UNKNOWN_CONFEDERATION, ReferenceData
from wcpred.data.results import normalize_name
from wcpred.features.elo import compute_elo
from wcpred.match import MatchContext, Segment

# (goals for, goals against, points) of a team's most recent played matches
History = deque[tuple[float, float, float]]
_FORM_STATS = [f"{stat}{w}" for w in C.FORM_WINDOWS for stat in ("gf", "ga", "ppg")]


class UnknownTeamError(KeyError):
    """A team name that never appears in the match data."""

    def __init__(self, name: str, suggestions: Sequence[str]) -> None:
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"unknown team {name!r}.{hint}")
        self.name = name
        self.suggestions = list(suggestions)

    def __str__(self) -> str:
        return str(self.args[0])


def form_stats(history: Iterable[tuple[float, float, float]]) -> dict[str, float]:
    """Mean goals for/against and points per game over each form window."""
    items = list(history)
    out: dict[str, float] = {}
    for w in C.FORM_WINDOWS:
        recent = items[-w:]
        k = len(recent)
        out[f"gf{w}"] = sum(x[0] for x in recent) / k if k else np.nan
        out[f"ga{w}"] = sum(x[1] for x in recent) / k if k else np.nan
        out[f"ppg{w}"] = sum(x[2] for x in recent) / k if k else np.nan
    out["nprior"] = float(len(items))  # capped at max(FORM_WINDOWS)
    return out


def perspective_features(
    team: Mapping[str, Any], opp: Mapping[str, Any], ctx: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble one perspective's features from scalars or equal-length arrays."""
    mv_t, mv_o = np.log1p(team["mv"]), np.log1p(opp["mv"])
    f: dict[str, Any] = {
        "elo_team": team["elo"],
        "elo_opp": opp["elo"],
        "elo_diff": team["elo"] - opp["elo"],
        "mv_team_log": mv_t,
        "mv_opp_log": mv_o,
        "mv_diff_log": mv_t - mv_o,
        "rank_pts_team": team["rank"],
        "rank_pts_opp": opp["rank"],
        "rank_pts_diff": team["rank"] - opp["rank"],
        "is_home": ctx["is_home"],
        "is_neutral": ctx["is_neutral"],
        "at_home_country": ctx["at_home_country"],
        "alt_match": ctx["alt_match"],
        "team_home_alt": team["alt_home"],
        "alt_gain": ctx["alt_match"] - team["alt_home"],
        "temp_match": ctx["temp_match"],
        "team_home_temp": team["home_temp"],
        "heat_gain": ctx["temp_match"] - team["home_temp"],
        "humidity_match": ctx["humidity_match"],
        "team_avg_age": team["avg_age"],
        "opp_avg_age": opp["avg_age"],
        "age_diff": team["avg_age"] - opp["avg_age"],
        "team_days_since": team["days"],
        "opp_days_since": opp["days"],
        "team_nprior": team["nprior"],
        "opp_nprior": opp["nprior"],
        "same_confederation": np.asarray(team["confed"] == opp["confed"], dtype=float),
        "confed_team": team["confed"],
        "confed_opp": opp["confed"],
        "importance": ctx["importance"],
        "is_knockout": ctx["is_knockout"],
        "year_frac": ctx["year_frac"],
    }
    for stat in _FORM_STATS:
        f[f"team_{stat}"] = team[stat]
        f[f"opp_{stat}"] = opp[stat]
    return {c: f[c] for c in C.ALL_FEATURES}


def _year_frac(dates: pd.Series | pd.Timestamp) -> Any:
    if isinstance(dates, pd.Timestamp):
        return dates.year + dates.dayofyear / 365.0
    return (dates.dt.year + dates.dt.dayofyear / 365.0).to_numpy(dtype=float)


@dataclass
class TeamState:
    """A team's rolling state after its most recent played match."""

    elo: float = C.ELO_START
    history: History = field(default_factory=lambda: deque(maxlen=max(C.FORM_WINDOWS)))
    last_played: pd.Timestamp | None = None


@dataclass
class FeatureWorld:
    """Team states at the end of the data, able to featurise any fixture."""

    states: dict[str, TeamState]
    ref: ReferenceData
    last_played: pd.Timestamp | None
    _rank_index: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for team, rows in self.ref.fifa_rankings.groupby("team"):
            self._rank_index[str(team)] = (
                rows["date"].to_numpy(dtype="datetime64[ns]"),
                rows["total_points"].to_numpy(dtype=float),
            )

    @property
    def teams(self) -> list[str]:
        return sorted(self.states)

    def resolve_team(self, name: str) -> str:
        """Normalise a user-supplied team name, raising with suggestions if unknown."""
        key = normalize_name(name)
        if key in self.states:
            return key
        lookup = {t.lower(): t for t in self.states}
        if key.lower() in lookup:
            return lookup[key.lower()]
        close = difflib.get_close_matches(key, self.states.keys(), n=3, cutoff=0.6)
        raise UnknownTeamError(name, close)

    def _ranking_points(self, team: str, date: pd.Timestamp) -> float:
        if team not in self._rank_index:
            return np.nan
        dates, points = self._rank_index[team]
        i = int(np.searchsorted(dates, date.to_datetime64(), side="right")) - 1
        return float(points[i]) if i >= 0 else np.nan

    def _side(self, team: str, ctx: MatchContext) -> dict[str, Any]:
        state = self.states[team]
        is_wc = ctx.tournament == C.WORLD_CUP
        side: dict[str, Any] = {
            "elo": state.elo,
            "mv": self.ref.market_value.get(team, np.nan),
            "rank": self._ranking_points(team, ctx.date),
            "alt_home": self.ref.team_altitude.get(team, C.DEFAULT_ALTITUDE_M),
            "home_temp": self.ref.home_temp(team),
            "confed": float(self.ref.confederation.get(team, UNKNOWN_CONFEDERATION)),
            "avg_age": self.ref.world_cup_squad_age(ctx.date.year, team) if is_wc else np.nan,
            "days": (
                np.nan if state.last_played is None else float((ctx.date - state.last_played).days)
            ),
        }
        side.update(form_stats(state.history))
        return side

    def matchup_row(
        self, team: str, opp: str, ctx: MatchContext, *, is_home: bool
    ) -> dict[str, float]:
        """Features for `team` (scoring against `opp`); `is_home` = listed home side."""
        team, opp = self.resolve_team(team), self.resolve_team(opp)
        home_team = team if is_home else opp
        temp, humidity = self.ref.climate(ctx.city, ctx.country, home_team)
        context = {
            "is_home": float(is_home and not ctx.neutral),
            "is_neutral": float(ctx.neutral),
            "at_home_country": float(ctx.country == team),
            "alt_match": self.ref.altitude(ctx.city, ctx.country, home_team),
            "temp_match": temp,
            "humidity_match": humidity,
            "is_knockout": float(ctx.knockout),
            "importance": ctx.importance,
            "year_frac": _year_frac(ctx.date),
        }
        row = perspective_features(self._side(team, ctx), self._side(opp, ctx), context)
        return {k: float(v) for k, v in row.items()}

    def matchup_frame(self, fixtures: Sequence[tuple[str, str, MatchContext]]) -> pd.DataFrame:
        """Two rows per fixture, interleaved: home perspective, then away."""
        rows = []
        for home, away, ctx in fixtures:
            rows.append(self.matchup_row(home, away, ctx, is_home=True))
            rows.append(self.matchup_row(away, home, ctx, is_home=False))
        return pd.DataFrame(rows, columns=C.ALL_FEATURES)


def _rolling_form(
    results: pd.DataFrame,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, History], dict[str, int]]:
    """Pre-match form for both sides of every row, plus each team's final history."""
    n = len(results)
    window = max(C.FORM_WINDOWS)
    keys = [*_FORM_STATS, "nprior", "days"]
    out = {side: {k: np.full(n, np.nan) for k in keys} for side in ("h", "a")}
    history: dict[str, History] = {}
    last_day: dict[str, int] = {}

    home = results["home_team"].to_numpy()
    away = results["away_team"].to_numpy()
    hs = results["home_score"].to_numpy(dtype=float)
    as_ = results["away_score"].to_numpy(dtype=float)
    played = results["played"].to_numpy(dtype=bool)
    day = results["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    empty: History = deque()

    for i in range(n):
        for side, team in (("h", home[i]), ("a", away[i])):
            arrays = out[side]
            for k, v in form_stats(history.get(team, empty)).items():
                arrays[k][i] = v
            if team in last_day:
                arrays["days"][i] = float(day[i] - last_day[team])
        if played[i]:
            points_h = 3.0 if hs[i] > as_[i] else 1.0 if hs[i] == as_[i] else 0.0
            points_a = 3.0 if as_[i] > hs[i] else 1.0 if hs[i] == as_[i] else 0.0
            history.setdefault(home[i], deque(maxlen=window)).append((hs[i], as_[i], points_h))
            history.setdefault(away[i], deque(maxlen=window)).append((as_[i], hs[i], points_a))
            last_day[home[i]] = last_day[away[i]] = int(day[i])
    return out, history, last_day


def build_features(results: pd.DataFrame, ref: ReferenceData) -> tuple[pd.DataFrame, FeatureWorld]:
    """Return (long feature table, FeatureWorld) for a cleaned results table.

    Long-table columns: ALL_FEATURES plus match_id, date, team, opp,
    is_home_persp, played, goals, segment and tournament. Unplayed rows are fixtures.
    """
    results = results.reset_index(drop=True)
    n = len(results)
    elo = compute_elo(results)
    form, history, last_day = _rolling_form(results)

    home = results["home_team"].to_numpy()
    away = results["away_team"].to_numpy()
    dates = results["date"]
    is_wc = (results["tournament"] == C.WORLD_CUP).to_numpy()
    years = dates.dt.year.to_numpy()

    def static_side(teams: np.ndarray, side: str, pre_elo: np.ndarray) -> dict[str, Any]:
        s: dict[str, Any] = {
            "elo": pre_elo,
            "mv": np.array([ref.market_value.get(t, np.nan) for t in teams]),
            "rank": ref.ranking_points(teams, dates.to_numpy()),
            "alt_home": np.array([ref.team_altitude.get(t, C.DEFAULT_ALTITUDE_M) for t in teams]),
            "home_temp": np.array([ref.home_temp(t) for t in teams]),
            "confed": np.array(
                [ref.confederation.get(t, UNKNOWN_CONFEDERATION) for t in teams], dtype=float
            ),
            "avg_age": np.array(
                [
                    ref.world_cup_squad_age(int(y), t) if wc else np.nan
                    for y, t, wc in zip(years, teams, is_wc, strict=True)
                ]
            ),
        }
        s.update(form[side])
        return s

    home_side = static_side(home, "h", elo.pre_home)
    away_side = static_side(away, "a", elo.pre_away)

    venue_cache: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for key in zip(results["city"], results["country"], home, strict=True):
        if key not in venue_cache:
            venue_cache[key] = (ref.altitude(*key), *ref.climate(*key))
    venue = np.array(
        [venue_cache[k] for k in zip(results["city"], results["country"], home, strict=True)]
    )

    neutral = results["neutral"].to_numpy(dtype=bool)
    country = results["country"].to_numpy()
    shared = {
        "is_neutral": neutral.astype(float),
        "alt_match": venue[:, 0] if n else np.array([]),
        "temp_match": venue[:, 1] if n else np.array([]),
        "humidity_match": venue[:, 2] if n else np.array([]),
        "is_knockout": results["is_knockout"].to_numpy(dtype=float),
        "importance": results["importance"].to_numpy(dtype=float),
        "year_frac": _year_frac(dates),
    }
    ctx_home = {
        **shared,
        "is_home": (~neutral).astype(float),
        "at_home_country": (country == home).astype(float),
    }
    ctx_away = {
        **shared,
        "is_home": np.zeros(n),
        "at_home_country": (country == away).astype(float),
    }

    segment = np.select(
        [
            results["is_knockout"].to_numpy(dtype=bool),
            results["final_tournament"].to_numpy(dtype=bool),
            results["importance"].to_numpy() <= 1.0,
        ],
        [Segment.KNOCKOUT.value, Segment.TOURNAMENT.value, Segment.FRIENDLY.value],
        default=Segment.COMPETITIVE.value,
    )

    def frame(
        feats: dict[str, Any],
        team: np.ndarray,
        opp: np.ndarray,
        goals: pd.Series,
        is_home_persp: int,
    ) -> pd.DataFrame:
        d = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in feats.items()})
        d["match_id"] = np.arange(n)
        d["date"] = dates.to_numpy()
        d["team"] = team
        d["opp"] = opp
        d["is_home_persp"] = is_home_persp
        d["played"] = results["played"].to_numpy(dtype=bool)
        d["goals"] = goals.to_numpy(dtype=float)
        d["segment"] = segment
        d["tournament"] = results["tournament"].to_numpy()
        return d

    long = pd.concat(
        [
            frame(
                perspective_features(home_side, away_side, ctx_home),
                home,
                away,
                results["home_score"],
                1,
            ),
            frame(
                perspective_features(away_side, home_side, ctx_away),
                away,
                home,
                results["away_score"],
                0,
            ),
        ],
        ignore_index=True,
    )

    epoch = np.datetime64("1970-01-01", "D")
    states = {
        team: TeamState(
            elo=elo.final.get(team, C.ELO_START),
            history=history.get(team, deque(maxlen=max(C.FORM_WINDOWS))),
            last_played=(
                pd.Timestamp(epoch + np.timedelta64(last_day[team], "D"))
                if team in last_day
                else None
            ),
        )
        for team in set(home) | set(away)
    }
    played_dates = dates[results["played"]]
    world = FeatureWorld(
        states=states,
        ref=ref,
        last_played=played_dates.max() if len(played_dates) else None,
    )
    return long, world

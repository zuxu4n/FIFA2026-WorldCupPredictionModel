"""Glue between the trained goals model and the feature world: predict fixtures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wcpred.features.build import FeatureWorld
from wcpred.match import MatchContext
from wcpred.models.goals import GoalsModel
from wcpred.models.scoreline import (
    advance_probability,
    outcome_probs,
    score_matrix,
    top_scorelines,
)

Fixture = tuple[str, str, MatchContext]


@dataclass(frozen=True)
class MatchPrediction:
    home: str
    away: str
    context: MatchContext
    xg_home: float
    xg_away: float
    p_home_win: float
    p_draw: float
    p_away_win: float
    p_home_advance: float  # knockout: after extra time and penalties
    top_scorelines: list[tuple[tuple[int, int], float]]


class Predictor:
    """A trained GoalsModel plus the team states it predicts from."""

    def __init__(self, model: GoalsModel, world: FeatureWorld) -> None:
        self.model = model
        self.world = world

    @property
    def rho(self) -> float:
        return self.model.calibration.rho

    def default_date(self) -> pd.Timestamp:
        """Day after the last played match: predictions reflect all known results."""
        if self.world.last_played is None:
            raise ValueError("no played matches in the data")
        return self.world.last_played + pd.Timedelta(days=1)

    def expected_goals(self, fixtures: Sequence[Fixture]) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated (xg_home, xg_away) for many fixtures in one model call."""
        if not fixtures:
            return np.zeros(0), np.zeros(0)
        raw = self.model.predict_raw(self.world.matchup_frame(fixtures))
        segments = [ctx.segment.value for _, _, ctx in fixtures]
        return self.model.calibration.scale(raw[0::2], raw[1::2], segments)

    def predict(self, home: str, away: str, ctx: MatchContext) -> MatchPrediction:
        home, away = self.world.resolve_team(home), self.world.resolve_team(away)
        lh, la = self.expected_goals([(home, away, ctx)])
        m = score_matrix(float(lh[0]), float(la[0]), self.rho)
        p_home, p_draw, p_away = outcome_probs(m)
        return MatchPrediction(
            home=home,
            away=away,
            context=ctx,
            xg_home=float(lh[0]),
            xg_away=float(la[0]),
            p_home_win=float(p_home),
            p_draw=float(p_draw),
            p_away_win=float(p_away),
            p_home_advance=float(advance_probability(lh, la, self.rho)[0]),
            top_scorelines=top_scorelines(m, 5),
        )

    def explain(
        self, home: str, away: str, ctx: MatchContext, top: int = 6
    ) -> dict[str, list[tuple[str, float]]]:
        """Largest SHAP drivers of each side's expected goals, as % effects on xG."""
        home, away = self.world.resolve_team(home), self.world.resolve_team(away)
        contrib = self.model.contributions(self.world.matchup_frame([(home, away, ctx)]))
        out = {}
        for team, row in ((home, contrib.iloc[0]), (away, contrib.iloc[1])):
            ranked = row.reindex(row.abs().sort_values(ascending=False).index)[:top]
            out[team] = [(str(f), float(np.expm1(v))) for f, v in ranked.items()]
        return out

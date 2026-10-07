"""Small domain types shared by feature engineering, prediction and simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from wcpred import config as C
from wcpred.data.results import classify_tournament, normalize_name


class Segment(str, Enum):
    """Match categories with different scoring levels (used by totals calibration)."""

    KNOCKOUT = "knockout"
    TOURNAMENT = "tournament"  # group stage of a final tournament
    COMPETITIVE = "competitive"  # qualifiers, Nations League, other competitions
    FRIENDLY = "friendly"


def segment_of(knockout: bool, final_tournament: bool, importance: float) -> Segment:
    if knockout:
        return Segment.KNOCKOUT
    if final_tournament:
        return Segment.TOURNAMENT
    if importance <= 1.0:
        return Segment.FRIENDLY
    return Segment.COMPETITIVE


@dataclass(frozen=True)
class MatchContext:
    """When, where and in which competition a fixture is played."""

    date: pd.Timestamp
    city: str
    country: str
    neutral: bool = True
    tournament: str = C.WORLD_CUP
    knockout: bool = False

    def __post_init__(self) -> None:
        # Normalise so contexts built from CLI input match the reference tables.
        object.__setattr__(self, "date", pd.Timestamp(self.date).normalize())
        object.__setattr__(self, "city", normalize_name(self.city))
        object.__setattr__(self, "country", normalize_name(self.country))

    @property
    def importance(self) -> float:
        return classify_tournament(self.tournament).importance

    @property
    def segment(self) -> Segment:
        rule = classify_tournament(self.tournament)
        return segment_of(self.knockout, rule.final_tournament, rule.importance)

"""The 2026 World Cup format: 12 groups of four, then a 32-team knockout bracket.

Data files (data/reference/):

* wc2026_groups_official.csv - group letter -> team
* wc2026_bracket.csv - matches 73-104 with date, venue and slot codes:
  "1A" group A winner, "2A" runner-up, "3ABCDF" a best third-placed team from
  one of those groups, "W74"/"L101" winner/loser of an earlier match
* wc2026_third_place_table.csv - FIFA's allocation of the eight qualifying
  third-placed teams to group winners, for each of the 495 combinations
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wcpred import config as C
from wcpred.data.results import DataError, normalize_name

GROUP_LETTERS = "ABCDEFGHIJKL"
ROUND_ORDER = ("R32", "R16", "QF", "SF", "3P", "Final")
_SLOT = re.compile(r"^(?:[12][A-L]|3[A-L]{2,}|[WL]\d+)$")


@dataclass(frozen=True)
class BracketMatch:
    number: int
    round: str
    date: pd.Timestamp
    home_slot: str
    away_slot: str
    city: str
    country: str


@dataclass(frozen=True)
class TournamentFormat:
    groups: dict[str, list[str]]
    matches: tuple[BracketMatch, ...]
    # qualified third-place groups (sorted letters) -> {group-winner slot: third's group}
    third_place_table: dict[str, dict[str, str]]

    @property
    def teams(self) -> list[str]:
        return [t for g in sorted(self.groups) for t in self.groups[g]]

    @classmethod
    def load(cls, directory: Path | None = None) -> TournamentFormat:
        directory = directory or C.PATHS.reference_dir
        groups_df = pd.read_csv(directory / "wc2026_groups_official.csv", encoding="utf-8")
        groups = {
            str(g): [normalize_name(t) for t in sub["team"]]
            for g, sub in groups_df.groupby("group", sort=True)
        }
        bracket_df = pd.read_csv(directory / "wc2026_bracket.csv", encoding="utf-8")
        matches = tuple(
            BracketMatch(
                number=int(r.match),
                round=str(r.round),
                date=pd.Timestamp(r.date),
                home_slot=str(r.home_slot),
                away_slot=str(r.away_slot),
                city=normalize_name(r.city),
                country=normalize_name(r.country),
            )
            for r in bracket_df.sort_values("match").itertuples(index=False)
        )
        table_df = pd.read_csv(
            directory / "wc2026_third_place_table.csv", encoding="utf-8", dtype=str
        )
        slot_cols = [c for c in table_df.columns if c not in ("option", "qualified")]
        table = {
            str(row["qualified"]): {col: str(row[col]) for col in slot_cols}
            for _, row in table_df.iterrows()
        }
        fmt = cls(groups=groups, matches=matches, third_place_table=table)
        fmt.validate()
        return fmt

    def validate(self) -> None:
        """Fail fast on malformed reference data instead of mid-simulation."""
        if sorted(self.groups) != list(GROUP_LETTERS) or any(
            len(ts) != 4 for ts in self.groups.values()
        ):
            raise DataError("expected 12 groups (A-L) of 4 teams")
        if len(set(self.teams)) != 48:
            raise DataError("a team appears in more than one group")
        numbers = [m.number for m in self.matches]
        if numbers != list(range(73, 105)):
            raise DataError("bracket must contain matches 73-104")
        for m in self.matches:
            for slot in (m.home_slot, m.away_slot):
                if not _SLOT.match(slot):
                    raise DataError(f"match {m.number}: bad slot code {slot!r}")
                if slot[0] in "WL" and int(slot[1:]) >= m.number:
                    raise DataError(f"match {m.number} references a later match: {slot}")
        r32_slots = [
            s for m in self.matches if m.round == "R32" for s in (m.home_slot, m.away_slot)
        ]
        expected = {f"{p}{g}" for p in "12" for g in GROUP_LETTERS}
        if {s for s in r32_slots if s[0] in "12"} != expected or len(r32_slots) != 32:
            raise DataError("Round of 32 must contain every group winner and runner-up once")
        third_opponents = {
            winner_slot: third_slot for winner_slot, third_slot in self.third_place_slots()
        }
        if len(self.third_place_table) != 495:
            raise DataError("third-place table must cover all 495 combinations")
        for qualified, assignment in self.third_place_table.items():
            if sorted(assignment.values()) != sorted(qualified):
                raise DataError(f"third-place row {qualified}: not a permutation of its groups")
            for winner_slot, group in assignment.items():
                if group not in third_opponents[winner_slot][1:]:
                    raise DataError(
                        f"third-place row {qualified}: 3{group} not eligible to play {winner_slot}"
                    )

    def third_place_slots(self) -> list[tuple[str, str]]:
        """(group-winner slot, third-place slot) for the 8 R32 ties against a third."""
        pairs = []
        for m in self.matches:
            if m.away_slot.startswith("3"):
                pairs.append((m.home_slot, m.away_slot))
            elif m.home_slot.startswith("3"):
                pairs.append((m.away_slot, m.home_slot))
        return pairs

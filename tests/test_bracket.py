import dataclasses

import pytest

from wcpred.data import DataError


def test_format_structure(wc_format):
    assert len(wc_format.teams) == 48
    rounds = [m.round for m in wc_format.matches]
    assert [rounds.count(r) for r in ("R32", "R16", "QF", "SF", "3P", "Final")] == [
        16,
        8,
        4,
        2,
        1,
        1,
    ]
    final = wc_format.matches[-1]
    assert (final.number, final.home_slot, final.away_slot) == (104, "W101", "W102")
    third_place = next(m for m in wc_format.matches if m.round == "3P")
    assert {third_place.home_slot, third_place.away_slot} == {"L101", "L102"}


def test_every_team_can_only_come_from_one_slot(wc_format):
    r32 = [s for m in wc_format.matches if m.round == "R32" for s in (m.home_slot, m.away_slot)]
    assert len(r32) == len(set(r32)) == 32
    assert sum(s.startswith("3") for s in r32) == 8


def test_official_third_place_allocation_for_2026(wc_format):
    # The real 2026 case: thirds from B, D, E, F, I, J, K, L qualified (FIFA option 67).
    assignment = wc_format.third_place_table["BDEFIJKL"]
    assert assignment == {
        "1A": "E",
        "1B": "J",
        "1D": "B",
        "1E": "D",
        "1G": "I",
        "1I": "F",
        "1K": "L",
        "1L": "K",
    }


def test_third_place_table_respects_eligibility(wc_format):
    # validate() already enforces this on load; spot-check the slot sets themselves
    slots = dict(wc_format.third_place_slots())
    assert slots["1E"] == "3ABCDF"
    for assignment in list(wc_format.third_place_table.values())[:50]:
        for winner, group in assignment.items():
            assert group in slots[winner][1:]


def test_validation_rejects_bad_data(wc_format):
    groups = dict(wc_format.groups)
    groups["A"] = groups["A"][:3]
    with pytest.raises(DataError, match="12 groups"):
        dataclasses.replace(wc_format, groups=groups).validate()

    matches = list(wc_format.matches)
    matches[-1] = dataclasses.replace(matches[-1], home_slot="W104")
    with pytest.raises(DataError, match="later match"):
        dataclasses.replace(wc_format, matches=tuple(matches)).validate()

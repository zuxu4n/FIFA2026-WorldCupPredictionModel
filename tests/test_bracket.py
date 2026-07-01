"""Tests for the official 2026 bracket parser + structure."""
import os
import pytest
from wcpred.bracket import (
    load_bracket, third_slots, load_official_groups, FINALS_TXT, GROUPS_CSV,
)

pytestmark = pytest.mark.skipif(
    not (os.path.exists(FINALS_TXT) and os.path.exists(GROUPS_CSV)),
    reason="official bracket not imported (run scripts/import_bracket.py)",
)


def test_bracket_has_32_matches():
    b = load_bracket()
    assert len(b) == 32
    assert min(b) == 73 and max(b) == 104


def test_r32_slot_counts():
    b = load_bracket()
    winners = runners = thirds = 0
    for no in range(73, 89):
        for r in (b[no].left, b[no].right):
            if r.kind == "pos" and r.place == 1:
                winners += 1
            elif r.kind == "pos" and r.place == 2:
                runners += 1
            elif r.kind == "third":
                thirds += 1
    assert (winners, runners, thirds) == (12, 12, 8)


def test_final_references_semifinal_winners():
    b = load_bracket()
    assert b[104].left.kind == "winner" and b[104].right.kind == "winner"
    assert {b[104].left.match, b[104].right.match} == {101, 102}


def test_eight_third_slots():
    assert len(third_slots(load_bracket())) == 8


def test_official_groups_complete():
    g = load_official_groups()
    assert len(g) == 12
    assert all(len(v) == 4 for v in g.values())
    assert sum(len(v) for v in g.values()) == 48

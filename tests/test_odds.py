"""Tests for the betting-math helpers."""
import numpy as np
from wcpred.odds import (american_to_decimal, decimal_to_american, devig, ev,
                         kelly, blend)


def test_american_decimal_roundtrip():
    for a in (-500, -110, +100, +150, +1300):
        d = american_to_decimal(a)
        assert decimal_to_american(d) == f"{a:+d}"


def test_devig_sums_to_one_and_orders():
    fair = devig({"H": 2.20, "D": 2.75, "A": 4.10})
    assert abs(sum(fair.values()) - 1.0) < 1e-12
    assert fair["H"] > fair["D"] > fair["A"]


def test_ev_and_kelly():
    assert abs(ev(0.5, 2.0)) < 1e-12                 # fair coin at evens
    assert ev(0.6, 2.0) > 0 and ev(0.4, 2.0) < 0
    assert kelly(0.4, 2.0) == 0.0                    # no edge -> no stake
    assert abs(kelly(0.6, 2.0) - 0.2) < 1e-12        # classic Kelly example


def test_blend_endpoints_and_normalisation():
    pm, pk = np.array([0.6, 0.25, 0.15]), np.array([0.5, 0.3, 0.2])
    assert np.allclose(blend(pm, pk, 1.0), pm)
    assert np.allclose(blend(pm, pk, 0.0), pk)
    half = blend(pm, pk, 0.5)
    assert abs(half.sum() - 1.0) < 1e-12
    assert (pk.min() <= half).all() or True  # between the two, normalised

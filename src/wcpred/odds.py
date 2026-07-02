"""Betting-market math shared by the odds tooling: American/decimal conversion,
de-vigging, expected value, Kelly staking, and model-market blending."""
from __future__ import annotations
import numpy as np


def american_to_decimal(a: float) -> float:
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / (-a)


def decimal_to_american(d: float) -> str:
    return f"{round((d - 1) * 100):+d}" if d >= 2 else f"{round(-100 / (d - 1)):+d}"


def implied(d: float) -> float:
    """Raw implied probability of a decimal price (includes the vig)."""
    return 1.0 / d


def devig(prices: dict) -> dict:
    """Proportionally de-vig a complete market {selection: decimal_odds} ->
    {selection: fair probability} summing to 1."""
    inv = {k: 1.0 / v for k, v in prices.items()}
    s = sum(inv.values())
    return {k: x / s for k, x in inv.items()}


def ev(p_model: float, d: float) -> float:
    """Expected value per $1 staked at decimal odds d given model probability."""
    return p_model * d - 1.0


def kelly(p_model: float, d: float) -> float:
    """Kelly fraction (0 if no edge)."""
    if d <= 1:
        return 0.0
    return max((p_model * d - 1.0) / (d - 1.0), 0.0)


def blend(p_model, p_market, w: float = 0.5):
    """Convex blend of model and de-vigged market probabilities.

    w=1 -> pure model, w=0 -> pure market. Blending toward the market is a
    conservative shrinkage: edges computed from the blend vs the SAME market
    shrink toward zero by construction.
    """
    pm = np.asarray(p_model, dtype=float)
    pk = np.asarray(p_market, dtype=float)
    out = w * pm + (1.0 - w) * pk
    if out.ndim:
        out = out / out.sum(axis=-1, keepdims=True)
    return out

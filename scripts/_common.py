"""Shared helpers for the CLI scripts."""
from __future__ import annotations
import sys
import pandas as pd
from wcpred import data as D
from wcpred.features import build_world
from wcpred.model import load_model


def load_everything():
    """Load matches, build the feature world, and load the trained model."""
    try:
        booster, meta = load_model()
    except Exception:
        sys.exit("No trained model found. Run:  python scripts/train.py")
    results = D.load_results()
    _, world = build_world(results)
    return results, world, booster, meta

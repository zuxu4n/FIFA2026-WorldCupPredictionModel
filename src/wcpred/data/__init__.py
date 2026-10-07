"""Data loading: match results, reference tables, and dataset download."""

from wcpred.data.reference import ReferenceData
from wcpred.data.results import (
    DataError,
    as_of,
    classify_tournament,
    clean_results,
    load_results,
    load_shootouts,
    normalize_name,
)

__all__ = [
    "DataError",
    "ReferenceData",
    "as_of",
    "classify_tournament",
    "clean_results",
    "load_results",
    "load_shootouts",
    "normalize_name",
]

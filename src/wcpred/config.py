"""Central configuration: paths, constants, model + Elo hyper-parameters."""
from __future__ import annotations
import os

# --- paths -----------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PKG_DIR, "..", ".."))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_REF = os.path.join(ROOT, "data", "reference")
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
OUTPUTS = os.path.join(ROOT, "outputs")
for _d in (DATA_RAW, DATA_REF, DATA_PROC, MODELS, OUTPUTS):
    os.makedirs(_d, exist_ok=True)

RESULTS_CSV = os.path.join(DATA_RAW, "results.csv")
SHOOTOUTS_CSV = os.path.join(DATA_RAW, "shootouts.csv")
TEAM_META_CSV = os.path.join(DATA_REF, "team_meta.csv")
VENUES_CSV = os.path.join(DATA_REF, "venues.csv")
MARKET_VALUES_CSV = os.path.join(DATA_REF, "market_values.csv")
FIFA_RANKINGS_CSV = os.path.join(DATA_REF, "fifa_rankings.csv")  # optional

MODEL_PATH = os.path.join(MODELS, "xgb_poisson.json")
MODEL_META_PATH = os.path.join(MODELS, "xgb_poisson_meta.json")

# --- public data source ----------------------------------------------------
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

# --- tournament context ----------------------------------------------------
WC2026_HOSTS = {"United States", "Canada", "Mexico"}
WC_TOURNAMENT_NAME = "FIFA World Cup"

# Match importance: higher weight = the model trusts these results more.
# Keyed by substrings searched (case-insensitive) in the `tournament` column.
TOURNAMENT_IMPORTANCE = [
    ("fifa world cup qualification", 2.5),
    ("fifa world cup", 4.0),
    ("confederations cup", 3.5),
    ("uefa euro", 3.5), ("copa am", 3.5), ("african cup", 3.0),
    ("afc asian cup", 3.0), ("gold cup", 2.8), ("nations league", 3.0),
    ("qualification", 2.2),
    ("friendly", 1.0),
]
DEFAULT_IMPORTANCE = 2.0

# --- Elo parameters --------------------------------------------------------
ELO_START = 1500.0
ELO_K = 40.0            # base K-factor (scaled up by match importance)
ELO_HOME_ADV = 65.0     # Elo points added to the home side (non-neutral)
ELO_REVERT = 0.0        # season reversion (0 = none; kept simple)

# --- feature engineering ---------------------------------------------------
FORM_WINDOWS = (5, 10)           # rolling form windows (matches)
RECENCY_HALFLIFE_YEARS = 6.0     # sample-weight half-life for recency decay
MIN_TRAIN_YEAR = 1990            # ignore very old matches for training

# --- XGBoost (count:poisson) ----------------------------------------------
XGB_PARAMS = {
    "objective": "count:poisson",
    "eval_metric": "poisson-nloglik",
    "eta": 0.03,
    "max_depth": 5,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "max_delta_step": 0.7,   # recommended for poisson stability
    "tree_method": "hist",
    "seed": 42,
}
XGB_NUM_ROUNDS = 1200
XGB_EARLY_STOPPING = 60
VALID_SINCE = "2023-01-01"   # time-based validation cut-off

# --- prediction ------------------------------------------------------------
MAX_GOALS = 12               # score-matrix truncation
DIXON_COLES_RHO = -0.05      # low-score dependency correction

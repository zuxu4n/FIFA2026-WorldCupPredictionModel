"""Import a historical FIFA-ranking dump into data/reference/fifa_rankings.csv.

    python scripts/import_fifa_rankings.py PATH

PATH may be a .zip (e.g. a download of github.com/Dato-Futbol/fifa-ranking) or a .csv.

It maps the ranking source's team spellings onto the match-dataset spellings
(IR Iran -> Iran, USA -> United States, Korea Republic -> South Korea, ...),
derives a per-date rank, and writes the schema the pipeline expects:
    date, team, rank, total_points
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

import pandas as pd

from wcpred import config as C
from wcpred.data import load_results, normalize_name

OUT = C.PATHS.reference_dir / "fifa_rankings.csv"

# FIFA-ranking spelling (accent-stripped) -> match-dataset spelling
ALIASES = {
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Chinese Taipei": "Taiwan",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Brunei Darussalam": "Brunei",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
}


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        cand = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not cand:
            sys.exit(f"no .csv found inside {path}")
        # prefer a file that looks like the historical ranking
        name = next((n for n in cand if "rank" in n.lower()), cand[0])
        print(f"reading {name} from {os.path.basename(path)}")
        with z.open(name) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
    print(f"reading {path}")
    return pd.read_csv(path, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")
    df = _read_any(path)

    cols = {c.lower(): c for c in df.columns}
    if "points" in cols and "total_points" not in cols:
        df = df.rename(columns={cols["points"]: "total_points"})
    for need in ("team", "total_points", "date"):
        if need not in df.columns:
            sys.exit(f"source is missing required column '{need}' (has: {list(df.columns)})")

    df = df[["team", "total_points", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df = df.dropna(subset=["date", "total_points"])

    # normalise team names onto the match-dataset spelling
    df["team"] = df["team"].map(normalize_name).map(lambda t: ALIASES.get(t, t))
    # derive rank within each ranking snapshot
    df["rank"] = df.groupby("date")["total_points"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values(["date", "rank"])[["date", "team", "rank", "total_points"]]

    df["date"] = df["date"].dt.date
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(
        f"wrote {OUT}  ({len(df):,} rows, "
        f"{df['date'].nunique()} snapshots, "
        f"{df['date'].min()} -> {df['date'].max()})"
    )

    # coverage report against the WC2026 finalists
    res = load_results()
    fin = res[(res.tournament == C.WORLD_CUP) & (res.date >= "2026-06-01")]
    wc = sorted(set(fin.home_team) | set(fin.away_team))
    have = set(df["team"].map(normalize_name))
    missing = [t for t in wc if normalize_name(t) not in have]
    print(
        f"WC2026 coverage: {len(wc) - len(missing)}/{len(wc)} matched"
        + (f"; still missing: {missing}" if missing else " (all matched)")
    )


if __name__ == "__main__":
    main()

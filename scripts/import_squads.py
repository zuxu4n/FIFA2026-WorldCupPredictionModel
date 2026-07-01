"""Import squad rosters (all World Cups) -> data/reference/squad_ages.csv.

    python scripts/import_squads.py [PATH]

PATH is the openfootball world-cup zip or extracted dir (must contain
more/<year>_squads.txt files). Default: the zip in Downloads.

For every (tournament year, team) it computes the average player age at that
tournament (and the age spread), so the feature can be joined *temporally* to
historical matches without leakage. Schema:
    year, team, avg_age, age_std, n_players
"""
from __future__ import annotations
import datetime as dt
import os
import re
import sys
import zipfile

from wcpred import data as D
from wcpred import config as C

DEFAULT_PATH = os.path.expanduser(r"~\Downloads\worldcup-master (1).zip")
OUT = os.path.join(C.DATA_REF, "squad_ages.csv")

ALIASES = {
    "USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea", "IR Iran": "Iran",
    "Republic of Ireland": "Ireland",
}
HEADER_RE = re.compile(r"^==\s*(.+?)\s*(?:#.*)?$")
BIRTH_RE = re.compile(r"\bb\.\s*(\d{4})/(\d{2})/(\d{2})")


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _squad_files(path: str) -> list[tuple[int, str]]:
    """Return (year, text) for each <year>_squads.txt found."""
    out = []
    rx = re.compile(r"(\d{4})_squads\.txt$")
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            m = rx.search(n)
            if m:
                out.append((int(m.group(1)), _decode(z.read(n))))
    else:
        for root, _, files in os.walk(path):
            for f in files:
                m = rx.search(f)
                if m:
                    out.append((int(m.group(1)), _decode(open(os.path.join(root, f), "rb").read())))
    return sorted(out)


def _ref_date(year: int) -> dt.date:
    return dt.date(2026, 6, 11) if year == 2026 else dt.date(year, 6, 1)


def _parse(year: int, text: str) -> list[tuple[str, float]]:
    """Yield (team, avg_age, age_std, n) per team in one squads file."""
    ref = _ref_date(year)
    rows, team, ages = [], None, []

    def flush():
        if team and ages:
            mean = sum(ages) / len(ages)
            var = sum((a - mean) ** 2 for a in ages) / len(ages)
            rows.append((team, round(mean, 2), round(var ** 0.5, 2), len(ages)))

    for line in text.splitlines():
        h = HEADER_RE.match(line)
        if h:
            flush()
            name = ALIASES.get(h.group(1).strip(), h.group(1).strip())
            team, ages = D.strip_accents(name), []
            continue
        b = BIRTH_RE.search(line)
        if b and team:
            y, mo, d = map(int, b.groups())
            try:
                born = dt.date(y, mo, d)
            except ValueError:
                continue
            ages.append((ref - born).days / 365.25)
    flush()
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        sys.exit(f"path not found: {path}")
    files = _squad_files(path)
    if not files:
        sys.exit(f"no *_squads.txt found in {path}")

    n_rows = 0
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("year,team,avg_age,age_std,n_players\n")
        for year, text in files:
            for team, avg, std, n in _parse(year, text):
                f.write(f"{year},{team},{avg},{std},{n}\n")
                n_rows += 1
    print(f"wrote {OUT}  ({n_rows} (year,team) rows across {len(files)} tournaments: "
          f"{files[0][0]}-{files[-1][0]})")

    # quick look at 2026
    import pandas as pd
    df = pd.read_csv(OUT)
    cur = df[df.year == 2026].sort_values("avg_age")
    print(f"\n2026 squads: {len(cur)} teams, mean age "
          f"{cur.avg_age.mean():.1f} (youngest {cur.iloc[0].team} {cur.iloc[0].avg_age}, "
          f"oldest {cur.iloc[-1].team} {cur.iloc[-1].avg_age})")


if __name__ == "__main__":
    main()

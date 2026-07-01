"""Vendor the official 2026 bracket + group draw from the openfootball dataset.

    python scripts/import_bracket.py [PATH]

PATH is the openfootball world-cup zip or an extracted dir (it must contain
2026--usa/cup_finals.txt and 2026--usa/cup.txt). Default: the zip in Downloads.

Writes:
    data/reference/wc2026_finals.txt          (the knockout bracket, verbatim)
    data/reference/wc2026_groups_official.csv  (official A-L group -> team)
"""
from __future__ import annotations
import os
import re
import sys
import zipfile

from wcpred import data as D
from wcpred import config as C

DEFAULT_PATH = os.path.expanduser(r"~\Downloads\worldcup-master (1).zip")
FINALS_OUT = os.path.join(C.DATA_REF, "wc2026_finals.txt")
GROUPS_OUT = os.path.join(C.DATA_REF, "wc2026_groups_official.csv")

# openfootball spelling -> match-dataset spelling
ALIASES = {
    "USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea", "IR Iran": "Iran",
}


def _read(path: str, member_suffix: str) -> str:
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        name = next((n for n in z.namelist() if n.endswith(member_suffix)), None)
        if not name:
            sys.exit(f"{member_suffix} not found in {path}")
        raw = z.read(name)
    else:
        full = None
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.join(root, f).replace("\\", "/").endswith(member_suffix):
                    full = os.path.join(root, f)
        if not full:
            sys.exit(f"{member_suffix} not found under {path}")
        raw = open(full, "rb").read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _alias(name: str) -> str:
    name = name.strip()
    name = ALIASES.get(name, name)
    return D.strip_accents(name)


def parse_groups(cup_txt: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in cup_txt.splitlines():
        m = re.match(r"\s*Group\s+([A-L])\s*\|\s*(.+)$", line)
        if not m:
            continue
        letter, rest = m.group(1), m.group(2)
        teams = [_alias(t) for t in re.split(r"\s{2,}", rest.strip()) if t.strip()]
        if len(teams) == 4:
            groups[letter] = teams
    return groups


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        sys.exit(f"path not found: {path}")

    finals = _read(path, "2026--usa/cup_finals.txt")
    cup = _read(path, "2026--usa/cup.txt")

    with open(FINALS_OUT, "w", encoding="utf-8") as f:
        f.write(finals)
    print(f"wrote {FINALS_OUT}  ({len(finals.splitlines())} lines)")

    groups = parse_groups(cup)
    with open(GROUPS_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("group,team\n")
        for g in sorted(groups):
            for t in groups[g]:
                f.write(f"{g},{t}\n")
    print(f"wrote {GROUPS_OUT}  ({len(groups)} groups, "
          f"{sum(len(v) for v in groups.values())} teams)")

    # validate against teams in the live results data
    res = D.load_results()
    fin = res[(res.tournament == C.WC_TOURNAMENT_NAME) & (res.date >= "2026-06-01")]
    live = set(map(D.strip_accents, set(fin.home_team) | set(fin.away_team)))
    official = {D.strip_accents(t) for v in groups.values() for t in v}
    missing = official - live
    print(f"group teams matching live data: {len(official & live)}/48"
          + (f"  UNMATCHED: {missing}" if missing else "  (all matched)"))


if __name__ == "__main__":
    main()

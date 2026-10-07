"""Import FIFA's Round-of-32 third-place allocation table (2026 regulations, Annex C).

    python scripts/import_third_place_table.py [WIKITEXT_FILE]

Which group winner plays which third-placed team depends on *which* eight of the
twelve third-placed teams qualify; FIFA publishes all C(12, 8) = 495 cases. This
script reads the table from the wikitext of Wikipedia's
"Template:2026 FIFA World Cup third-place table" (downloaded if no file is given)
and writes data/reference/wc2026_third_place_table.csv:

    option, qualified, 1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L

`qualified` is the sorted string of the eight groups whose third-placed teams
advance; each slot column holds the group whose third-placed team that group
winner faces.
"""

from __future__ import annotations

import csv
import re
import sys
import urllib.request
from pathlib import Path

from wcpred.config import PATHS

SOURCE_URL = (
    "https://en.wikipedia.org/w/index.php?"
    "title=Template:2026_FIFA_World_Cup_third-place_table&action=raw"
)
SLOT_COLUMNS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]
GROUPS = "ABCDEFGHIJKL"


def parse_table(wikitext: str) -> list[list[str]]:
    rows: list[list[str]] = []
    # Each row starts with "! scope="row" | <n>" followed by 12 group cells and 8 slot cells.
    for block in re.split(r'^!\s*scope="row"\s*\|\s*', wikitext, flags=re.M)[1:]:
        option = int(block.split("\n", 1)[0].strip().rstrip("*"))  # "*" marks the 2026 case
        body = re.sub(r'^!\s*rowspan="\d+"\s*\|\s*$', "", block, flags=re.M)
        body = body.split("|-")[0].split("|}")[0]
        cells = [c.strip() for c in re.split(r"\|\||\n\|", body.split("\n", 1)[1])]
        cells = [c.lstrip("|").strip() for c in cells]
        group_cells, slot_cells = cells[:12], [c for c in cells[12:] if c]
        qualified = "".join(
            g for g, c in zip(GROUPS, group_cells, strict=True) if c.strip("'") == g
        )
        assigned = [c.removeprefix("3") for c in slot_cells]
        if len(qualified) != 8 or len(assigned) != 8:
            raise ValueError(f"option {option}: could not parse row ({qualified=}, {assigned=})")
        if sorted(assigned) != sorted(qualified):
            raise ValueError(f"option {option}: assignment {assigned} != qualified {qualified}")
        rows.append([str(option), qualified, *assigned])
    return rows


def main() -> None:
    if len(sys.argv) > 1:
        wikitext = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "wcpred"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            wikitext = resp.read().decode("utf-8")

    rows = parse_table(wikitext)
    if len(rows) != 495 or len({r[1] for r in rows}) != 495:
        sys.exit(f"expected 495 distinct combinations, parsed {len(rows)}")

    out = PATHS.reference_dir / "wc2026_third_place_table.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["option", "qualified", *SLOT_COLUMNS])
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} combinations)")


if __name__ == "__main__":
    main()

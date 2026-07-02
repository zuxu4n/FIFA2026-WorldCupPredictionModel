"""Parse the official 2026 knockout bracket (data/reference/wc2026_finals.txt)
and run a venue-aware Monte-Carlo along the *real* tournament paths.

Each Round-of-32 slot is one of: a group winner (1X), a group runner-up (2X),
or a best third-placed team drawn from a set of groups (e.g. 3A/B/C/D/F). Later
rounds reference earlier match winners/losers (W74, L101). The qualifying thirds
are assigned to their slots by bipartite matching that respects each slot's
eligible-group set (a faithful relaxation of FIFA's official placement table).
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import os
import re
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .predict import knockout_advance, predict_lambdas
from .simulate import (current_standings, remaining_group_fixtures,
                       _sample_scoreline, KNOCKOUT_START)

FINALS_TXT = os.path.join(C.DATA_REF, "wc2026_finals.txt")
GROUPS_CSV = os.path.join(C.DATA_REF, "wc2026_groups_official.csv")

ROUND_OF = {  # match number -> round label
    **{n: "R32" for n in range(73, 89)},
    **{n: "R16" for n in range(89, 97)},
    **{n: "QF" for n in range(97, 101)},
    **{n: "SF" for n in (101, 102)},
    103: "3P", 104: "Final",
}

# openfootball venue string -> (venues.csv city, country)
VENUE_MAP = {
    "Los Angeles (Inglewood)": ("Inglewood", "United States"),
    "Boston (Foxborough)": ("Foxborough", "United States"),
    "Monterrey (Guadalupe)": ("Guadalupe", "Mexico"),
    "Houston": ("Houston", "United States"),
    "New York/New Jersey (East Rutherford)": ("East Rutherford", "United States"),
    "Dallas (Arlington)": ("Arlington", "United States"),
    "Mexico City": ("Mexico City", "Mexico"),
    "Atlanta": ("Atlanta", "United States"),
    "San Francisco Bay Area (Santa Clara)": ("Santa Clara", "United States"),
    "Seattle": ("Seattle", "United States"),
    "Toronto": ("Toronto", "Canada"),
    "Miami (Miami Gardens)": ("Miami Gardens", "United States"),
    "Vancouver": ("Vancouver", "Canada"),
    "Kansas City": ("Kansas City", "United States"),
    "Philadelphia": ("Philadelphia", "United States"),
}


@dataclass(frozen=True)
class Ref:
    """A bracket slot source."""
    kind: str            # 'pos' | 'third' | 'winner' | 'loser'
    place: int = 0       # for 'pos': 1 or 2
    group: str = ""      # for 'pos': group letter
    groups: frozenset = frozenset()  # for 'third': eligible groups
    match: int = 0       # for 'winner'/'loser': match number


@dataclass
class BracketMatch:
    no: int
    left: Ref
    right: Ref
    city: str
    country: str


def _parse_ref(tok: str, comment_codes: list[str]) -> Ref:
    tok = tok.strip()
    if re.fullmatch(r"[12][A-L]", tok):
        return Ref("pos", int(tok[0]), tok[1])
    if re.fullmatch(r"3[A-L](?:/[A-L])+", tok):
        return Ref("third", groups=frozenset(tok[1:].split("/")))
    if re.fullmatch(r"W\d+", tok):
        return Ref("winner", match=int(tok[1:]))
    if re.fullmatch(r"L\d+", tok):
        return Ref("loser", match=int(tok[1:]))
    # resolved team name -> take its slot code from the trailing comment
    if comment_codes:
        code = comment_codes.pop(0)
        return Ref("pos", int(code[0]), code[1])
    raise ValueError(f"cannot resolve bracket token {tok!r}")


def load_bracket() -> dict[int, BracketMatch]:
    text = open(FINALS_TXT, encoding="utf-8").read()
    line_re = re.compile(
        r"\s*\((\d+)\)\s+[\d:]+\s+UTC[+\-]\d+\s+(.+?)\s+@\s+(.+?)\s*(?:##\s*(.*))?$"
    )
    out: dict[int, BracketMatch] = {}
    for line in text.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        no = int(m.group(1))
        matchup, city_raw, comment = m.group(2), m.group(3).strip(), m.group(4) or ""
        codes = re.findall(r"[12][A-L]", comment)
        left_raw, right_raw = re.split(r"\s+v\s+", matchup, maxsplit=1)
        left = _parse_ref(left_raw, codes)
        right = _parse_ref(right_raw, codes)
        city, country = VENUE_MAP.get(city_raw, (city_raw, "United States"))
        out[no] = BracketMatch(no, left, right, city, country)
    return out


def load_official_groups() -> dict[str, list[str]]:
    df = pd.read_csv(GROUPS_CSV, encoding="utf-8")
    df["team"] = df["team"].map(D.strip_accents)
    return {g: list(sub["team"]) for g, sub in df.groupby("group")}


def third_slots(bracket: dict[int, BracketMatch]) -> list[tuple[int, str, frozenset]]:
    """List of (match_no, side, eligible_groups) for every best-third slot."""
    slots = []
    for no, bm in bracket.items():
        for side, ref in (("left", bm.left), ("right", bm.right)):
            if ref.kind == "third":
                slots.append((no, side, ref.groups))
    return slots


def _match_thirds(slots, qualified: set[str], rng,
                  pinned: dict | None = None) -> dict[tuple[int, str], str]:
    """Bipartite-match qualifying third-place groups to eligible slots (Kuhn's).

    `pinned` maps (match_no, side) -> group for slots whose real assignment is
    already known from played matches; those are fixed and excluded from the
    matching, along with their groups.
    """
    pinned = pinned or {}
    fixed = {k: g for k, g in pinned.items()}
    used_groups = set(fixed.values())
    slots = [s for s in slots if (s[0], s[1]) not in fixed]
    qualified = set(qualified) - used_groups
    order = list(range(len(slots)))
    rng.shuffle(order)               # randomise to vary among valid assignments
    groups = list(qualified)
    rng.shuffle(groups)
    match_slot_to_group: dict[int, str] = {}
    match_group_to_slot: dict[str, int] = {}

    def try_assign(si, seen):
        for g in groups:
            if g in slots[si][2] and g not in seen:
                seen.add(g)
                if g not in match_group_to_slot or try_assign(match_group_to_slot[g], seen):
                    match_slot_to_group[si] = g
                    match_group_to_slot[g] = si
                    return True
        return False

    for si in order:
        try_assign(si, set())
    # fall back: drop any unmatched groups into leftover slots
    used_slots = set(match_slot_to_group)
    leftover_groups = [g for g in groups if g not in match_group_to_slot]
    for si in order:
        if si not in used_slots and leftover_groups:
            match_slot_to_group[si] = leftover_groups.pop()
    out = {(slots[si][0], slots[si][1]): g for si, g in match_slot_to_group.items()}
    out.update(fixed)
    return out


def _shootout_winners() -> dict:
    """(date, frozenset(team pair)) -> winner, from shootouts.csv if present."""
    if not os.path.exists(C.SHOOTOUTS_CSV):
        return {}
    s = pd.read_csv(C.SHOOTOUTS_CSV, encoding="utf-8")
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    for col in ("home_team", "away_team", "winner"):
        s[col] = s[col].astype(str).map(D.strip_accents)
    return {(r.date, frozenset((r.home_team, r.away_team))): r.winner
            for r in s.itertuples(index=False)}


R16_START = pd.Timestamp("2026-07-04")


def played_knockouts(results: pd.DataFrame) -> dict:
    """frozenset(team pair) -> (winner, date) for completed knockout matches.

    Draws (after extra time) are resolved via shootouts.csv; if the shootout
    winner is unknown the match is left to the simulator.
    """
    f = results[(results.tournament == C.WC_TOURNAMENT_NAME)
                & results.played & (results.date >= KNOCKOUT_START)]
    so = _shootout_winners() if (f.home_score == f.away_score).any() else {}
    lock = {}
    for r in f.itertuples(index=False):
        pair = frozenset((r.home_team, r.away_team))
        if r.home_score > r.away_score:
            w = r.home_team
        elif r.away_score > r.home_score:
            w = r.away_team
        else:
            w = so.get((r.date, pair))
            if w is None or w not in pair:
                continue
        lock[pair] = (w, r.date)
    return lock


def det_order(ts, st):
    """Deterministic group ordering: pts, gd, gf, then name (name only breaks
    exact ties; FIFA uses head-to-head/fair-play there — rare and unknowable)."""
    return sorted(ts, key=lambda t: (st[t]["pts"], st[t]["gd"], st[t]["gf"], t),
                  reverse=True)


def fixed_group_phase(results, groups, bracket, tslots):
    """When the group stage is complete, positions are deterministic and the
    scheduled/played R32 fixtures reveal the real third-slot assignments.

    Returns (winners, runners, thirds, best8, pin_slot) or None if incomplete.
    """
    from .simulate import current_standings, _group_stage
    st = current_standings(results, groups)
    teams = [t for ts in groups.values() for t in ts]
    if any(st[t]["played"] < 3 for t in teams):
        return None
    winners, runners, thirds = {}, {}, {}
    for g, ts in groups.items():
        o = det_order(ts, st)
        winners[g], runners[g], thirds[g] = o[0], o[1], o[2]
    ranked = sorted(thirds, key=lambda g: (st[thirds[g]]["pts"], st[thirds[g]]["gd"],
                                           st[thirds[g]]["gf"], thirds[g]),
                    reverse=True)
    best8 = ranked[:8]

    # pin third slots from all KO-window fixtures (played or scheduled)
    ko = results[(results.tournament == C.WC_TOURNAMENT_NAME)
                 & (results.date >= KNOCKOUT_START) & (results.date < R16_START)]
    pairs = [frozenset((r.home_team, r.away_team)) for r in ko.itertuples(index=False)]
    pin_slot = {}
    for no, side, elig in tslots:
        bm = bracket[no]
        other_ref = bm.right if side == "left" else bm.left
        if other_ref.kind != "pos":
            continue
        other_team = (winners if other_ref.place == 1 else runners)[other_ref.group]
        for pair in pairs:
            if other_team in pair:
                partner = next(t for t in pair if t != other_team)
                for g in elig:
                    if g in best8 and thirds[g] == partner:
                        pin_slot[(no, side)] = g
                        break
    return winners, runners, thirds, best8, pin_slot


def simulate_official(booster, world, results: pd.DataFrame, n_sims: int = 5000,
                      seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = load_official_groups()
    bracket = load_bracket()
    tslots = third_slots(bracket)
    base = current_standings(results, groups)
    rem = remaining_group_fixtures(results, groups, booster, world)
    ko_lock = played_knockouts(results)   # completed KO results are fixed
    fixed = fixed_group_phase(results, groups, bracket, tslots)
    teams = [t for ts in groups.values() for t in ts]
    team_group = {t: g for g, ts in groups.items() for t in ts}
    order = sorted(bracket)

    # cache knockout advance prob per (team, opp, city) — venue affects altitude
    adv_cache: dict[tuple[str, str, str], float] = {}

    def p_adv(a, b, city, country):
        key = (a, b, city)
        if key not in adv_cache:
            lh, la = predict_lambdas(booster, world, a, b, city=city,
                                     country=country, neutral=True, importance=4.0)
            adv_cache[key] = knockout_advance(lh, la)[0]
        return adv_cache[key]

    counters = {t: defaultdict(int) for t in teams}

    def rank_key(t, s):
        return (s["pts"], s["gd"], s["gf"], rng.random())

    for _ in range(n_sims):
        st = {t: dict(base[t]) for t in teams}
        for fx in rem:
            gh, ga = _sample_scoreline(fx["M"], rng)
            for t, f_, a_ in ((fx["home"], gh, ga), (fx["away"], ga, gh)):
                s = st[t]
                s["gf"] += f_; s["ga"] += a_; s["gd"] += f_ - a_
                s["pts"] += 3 if f_ > a_ else (1 if f_ == a_ else 0)

        if fixed is not None:
            # group stage complete: deterministic positions + pinned slots
            winners, runners, thirds, best, pin_slot = fixed
            winners, runners, thirds = dict(winners), dict(runners), dict(thirds)
        else:
            winners, runners, thirds = {}, {}, {}
            for g, ts in groups.items():
                o = sorted(ts, key=lambda t: rank_key(t, st[t]), reverse=True)
                winners[g], runners[g], thirds[g] = o[0], o[1], o[2]
            best = sorted(thirds, key=lambda g: rank_key(thirds[g], st[thirds[g]]),
                          reverse=True)[:8]
            pin_slot = {}
        for g in groups:
            counters[winners[g]]["win_group"] += 1
        slot_group = _match_thirds(tslots, set(best), rng, pinned=pin_slot)

        won, lost, parts = {}, {}, defaultdict(set)

        def resolve(ref, no, side):
            if ref.kind == "pos":
                return (winners if ref.place == 1 else runners)[ref.group]
            if ref.kind == "third":
                return thirds[slot_group[(no, side)]]
            if ref.kind == "winner":
                return won[ref.match]
            return lost[ref.match]

        for no in order:
            bm = bracket[no]
            a = resolve(bm.left, no, "left")
            b = resolve(bm.right, no, "right")
            parts[ROUND_OF[no]].add(a); parts[ROUND_OF[no]].add(b)
            locked = ko_lock.get(frozenset((a, b)))
            if locked is not None:                       # already played: fixed
                wt = locked[0]
                won[no], lost[no] = wt, (b if wt == a else a)
            elif rng.random() < p_adv(a, b, bm.city, bm.country):
                won[no], lost[no] = a, b
            else:
                won[no], lost[no] = b, a

        for rnd, ps in parts.items():
            for t in ps:
                counters[t][rnd] += 1
        for t in (won[104],):
            counters[t]["champion"] += 1

    n = n_sims
    rows = []
    for t in teams:
        c = counters[t]
        rows.append({
            "team": t, "group": team_group[t], "elo": round(world.state(t).elo),
            "P_win_group": c["win_group"] / n, "P_advance": c["R32"] / n,
            "P_R16": c["R16"] / n, "P_QF": c["QF"] / n, "P_SF": c["SF"] / n,
            "P_final": c["Final"] / n, "P_champion": c["champion"] / n,
        })
    table = pd.DataFrame(rows).sort_values("P_champion", ascending=False)
    return {"groups": groups, "table": table, "bracket": bracket, "n_sims": n}

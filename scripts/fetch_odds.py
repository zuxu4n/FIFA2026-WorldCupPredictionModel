"""Pull live odds from The Odds API, de-vig them, and compute the model's
expected value (EV) on each outcome — i.e. find genuine value bets.

    set ODDS_API_KEY=...           (or pass --key)
    python scripts/fetch_odds.py --book fanduel --min-ev 0.02

EV per $1 staked = model_prob * decimal_odds - 1  (positive => +EV vs that price).
Kelly fraction   = (model_prob*decimal - 1) / (decimal - 1).
Fair prob        = de-vigged market implied probability (context only).

One /odds call = 1 credit (1 market x 1 region). Raw response is cached to
outputs/odds_raw.json so re-runs with --from-cache cost nothing.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request
import pandas as pd
from _common import load_everything
from wcpred import config as C
from wcpred import data as D
from wcpred.predict import predict_fixture

RAW = os.path.join(C.OUTPUTS, "odds_raw.json")
# odds-API name -> our data spelling (strip_accents handles the rest)
ALIAS = {"USA": "United States"}


def norm(t):
    t = t.replace(" & ", " and ")   # "Bosnia & Herzegovina" -> "... and ..."
    t = ALIAS.get(t, t)
    return D.strip_accents(t)


def american(d):
    return f"{round((d-1)*100):+d}" if d >= 2 else f"{round(-100/(d-1)):+d}"


def fetch(key, sport, regions, markets):
    url = (f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={key}"
           f"&regions={regions}&markets={markets}&oddsFormat=decimal")
    req = urllib.request.Request(url, headers={"User-Agent": "wcpred/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        remaining = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        data = json.loads(r.read())
    json.dump(data, open(RAW, "w"))
    print(f"fetched {len(data)} events | credits used={used} remaining={remaining}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("ODDS_API_KEY"))
    ap.add_argument("--sport", default="soccer_fifa_world_cup")
    ap.add_argument("--regions", default="us")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--min-ev", type=float, default=0.0, help="flag bets above this EV")
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()

    if args.from_cache and os.path.exists(RAW):
        data = json.load(open(RAW)); print(f"loaded {len(data)} cached events")
    else:
        if not args.key:
            sys.exit("no API key: set ODDS_API_KEY or pass --key")
        data = fetch(args.key, args.sport, args.regions, "h2h")

    results, world, booster, meta = load_everything()
    # index ONLY real upcoming (unplayed) WC fixtures, by normalised team-set
    up = results[(results.tournament == C.WC_TOURNAMENT_NAME) & (~results.played)]
    fx = {frozenset((norm(r.home_team), norm(r.away_team))): r
          for r in up.itertuples(index=False)}

    rows, skipped = [], []
    for ev in data:
        home, away = norm(ev["home_team"]), norm(ev["away_team"])
        f = fx.get(frozenset((home, away)))
        if f is None:               # not a recognised current WC2026 fixture
            skipped.append(f"{ev['home_team']} v {ev['away_team']}")
            continue
        book = next((b for b in ev.get("bookmakers", []) if b["key"] == args.book), None)
        h2h = next((m for m in book["markets"] if m["key"] == "h2h"), None) if book else None
        if not h2h:
            skipped.append(f"{ev['home_team']} v {ev['away_team']} (no {args.book} h2h)")
            continue
        price = {norm(o["name"]) if o["name"] != "Draw" else "Draw": o["price"]
                 for o in h2h["outcomes"]}
        fh, fa = norm(f.home_team), norm(f.away_team)   # data-spelling teams
        if fh not in price or fa not in price or "Draw" not in price:
            skipped.append(f"{ev['home_team']} v {ev['away_team']} (odds/team mismatch)")
            continue
        inv = {k: 1 / v for k, v in price.items()}
        over = sum(inv.values())
        fair = {k: inv[k] / over for k in price}       # de-vigged market prob

        # predict with the REAL fixture (correct spelling, venue, host/neutral)
        p = predict_fixture(booster, world, f.home_team, f.away_team,
                            city=f.city, country=f.country, neutral=bool(f.neutral),
                            importance=4.0)
        model = {fh: p.p_home_win, "Draw": p.p_draw, fa: p.p_away_win}

        for sel in (fh, "Draw", fa):
            d = price[sel]; pm = model[sel]
            ev_ = pm * d - 1
            kelly = (pm * d - 1) / (d - 1) if d > 1 else 0
            rows.append({
                "match": f"{fh} v {fa}", "venue": f.city, "selection": sel,
                "FD": american(d), "dec": round(d, 2),
                "model%": round(pm * 100, 1), "fair%": round(fair[sel] * 100, 1),
                "EV%": round(ev_ * 100, 1), "kelly%": round(max(kelly, 0) * 100, 1),
            })
    if skipped:
        print("skipped (not a recognised upcoming WC2026 fixture, or no odds):")
        for s in skipped:
            print("   -", s)

    if not rows:
        print(f"no {args.book} h2h odds matched. Books present:",
              sorted({b['key'] for ev in data for b in ev.get('bookmakers', [])}))
        return
    df = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUTS, "odds_edges.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== all selections ({args.book}) ===")
    print(df.to_string(index=False))
    val = df[df["EV%"] > args.min_ev * 100].sort_values("EV%", ascending=False)
    print(f"\n=== +EV bets (model EV% > {args.min_ev*100:.0f}%) ===")
    print(val.to_string(index=False) if len(val) else "none")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

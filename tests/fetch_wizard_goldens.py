"""Fetch golden EV values from the Wizard of Odds hand-calculator backend.

One-shot, throttled tool — run manually to (re)generate the GOLDENS constant
in test_ev_engine.py. Never call this from the app itself: the endpoint is
undocumented, ~0.8 s per request, and has no SLA (see FEATURES.md Appendix B).

Usage:  .venv\\Scripts\\python tests\\fetch_wizard_goldens.py
Prints a JSON list of {case, params, response} objects to stdout.
"""

import json
import sys
import time

import requests

URL = "https://wizardofodds.com/calculators-js/blackjack/calculate/"
HEADERS = {"User-Agent": "Mozilla/5.0 (golden-value test oracle; one-shot)"}

# comp indices here follow the app's convention: [A, 2, 3, 4, 5, 6, 7, 8, 9, T]
FULL_8 = [32, 32, 32, 32, 32, 32, 32, 32, 32, 128]


def comp_minus(comp, *ranks):
    """Remove cards (rank strings 'A','2'..'9','T') from a composition."""
    idx = {"A": 0, "T": 9, **{str(v): v - 1 for v in range(2, 10)}}
    out = list(comp)
    for r in ranks:
        out[idx[r]] -= 1
    return out


def params_for(comp, upcard, hand, peek=0, h17=0, das=1, splits=1, surrender=0):
    """comp = unseen composition AFTER the deal (k=1), app index order."""
    a_to_j = comp[1:9] + [comp[9], comp[0]]  # 2..9, T, A
    p = {chr(ord("a") + i): v for i, v in enumerate(a_to_j)}
    p.update({"k": 1, "l": 1.5, "m": peek, "n": h17, "o": 0,
              "p": splits, "q": splits, "r": 0, "s": das, "t": surrender,
              "u": upcard, "v": hand})
    return p


CASES = [
    # (name, comp(after deal), upcard, hand, extra-kwargs)
    ("16vT_full_enhc", comp_minus(FULL_8, "T", "6", "T"), "T", "T6", {}),
    ("16vT_full_peek", comp_minus(FULL_8, "T", "6", "T"), "T", "T6", {"peek": 1}),
    ("16vT_full_h17", comp_minus(FULL_8, "T", "6", "T"), "T", "T6", {"h17": 1}),
    ("A7v9_full_enhc", comp_minus(FULL_8, "A", "7", "9"), "9", "A7", {}),
    ("11vA_full_enhc", comp_minus(FULL_8, "6", "5", "A"), "A", "65", {}),
    ("11vA_full_peek", comp_minus(FULL_8, "6", "5", "A"), "A", "65", {"peek": 1}),
    ("88vT_full_enhc", comp_minus(FULL_8, "8", "8", "T"), "T", "88", {}),
    ("TTvA_full_enhc", comp_minus(FULL_8, "T", "T", "A"), "A", "TT", {}),
    ("9v6_full_enhc", comp_minus(FULL_8, "5", "4", "6"), "6", "54", {}),
    # Depleted, ten-rich shoe (mid-shoe): 16 vs T should flip toward Stand.
    ("16vT_rich_enhc",
     comp_minus([14, 8, 8, 8, 8, 8, 12, 12, 12, 80], "T", "6", "T"), "T", "T6", {}),
    ("16vT_rich_peek",
     comp_minus([14, 8, 8, 8, 8, 8, 12, 12, 12, 80], "T", "6", "T"), "T", "T6", {"peek": 1}),
    # Low-card-rich shoe: 12 vs 4 should flip toward Hit.
    ("12v4_poor_enhc",
     comp_minus([10, 26, 26, 26, 24, 24, 12, 12, 12, 60], "T", "2", "4"), "4", "T2", {}),
]


def main():
    out = []
    for name, comp, upcard, hand, kw in CASES:
        params = params_for(comp, upcard, hand, **kw)
        r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        body = r.json()
        out.append({"case": name, "comp": comp, "upcard": upcard, "hand": hand,
                    "kw": kw, "response": body})
        print(f"  {name}: {body}", file=sys.stderr)
        time.sleep(1.5)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()

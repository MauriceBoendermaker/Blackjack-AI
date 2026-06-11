"""Practice & replay trainer logic (V2 Feature 7). Pure — UI lives in
lib/interfaces/trainer_window.py.

Drills follow what the reference tools (Blackjack Apprenticeship, Casino
Vérité) converge on — deck countdown (benchmark: one deck < 30 s, perfect,
five times running), deviation flashcards, full-hand decisions — plus the
two things they can't do: flashcards generated from YOUR rules profile, and
replays of YOUR recorded rounds graded in EV lost per error by the exact
engine.
"""

import random

from ..common import constants
from . import cards, deviations, ev_engine, ev_offload
from .ev_engine import ACE, TEN

# Concrete two-card hands for the deviation table's abstract keys.
CARDS_FOR_KEY = {
    "16": ["10 of Hearts", "6 of Clubs"],
    "15": ["10 of Hearts", "5 of Clubs"],
    "14": ["10 of Hearts", "4 of Clubs"],
    "13": ["10 of Hearts", "3 of Clubs"],
    "12": ["10 of Hearts", "2 of Clubs"],
    "11": ["6 of Hearts", "5 of Clubs"],
    "10": ["6 of Hearts", "4 of Clubs"],
    "9": ["5 of Hearts", "4 of Clubs"],
    "10,10": ["10 of Hearts", "King of Clubs"],
}

ACTION_NAMES = {"S": "Stand", "H": "Hit", "D": "Double", "P": "Split",
                "R": "Surrender"}


# ------------------------------------------------------------- countdown

def fresh_deck(rng=None) -> list:
    deck = cards.all_card_names()
    (rng or random).shuffle(deck)
    return deck


def running_count(card_names) -> int:
    return sum(cards.hilo_delta(c) for c in card_names)


def grade_countdown(card_names, answer) -> dict:
    correct = running_count(card_names)
    return {"correct_count": correct, "answer": answer,
            "right": answer == correct}


# ------------------------------------------------------------ flashcards

def deviation_items(s17=None, surrender=None) -> list:
    """Every index play of the ACTIVE rules profile as a flashcard source."""
    if s17 is None:
        s17 = constants.RULES["s17"]
    if surrender is None:
        surrender = constants.RULES["surrender"]
    table = deviations.I18_S17 if s17 else deviations.I18_H17
    items = [{"hand": h, "dealer": d, "index": idx, "above": above,
              "below": below, "source": "I18"}
             for (h, d), (above, idx, below) in table.items()]
    if surrender:
        fab = deviations.FAB4_S17 if s17 else deviations.FAB4_H17
        items += [{"hand": h, "dealer": d, "index": idx, "above": "R",
                   "below": "H", "source": "Fab4"}
                  for (h, d), idx in fab.items()]
    return items


def make_flashcard(rng=None, items=None) -> dict:
    """One quiz card: a deviation hand at a true count just above or below
    its index — the player must know which side they're on."""
    rng = rng or random
    items = items or deviation_items()
    item = rng.choice(items)
    offset = rng.choice([-2.0, -1.0, 1.0, 2.0])
    tc = item["index"] + offset
    correct = item["above"] if tc >= item["index"] else item["below"]
    return {
        **item, "tc": tc, "correct": correct,
        "question": (f"{item['hand']} vs {item['dealer']}  ·  TC {tc:+g}"),
        "rule": (f"{item['source']}: {ACTION_NAMES[item['above']]} at "
                 f"TC ≥ {item['index']:+d}, else {ACTION_NAMES[item['below']]}"),
    }


def grade_flashcard(card, answer_code) -> bool:
    return answer_code == card["correct"]


def _comp_for_tc(target_tc, removed_indices, deck_count=8):
    """A realistic composition near a target TC (2-9 dealt out for positive
    counts, tens+aces 4:1 for negative) — same construction the deviation
    cross-check tests use."""
    hilo = {ACE: -1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, TEN: -1}
    comp = list(ev_engine.full_shoe(deck_count))
    rc = 0
    for idx in removed_indices:
        comp[idx] -= 1
        rc += hilo[idx]
    cycle = [1, 2, 3, 4, 5, 6, 7, 8] if target_tc > 0 else [TEN, TEN, TEN, TEN, ACE]
    i = 0
    while ((rc / (sum(comp) / 52.0)) < target_tc) if target_tc > 0 \
            else ((rc / (sum(comp) / 52.0)) > target_tc):
        idx = cycle[i % len(cycle)]
        i += 1
        if comp[idx] <= 0:
            if all(comp[j] <= 0 for j in cycle):
                break
            continue
        comp[idx] -= 1
        rc += hilo[idx]
    return tuple(comp)


def ev_cost(card, answer_code) -> float | None:
    """How much EV a wrong flashcard answer gives up, in units of bet —
    graded by the exact engine on a shoe built at the card's true count.
    None when the chosen action isn't comparable (e.g. surrender unoffered)."""
    hand_names = CARDS_FOR_KEY.get(card["hand"])
    if hand_names is None:
        return None
    hand = tuple(sorted(ev_engine.card_index(c) for c in hand_names))
    dealer = ev_engine.card_index(card["dealer"])
    comp = _comp_for_tc(card["tc"], list(hand) + [dealer])
    # peek=True is intentional (the index tables are peek-game derived), but
    # the dealer rule must match the profile that generated the flashcard —
    # H17 indices graded under S17 would contradict the verdict.
    rules = ev_engine.Rules(s17=constants.RULES["s17"], peek=True,
                            surrender=card["source"] == "Fab4")
    # Same worker process as live seat advice: the exact evaluation is pure
    # Python and would stutter the Tk thread if run on a thread here.
    result = ev_offload.run("advice", ev_engine.evaluate, hand, dealer, comp,
                            rules)
    evs = result["evs"]
    if answer_code not in evs:
        return None
    return max(0.0, evs[result["best"]] - evs[answer_code])


# ---------------------------------------------------------------- replay

def replay_items(store, limit=300) -> list:
    """Decision quizzes from your own recorded rounds: the seat's first two
    cards vs the dealer up-card at the count that was live at the time."""
    if store is None:
        return []
    try:
        rounds = store.sample_rounds(limit)
    except Exception:
        return []
    items = []
    for rec in rounds:
        dealer = rec.get("dealer")
        if not dealer:
            continue
        for seat in rec.get("seats", []):
            all_cards = [c for c in seat.get("cards", []) if c and c != "-"]
            hand = all_cards[:2]
            if len(hand) < 2 or seat.get("split"):
                continue
            if cards.hand_value(hand) >= 21:
                continue
            # The stored optimal line describes the FINAL hand — only show it
            # when the quizzed two cards were the whole hand.
            optimal = seat.get("optimal", "") if len(all_cards) == 2 else ""
            items.append({"cards": hand, "dealer": dealer,
                          "tc": rec.get("true_count", 0.0),
                          "optimal_text": optimal})
    return items


def grade_replay(item, answer_code, advisor) -> dict:
    """Grade the recorded decision WITH the recorded count: the book line
    plus any index deviation active at that true count (R rows fall back —
    no surrender online)."""
    action, _, _ = advisor.advice(item["cards"], item["dealer"])
    if not action:
        return {"right": None, "book": None}
    parts = action.split("/")
    book = parts[0]
    if book == "R":
        book = parts[1] if len(parts) > 1 else "S"
    correct = book
    dealer = cards.dealer_strategy_rank(item["dealer"])
    if dealer:
        dev = deviations.index_advice(cards.hand_key(item["cards"]), dealer,
                                      item.get("tc", 0.0), two_cards=True)
        if dev is not None:
            correct = dev["action"]
    return {"right": answer_code == correct, "book": correct,
            "book_name": ACTION_NAMES.get(correct, correct)}

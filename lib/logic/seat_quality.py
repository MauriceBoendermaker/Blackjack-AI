"""Seat play-quality scoring for Bet Behind (V2 Feature 8).

Detection never sees which button another player pressed, but at round end
the cards tell the story: replay the basic-strategy book over the hand's
actual draw sequence and check whether the number of cards taken matches.
Standing on 16 vs 10 (two cards where the book keeps hitting) or hitting a
hard 14 vs 6 (extra cards where the book stands) both show up as a count
mismatch.

Known blind spots, scored charitably or skipped:
  * a double is indistinguishable from a single hit (both = one extra card);
  * an unsplit pair the book says to split counts as not-book;
  * hands whose book row is missing return None and don't count.

Scores are session-scoped on purpose — seats change players online, so
yesterday's discipline says nothing about who sits there now.
"""

from . import cards


def follows_book(card_names, dealer_rank, advisor, post_split=False):
    """True/False when the hand's draw count matches book play; None when it
    can't be judged (too few cards, unknown dealer, missing CSV row)."""
    hand = [c for c in card_names if c and c != "-"]
    if len(hand) < 2 or not dealer_rank:
        return None
    current = hand[:2]
    taken = 2
    while True:
        total = cards.hand_value(current)
        if total >= 21:
            # Busted while drawing, natural, or made 21 — no decision left;
            # book-consistent iff no cards beyond this point.
            return taken == len(hand)
        action, _, _ = advisor.advice(current, dealer_rank, post_split=post_split)
        if action is None:
            return None
        parts = action.split("/")
        primary = parts[0]
        if primary == "R":  # surrender unavailable online -> the fallback play
            primary = parts[1] if len(parts) > 1 else "S"
        if primary == "P":
            # Book says split and this seat didn't (split seats are scored
            # per hand with post_split=True and never reach a P row).
            return False
        if primary == "D":
            if len(current) > 2:
                primary = parts[1] if len(parts) > 1 else "H"
            else:
                # Proper double spot: book play = exactly one more card.
                # (One card could also be a plain hit — scored charitably.)
                return len(hand) == 3
        if primary == "S":
            return taken == len(hand)
        # primary == "H"
        if taken >= len(hand):
            return False  # stood where the book keeps hitting
        current = current + [hand[taken]]
        taken += 1


def score_settled_round(settle, seats_snap, dealer_rank, advisor):
    """Verdicts for every hand of a settled round.

    `settle` is settlement.settle_round() output, `seats_snap` the snapshot
    seat dicts (for the split flag). Returns {seat_index: [bool, ...]} with
    un-judgeable hands omitted."""
    by_index = {s["index"]: s for s in seats_snap}
    out = {}
    for seat_entry in settle.get("seats", []):
        idx = seat_entry["index"]
        split = bool(by_index.get(idx, {}).get("split"))
        verdicts = []
        for hand in seat_entry.get("hands", []):
            verdict = follows_book(hand["cards"], dealer_rank, advisor,
                                   post_split=split)
            if verdict is not None:
                verdicts.append(verdict)
        if verdicts:
            out[idx] = verdicts
    return out

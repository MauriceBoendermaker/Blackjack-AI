"""Exact composition-dependent EV engine — the "Optimal" advice line.

Computes exact expected values for Stand / Hit / Double / Split / Surrender /
Insurance conditioned on the actual remaining-shoe composition (the cards not
yet seen this shoe), instead of the fixed-shoe averages behind the basic
strategy CSV. Pure math, no Tkinter, no I/O.

Composition model: a 10-tuple of unseen-card counts indexed by rank value - 1
(index 0 = Ace, 1..8 = ranks 2..9, 9 = the ten bucket 10/J/Q/K). Suits and the
J/Q/K identity are irrelevant to main-game EV, so CardCounter's per-rank totals
are a sufficient statistic. The composition must already exclude every seen
card — including the hand being evaluated and the dealer up-card — which is
exactly what `CardCounter.per_rank` tracks.

Dealer-blackjack handling: all player EVs are first computed conditioned on
the dealer NOT having blackjack (the hole card can't complete a natural, and
player draw probabilities carry the matching Bayes correction). For peek games
that conditional EV is the answer — the decision only arises after the peek.
For no-hole-card games (ENHC) the unconditional EV mixes the conditional value
with the dealer-blackjack branch by the law of total expectation:
    EV = P(BJ) * loss_on_bj(action) + (1 - P(BJ)) * EV_no_bj(action)
which is exact because every in-tree player choice (stand/hit) loses the same
1 unit to a dealer natural.

Splits use the split-once model (two hands, no resplit — the Evolution rule)
with the standard independence approximation between the two hands; split aces
receive one card unless the rules allow hitting them.

Validated against the Wizard of Odds hand-calculator backend (see
tests/test_ev_engine.py — golden values fetched by tests/fetch_wizard_goldens.py).
"""

import threading
from dataclasses import dataclass
from functools import lru_cache

from ..common import constants

ACE, TEN = 0, 9  # composition indices; value of index i is i + 1 (ace low)

#: Final-total slots returned by the dealer recursion.
_OUT_17, _OUT_18, _OUT_19, _OUT_20, _OUT_21, _OUT_BUST = range(6)


@dataclass(frozen=True)
class Rules:
    s17: bool = True              # dealer stands on all 17s
    peek: bool = False            # US hole-card peek; False = no hole card (ENHC)
    dealer_bj_takes: str = "all"  # ENHC: dealer BJ takes "all" bets or "obo" (original bet only)
    das: bool = True              # double after split
    double_on: str = "any"        # "any" | "9-11" | "10-11" (hard totals)
    hit_split_aces: bool = False
    surrender: bool = False       # late surrender offered
    bj_pays: float = 1.5          # 3:2; 6:5 tables use 1.2


DEFAULT_RULES = Rules(**constants.RULES)


def current_rules() -> Rules:
    """Rules as configured right now (constants.RULES is runtime-mutable via
    the settings dialog). Rules is a frozen dataclass hashed by value, so
    lru-cached evaluations key correctly across settings changes."""
    return Rules(**constants.RULES)


# ------------------------------------------------------------- composition

def full_shoe(deck_count: int = constants.DECK_COUNT) -> tuple:
    comp = [4 * deck_count] * 10
    comp[TEN] = 16 * deck_count
    return tuple(comp)


_PER_RANK_INDEX = {"Ace": ACE, "10": TEN, **{str(v): v - 1 for v in range(2, 10)}}


def comp_from_per_rank(per_rank: dict, deck_count: int = constants.DECK_COUNT) -> tuple:
    """Remaining-shoe composition from CardCounter.per_rank ('Ace', '2'..'10')."""
    comp = list(full_shoe(deck_count))
    for key, seen in per_rank.items():
        idx = _PER_RANK_INDEX[key]
        comp[idx] = max(0, comp[idx] - seen)
    return tuple(comp)


def card_index(card_name: str) -> int:
    """'Ace of Spades' / 'King' / '7' -> composition index."""
    rank = str(card_name).split(" ")[0]
    aliases = {"J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
    rank = aliases.get(rank, rank)
    if rank == "Ace":
        return ACE
    if rank in ("10", "Jack", "Queen", "King"):
        return TEN
    return int(rank) - 1


def _minus(comp: tuple, idx: int) -> tuple:
    return comp[:idx] + (comp[idx] - 1,) + comp[idx + 1:]


def _hand_add(total: int, soft: bool, idx: int):
    """Add card `idx` to a (total, soft) hand; soft = an ace counted as 11."""
    if idx == ACE and total + 11 <= 21:
        return total + 11, True
    t = total + (1 if idx == ACE else idx + 1)
    if t > 21 and soft:
        return t - 10, False
    return t, soft


def hand_state(indices) -> tuple:
    """(total, soft) for a sequence of composition indices."""
    total, soft = 0, False
    for idx in indices:
        total, soft = _hand_add(total, soft, idx)
    return total, soft


# ------------------------------------------------------------ dealer hand

# Manual memos instead of lru_cache: deep ace-up evaluations can touch a few
# hundred thousand dealer states, and LRU eviction at that size THRASHES
# (entries still needed by sibling subtrees get evicted and recomputed).
# A plain dict never evicts mid-recursion; boundary code clears it wholesale
# only between evaluations.
#
# The caches are THREAD-LOCAL (review-verified): three threads evaluate
# concurrently — the advice thread (seat advice + warm-up), the predeal
# thread (~15 s sweeps), and trainer EV grading — and a shared dict let one
# thread's boundary clear wipe another's in-flight working set (measured
# 2.8x slowdown on the sweep). Per-thread dicts restore the invariant
# exactly; Feature 9's cross-seat sharing survives because all seat advice
# runs on the single advice-pool thread.
_DEALER_CACHE_LIMIT = 600_000
_PLAYER_MEMO_LIMIT = 400_000
_TLS = threading.local()


def _thread_caches():
    """(dealer_cache, player_memo) for the calling thread."""
    caches = getattr(_TLS, "caches", None)
    if caches is None:
        caches = ({}, {})
        _TLS.caches = caches
    return caches


def clear_thread_caches():
    """Drop the calling thread's memos (tests/benchmarks)."""
    _TLS.caches = ({}, {})


def _dealer_final(comp: tuple, total: int, soft: bool, s17: bool) -> tuple:
    """P(final total = 17/18/19/20/21/bust) for a dealer who keeps drawing."""
    if total > 21:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    if total >= 18 or (total == 17 and (s17 or not soft)):
        out = [0.0] * 6
        out[total - 17] = 1.0
        return tuple(out)
    cache = _thread_caches()[0]
    key = (comp, total, soft, s17)
    cached = cache.get(key)
    if cached is not None:
        return cached
    n = sum(comp)
    if n == 0:  # degenerate; unreachable in practice
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    acc = [0.0] * 6
    for idx in range(10):
        c = comp[idx]
        if not c:
            continue
        t2, s2 = _hand_add(total, soft, idx)
        sub = _dealer_final(_minus(comp, idx), t2, s2, s17)
        p = c / n
        for i in range(6):
            acc[i] += p * sub[i]
    result = tuple(acc)
    cache[key] = result
    return result


@lru_cache(maxsize=50_000)
def _dealer_dist(comp: tuple, up_idx: int, excl_idx, s17: bool) -> tuple:
    """Dealer final-total distribution from the up-card, drawing the hole card
    (and all further cards) from `comp`. `excl_idx` excludes one rank as the
    hole card (no-blackjack conditioning); the dealer's later draws are
    unconditional."""
    total, soft = (11, True) if up_idx == ACE else (up_idx + 1, False)
    n = sum(comp)
    denom = n - (comp[excl_idx] if excl_idx is not None else 0)
    if denom <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    acc = [0.0] * 6
    for idx in range(10):
        if idx == excl_idx:
            continue
        c = comp[idx]
        if not c:
            continue
        t2, s2 = _hand_add(total, soft, idx)
        sub = _dealer_final(_minus(comp, idx), t2, s2, s17)
        p = c / denom
        for i in range(6):
            acc[i] += p * sub[i]
    return tuple(acc)


# ------------------------------------------------------------- player hand

class _Evaluator:
    """One evaluation context: fixed up-card and rules, varying composition.

    All EVs are conditioned on no dealer blackjack. With an A or T up-card the
    hole card excludes the completing rank, and player draw probabilities get
    the exact Bayes correction: drawing rank c shifts P(hole != completer), so
        P(draw c | no BJ) = P(draw c) * P(no BJ | c drawn) / P(no BJ).
    """

    def __init__(self, up_idx: int, rules: Rules):
        self.up = up_idx
        self.rules = rules
        self.excl = TEN if up_idx == ACE else (ACE if up_idx == TEN else None)
        self._memo_prefix = (up_idx, rules)

    def draw_probs(self, comp: tuple):
        n = sum(comp)
        excl = self.excl
        if excl is None or comp[excl] == 0 or comp[excl] >= n:
            # No conditioning possible/needed. comp[excl] >= n means only
            # BJ-completing cards remain — P(no BJ) = 0, so the conditional
            # branch carries zero weight; plain probabilities avoid a /0.
            return [(i, comp[i] / n) for i in range(10) if comp[i]]
        h = comp[excl]
        out = []
        for i in range(10):
            c = comp[i]
            if not c:
                continue
            h2 = h - (1 if i == excl else 0)
            ratio = (1 - h2 / (n - 1)) / (1 - h / n)
            out.append((i, c / n * ratio))
        return out

    def ev_stand(self, total: int, comp: tuple) -> float:
        dist = _dealer_dist(comp, self.up, self.excl, self.rules.s17)
        ev = dist[_OUT_BUST]
        for d in range(17, 22):
            p = dist[d - 17]
            if total > d:
                ev += p
            elif total < d:
                ev -= p
        return ev

    def ev_best(self, total: int, soft: bool, comp: tuple) -> float:
        """EV of optimal stand/hit play from this state (thread-shared memo)."""
        memo = _thread_caches()[1]
        key = (self._memo_prefix, total, soft, comp)
        cached = memo.get(key)
        if cached is not None:
            return cached
        ev = max(self.ev_stand(total, comp), self.ev_hit(total, soft, comp))
        memo[key] = ev
        return ev

    def ev_hit(self, total: int, soft: bool, comp: tuple) -> float:
        acc = 0.0
        for i, p in self.draw_probs(comp):
            t2, s2 = _hand_add(total, soft, i)
            acc += p * (-1.0 if t2 > 21 else self.ev_best(t2, s2, _minus(comp, i)))
        return acc

    def ev_double(self, total: int, soft: bool, comp: tuple) -> float:
        acc = 0.0
        for i, p in self.draw_probs(comp):
            t2, _ = _hand_add(total, soft, i)
            acc += p * (-2.0 if t2 > 21 else 2.0 * self.ev_stand(t2, _minus(comp, i)))
        return acc

    def _can_double(self, total: int, soft: bool) -> bool:
        rule = self.rules.double_on
        if rule == "any":
            return True
        lo = 9 if rule == "9-11" else 10
        return not soft and lo <= total <= 11

    def ev_split(self, pair_idx: int, comp: tuple) -> float:
        """Split once into two hands (the Evolution rule); the two hands are
        treated as independent draws from the same composition (standard
        one-card-removed approximation)."""
        start = (11, True) if pair_idx == ACE else (pair_idx + 1, False)
        one_card_only = pair_idx == ACE and not self.rules.hit_split_aces
        acc = 0.0
        for i, p in self.draw_probs(comp):
            c2 = _minus(comp, i)
            t2, s2 = _hand_add(start[0], start[1], i)
            if one_card_only:
                acc += p * self.ev_stand(t2, c2)
                continue
            best = self.ev_best(t2, s2, c2)
            if self.rules.das and self._can_double(t2, s2):
                best = max(best, self.ev_double(t2, s2, c2))
            acc += p * best
        return 2.0 * acc

    def p_bust_on_double_given_bj(self, total: int, soft: bool, comp: tuple) -> float:
        """P(the double draw busts | the hole card IS the completer)."""
        excl = self.excl
        n = sum(comp)
        h = comp[excl]
        if h == 0:
            return 0.0
        acc = 0.0
        for i in range(10):
            c = comp[i]
            if not c:
                continue
            h2 = h - (1 if i == excl else 0)
            p = (c / n) * (h2 / (n - 1)) / (h / n)  # P(draw i | hole = completer)
            t2, _ = _hand_add(total, soft, i)
            if t2 > 21:
                acc += p
        return acc


# ---------------------------------------------------------------- top level

def insurance_ev(comp: tuple) -> tuple:
    """(P(hole is a ten), insurance EV per unit of insurance bet).

    Matches the Wizard of Odds convention: insurance pays 2:1, so the EV per
    insured unit is 3p - 1 and the bet is +EV iff the unseen tens fraction
    exceeds 1/3."""
    n = sum(comp)
    if n == 0:
        return 0.0, -1.0
    p = comp[TEN] / n
    return p, 3.0 * p - 1.0


def insurance_advice(per_rank: dict, deck_count: int | None = None,
                     rules: Rules | None = None) -> dict:
    """The exact insurance / even-money call from the live shoe composition.

    Insurance is a solved decision: take iff the unseen tens fraction exceeds
    1/3 (regardless of the player's hand). Even money on a blackjack is the
    same bet in disguise — its edge over declining is 1 - bj_pays * (1 - p),
    which for 3:2 tables goes positive at exactly the same p > 1/3."""
    if deck_count is None:
        deck_count = constants.DECK_COUNT
    if rules is None:
        rules = current_rules()
    p, ev = insurance_ev(comp_from_per_rank(per_rank, deck_count))
    return {
        "take": ev > 0,
        "p_ten": p,
        "ev": ev,                                        # per unit of insurance bet
        "even_money_edge": 1.0 - rules.bj_pays * (1.0 - p),
    }


@lru_cache(maxsize=20_000)
def evaluate(hand: tuple, up_idx: int, comp: tuple, rules: Rules = DEFAULT_RULES,
             post_split: bool = False) -> dict:
    """Exact EVs for a hand (tuple of composition indices, any order) against
    `up_idx`, with `comp` the unseen composition (hand + up-card excluded).

    Returns {"evs": {code: ev}, "best": code, "p_dealer_bj": float} with codes
    S/H/D/P/R; D, P, R appear only when the action is available. EVs are in
    units of the initial bet and — for ENHC rules — include the
    dealer-blackjack branch. `post_split` hands can't resplit or surrender
    and may double only under DAS."""
    dealer_cache, player_memo = _thread_caches()
    if len(dealer_cache) > _DEALER_CACHE_LIMIT:
        dealer_cache.clear()  # boundary clear: never evicts mid-recursion
    if len(player_memo) > _PLAYER_MEMO_LIMIT:
        player_memo.clear()
    ev = _Evaluator(up_idx, rules)
    evs, p_bj = _action_evs(ev, hand, comp, rules, post_split)
    best = max(evs, key=evs.get)
    return {"evs": evs, "best": best, "p_dealer_bj": p_bj}


def _action_evs(ev: "_Evaluator", hand: tuple, comp: tuple, rules: Rules,
                post_split: bool = False):
    """(evs dict, p_dealer_bj) for a hand using a (possibly shared) evaluator.
    Sharing one _Evaluator across many hands of the same up-card lets their
    player-tree memos overlap — the key speedup for the pre-deal sweep."""
    total, soft = hand_state(hand)
    two_cards = len(hand) == 2

    evs = {
        "S": ev.ev_stand(total, comp),
        "H": ev.ev_hit(total, soft, comp),
    }
    if two_cards:
        if ev._can_double(total, soft) and (not post_split or rules.das):
            evs["D"] = ev.ev_double(total, soft, comp)
        if hand[0] == hand[1] and not post_split:
            evs["P"] = ev.ev_split(hand[0], comp)
        if rules.surrender and not post_split:
            evs["R"] = -0.5

    # ENHC: mix in the dealer-blackjack branch (peek games are already
    # conditioned on no blackjack — the decision happens after the peek).
    p_bj = 0.0
    if ev.excl is not None:
        n = sum(comp)
        p_unc = comp[ev.excl] / n if n else 0.0
        if not rules.peek and p_unc:
            p_bj = p_unc
            lose_all = rules.dealer_bj_takes == "all"
            for code in list(evs):
                if code == "S" or code == "H":
                    loss = -1.0
                elif code == "D":
                    if lose_all:
                        loss = -2.0
                    else:  # OBO: the doubled half is returned unless already bust
                        loss = -1.0 - ev.p_bust_on_double_given_bj(total, soft, comp)
                elif code == "P":
                    # OBO approximation: post-split doubles in the BJ branch
                    # are treated as returned.
                    loss = -2.0 if lose_all else -1.0
                else:  # R — late surrender still loses the full bet to a natural
                    loss = -1.0
                evs[code] = p_bj * loss + (1.0 - p_bj) * evs[code]
    return evs, p_bj


@lru_cache(maxsize=32)
def predeal_ev(comp: tuple, rules: Rules) -> float:
    """Exact EV of the NEXT round played optimally, per unit bet, from the
    current pre-deal composition (the dealer's cards are still in the shoe).

    Enumerates dealer up-card x the 55 unordered starting hands without
    replacement; each hand is solved by evaluate() (split-once model). For
    peek rules evaluate() is conditioned on no dealer blackjack, so the
    dealer-BJ branch (player loses 1, naturals push) is mixed back in here;
    ENHC results already include it. This is the honest replacement for the
    linear true-count edge model — it sees ten/ace density and shoe depth
    that a single scalar count cannot."""
    n = sum(comp)
    if n < 20:
        return 0.0
    dealer_cache, player_memo = _thread_caches()
    if len(dealer_cache) > _DEALER_CACHE_LIMIT:
        dealer_cache.clear()
    if len(player_memo) > _PLAYER_MEMO_LIMIT:
        player_memo.clear()
    total_ev = 0.0
    for u in range(10):
        if not comp[u]:
            continue
        p_u = comp[u] / n
        comp_u = _minus(comp, u)
        n1 = sum(comp_u)
        pair_div = n1 * (n1 - 1)
        completer = TEN if u == ACE else (ACE if u == TEN else None)
        # ONE evaluator per up-card: all 55 hands share its player-tree memo.
        ev = _Evaluator(u, rules)
        for c1 in range(10):
            if not comp_u[c1]:
                continue
            for c2 in range(c1, 10):
                if c1 == c2:
                    weight = comp_u[c1] * (comp_u[c1] - 1) / pair_div
                else:
                    if not comp_u[c2]:
                        continue
                    weight = 2.0 * comp_u[c1] * comp_u[c2] / pair_div
                if weight <= 0:
                    continue
                comp_rest = _minus(_minus(comp_u, c1), c2)
                n_rest = sum(comp_rest)
                p_bj = (comp_rest[completer] / n_rest
                        if completer is not None and n_rest else 0.0)
                total, _ = hand_state((c1, c2))
                if total == 21:  # natural: paid bj_pays unless dealer also has one
                    ev_hand = (1.0 - p_bj) * rules.bj_pays
                else:
                    evs, _ = _action_evs(ev, (c1, c2), comp_rest, rules)
                    ev_hand = max(evs.values())
                    if rules.peek and p_bj:
                        # _action_evs is conditional on no dealer BJ for peek.
                        ev_hand = p_bj * -1.0 + (1.0 - p_bj) * ev_hand
                total_ev += p_u * weight * ev_hand
    if len(dealer_cache) > _DEALER_CACHE_LIMIT:
        dealer_cache.clear()  # the sweep inflates it well past the limit
    if len(player_memo) > _PLAYER_MEMO_LIMIT:
        player_memo.clear()
    return total_ev


# ------------------------------------------------- explainability (V3 F8)

class _OutcomeEval:
    """Per-action P(win / push / lose) under the same EV-optimal policy the
    advice line recommends — the probabilities behind the EV number.

    Reuses the _Evaluator for every DECISION (so the policy is exactly the
    advised one, Bayes-corrected draws included) and propagates outcome
    triples through the same tree. Doubles inside the post-draw best play
    are folded into hit/stand for the outcome view (they change stake, not
    the outcome class); the split view reports the NET of the two
    approximately-independent hands."""

    def __init__(self, ev: _Evaluator):
        self.ev = ev
        self._memo = {}

    def stand(self, total: int, comp: tuple) -> tuple:
        dist = _dealer_dist(comp, self.ev.up, self.ev.excl, self.ev.rules.s17)
        w, p, l = dist[_OUT_BUST], 0.0, 0.0
        for d in range(17, 22):
            q = dist[d - 17]
            if total > d:
                w += q
            elif total < d:
                l += q
            else:
                p += q
        return (w, p, l)

    def best(self, total: int, soft: bool, comp: tuple) -> tuple:
        key = (total, soft, comp)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if self.ev.ev_stand(total, comp) >= self.ev.ev_hit(total, soft, comp):
            out = self.stand(total, comp)
        else:
            out = self.hit(total, soft, comp)
        self._memo[key] = out
        return out

    def hit(self, total: int, soft: bool, comp: tuple) -> tuple:
        w = p = l = 0.0
        for i, q in self.ev.draw_probs(comp):
            t2, s2 = _hand_add(total, soft, i)
            if t2 > 21:
                l += q
            else:
                sw, sp, sl = self.best(t2, s2, _minus(comp, i))
                w += q * sw
                p += q * sp
                l += q * sl
        return (w, p, l)

    def double(self, total: int, soft: bool, comp: tuple) -> tuple:
        w = p = l = 0.0
        for i, q in self.ev.draw_probs(comp):
            t2, _ = _hand_add(total, soft, i)
            if t2 > 21:
                l += q
            else:
                sw, sp, sl = self.stand(t2, _minus(comp, i))
                w += q * sw
                p += q * sp
                l += q * sl
        return (w, p, l)

    def split_net(self, pair_idx: int, comp: tuple) -> tuple:
        start = (11, True) if pair_idx == ACE else (pair_idx + 1, False)
        one_card = pair_idx == ACE and not self.ev.rules.hit_split_aces
        w = p = l = 0.0
        for i, q in self.ev.draw_probs(comp):
            c2 = _minus(comp, i)
            t2, s2 = _hand_add(start[0], start[1], i)
            sw, sp, sl = (self.stand(t2, c2) if one_card
                          else self.best(t2, s2, c2))
            w += q * sw
            p += q * sp
            l += q * sl
        # Net over the two (approx. independent, identically distributed)
        # hands: win = more wins than losses, push = balanced.
        return (w * w + 2 * w * p, p * p + 2 * w * l, l * l + 2 * l * p)


def _mix_bj_outcome(triple: tuple, p_bj: float) -> tuple:
    """ENHC: the dealer-blackjack branch turns every outcome into a loss."""
    w, p, l = triple
    return ((1 - p_bj) * w, (1 - p_bj) * p, (1 - p_bj) * l + p_bj)


def action_outcomes(hand: tuple, up_idx: int, comp: tuple,
                    rules: Rules = DEFAULT_RULES,
                    post_split: bool = False) -> dict:
    """{code: (p_win, p_push, p_lose)} for the available actions, matching
    evaluate()'s conditioning: peek games are conditional on no dealer
    blackjack; ENHC mixes the blackjack branch in (a loss for every
    action). Surrender 'loses' its half bet by definition."""
    total, soft = hand_state(hand)
    ev = _Evaluator(up_idx, rules)
    oc = _OutcomeEval(ev)
    out = {"S": oc.stand(total, comp), "H": oc.hit(total, soft, comp)}
    if len(hand) == 2:
        if ev._can_double(total, soft) and (not post_split or rules.das):
            out["D"] = oc.double(total, soft, comp)
        if hand[0] == hand[1] and not post_split:
            out["P"] = oc.split_net(hand[0], comp)
        if rules.surrender and not post_split:
            out["R"] = (0.0, 0.0, 1.0)
    if ev.excl is not None and not rules.peek:
        n = sum(comp)
        p_bj = comp[ev.excl] / n if n else 0.0
        if p_bj:
            out = {code: _mix_bj_outcome(t, p_bj) for code, t in out.items()}
    return out


def dealer_distribution(comp: tuple, up_idx: int,
                        rules: Rules = DEFAULT_RULES) -> dict:
    """Display-ready dealer final-total distribution from the up-card:
    {"17".."21", "bj", "bust"}. The 21 slot is multi-card 21s only; "bj"
    is the natural (pre-peek probability — shown for peek games too, since
    that's what the player faces when the up-card lands)."""
    excl = TEN if up_idx == ACE else (ACE if up_idx == TEN else None)
    n = sum(comp)
    p_bj = comp[excl] / n if excl is not None and n else 0.0
    cond = _dealer_dist(comp, up_idx, excl, rules.s17)
    out = {str(total): cond[total - 17] * (1.0 - p_bj)
           for total in range(17, 22)}
    out["bj"] = p_bj
    out["bust"] = cond[_OUT_BUST] * (1.0 - p_bj)
    return out


#: Rank labels for the composition-driver panel, by composition index.
RANK_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10/J/Q/K"]


def composition_drivers(comp: tuple, deck_count: int) -> list:
    """Per-rank live density vs the fresh-shoe baseline — the ranks whose
    depletion/excess moved the call. [{label, live, baseline, delta}]."""
    n = sum(comp)
    base = full_shoe(deck_count)
    bn = sum(base)
    out = []
    for i in range(10):
        live = comp[i] / n if n else 0.0
        baseline = base[i] / bn
        out.append({"label": RANK_LABELS[i], "live": live,
                    "baseline": baseline, "delta": live - baseline})
    return out


def inspect_hand(player_cards, dealer_rank, comp: tuple,
                 deck_count: int | None = None, rules: Rules | None = None,
                 post_split: bool = False):
    """Everything the 'why this play?' inspector shows, in one picklable
    job (V3 F8): the full EV dict, per-action outcome probabilities, the
    dealer final-total distribution, composition drivers, and the
    fresh-shoe verdict for the flip indicator. `comp` is the unseen
    composition EXCLUDING hand and up-card (sandbox callers edit it
    directly). None when no decision applies — same rules as advise()."""
    if deck_count is None:
        deck_count = constants.DECK_COUNT
    if rules is None:
        rules = current_rules()
    hand = [c for c in player_cards if c and c != "-"]
    if len(hand) < 2 or not dealer_rank:
        return None
    indices = tuple(sorted(card_index(c) for c in hand))
    total, _ = hand_state(indices)
    if total >= 21:
        return None
    if any(c < 0 for c in comp) or sum(comp) <= 1:
        return None
    up = card_index(dealer_rank)
    result = evaluate(indices, up, comp, rules, post_split)
    # Fresh-shoe counterfactual: same hand, baseline composition — when the
    # verdicts differ, the composition (not the totals) made the call.
    baseline = list(full_shoe(deck_count))
    for idx in indices + (up,):
        baseline[idx] -= 1
    baseline_result = (evaluate(indices, up, tuple(baseline), rules,
                                post_split)
                       if all(c >= 0 for c in baseline) else None)
    return {
        "evs": result["evs"],
        "best": result["best"],
        "p_dealer_bj": result["p_dealer_bj"],
        "outcomes": action_outcomes(indices, up, comp, rules, post_split),
        "dealer_dist": dealer_distribution(comp, up, rules),
        "drivers": composition_drivers(comp, deck_count),
        "baseline_best": baseline_result["best"] if baseline_result else None,
        "baseline_evs": baseline_result["evs"] if baseline_result else None,
        "hand_total": hand_state(indices),
        "comp": comp,
    }


def inspect_from_per_rank(player_cards, dealer_rank, per_rank: dict,
                          deck_count: int | None = None,
                          rules: Rules | None = None,
                          post_split: bool = False):
    """inspect_hand from CardCounter.per_rank seen-counts (the live path —
    the snapshot's count block carries per_rank)."""
    if deck_count is None:
        deck_count = constants.DECK_COUNT
    return inspect_hand(player_cards, dealer_rank,
                        comp_from_per_rank(per_rank, deck_count),
                        deck_count, rules, post_split)


def advise(player_cards, dealer_rank, per_rank: dict,
           deck_count: int | None = None, rules: Rules | None = None,
           post_split: bool = False):
    """Card names + CardCounter per-rank totals -> evaluate() result, or None
    when no decision applies (too few cards, bust/21, unknown dealer card)."""
    if deck_count is None:
        deck_count = constants.DECK_COUNT
    if rules is None:
        rules = current_rules()
    hand = [c for c in player_cards if c and c != "-"]
    if len(hand) < 2 or not dealer_rank:
        return None
    indices = tuple(sorted(card_index(c) for c in hand))
    total, _ = hand_state(indices)
    if total >= 21:
        return None
    comp = comp_from_per_rank(per_rank, deck_count)
    if any(c < 0 for c in comp) or sum(comp) <= 1:
        return None
    return evaluate(indices, card_index(dealer_rank), comp, rules, post_split)

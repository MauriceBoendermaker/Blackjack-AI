"""v3.1 UI overhaul: dealer-centred seat fan — the pure geometry, no Tk root.

Run:  .venv\\Scripts\\python -m unittest tests.test_table_layout -v
"""

import unittest

from lib.common import constants
from lib.interfaces.table_view import seat_positions

# (w, h) canvas sizes to verify: the default window, a 27" monitor at full
# scale, and just under the app's minimum window size.
SIZES = [(1500, 950), (2560, 1340), (1150, 700)]
N = 7
CARD_W, CARD_H = constants.CARD_RENDER_SIZE       # (60, 88) baseline
SCALED_CARD = (90, 132)                           # the same card at 150% DPI


def cases():
    """Every (w, h, card_w, card_h) combination under test."""
    for w, h in SIZES:
        yield w, h, CARD_W, CARD_H
    yield 2560, 1340, SCALED_CARD[0], SCALED_CARD[1]


class SeatFan(unittest.TestCase):
    def test_middle_seat_is_lowest(self):
        for w, h, cw, ch in cases():
            pos = seat_positions(w, h, N, cw, ch)
            middle_y = pos[N // 2][1]
            for i, (_, y) in enumerate(pos):
                if i != N // 2:
                    self.assertLess(y, middle_y, f"seat {i} at {w}x{h}")

    def test_symmetric_about_centre(self):
        for w, h, cw, ch in cases():
            pos = seat_positions(w, h, N, cw, ch)
            for i in range(N):
                x_i, y_i = pos[i]
                x_j, y_j = pos[N - 1 - i]
                self.assertAlmostEqual(y_i, y_j, delta=1.0, msg=f"y {i} at {w}x{h}")
                self.assertAlmostEqual(x_i + x_j, w, delta=1.0, msg=f"x {i} at {w}x{h}")

    def test_x_strictly_increasing(self):
        # Player 1 leftmost through Player 7 rightmost.
        for w, h, cw, ch in cases():
            xs = [x for x, _ in seat_positions(w, h, N, cw, ch)]
            for i in range(N - 1):
                self.assertLess(xs[i], xs[i + 1], f"seats {i},{i + 1} at {w}x{h}")

    def test_edge_seats_inside_margins(self):
        # Cards and the 118 px optimal-label wraplength must keep >= ~40 px
        # of clear space to the canvas sides.
        for w, h, cw, ch in cases():
            s = ch / float(CARD_H)
            half_extent = max(cw / 2.0, 118.0 * s / 2.0)
            margin = 40.0 * s
            for i, (x, _) in enumerate(seat_positions(w, h, N, cw, ch)):
                self.assertGreaterEqual(x - half_extent, margin, f"seat {i} at {w}x{h}")
                self.assertLessEqual(x + half_extent, w - margin, f"seat {i} at {w}x{h}")

    def test_no_seat_in_dealer_zone(self):
        # The dealer card sits at y=46 (scaled) with the insurance/playout
        # labels beside it; every seat's card top must clear that area.
        dealer_h = constants.DEALER_CARD_RENDER_SIZE[1]
        for w, h, cw, ch in cases():
            s = ch / float(CARD_H)
            dealer_bottom = (46 + dealer_h) * s
            for i, (_, y) in enumerate(seat_positions(w, h, N, cw, ch)):
                self.assertGreater(y, dealer_bottom + 10 * s, f"seat {i} at {w}x{h}")

    def test_middle_seat_label_stack_fits(self):
        # Cards plus the ~150 px total/advice/optimal/index/name stack must
        # stay on screen for the lowest (middle) seat.
        for w, h, cw, ch in cases():
            s = ch / float(CARD_H)
            y_mid = seat_positions(w, h, N, cw, ch)[N // 2][1]
            self.assertLessEqual(y_mid + ch + 150 * s, h, f"middle seat at {w}x{h}")


if __name__ == "__main__":
    unittest.main()

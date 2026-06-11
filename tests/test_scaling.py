"""v3.1 UI overhaul: per-monitor DPI scaling — the pure part, no Tk root.

Run:  .venv\\Scripts\\python -m unittest tests.test_scaling -v
"""

import unittest

from lib.interfaces import scaling


class ResolveScale(unittest.TestCase):
    def test_auto_from_dpi(self):
        self.assertEqual(scaling.resolve_scale(96, 0), 1.0)
        self.assertEqual(scaling.resolve_scale(120, 0), 1.25)
        self.assertEqual(scaling.resolve_scale(144, 0), 1.5)
        self.assertEqual(scaling.resolve_scale(192, 0), 2.0)

    def test_falsy_override_means_auto(self):
        for override in (0, None, 0.0, ""):
            self.assertEqual(scaling.resolve_scale(144, override), 1.5)

    def test_override_wins_over_dpi(self):
        self.assertEqual(scaling.resolve_scale(192, 100), 1.0)
        self.assertEqual(scaling.resolve_scale(96, 150), 1.5)
        self.assertEqual(scaling.resolve_scale(144, 125), 1.25)

    def test_override_clamped(self):
        self.assertEqual(scaling.resolve_scale(96, 50), 0.75)
        self.assertEqual(scaling.resolve_scale(96, 400), 3.0)

    def test_auto_dpi_clamped(self):
        self.assertEqual(scaling.resolve_scale(24, 0), 0.75)
        self.assertEqual(scaling.resolve_scale(960, 0), 3.0)


class PxAndSize(unittest.TestCase):
    """px/size read the module-level factor; pin it for deterministic tests."""

    def setUp(self):
        self._saved = scaling._scale

    def tearDown(self):
        scaling._scale = self._saved

    def test_px_rounds_to_int(self):
        scaling._scale = 1.25
        self.assertEqual(scaling.px(340), 425)
        self.assertEqual(scaling.px(8), 10)
        self.assertIsInstance(scaling.px(320), int)

    def test_px_identity_at_scale_one(self):
        scaling._scale = 1.0
        for n in (0, 1, 118, 1500):
            self.assertEqual(scaling.px(n), n)

    def test_size_scales_both_dimensions(self):
        scaling._scale = 1.5
        self.assertEqual(scaling.size((60, 88)), (90, 132))
        self.assertEqual(scaling.size((56, 82)), (84, 123))

    def test_size_is_hashable_tuple_of_ints(self):
        # The table view keys its image cache on (name, size) — the scaled
        # size must stay a hashable tuple of ints.
        scaling._scale = 2.0
        result = scaling.size((84, 123))
        self.assertEqual(result, (168, 246))
        self.assertEqual({("back", result): 1}[("back", result)], 1)

    def test_scale_reports_current_factor(self):
        scaling._scale = 1.75
        self.assertEqual(scaling.scale(), 1.75)


class OnChangeRegistry(unittest.TestCase):
    def setUp(self):
        self._saved = list(scaling._callbacks)
        scaling._callbacks.clear()

    def tearDown(self):
        scaling._callbacks[:] = self._saved

    def test_registered_callbacks_fire_in_order(self):
        calls = []
        scaling.on_change(lambda: calls.append("a"))
        scaling.on_change(lambda: calls.append("b"))
        scaling._fire_callbacks()
        self.assertEqual(calls, ["a", "b"])

    def test_one_failing_callback_does_not_block_the_rest(self):
        calls = []

        def boom():
            raise RuntimeError("listener bug")

        scaling.on_change(boom)
        scaling.on_change(lambda: calls.append("survivor"))
        scaling._fire_callbacks()  # must not raise
        self.assertEqual(calls, ["survivor"])


if __name__ == "__main__":
    unittest.main()

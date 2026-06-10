"""Item 9: keystroke-level numeric validation — the pure part, no Tk.

Run:  .venv\\Scripts\\python -m unittest tests.test_validation -v
"""

import unittest

from lib.interfaces.validation import valid_partial_number


class ValidPartialNumber(unittest.TestCase):
    def test_empty_allowed_while_typing(self):
        self.assertTrue(valid_partial_number(""))
        self.assertTrue(valid_partial_number("", integer=True))

    def test_plain_digits(self):
        for text in ("0", "5", "50", "5000", "10000000"):
            self.assertTrue(valid_partial_number(text), text)
            self.assertTrue(valid_partial_number(text, integer=True), text)

    def test_one_separator_anywhere(self):
        # Partial forms like "5." and bare "." must survive the keystroke;
        # the commit handler does the strict parse.
        for text in ("5.", ".5", "5.5", "5,", ",5", "5,5", ".", ","):
            self.assertTrue(valid_partial_number(text), text)

    def test_second_separator_rejected(self):
        for text in ("5..5", "5,,5", "1.2.3", "1,2.3", "1.2,3", "..", ",.", "5.5,"):
            self.assertFalse(valid_partial_number(text), text)

    def test_non_numeric_rejected(self):
        # No sign needed for bankroll/bet fields; spaces and units are noise.
        for text in ("abc", "5a", "a5", "-5", "+5", " 5", "5 ", "€5", "5e3"):
            self.assertFalse(valid_partial_number(text), text)

    def test_unicode_digit_lookalikes_rejected(self):
        # str.isdigit() would accept these; float() would not.
        for text in ("²", "٥", "5²"):
            self.assertFalse(valid_partial_number(text), text)

    def test_integer_mode_rejects_separators(self):
        for text in ("5.", "5,", "5.5", ".", ","):
            self.assertFalse(valid_partial_number(text, integer=True), text)


if __name__ == "__main__":
    unittest.main()

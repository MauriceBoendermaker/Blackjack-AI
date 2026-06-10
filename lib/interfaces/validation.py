"""Keystroke-level numeric validation for Entry/Spinbox widgets (item 9).

valid_partial_number() is a pure function — this module never imports
tkinter, so the logic stays unit-testable headless. attach_numeric_entry()
wires it to a live widget via Tk's validate='key' hook: permissive while
typing (partial input like "5." must survive the next keystroke), while the
caller's existing <Return>/<FocusOut> handlers keep the strict parse-and-
clamp on commit.
"""

_DIGITS = set("0123456789")
_SEPARATORS = set(",.")


def valid_partial_number(text, integer=False):
    """True when `text` could still become a non-negative number: digits
    plus at most one decimal separator (',' or '.', EU and US styles).
    Empty is allowed — clearing the field mid-edit is normal typing.
    `integer=True` rejects separators too (whole-number fields)."""
    if text == "":
        return True
    allowed = _DIGITS if integer else _DIGITS | _SEPARATORS
    if any(ch not in allowed for ch in text):
        return False
    return sum(text.count(s) for s in _SEPARATORS) <= 1


def attach_numeric_entry(entry, integer=False):
    """Restrict a tk.Entry/tk.Spinbox to partial numeric input per keystroke.

    The validatecommand must stay pure and synchronous (it runs on the Tk
    thread inside the keystroke), so it only filters syntax — bounds and
    parsing remain with the widget's commit handlers. Tk silently flips
    validate to 'none' if a programmatic textvariable set fails validation
    (e.g. a '1e+07'-style reset); the <FocusIn> binding re-arms it before
    the user can type again.
    """
    vcmd = (entry.register(lambda text: valid_partial_number(text, integer)),
            "%P")
    entry.configure(validate="key", validatecommand=vcmd)
    entry.bind("<FocusIn>", lambda e: entry.configure(validate="key"), add="+")

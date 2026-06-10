"""Blackjack AI — entry point.

Run with:  python main.py
"""

from lib.common import constants, settings
from lib.logic.log_manager import LogManager, install_redirectors


def main():
    # Per-monitor DPI awareness BEFORE any window exists, so screen-capture
    # coordinates, calibrated regions, and the overlay HUD line up on
    # scaled displays.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    # Logging first, so every later print is captured (stderr included).
    log_manager = LogManager()
    install_redirectors(log_manager)

    print(f"Starting {constants.TITLE}")
    if settings.load_and_apply():
        print("Loaded table profile from output/settings.json.")

    # Imported after logging is live; the GUI starts instantly because the
    # detection models are only initialized when detection is started.
    from lib.interfaces.modern_gui import ModernBlackjackGUI

    gui = ModernBlackjackGUI(log_manager)
    print("UI ready.")
    gui.mainloop()


if __name__ == "__main__":
    main()

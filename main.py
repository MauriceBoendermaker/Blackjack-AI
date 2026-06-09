"""Blackjack AI — entry point.

Run with:  python main.py
"""

from lib.common import constants
from lib.logic.log_manager import LogManager, install_redirectors


def main():
    # Logging first, so every later print is captured (stderr included).
    log_manager = LogManager()
    install_redirectors(log_manager)

    print(f"Starting {constants.TITLE}")

    # Imported after logging is live; the GUI starts instantly because the
    # detection models are only initialized when detection is started.
    from lib.interfaces.modern_gui import ModernBlackjackGUI

    gui = ModernBlackjackGUI(log_manager)
    print("UI ready.")
    gui.mainloop()


if __name__ == "__main__":
    main()

# TODO: Implement system to detect splitted hands
# TODO: Fix 2x Ace being counted as "12", instead of 2 (and thus split)
# TODO: Finetune accuracy for both models (adjust parameters)
# TODO: Refine player regions

# UI:
# TODO: Fix player count
# TODO: Count total per-card amount
# TODO: Implement advice per player according to default cheat sheet in
# TODO: Implement OCR for Saldo and Current bet

import threading
import sys

import lib.common.constants as constants
from lib.interfaces.modern_gui import ModernBlackjackGUI
from lib.logic.background import BackgroundProcessor
from lib.logic.log_manager import LogManager, PrintRedirector


def init_gui(log_manager):
    """Initialize the modern GUI"""
    gui = ModernBlackjackGUI(log_manager)
    return gui


def main():
    # Initialize logging system BEFORE any print statements
    log_manager = LogManager()
    sys.stdout = PrintRedirector(log_manager, sys.stdout)

    print(f"Starting {constants.TITLE}")
    print("🎰 Loading modern interface...")

    gui = init_gui(log_manager)
    print("✓ Modern UI loaded successfully!")

    gui.mainloop()


if __name__ == "__main__":
    main()

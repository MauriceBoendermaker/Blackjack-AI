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

import lib.common.constants as constants
from lib.interfaces.gui import GraphicalUserInterface
from lib.logic.background import BackgroundProcessor


def init_gui():
    gui = GraphicalUserInterface()
    return gui


def start_background_processing(gui):
    if not gui.background_processor:
        gui.background_processor = BackgroundProcessor(gui.update_ui_callback, gui)
        gui.background_processor.blackjack_logic.set_monitor(gui.monitor_utils.monitor)
    gui.background_processor.start()


def main():
    print(f"Starting {constants.TITLE}")
    gui = init_gui()

    # Start background processing in a separate thread after monitor is confirmed
    def start_when_ready():
        gui.start()
        detection_thread = threading.Thread(target=start_background_processing, args=(gui,), daemon=True)
        detection_thread.start()

    gui.start_button.config(command=lambda: [gui.confirm_monitor_selection(), start_when_ready()])

    # Start the main GUI loop
    gui.mainloop()


if __name__ == "__main__":
    main()

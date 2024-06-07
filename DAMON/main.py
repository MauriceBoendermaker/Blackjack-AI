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

    gui.confirm_button.config(command=lambda: [gui.confirm_monitor_selection(), start_when_ready()])

    # Start the main GUI loop
    gui.mainloop()


if __name__ == "__main__":
    main()

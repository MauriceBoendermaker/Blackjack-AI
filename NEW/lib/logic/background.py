import time
import queue
import threading

from ..common import constants
from .card_utils import get_card_utils
from .blackjack import BlackjackLogic

class BackgroundProcessor:
    def __init__(self, update_ui_callback, gui):
        self.gui = gui
        self.update_ui_callback = update_ui_callback
        self.update_queue = queue.Queue()
        self.blackjack_logic = BlackjackLogic(gui)
        self.card_utils = get_card_utils(gui)

    def start(self):
        threading.Thread(target=self.background_processing, daemon=True).start()
        self.gui.after(constants.QUEUE_POLL_MS, self.check_for_updates)

    def background_processing(self):
        while True:
            detection_state = self.blackjack_logic.capture_screen_and_track_cards()
            self.update_queue.put("update")

            # Adaptive sleep timing based on detection activity
            if detection_state == "active_dealing":
                sleep_time = 0.5  # Fast checks during card dealing
            elif detection_state == "round_complete":
                sleep_time = 1.0  # Medium checks when round is complete
            else:  # "waiting"
                sleep_time = 1.5  # Slower checks when waiting for new round

            time.sleep(sleep_time)

    def check_for_updates(self):
        while not self.update_queue.empty():
            data = self.update_queue.get_nowait()
            if data == "update":
                self.update_ui_callback()
        self.gui.after(constants.QUEUE_POLL_MS, self.check_for_updates)

    def update_gui_from_queue(self):
        try:
            while not self.update_queue.empty():
                data = self.update_queue.get_nowait()
                if data == "update":
                    self.update_ui_callback()
        finally:
            self.gui.after(constants.QUEUE_POLL_MS, self.update_gui_from_queue)

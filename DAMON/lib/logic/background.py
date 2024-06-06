import queue
import threading

from .blackjack import BlackjackLogic
from ..common import constants
from .card_utils import CardUtils


class BackgroundProcessor:
    def __init__(self, update_ui_callback, gui):
        self.update_ui_callback = update_ui_callback
        self.update_queue = queue.Queue()
        self.blackjack_logic = BlackjackLogic(gui)
        self.gui = gui
        self.card_utils = CardUtils()

    def start(self):
        threading.Thread(target=self.background_processing, daemon=True).start()

    def background_processing(self):
        while True:
            self.blackjack_logic.capture_screen_and_track_cards()
            self.update_ui_callback()

    def check_for_updates(self):
        if not self.update_queue.empty():
            data = self.update_queue.get()
            if 'dealer_card' in data:
                self.blackjack_logic.update_dealer_card_display(data['dealer_card'])
        self.gui.after(10, self.check_for_updates)

    def update_gui_from_queue(self):
        try:
            while not self.update_queue.empty():
                data = self.update_queue.get_nowait()
                if isinstance(data, dict):
                    if 'dealer_card' in data:
                        self.blackjack_logic.update_dealer_card_display(data['dealer_card'])
                    elif 'players_cards' in data:
                        self.blackjack_logic.update_player_cards_display(data['players_cards'],
                                                                         self.blackjack_logic.dealer_up_card,
                                                                         self.card_utils.true_count,
                                                                         constants.BASE_BET)
        finally:
            self.gui.after(10, self.update_gui_from_queue)

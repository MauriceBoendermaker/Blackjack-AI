import os

from roboflow import Roboflow
from ..common import constants


class Utils:
    def __init__(self):
        pass

    # Initialize prediction model for player cards
    def initialize_player_model(self):
        rf_players = Roboflow(api_key=constants.ROBOFLOW_API_KEY)
        project_players = rf_players.workspace().project(constants.PROJECT_ID_PLAYERS)
        return project_players.version(constants.MODEL_VERSION_PLAYERS).model

    # Initialize prediction model for dealer cards
    def initialize_dealer_model(self):
        rf_dealer = Roboflow(api_key=constants.ROBOFLOW_API_KEY)
        project_dealer = rf_dealer.workspace().project(constants.PROJECT_ID_DEALER)
        return project_dealer.version(constants.MODEL_VERSION_DEALER).model

    def generate_card_image_path(self, card):
        card = card.replace(" ", "_")
        return os.path.join(constants.CARD_FOLDER_PATH, f"{card}.png")

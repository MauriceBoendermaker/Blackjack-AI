import os
import cv2
import time
import numpy as np
from PIL import Image
from roboflow import Roboflow
from mss import mss
from screeninfo import get_monitors
from matplotlib.path import Path
from collections import defaultdict
from collections import OrderedDict
import tempfile

from class_mapping import class_mapping

# TODO: Convert player_regions to 1920x1080 resolution (now 2560x1440).
# Define player regions with numpy arrays or convert from your provided points
player_regions = [
    Path(np.array([[652, 1026], [760, 1126], [1112, 994], [996, 902], [652, 1026]])),
    Path(np.array([[776, 1130], [924, 1218], [1184, 1026], [1072, 962], [776, 1130]])),
    Path(np.array([[920, 1218], [1168, 1254], [1292, 1050], [1176, 1018], [920, 1218]])),
    Path(np.array([[1168, 1254], [1420, 1250], [1412, 1042], [1276, 1046], [1168, 1254]])),
    Path(np.array([[1412, 1246], [1616, 1202], [1524, 1010], [1392, 1042], [1412, 1246]])),
    Path(np.array([[1616, 1202], [1796, 1130], [1616, 958], [1512, 1014], [1616, 1202]])),
    Path(np.array([[1796, 1130], [1924, 1018], [1696, 894], [1608, 962], [1796, 1130]]))
]

# Initialize Roboflow
api_key = "WBy7jG6AiiqjzifOfiNH"  # Replace with your actual API key
project_id = "dey022"  # Replace with your actual project ID
rf = Roboflow(api_key=api_key)
project = rf.workspace().project(project_id)
model = project.version(1).model


# Function to get card name from class label
def get_card_name(class_label):
    return class_mapping.get(class_label, "Unknown")


# Initialize a dictionary to store the locked cards and their highest confidence for each player
# locked_cards = defaultdict(lambda: {"cards": [], "max_confidence": 0.0})

# Initialize a dictionary to store the guessed cards for each player
guessed_cards = {}


# Function to print the highest confident card for each player
# def print_highest_confidence_cards():
#     for player, data in highest_confidence_cards.items():
#         if data["card"]:  # Check if a card is stored for this player
#             print(f"Player {player + 1}: {data['card']} (Confidence: {data['confidence']:.2f})")


# Continuously capture screen and perform object detection
def capture_screen_and_track_cards(monitor_number=0):
    global cards_printed_count  # Declare cards_printed_count as global

    while True:
        # Capture screen
        monitor_info = get_monitors()[monitor_number]
        monitor = {
            "left": monitor_info.x,
            "top": monitor_info.y,
            "width": monitor_info.width,
            "height": monitor_info.height
        }
        sct = mss()
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            img.save(temp_file_path)

        # Predict using object detection model
        predictions = model.predict(temp_file_path, confidence=40, overlap=100).json()['predictions']

        # Process predictions
        for prediction in predictions:
            x, y = prediction['x'], prediction['y']
            class_label = prediction['class']
            confidence = prediction['confidence']  # Extract confidence level
            card_name = get_card_name(class_label)

            # Check if the detected object is within any player region
            for i, region in enumerate(player_regions):
                if region.contains_point([x, y]):
                    if i not in guessed_cards:
                        # If no card detected yet for this player, store the first one
                        guessed_cards[i] = {"cards": [card_name], "confidences": [confidence]}
                    elif len(guessed_cards[i]['cards']) == 1:
                        # If the first card has been detected for this player
                        guessed_cards[i]['cards'].append(card_name)
                        guessed_cards[i]['confidences'].append(confidence)
                    elif len(guessed_cards[i]['cards']) == 2:
                        # If the second card has been detected for this player
                        # Print the guessed cards and their confidences
                        print_player_cards(i)
                        # Reset the guessed cards for this player
                        guessed_cards[i] = {"cards": [], "confidences": []}
                    break

        # Wait for 2 seconds before the next detection
        # time.sleep(2)

        # Delete temporary file
        os.unlink(temp_file_path)


def print_player_cards(player_index):
    cards = guessed_cards[player_index]['cards']
    confidences = guessed_cards[player_index]['confidences']
    card_info = ' // '.join([f"{card} (C: {confidence * 100:.2f}%)" for card, confidence in zip(cards, confidences)])
    print(f"Player {player_index + 1}: {card_info}")


# Main function to start tracking cards
if __name__ == "__main__":
    capture_screen_and_track_cards()

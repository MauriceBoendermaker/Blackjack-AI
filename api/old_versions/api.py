import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
from mss import mss
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from matplotlib.path import Path

from class_mapping import class_mapping

# Initialize Roboflow
rf = Roboflow(api_key="WBy7jG6AiiqjzifOfiNH")
project = rf.workspace().project("dey022")
model = project.version(1).model

# Dictionary to store card ownership
card_ownership = defaultdict(list)

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


# Function to predict on image
def predict(image):
    # Perform prediction
    prediction = model.predict(image, confidence=40, overlap=30).json()

    # Extract class labels and map them to their corresponding names
    class_labels = [class_mapping.get(item["class"], "Unknown") for item in prediction["predictions"]]
    print("Class labels:", class_labels)

    return prediction


# Function to calculate Euclidean distance between two points
# def euclidean_distance(p1, p2):
#     return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to check if two bounding boxes overlap
def check_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False


# Function to detect stacked cards
def detect_stacked_cards(predictions):
    stacked_cards = []
    for i, card1 in enumerate(predictions):
        for j, card2 in enumerate(predictions):
            if i != j:
                bbox1 = card1.get("bbox", (0, 0, 0, 0))
                bbox2 = card2.get("bbox", (0, 0, 0, 0))
                if check_overlap(bbox1, bbox2):
                    stacked_cards.append((i, j))
    return stacked_cards


# Updated Function to associate cards with players, handling polygon regions
def associate_cards_with_players(predictions, player_regions, card_ownership):
    print("Starting card association...")
    for card in predictions:
        bbox = card.get("bbox", (0, 0, 0, 0))
        centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        print(f"Card centroid: {centroid}")
        found_region = False
        for player_idx, region in enumerate(player_regions):
            if region.contains_point(centroid):
                found_region = True
                card_label = class_mapping.get(card['class'], "Unknown")
                card_ownership[player_idx].append(card_label)
                print(f"Card '{card_label}' associated with Player {player_idx}")
                break
        if not found_region:
            print("Card centroid:", centroid)
            print("Card not associated with any player.")

    # After trying to associate all cards, print out a summary
    if all(len(cards) == 0 for cards in card_ownership.values()):
        print("No cards were associated with any players. Please check the defined player regions and card centroids.")


# Updated Function to associate cards with players, handling stacked cards
# def associate_cards_with_players(predictions, player_regions, card_ownership):
#     for card in predictions:
#         bbox = card.get("bbox", (0, 0, 0, 0))
#         centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
#         min_distance = float('inf')
#         assigned_player = None
#         for player, region in player_regions.items():
#             distance = euclidean_distance(centroid, region)
#             if distance < min_distance:
#                 min_distance = distance
#                 assigned_player = player
#         if assigned_player is not None:
#             found = False
#             for existing_card in card_ownership[assigned_player]:
#                 if 'stacked_cards' not in existing_card:
#                     existing_card['stacked_cards'] = []
#                 if check_overlap(existing_card['bbox'], bbox):
#                     existing_card['stacked_cards'].append(card)
#                     found = True
#                     break
#             if not found:
#                 card_ownership[assigned_player].append({'bbox': bbox, 'stacked_cards': [card]})
#         else:
#             print(f"Card at centroid {centroid} could not be assigned to a player.")


# Function to print the cards for each player
def print_player_cards(card_ownership):
    for player in range(8):  # Assuming there are 8 players
        cards = card_ownership.get(player, [])
        unique_cards = set()
        for card_info in cards:
            # Ensure 'stacked_cards' key exists
            stacked_cards = card_info.get('stacked_cards', [])
            for card in stacked_cards:
                card_label = class_mapping.get(card['class'], "Unknown")
                unique_cards.add(card_label)
        unique_card_list = sorted(list(unique_cards))
        if unique_card_list:
            print(f"Player {player}: {', '.join(unique_card_list)}")
        else:
            print(f"Player {player}: No cards")


def draw_prediction_and_players(image, predictions, player_regions, card_ownership):
    # Ensure coordinates are within image boundaries
    def clamp(val, min_val, max_val):
        return max(min_val, min(val, max_val))

    # Draw detected cards
    for item in predictions:
        bbox = item.get("bbox", [0, 0, 0, 0])
        class_id = item.get("class", "Unknown")
        label = class_mapping.get(class_id, "Unknown")

        # Ensure bbox coordinates are integers and within image boundaries
        x, y, w, h = map(int, bbox)
        x = clamp(x, 0, image.shape[1])
        y = clamp(y, 0, image.shape[0])
        w = clamp(w, 0, image.shape[1] - x)
        h = clamp(h, 0, image.shape[0] - y)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate position for the label
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y, label_size[1] + 10)  # Ensure label is within image

        # Draw label background
        cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y + base_line - 10), (0, 255, 0),
                      cv2.FILLED)
        # Draw label text
        cv2.putText(image, label, (x, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw player areas
    for region in player_regions:
        points = np.array(region.vertices, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Draw card ownership
    for player_idx, cards in card_ownership.items():
        for card_label in cards:
            # Define position for the card ownership label
            ownership_label = f"Player {player_idx}"
            text_size, _ = cv2.getTextSize(ownership_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = x + 10  # Adjust as needed
            text_y = y + 10  # Adjust as needed

            # Draw ownership label background
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y + 5), (255, 255, 255), cv2.FILLED)

            # Draw ownership label text
            cv2.putText(image, ownership_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


# Then, in your capture_screen function, save the image with everything drawn
def capture_screen(mon_num=0):
    monitor_info = get_monitors()[mon_num]
    monitor = {
        "left": monitor_info.x,
        "top": monitor_info.y,
        "width": monitor_info.width,
        "height": monitor_info.height
    }
    while True:
        screenshot = mss().grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Obtain predictions
        prediction = predict(img_np)
        predictions = prediction.get("predictions", [])

        # Draw predictions and player areas on the image
        img_with_everything = draw_prediction_and_players(img_np, predictions, player_regions, card_ownership)

        # Save the image to a file
        cv2.imwrite('detected_cards_and_players.png', img_with_everything)

        break  # Exiting the loop after saving the image


if __name__ == "__main__":
    capture_screen()

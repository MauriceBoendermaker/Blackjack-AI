import os
import cv2
import queue
import tkinter as tk

from PIL import Image, ImageTk
from api13 import reset_for_new_round

window = None
dealer_value_label = None
player_card_labels = []
player_advice_labels = []
canvas = None
round_label = None
reset_button = None
round_count = 0

CARD_WIDTH = 73
CARD_HEIGHT = 98
CARD_SPACING = 10
CARD_FOLDER = "cards"
DEFAULT_CARD_IMAGE_PATH = "cards/red card.jpg"

# Assuming this queue is accessible by both api12.py and display1.py
update_queue = queue.Queue()


def clear_player_cards():
    global player_card_labels
    for label in player_card_labels:
        label.destroy()  # This removes the label from the canvas
    player_card_labels.clear()  # Clear the list for the next round


def initialize_gui():
    global window, dealer_value_label, player_card_labels, player_advice_labels, canvas, round_label, reset_button, round_count
    window = tk.Tk()
    window.title("Blackjack AI")
    window.geometry("1000x700")

    canvas = tk.Canvas(window)
    canvas.pack(fill=tk.BOTH, expand=True)

    dealer_value_label = tk.Label(canvas, text="Dealer has: ")
    dealer_value_label.pack()

    round_label = tk.Label(window, text=f"Round: {round_count}")
    round_label.place(x=10, y=10)

    reset_button = tk.Button(window, text="Reset Round", command=reset_round)
    reset_button.pack()
    reset_button.place(x=100, y=10)


def reset_round():
    global player_card_labels
    canvas.delete("all")  # This clears everything from the canvas
    reset_for_new_round()  # Assuming this resets the backend data structure
    player_card_labels = []  # Reset the list holding references to the card images


def check_for_updates():
    global cards_displayed

    if not update_queue.empty():
        data = update_queue.get()
        if 'dealer_card' in data:
            update_dealer_card_display(data['dealer_card'])
    # Cancel any previously scheduled executions and schedule the function to run again
    window.after(10, check_for_updates)


def update_dealer_card_display(dealer_value):
    dealer_value_label.config(text=f"Dealer has: {dealer_value}")


def cv2_to_pil(image):
    # Convert an OpenCV image (BGR) to a PIL image (RGB)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def generate_card_image_path(card):
    card = card.replace(" ", "_")
    return os.path.join(CARD_FOLDER, f"{card}.png")


def update_player_cards_display(player_data_list):
    global player_card_labels

    clear_player_cards()  # Clear the displayed cards

    start_y = 10  # Fixed starting y-position for the cards

    # Calculate total width required for all players' cards
    total_width = 7 * (CARD_WIDTH + CARD_SPACING)  # 7 columns
    total_width += 8  # 1px border around each column
    column_width = (total_width - 8) // 7  # Width of each column without borders

    # Iterate over each player's data
    for i, player_data in enumerate(player_data_list):
        # Calculate starting x-position for the player's cards column
        start_x = i * (column_width + CARD_SPACING) + CARD_SPACING

        # Draw border around the column
        canvas.create_rectangle(start_x, start_y, start_x + column_width, start_y + window.winfo_height(),
                                outline="black", width=1)

        cards = player_data.get('cards', ['-'])  # Default card if no cards detected
        player_number = i + 1  # Player index (1-indexed)

        # Create label for player number and position it at the bottom of the column
        player_label = tk.Label(canvas, text=f"Player {player_number}")
        player_label.place(x=start_x + column_width // 2, y=window.winfo_height() - 10, anchor="s")

        for card in cards:  # Iterate over each card for the player
            card_image_path = generate_card_image_path(card)  # Generate file path for the card image

            if os.path.exists(card_image_path):  # Check if the card image exists, otherwise use default image
                card_image = cv2.imread(card_image_path)
            else:
                print(f"Card image not found: {card_image_path}. Using default image.")
                card_image = cv2.imread(DEFAULT_CARD_IMAGE_PATH)

            card_image_rgb = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB

            pil_image = Image.fromarray(card_image_rgb)  # Convert RGB image to PIL Image

            resized_image = pil_image.resize((CARD_WIDTH, CARD_HEIGHT))  # Resize the image with anti-aliasing

            photo_img = ImageTk.PhotoImage(resized_image)  # Convert PIL image to PhotoImage

            # Create image label and position it
            card_label = tk.Label(canvas, image=photo_img)
            card_label.image = photo_img  # Keep a reference to prevent garbage collection
            card_label.place(x=start_x, y=start_y)

            player_card_labels.append(card_label)  # Store the label in a list

            # Update starting y-position for the next card (stacked upwards)
            start_y += CARD_HEIGHT + 10

        start_y = 10  # Reset starting y-position for the next player's cards


def update_gui_from_queue():
    try:
        while not update_queue.empty():
            data = update_queue.get_nowait()  # Use get_nowait() to avoid blocking
            if isinstance(data, dict):
                if 'dealer_card' in data:
                    update_dealer_card_display(data['dealer_card'])
                elif 'players_cards' in data:
                    update_player_cards_display(data['players_cards'])
    finally:
        window.after(10, update_gui_from_queue)


def go_to_next_round():
    global round_count, player_card_labels, cards_displayed

    # Clear player card images
    for label in player_card_labels:
        label.destroy()  # Assuming tkinter labels for card images
    player_card_labels.clear()

    # Clear dealer card display
    dealer_value_label.config(text="Dealer has: ")

    # Increment the round count
    round_count += 1
    round_label.config(text=f"Round: {round_count}")

    # Update the round display
    if round_label is not None:
        round_label.config(text=f"Round: {round_count}")


def run_gui():
    window.mainloop()


# Setup the GUI and call check_for_updates() to start the loop
initialize_gui()  # Ensure this initializes your window and GUI components
window.after(10, check_for_updates())
run_gui()  # Starts the main loop

import tkinter as tk

import lib.common.constants as constants


class GraphicalUserInterface(tk.Tk):
    """
    Class for our graphical user interface for BlackJack AI
    """

    def __init__(self):
        super().__init__()
        self.title(constants.TITLE)
        self.geometry(constants.SIZE)

        self.canvas = tk.Canvas()
        self.draw_canvas()
        self.add_buttons()

    def draw_canvas(self):
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_rectangle(50, 50, 200, 100, fill="black", outline="white")
        self.canvas.create_text(125, 75, text="Dealer", fill="white")
        self.canvas.create_rectangle(50, 400, 200, 450, fill="black", outline="white")
        self.canvas.create_text(125, 425, text="Player", fill="white")

    def add_buttons(self):
        self.start_button = tk.Button(
            self, text="Start Game", command=self.test)
        self.canvas.create_window(300, 550, window=self.start_button)

        # dealer_value_label = tk.Label(canvas, text="Dealer has: ")
        # dealer_value_label.place(relx=1.0, rely=0.0, x=-50, y=0, anchor='ne')  # Adjusted for top-right with padding

        # round_label = tk.Label(window, text=f"Round: {round_count}")
        # round_label.place(x=10, y=5)

        # reset_button = tk.Button(window, text="Reset Round", command=reset_for_new_round)
        # reset_button.pack()
        # reset_button.place(x=100, y=5)

        # start_y = 750  # Start position from the bottom of an 800px tall window
        # label_height = 20  # Estimated height of each label
        # spacing = 5  # Spacing between labels

        # for card_value in card_values:  # No need to reverse since we're starting from the bottom now
        #     text = f"{card_value}: 0x"
        #     label = tk.Label(canvas, text=text, anchor='e')
        #     label.place(relx=0.95, y=start_y, anchor="se")  # Adjust relx and anchor if needed
        #     card_counter_labels.append(label)
        #     start_y -= (label_height + spacing)  # Move up for the next label

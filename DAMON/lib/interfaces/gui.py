import tkinter as tk

from ..common import constants
from ..player_boxes import generator as pbox_generator


class GraphicalUserInterface(tk.Tk):
    """
    Class for our graphical user interface for BlackJack AI
    """

    def __init__(self):
        super().__init__()
        self.title(constants.TITLE)
        self.geometry(constants.SIZE)
        self.pbox_generator = pbox_generator.PlayerBoxGenerator()

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
        self.pbox_gen_button = tk.Button(
            self, text="Generate Player Boxes", command=self.pbox_generator.generate
        )

        self.canvas.create_window(300, 550, window=self.pbox_gen_button)

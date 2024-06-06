import screeninfo
import tkinter as tk
from tkinter import ttk, messagebox

from ..common import constants
from ..player_boxes import generator as pbox_generator
from ..logic.monitor_utils import MonitorUtils
from ..logic.background import BackgroundProcessor


class GraphicalUserInterface(tk.Tk):
    """
    Class for our graphical user interface for BlackJack AI
    """

    def __init__(self):
        super().__init__()
        self.title(constants.TITLE)
        self.geometry(constants.SIZE)
        self.resizable(False, False)  # Prevent window resizing

        self.round_label = None
        self.reset_button = None
        self.dealer_value_label = None

        self.pbox_generator = pbox_generator.PlayerBoxGenerator(self)
        self.monitor_utils = MonitorUtils()  # Initialize MonitorUtils instance

        self.monitor_var = tk.StringVar()
        self.monitor_selection_frame = tk.Frame(self)
        self.monitor_selection_frame.pack()

        self.monitor_label = tk.Label(self.monitor_selection_frame, text="Select Monitor:")
        self.monitor_label.pack(side=tk.LEFT)

        self.monitor_combo = ttk.Combobox(
            self.monitor_selection_frame, textvariable=self.monitor_var, state="readonly"
        )
        self.monitor_combo.pack(side=tk.LEFT)

        self.resolution_label = tk.Label(self.monitor_selection_frame, text="Resolution: ")
        self.resolution_label.pack(side=tk.LEFT)

        self.resolution_var = tk.StringVar()
        self.resolution_display = tk.Label(self.monitor_selection_frame, textvariable=self.resolution_var)
        self.resolution_display.pack(side=tk.LEFT)

        self.confirm_button = tk.Button(
            self.monitor_selection_frame, text="Confirm", command=self.confirm_monitor_selection
        )
        self.confirm_button.pack(side=tk.LEFT)

        self.pbox_gen_button = tk.Button(
            self, text="Generate Player Boxes", command=self.pbox_generator.generate, state=tk.DISABLED
        )
        self.pbox_gen_button.pack()

        self.start_button = tk.Button(
            self, text="Start", command=self.start
        )
        self.start_button.pack(pady=20)

        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.background_processor = None

        self.draw_canvas()
        self.populate_monitors()

        self.bind("<Configure>", self.on_resize)  # Bind the resize event

    def draw_canvas(self):
        self.canvas.create_rectangle(50, 50, 200, 100, fill="black", outline="white")
        self.canvas.create_text(125, 75, text="Dealer", fill="white")
        self.canvas.create_rectangle(50, 400, 200, 450, fill="black", outline="white")
        self.canvas.create_text(125, 425, text="Player", fill="white")

    def populate_monitors(self):
        monitors = screeninfo.get_monitors()
        self.monitor_combo["values"] = [f"Monitor {i + 1}: {monitor.name}" for i, monitor in enumerate(monitors)]

    def confirm_monitor_selection(self):
        selected_monitor_index = self.monitor_combo.current()
        if selected_monitor_index == -1:
            messagebox.showerror("Error", "Please select a monitor.")
            return

        monitors = screeninfo.get_monitors()
        selected_monitor = monitors[selected_monitor_index]
        self.pbox_generator.set_monitor(selected_monitor)
        self.monitor_utils.set_monitor(selected_monitor)  # Pass the monitor to MonitorUtils
        resolution_text = f"{selected_monitor.width}x{selected_monitor.height}"
        self.resolution_var.set(resolution_text)
        self.pbox_gen_button.config(state=tk.NORMAL)

    def on_resize(self, event):
        if hasattr(self.pbox_generator, 'current_image_path') and self.pbox_generator.current_image_path:
            self.pbox_generator._display_image(self.pbox_generator.current_image_path)

    def clear_screen(self):
        self.canvas.delete("all")
        print("Screen cleared.")

    def start(self):
        if not self.monitor_utils.monitor:
            messagebox.showerror("Error", "Please confirm monitor selection before starting the game.")
            return
        self.clear_screen()
        if not self.background_processor:
            self.background_processor = BackgroundProcessor(self.update_ui_callback, self)
            self.background_processor.blackjack_logic.set_monitor(self.monitor_utils.monitor)
        self.background_processor.start()

    def update_ui_callback(self):
        self.background_processor.update_gui_from_queue()


if __name__ == "__main__":
    root = GraphicalUserInterface()
    root.mainloop()

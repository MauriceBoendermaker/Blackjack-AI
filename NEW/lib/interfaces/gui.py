import threading
import screeninfo
import tkinter as tk

from tkinter import ttk, messagebox

from ..common import constants
from ..logic.monitor_utils import MonitorUtils
from ..logic.background import BackgroundProcessor
from ..player_boxes import generator as pbox_generator


class GraphicalUserInterface(tk.Tk):
    # Class for graphical user interface for BlackJack AI

    def __init__(self):
        super().__init__()
        self.title(constants.TITLE)
        self.geometry(constants.SIZE)
        self.resizable(False, False)

        self.reset_button = None
        self.background_processor = None

        self.player_decision_labels = {}
        self.last_decision_states = {}

        self.pbox_generator = pbox_generator.PlayerBoxGenerator(self)
        self.monitor_utils = MonitorUtils()

        self.monitor_var = tk.StringVar()
        self.monitor_selection_frame = ttk.Frame(self)
        self.monitor_selection_frame.pack(padx=30, pady=20)
        self.monitor_label = ttk.Label(self.monitor_selection_frame, text="Select Monitor:")
        self.monitor_label.pack(side=tk.LEFT)
        self.monitor_combo = ttk.Combobox(self.monitor_selection_frame, textvariable=self.monitor_var, state="readonly")
        self.monitor_combo.pack(side=tk.LEFT)

        self.resolution_label = ttk.Label(self.monitor_selection_frame, text="Resolution: ")
        self.resolution_label.pack(side=tk.LEFT)
        self.resolution_var = tk.StringVar()
        self.resolution_display = ttk.Label(self.monitor_selection_frame, textvariable=self.resolution_var)
        self.resolution_display.pack(side=tk.LEFT)

        self.confirm_button = ttk.Button(self.monitor_selection_frame, text="Confirm",
                                         command=self.confirm_monitor_selection)
        self.confirm_button.pack(side=tk.LEFT, padx=10)

        self.pbox_gen_button = ttk.Button(self, text="Generate Player Boxes", command=self.pbox_generator.generate,
                                          state=tk.DISABLED)
        self.pbox_gen_button.pack(padx=30, pady=20)

        self.start_button = ttk.Button(self, text="Start", command=self.start)
        self.start_button.pack(pady=30)

        self.reset_button = ttk.Button(self, text="Reset Round", command=self.reset_round)
        self.reset_button.pack(pady=10)

        self.refresh_button = ttk.Button(self, text="Refresh Counters", command=self.force_refresh_counters)
        self.refresh_button.pack(pady=5)

        self.round_label = ttk.Label(self, text=f"Round: 0", font=("Helvetica", 14))
        self.round_label.place(x=10, y=5)

        self.dealer_value_label = ttk.Label(self, text="Dealer has: ", font=("Helvetica", 14))
        self.dealer_value_label.place(relx=1.0, rely=0.0, x=-50, y=0, anchor='ne')

        self.counter_frame = ttk.LabelFrame(self, text="Card Counters")
        self.counter_frame.pack(pady=10)

        self.card_counter_widgets = {}
        self.create_card_counter_widgets()

        self.canvas = tk.Canvas(self, bg="#ffffff")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        self.draw_canvas()
        self.populate_monitors()

        self.bind("<Configure>", self.on_resize)

    def draw_canvas(self):
        self.canvas.create_rectangle(50, 50, 200, 100, fill="black", outline="white")
        self.canvas.create_text(125, 75, text="Dealer", fill="white")
        for i in range(7):
            x1 = 50 + i * (constants.CARD_WIDTH + constants.CARD_SPACING)
            y1 = 150
            x2 = x1 + constants.CARD_WIDTH
            y2 = y1 + constants.CARD_HEIGHT
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")
            self.canvas.create_text((x1 + x2) // 2, y2 + 30, text=f"Player {i + 1}", fill="black")

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
        self.monitor_utils.set_monitor(selected_monitor)
        resolution_text = f"{selected_monitor.width}x{selected_monitor.height}"
        self.resolution_var.set(resolution_text)
        self.pbox_gen_button.config(state=tk.NORMAL)

    def update_player_decision_labels(self, player_decisions):
        for player_num, decisions in player_decisions.items():
            key = f"{decisions['decision']}|{decisions['second']}"
            if self.last_decision_states.get(player_num) != key:
                self.last_decision_states[player_num] = key

                if player_num in self.player_decision_labels:
                    for label in self.player_decision_labels[player_num]:
                        label.destroy()

                decision_label = tk.Label(self.canvas, text=f"{decisions['decision']}", fg="green",
                                          font=("Helvetica", 12, "bold"))
                second_label = tk.Label(self.canvas, text=f"Optimal: {decisions['second']}", fg="blue",
                                        font=("Helvetica", 10, "bold"))

                x = 50 + (player_num - 1) * (constants.CARD_WIDTH + constants.CARD_SPACING)
                y = 280

                self.canvas.create_window(x, y, anchor="nw", window=decision_label)
                self.canvas.create_window(x, y + 25, anchor="nw", window=second_label)

                self.player_decision_labels[player_num] = [decision_label, second_label]

    def create_card_counter_widgets(self):
        numbers = [str(n) for n in range(2, 11)]
        faces = ['Jack', 'Queen', 'King', 'Ace']

        self.card_value_map = {
            'Jack': '10',
            'Queen': '10',
            'King': '10',
            'Ace': 'Ace'
        }

        full_order = numbers + faces

        for idx, card in enumerate(numbers + faces):
            if card in numbers:
                row = numbers.index(card)
                col = 0
            else:
                row = faces.index(card)
                col = 3

            label_text = f"{card}: 0x"

            minus_btn = tk.Button(self.counter_frame, text="-", fg="white", bg="red", width=3,
                                  command=lambda c=card: self.adjust_card_count(c, -1))
            minus_btn.grid(row=row, column=col)

            label = tk.Label(self.counter_frame, text=label_text, width=10, anchor="w")
            label.grid(row=row, column=col + 1)

            plus_btn = tk.Button(self.counter_frame, text="+", fg="white", bg="green", width=3,
                                 command=lambda c=card: self.adjust_card_count(c, 1))
            plus_btn.grid(row=row, column=col + 2)

            self.card_counter_widgets[card] = label

            if self.background_processor and self.background_processor.blackjack_logic:
                logic = self.background_processor.blackjack_logic
                logic.card_utils.card_counter_labels[card] = label

    def adjust_card_count(self, card_name, increment):
        if not self.background_processor or not self.background_processor.blackjack_logic:
            return

        logic = self.background_processor.blackjack_logic
        card_utils = logic.card_utils

        mapped = self.card_value_map.get(card_name, card_name)
        card_utils.update_card_counter(card_name, increment)

        count = card_utils.card_counters.get(mapped, 0)
        label = self.card_counter_widgets[card_name]
        label.config(text=f"{card_name}: {count}x")

    def on_resize(self, event):
        if hasattr(self.pbox_generator, 'current_image_path') and self.pbox_generator.current_image_path:
            self.pbox_generator._display_image(self.pbox_generator.current_image_path)

    def clear_screen(self):
        self.canvas.delete("all")
        print("Screen cleared.")

    def force_refresh_counters(self):
        if self.background_processor and self.background_processor.blackjack_logic:
            card_utils = self.background_processor.blackjack_logic.card_utils

            for card_name, label in self.card_counter_widgets.items():
                mapped = self.card_value_map.get(card_name, card_name)
                count = card_utils.card_counters.get(mapped, 0)
                label.config(text=f"{card_name}: {count}x")

            self.counter_frame.update_idletasks()
            self.counter_frame.update()
            self.update_idletasks()
            self.update()

            print("[REFRESH] Counter labels updated manually.")

    def start(self):
        if not self.monitor_utils.monitor:
            messagebox.showerror("Error", "Please confirm monitor selection before starting the game.")
            return
        self.clear_screen()
        self.draw_canvas()
        if not self.background_processor:
            self.background_processor = BackgroundProcessor(self.update_ui_callback, self)
            self.background_processor.blackjack_logic.set_monitor(self.monitor_utils.monitor)
        self.background_processor.start()

        if not self.background_processor:
            self.background_processor = BackgroundProcessor(self.update_ui_callback, self)
            self.background_processor.blackjack_logic.set_monitor(self.monitor_utils.monitor)

    def reset_round(self):
        if self.background_processor and self.background_processor.blackjack_logic:
            threading.Thread(target=self.run_reset_process, daemon=True).start()

    def run_reset_process(self):
        self.background_processor.blackjack_logic.reset_for_new_round()
        logic = self.background_processor.blackjack_logic
        logic.card_utils.counted_cards_this_round.clear()
        self.gui_reset_update()

    def gui_reset_update(self):
        self.background_processor.blackjack_logic.reset_gui_elements()
        self.update_ui_callback()

    def update_ui_callback(self):
        if self.background_processor:
            logic = self.background_processor.blackjack_logic
            decisions = logic.get_current_player_decisions()
            self.update_player_decision_labels(decisions)

            for label in self.card_counter_widgets.values():
                label.update_idletasks()
                label.update()

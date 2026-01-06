import threading
import screeninfo
import tkinter as tk

from tkinter import ttk, messagebox

from ..common import constants
from ..logic.monitor_utils import MonitorUtils
from ..logic.background import BackgroundProcessor
from ..player_boxes import generator as pbox_generator
from .tooltip import ToolTip


class GraphicalUserInterface(tk.Tk):
    # Class for graphical user interface for BlackJack AI

    def __init__(self):
        super().__init__()
        self.title(constants.TITLE)
        self.geometry(constants.SIZE)
        self.resizable(True, True)

        self.style = ttk.Style()
        self.style.configure("TLabel", font=constants.DEFAULT_FONT)
        self.style.configure("TButton", font=constants.DEFAULT_FONT)
        self.style.configure("Status.TLabel", relief="sunken", anchor="w")

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.reset_button = None
        self.background_processor = None

        self.player_decision_labels = {}
        self.last_decision_states = {}

        self.pbox_generator = pbox_generator.PlayerBoxGenerator(self)
        self.monitor_utils = MonitorUtils()

        self.monitor_var = tk.StringVar()

        self.left_frame = ttk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.rowconfigure(2, weight=1)

        self.monitor_selection_frame = ttk.Frame(self.left_frame)
        self.monitor_selection_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.monitor_selection_frame.columnconfigure(1, weight=1)

        self.monitor_label = ttk.Label(self.monitor_selection_frame, text="Select Monitor:")
        self.monitor_label.grid(row=0, column=0, sticky="w")
        self.monitor_combo = ttk.Combobox(
            self.monitor_selection_frame, textvariable=self.monitor_var, state="readonly"
        )
        self.monitor_combo.grid(row=0, column=1, sticky="ew")

        self.resolution_label = ttk.Label(self.monitor_selection_frame, text="Resolution: ")
        self.resolution_label.grid(row=1, column=0, sticky="w")
        self.resolution_var = tk.StringVar()
        self.resolution_display = ttk.Label(
            self.monitor_selection_frame, textvariable=self.resolution_var
        )
        self.resolution_display.grid(row=1, column=1, sticky="w")

        self.confirm_button = ttk.Button(
            self.monitor_selection_frame,
            text="Confirm",
            command=self.confirm_monitor_selection,
        )
        self.confirm_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        ToolTip(self.confirm_button, "Confirm selected monitor")

        self.action_frame = ttk.Frame(self.left_frame)
        self.action_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.pbox_gen_button = ttk.Button(
            self,
            text="Generate Player Boxes",
            command=self.pbox_generator.generate_async,
            state=tk.DISABLED,
        )
        # Position screenshot button at the bottom-right corner of the window
        self.pbox_gen_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

        self.start_button = ttk.Button(self.action_frame, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, pady=5, sticky="ew")

        self.reset_button = ttk.Button(self.action_frame, text="Reset Round", command=self.reset_round)
        self.reset_button.grid(row=1, column=0, pady=5, sticky="ew")

        self.refresh_button = ttk.Button(
            self.action_frame, text="Refresh Counters", command=self.force_refresh_counters
        )
        self.refresh_button.grid(row=2, column=0, pady=5, sticky="ew")

        self.counter_container = ttk.LabelFrame(self.left_frame, text="Card Counters")
        self.counter_container.grid(row=2, column=0, sticky="nsew")
        self.counter_container.rowconfigure(0, weight=1)
        self.counter_container.columnconfigure(0, weight=1)

        self.counter_canvas = tk.Canvas(self.counter_container, highlightthickness=0)
        self.counter_scrollbar = ttk.Scrollbar(
            self.counter_container, orient="vertical", command=self.counter_canvas.yview
        )
        self.counter_canvas.configure(yscrollcommand=self.counter_scrollbar.set)
        self.counter_canvas.grid(row=0, column=0, sticky="nsew")
        self.counter_scrollbar.grid(row=0, column=1, sticky="ns")

        self.counter_frame = ttk.Frame(self.counter_canvas)
        self.counter_canvas.create_window((0, 0), window=self.counter_frame, anchor="nw")
        self.counter_frame.bind(
            "<Configure>",
            lambda e: self.counter_canvas.configure(scrollregion=self.counter_canvas.bbox("all")),
        )

        self.round_label = ttk.Label(self, text=f"Round: 0", font=constants.LARGE_FONT)
        self.round_label.grid(row=0, column=1, sticky="ne", padx=10, pady=5)

        self.dealer_value_label = ttk.Label(self, text="Dealer has: ", font=constants.LARGE_FONT)
        self.dealer_value_label.grid(row=0, column=1, sticky="nw", padx=10, pady=5)

        self.card_counter_widgets = {}
        self.create_card_counter_widgets()

        # Create canvas with minimum size for semicircular player layout
        self.canvas = tk.Canvas(self, bg="#ffffff", width=900, height=650)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.decision_frame = ttk.Frame(self)
        self.decision_frame.grid(row=1, column=1, sticky="ew", padx=10)

        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Add FPS counter label in top-right corner of canvas
        self.fps_var = tk.StringVar(value="FPS: --")
        self.fps_label = tk.Label(self.canvas, textvariable=self.fps_var,
                                   bg="yellow", fg="black", font=("Arial", 10, "bold"),
                                   padx=5, pady=2)
        self.fps_label.place(relx=0.98, rely=0.02, anchor="ne")

        ToolTip(self.start_button, "Start detection and analysis")
        ToolTip(self.reset_button, "Reset current round")
        ToolTip(self.refresh_button, "Refresh card counters")
        ToolTip(self.pbox_gen_button, "Generate player regions on the monitor")

        self.draw_canvas()
        self.populate_monitors()

        # Pre-render player seats on initialization
        self.initialize_player_seats()

        self.bind("<Configure>", self.on_resize)

    def set_status(self, message):
        self.status_var.set(message)

    def update_fps_display(self, fps, cycle_time_ms):
        """Update the FPS counter display"""
        self.fps_var.set(f"FPS: {fps:.1f} | {cycle_time_ms:.0f}ms")

    def draw_canvas(self):
        import math

        # Canvas cleared - placeholder rectangles removed for cleaner UI
        # Player and dealer positions are now dynamically created by blackjack.py
        self.canvas.delete("all")
        self.update_idletasks()

    def initialize_player_seats(self):
        """Pre-render empty player seats and dealer position on startup"""
        from ..logic.background import BackgroundProcessor

        # Create background processor to access blackjack logic
        temp_processor = BackgroundProcessor(lambda: None, self)

        # Initialize empty player data
        empty_player_data = [{'player_index': i, 'cards': ['-', '-']} for i in range(7)]

        # Render the seats with empty cards
        temp_processor.blackjack_logic.update_player_cards_display(
            empty_player_data,
            dealer_up_card=None,
            true_count=0,
            base_bet=constants.BASE_BET
        )

        # Store reference for later use
        if not self.background_processor:
            self.background_processor = temp_processor

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

                decision_label = ttk.Label(
                    self.decision_frame,
                    text=f"{decisions['decision']}",
                    foreground="green",
                    font=constants.LARGE_FONT,
                )
                second_label = ttk.Label(
                    self.decision_frame,
                    text=f"Optimal: {decisions['second']}",
                    foreground="blue",
                    font=constants.DEFAULT_FONT,
                )

                col = player_num - 1
                decision_label.grid(row=0, column=col, padx=5)
                second_label.grid(row=1, column=col, padx=5)

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

            minus_btn = tk.Button(
                self.counter_frame,
                text="-",
                fg="white",
                bg="red",
                width=3,
                command=lambda c=card: self.adjust_card_count(c, -1),
            )
            minus_btn.grid(row=row, column=col)

            label = ttk.Label(self.counter_frame, text=label_text, width=10, anchor="w")
            label.grid(row=row, column=col + 1, padx=2)

            plus_btn = tk.Button(
                self.counter_frame,
                text="+",
                fg="white",
                bg="green",
                width=3,
                command=lambda c=card: self.adjust_card_count(c, 1),
            )
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
        self.set_status("Screen cleared")

    def force_refresh_counters(self):
        if self.background_processor and self.background_processor.blackjack_logic:
            card_utils = self.background_processor.blackjack_logic.card_utils

            for card_name, label in self.card_counter_widgets.items():
                mapped = self.card_value_map.get(card_name, card_name)
                count = card_utils.card_counters.get(mapped, 0)
                label.config(text=f"{card_name}: {count}x")



            self.set_status("Counters refreshed")

    def start(self):
        if not self.monitor_utils.monitor:
            messagebox.showerror("Error", "Please confirm monitor selection before starting the game.")
            return
        self.clear_screen()

        # Reuse existing background processor (created during init) or create new one
        if not self.background_processor:
            self.background_processor = BackgroundProcessor(self.update_ui_callback, self)

        # Set monitor for detection
        self.background_processor.blackjack_logic.set_monitor(self.monitor_utils.monitor)

        # Start background processing
        self.background_processor.start()
        self.set_status("Background processing started")

    def reset_round(self):
        if self.background_processor and self.background_processor.blackjack_logic:
            threading.Thread(target=self.run_reset_process, daemon=True).start()
            self.set_status("Round reset")

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

            self.set_status("UI updated")



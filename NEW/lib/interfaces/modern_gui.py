"""
Modern, Enhanced GUI for Blackjack AI
Completely redesigned interface with professional styling
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import screeninfo

from ..common import constants
from ..logic.monitor_utils import MonitorUtils
from ..logic.background import BackgroundProcessor
from ..player_boxes import generator as pbox_generator
from .tooltip import ToolTip


class ModernBlackjackGUI(tk.Tk):
    """Modern, professional GUI for Blackjack AI"""

    def __init__(self, log_manager=None):
        super().__init__()
        self.title(f"🎰 {constants.TITLE}")
        self.geometry("1600x900")
        self.resizable(True, True)
        self.configure(bg="#f8f9fa")  # Light grey background

        # Store log manager reference
        self.log_manager = log_manager

        # Professional minimalistic color scheme - Bright theme
        self.colors = {
            'bg_primary': '#f8f9fa',      # Light grey (almost white)
            'bg_secondary': '#ffffff',    # Pure white
            'bg_canvas': '#e9ecef',       # Light grey canvas
            'accent': '#0d6efd',          # Professional blue
            'accent_hover': '#0a58ca',    # Darker blue on hover
            'text_primary': '#212529',    # Dark grey (almost black)
            'text_secondary': '#6c757d',  # Medium grey
            'success': '#198754',         # Professional green
            'warning': '#ffc107',         # Professional amber
            'danger': '#dc3545',          # Professional red
            'border': '#dee2e6',          # Light border
            'card_bg': '#ffffff',         # White
            'dealer_area': '#f1f3f5',     # Very light grey
        }

        # Configure modern ttk styles
        self.setup_modern_styles()

        self.background_processor = None
        self.monitor_utils = MonitorUtils()
        self.pbox_generator = pbox_generator.PlayerBoxGenerator(self)

        # Build the modern UI
        self.build_modern_interface()

    def setup_modern_styles(self):
        """Configure professional minimalistic styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Professional button style
        style.configure("Modern.TButton",
                       background=self.colors['accent'],
                       foreground='white',
                       borderwidth=1,
                       bordercolor=self.colors['accent'],
                       focuscolor='none',
                       font=('Inter', 11, 'normal'),
                       padding=12)
        style.map("Modern.TButton",
                 background=[('active', self.colors['accent_hover'])])

        # Professional label style
        style.configure("Modern.TLabel",
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'],
                       font=('Inter', 11))

        # Frame styles
        style.configure("Modern.TFrame",
                       background=self.colors['bg_primary'])

        style.configure("Card.TFrame",
                       background=self.colors['bg_secondary'],
                       relief='flat',
                       borderwidth=1)

        # Status bar style
        style.configure("Status.TLabel",
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['text_primary'],
                       font=('Inter', 10),
                       padding=8)

    def build_modern_interface(self):
        """Build the complete modern interface"""
        # Configure grid weights
        self.columnconfigure(0, weight=0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left panel - Controls
        self.build_left_panel()

        # Right panel - Game area
        self.build_game_area()

        # Bottom status bar
        self.build_status_bar()

        # Pre-render player seats
        self.after(100, self.initialize_player_seats)

    def build_left_panel(self):
        """Build professional left control panel"""
        left_frame = tk.Frame(self, bg=self.colors['bg_secondary'],
                             highlightbackground=self.colors['border'],
                             highlightthickness=1)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Header - Clean and professional
        header = tk.Frame(left_frame, bg=self.colors['bg_secondary'], height=70)
        header.pack(fill=tk.X, pady=0)
        header.pack_propagate(False)

        title_label = tk.Label(header, text="Blackjack AI",
                              font=('Inter', 18, 'bold'),
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'])
        title_label.pack(pady=20)

        # Scrollable content area
        canvas_left = tk.Canvas(left_frame, bg=self.colors['bg_secondary'],
                               highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical",
                                 command=canvas_left.yview)
        scrollable_frame = tk.Frame(canvas_left, bg=self.colors['bg_secondary'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all"))
        )

        canvas_left.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_left.configure(yscrollcommand=scrollbar.set)

        canvas_left.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Monitor Selection Section
        self.build_monitor_section(scrollable_frame)

        # Action Buttons Section
        self.build_action_buttons(scrollable_frame)

        # Card Counters Section
        self.build_card_counters(scrollable_frame)

        # Round Info Section
        self.build_round_info(scrollable_frame)

    def build_monitor_section(self, parent):
        """Build professional monitor selection section"""
        section = self.create_section(parent, "Monitor Setup")

        self.monitor_var = tk.StringVar()
        monitor_combo = ttk.Combobox(section, textvariable=self.monitor_var,
                                    state="readonly", font=('Inter', 11))
        monitor_combo.pack(fill=tk.X, pady=(5, 10))

        # Populate monitors
        monitors = screeninfo.get_monitors()
        monitor_combo["values"] = [f"Monitor {i + 1}: {m.name}"
                                   for i, m in enumerate(monitors)]
        if monitors:
            monitor_combo.current(0)

        # Resolution display
        self.resolution_var = tk.StringVar(value="Resolution: Not selected")
        res_label = tk.Label(section, textvariable=self.resolution_var,
                           font=('Inter', 10),
                           bg=self.colors['bg_secondary'],
                           fg=self.colors['text_secondary'])
        res_label.pack(pady=(0, 12))

        # Confirm button - Professional style
        confirm_btn = tk.Button(section, text="Confirm Selection",
                               command=self.confirm_monitor_selection,
                               bg=self.colors['success'],
                               fg='white',
                               font=('Inter', 11, 'normal'),
                               relief='flat',
                               cursor='hand2',
                               borderwidth=0,
                               padx=20, pady=12)
        confirm_btn.pack(fill=tk.X)

        monitor_combo.bind("<<ComboboxSelected>>", self.on_monitor_change)

    def build_action_buttons(self, parent):
        """Build professional action buttons"""
        section = self.create_section(parent, "Controls")

        buttons = [
            ("Start Detection", self.start, self.colors['success'], 'white'),
            ("Reset Round", self.reset_round, self.colors['warning'], self.colors['text_primary']),
            ("Refresh Counters", self.refresh_counters, self.colors['accent'], 'white'),
            ("Generate Regions", self.generate_player_boxes, self.colors['bg_canvas'], self.colors['text_primary']),
            ("View Logs", self.open_logging_window, self.colors['text_secondary'], 'white'),
        ]

        for text, command, bg_color, fg_color in buttons:
            # Create button with rounded corners effect (using Frame)
            btn_frame = tk.Frame(section, bg=self.colors['bg_secondary'])
            btn_frame.pack(fill=tk.X, pady=6)

            btn = tk.Button(btn_frame, text=text,
                          command=command,
                          bg=bg_color,
                          fg=fg_color,
                          font=('Inter', 11, 'normal'),
                          relief='flat',
                          cursor='hand2',
                          borderwidth=0,
                          padx=20, pady=12)
            btn.pack(fill=tk.X)

            # Add subtle hover effect
            btn.bind('<Enter>', lambda e, b=btn, c=self.colors['accent_hover']: b.config(bg=c) if b['bg'] == self.colors['accent'] else None)
            btn.bind('<Leave>', lambda e, b=btn, c=bg_color: b.config(bg=c))

    def build_card_counters(self, parent):
        """Build professional card counters section"""
        section = self.create_section(parent, "Card Counters")

        # Card counter widgets will be added here
        self.card_counter_frame = tk.Frame(section, bg=self.colors['bg_secondary'])
        self.card_counter_frame.pack(fill=tk.BOTH, expand=True)

        self.card_counter_widgets = {}
        self.create_card_counter_widgets()

    def build_round_info(self, parent):
        """Build professional round information section"""
        section = self.create_section(parent, "Game Info")

        # Round counter
        self.round_var = tk.StringVar(value="Round: 0")
        round_label = tk.Label(section, textvariable=self.round_var,
                              font=('Inter', 13, 'bold'),
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'])
        round_label.pack(pady=8)

        # True count
        self.true_count_var = tk.StringVar(value="True Count: 0.0")
        tc_label = tk.Label(section, textvariable=self.true_count_var,
                           font=('Inter', 12),
                           bg=self.colors['bg_secondary'],
                           fg=self.colors['success'])
        tc_label.pack(pady=8)

    def create_section(self, parent, title):
        """Create a professional section with title"""
        container = tk.Frame(parent, bg=self.colors['bg_secondary'])
        container.pack(fill=tk.X, padx=20, pady=15)

        title_label = tk.Label(container, text=title,
                              font=('Inter', 12, 'bold'),
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'])
        title_label.pack(anchor='w', pady=(0, 12))

        return container

    def build_game_area(self):
        """Build the professional game display area"""
        game_frame = tk.Frame(self, bg=self.colors['bg_primary'])
        game_frame.grid(row=0, column=1, sticky="nsew")
        game_frame.rowconfigure(0, weight=1)
        game_frame.columnconfigure(0, weight=1)

        # Main canvas with professional styling
        self.canvas = tk.Canvas(game_frame,
                               bg=self.colors['bg_secondary'],
                               highlightbackground=self.colors['border'],
                               highlightthickness=1,
                               relief='flat')
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)

        # FPS Counter - Professional style
        self.fps_var = tk.StringVar(value="FPS: --")
        self.fps_label = tk.Label(self.canvas, textvariable=self.fps_var,
                                 bg=self.colors['accent'],
                                 fg='white',
                                 font=('Inter', 10, 'bold'),
                                 padx=12, pady=6,
                                 relief='flat')
        self.fps_label.place(relx=0.98, rely=0.02, anchor="ne")

        # Dealer label - Professional
        self.dealer_label = tk.Label(self.canvas, text="Dealer",
                                    font=('Inter', 16, 'bold'),
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'])
        self.dealer_label.place(relx=0.5, rely=0.06, anchor="center")

    def build_status_bar(self):
        """Build professional status bar"""
        status_frame = tk.Frame(self, bg=self.colors['bg_secondary'],
                               highlightbackground=self.colors['border'],
                               highlightthickness=1,
                               height=40)
        status_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        status_frame.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Inter', 10),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_secondary'],
                               anchor='w',
                               padx=15)
        status_label.pack(fill=tk.BOTH, expand=True)

    def create_card_counter_widgets(self):
        """Create modern card counter widgets"""
        from ..logic.card_utils import get_card_utils

        card_utils = get_card_utils(self)

        card_values = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10']

        for card_value in card_values:
            frame = tk.Frame(self.card_counter_frame,
                           bg=self.colors['bg_canvas'],
                           relief='flat',
                           padx=10, pady=5)
            frame.pack(fill=tk.X, pady=2)

            # Card value label
            label = tk.Label(frame, text=f"{card_value}:",
                           font=('Segoe UI', 10, 'bold'),
                           bg=self.colors['bg_canvas'],
                           fg=self.colors['text_primary'],
                           width=4,
                           anchor='w')
            label.pack(side=tk.LEFT)

            # Counter display
            count_var = tk.StringVar(value="0x")
            count_label = tk.Label(frame, textvariable=count_var,
                                  font=('Segoe UI', 10),
                                  bg=self.colors['bg_canvas'],
                                  fg=self.colors['success'],
                                  width=5,
                                  anchor='center')
            count_label.pack(side=tk.LEFT, padx=10)

            # +/- buttons
            btn_minus = tk.Button(frame, text="−",
                                command=lambda v=card_value: self.decrement_counter(v),
                                bg=self.colors['accent'],
                                fg='white',
                                font=('Segoe UI', 10, 'bold'),
                                width=3,
                                relief='flat',
                                cursor='hand2')
            btn_minus.pack(side=tk.LEFT, padx=2)

            btn_plus = tk.Button(frame, text="+",
                               command=lambda v=card_value: self.increment_counter(v),
                               bg=self.colors['success'],
                               fg='white',
                               font=('Segoe UI', 10, 'bold'),
                               width=3,
                               relief='flat',
                               cursor='hand2')
            btn_plus.pack(side=tk.LEFT, padx=2)

            self.card_counter_widgets[card_value] = {
                'label': count_label,
                'var': count_var
            }

    # Interface methods
    def set_status(self, message):
        """Update status bar with message"""
        self.status_var.set(message)

    def update_fps_display(self, fps, cycle_time_ms):
        """Update FPS display"""
        self.fps_var.set(f"{fps:.1f} FPS | {cycle_time_ms:.0f}ms")

    def on_monitor_change(self, event):
        """Update resolution display when monitor changes"""
        selection = self.monitor_var.get()
        if selection:
            monitor_index = int(selection.split()[1].rstrip(':')) - 1
            monitors = screeninfo.get_monitors()
            if 0 <= monitor_index < len(monitors):
                monitor = monitors[monitor_index]
                self.resolution_var.set(f"Resolution: {monitor.width}x{monitor.height}")

    def confirm_monitor_selection(self):
        """Confirm monitor selection"""
        selection = self.monitor_var.get()
        if not selection:
            messagebox.showerror("Error", "Please select a monitor first.")
            return

        monitor_index = int(selection.split()[1].rstrip(':')) - 1
        monitors = screeninfo.get_monitors()
        self.monitor_utils.monitor = monitors[monitor_index]
        self.set_status(f"Monitor {monitor_index + 1} selected")
        messagebox.showinfo("Success", f"Monitor {monitor_index + 1} confirmed!")

    def initialize_player_seats(self):
        """Pre-render empty player seats"""
        from ..logic.background import BackgroundProcessor

        temp_processor = BackgroundProcessor(lambda: None, self)
        empty_player_data = [{'player_index': i, 'cards': ['-', '-']} for i in range(7)]

        temp_processor.blackjack_logic.update_player_cards_display(
            empty_player_data,
            dealer_up_card=None,
            true_count=0,
            base_bet=constants.BASE_BET
        )

        if not self.background_processor:
            self.background_processor = temp_processor

        self.set_status("Player seats initialized")

    def start(self):
        """Start detection"""
        if not self.monitor_utils.monitor:
            messagebox.showerror("Error", "Please confirm monitor selection first.")
            return

        if not self.background_processor:
            self.background_processor = BackgroundProcessor(self.update_ui_callback, self)

        self.background_processor.blackjack_logic.set_monitor(self.monitor_utils.monitor)
        self.background_processor.start()
        self.set_status("Detection started")

    def reset_round(self):
        """Reset current round"""
        if self.background_processor and self.background_processor.blackjack_logic:
            threading.Thread(target=self.run_reset_process, daemon=True).start()

    def run_reset_process(self):
        """Run reset process in thread"""
        if self.background_processor and self.background_processor.blackjack_logic:
            self.background_processor.blackjack_logic.reset_for_new_round()
            self.background_processor.blackjack_logic.reset_gui_elements()
            self.after(0, lambda: self.set_status("Round reset"))

    def refresh_counters(self):
        """Refresh card counters"""
        if self.background_processor:
            self.background_processor.card_utils.refresh_card_counter_widgets()
            self.set_status("Counters refreshed")

    def generate_player_boxes(self):
        """Generate player box regions"""
        self.pbox_generator.generate()
        self.set_status("Player regions generated")

    def open_logging_window(self):
        """Open non-modal logging window"""
        # Check if window already exists and is open
        if hasattr(self, 'logging_window') and self.logging_window.winfo_exists():
            self.logging_window.lift()  # Bring to front
            self.set_status("Logging window brought to front")
            return

        # Import and create new window
        from .logging_window import LoggingWindow

        if not self.log_manager:
            messagebox.showerror("Error", "Logging system not initialized.")
            return

        self.logging_window = LoggingWindow(self, self.log_manager)
        self.set_status("Logging window opened")

    def increment_counter(self, card_value):
        """Increment card counter"""
        if self.background_processor:
            self.background_processor.card_utils.update_card_counter(f"{card_value} of Hearts", 1)

    def decrement_counter(self, card_value):
        """Decrement card counter"""
        if self.background_processor:
            self.background_processor.card_utils.update_card_counter(f"{card_value} of Hearts", -1)

    def update_ui_callback(self):
        """Callback for UI updates"""
        if self.background_processor and self.background_processor.blackjack_logic:
            self.background_processor.blackjack_logic.update_gui()

    @property
    def round_label(self):
        """Compatibility property"""
        class RoundLabel:
            def __init__(self, var):
                self.var = var
            def config(self, text):
                self.var.set(text)
        return RoundLabel(self.round_var)

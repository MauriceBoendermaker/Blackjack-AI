"""Main application window.

Architecture: the window never talks to models or the screen directly. A
DetectionController runs the engine on a worker thread; this GUI polls
`engine.get_snapshot()` from a Tk `after()` loop and renders the result.
Every user action calls a thread-safe engine method. No Tkinter object is
ever touched from the worker thread.
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import screeninfo

from ..common import constants
from ..logic.background import DetectionController
from ..logic.counting import COUNTER_KEYS
from ..logic.phase import PHASE_LABELS
from ..logic.region_preview import build_region_preview
from . import scaling
from .card_picker import CardPicker
from .table_view import TableView
from .tooltip import ToolTip
from .validation import attach_numeric_entry

C = constants.COLORS


class ModernBlackjackGUI(tk.Tk):
    def __init__(self, log_manager=None):
        super().__init__()
        # Scaling first: every widget below picks up the live FONT_* objects
        # and px() sizes resolved for this monitor's DPI.
        scaling.init(self)
        self.title(constants.TITLE)
        width, height = self._initial_geometry()
        self.geometry(f"{width}x{height}")
        self._minsize_applied = (min(scaling.px(1150), width),
                                 min(scaling.px(760), height))
        self.minsize(*self._minsize_applied)
        self.configure(bg=C["bg_primary"])

        self.log_manager = log_manager
        # add_log (not print) so engine WARNING/ERROR levels survive into the
        # log window instead of being folded into the message text.
        self.controller = DetectionController(
            log=log_manager.add_log if log_manager else print)
        self.monitor = None
        self._last_seq = -1
        self._preview_busy = False

        # Ghost-mode executor (V4 Feature 2): a second consumer of the same
        # snapshot. step() runs in the render path (pure dict math); clicks
        # only ever fire from the confirm hotkey in assist mode.
        engine = self.controller.engine
        from ..logic.executor import Executor
        self.executor = Executor(
            snapshot_fn=engine.get_snapshot,
            monitor_fn=lambda: self.monitor,
            store=engine.store,
            io_submit=engine.submit_io,
            log=log_manager.add_log if log_manager else print,
            running_fn=lambda: self.controller.running)
        self.ghost_marker = None
        self._hotkeys = None

        # Row 0 nav bar, row 1 sidebar + table (the stretchy row), row 2 status.
        self.columnconfigure(0, weight=0, minsize=scaling.px(340))
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_nav_bar()
        self._build_left_panel()
        self._build_table()
        self._build_status_bar()
        scaling.watch(self)
        # Fonts and the table rescale themselves on a DPI change; the fixed
        # px() chrome (sidebar width, nav height, minsize) must follow too.
        scaling.on_change(self._apply_chrome_scale)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(constants.SNAPSHOT_POLL_MS, self._poll_snapshot)
        self.after(300, lambda: self.table.preload_images())
        self.after(150, self._auto_select_monitor)
        self.after(600, self._maybe_restore_shoe)
        self.set_status("Ready — confirm a monitor, then press Start Detection.")

    def _apply_chrome_scale(self):
        """Re-apply the fixed px() dimensions after a DPI rescale (monitor
        move or a settings override) — fonts and the table already updated."""
        self.columnconfigure(0, minsize=scaling.px(340))
        self._sidebar_canvas.config(width=scaling.px(320))
        self.nav_bar.config(height=scaling.px(42))
        # Minsize follows the CURRENT monitor (not the primary) and is only
        # re-asserted on change — a redundant minsize can emit WM resizes
        # that re-feed the DPI watcher after its guard releases.
        _, _, work_w, work_h = scaling.workarea(self)
        minsize = (min(scaling.px(1150), int(work_w * 0.92)),
                   min(scaling.px(760), int(work_h * 0.92)))
        if minsize != self._minsize_applied:
            self._minsize_applied = minsize
            self.minsize(*minsize)
        ttk.Style(self).configure("Dark.Vertical.TScrollbar",
                                  width=scaling.px(10))

    def _initial_geometry(self):
        """The 1500x950 design size scaled for this monitor's DPI, clamped to
        ~92% of the monitor so the window never opens larger than the screen."""
        try:
            monitors = screeninfo.get_monitors()
            mon = next((m for m in monitors if getattr(m, "is_primary", False)),
                       monitors[0])
            max_w, max_h = int(mon.width * 0.92), int(mon.height * 0.92)
        except Exception:
            max_w, max_h = scaling.px(1500), scaling.px(950)
        return min(scaling.px(1500), max_w), min(scaling.px(950), max_h)

    # -------------------------------------------------------------- nav bar

    def _build_nav_bar(self):
        """Top bar: app name + version left, window-opening buttons right.
        Trims the sidebar to table actions only, so it rarely needs to scroll."""
        nav = tk.Frame(self, bg=C["bg_secondary"], height=scaling.px(42))
        nav.grid(row=0, column=0, columnspan=2, sticky="ew")
        nav.pack_propagate(False)
        # 1px bottom border (highlightthickness would also frame the sides).
        tk.Frame(nav, bg=C["border"], height=1).pack(side=tk.BOTTOM, fill=tk.X)
        self.nav_bar = nav

        tk.Label(nav, text="Blackjack AI", font=constants.FONT_SECTION,
                 bg=C["bg_secondary"], fg=C["text_primary"]
                 ).pack(side=tk.LEFT, padx=(14, 4))
        tk.Label(nav, text=f"v{constants.VERSION}", font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"]).pack(side=tk.LEFT)

        # Reversed so side=RIGHT packing shows them left-to-right as listed.
        for text, command, tooltip in reversed([
            ("🗒  Logs", self._open_logs, "Open the live log window"),
            ("📊  Stats", self._open_stats,
             "Round history, count distribution, CSV export"),
            ("🩺  Leaks", self._open_leaks,
             "What your mistakes cost — ranked, EV-priced, drillable"),
            ("🛡  Bankroll", self._open_bankroll,
             "Risk of ruin, Kelly risk table, Monte Carlo simulation"),
            ("🎓  Trainer", self._open_trainer,
             "Deck countdown, deviation flashcards, replay drills"),
            ("🎯  HUD", self._toggle_hud,
             "Compact always-on-top panel to park next to the stream"),
            ("⚙  Settings", self._open_settings,
             "Table rules, deck count, side-bet paytables"),
        ]):
            btn = self._nav_button(nav, text, command, tooltip)
            if command == self._toggle_hud:  # bound methods: == not `is`
                self.hud_nav_btn = btn

    def _nav_button(self, parent, text, command, tooltip):
        btn = self._button(parent, text, command, C["bg_secondary"],
                           fg=C["text_primary"], hover=C["bg_primary"],
                           tooltip=tooltip)
        btn.config(padx=scaling.px(10), pady=scaling.px(4))
        btn.pack(side=tk.RIGHT, padx=2)
        return btn

    # ------------------------------------------------------------ left panel

    def _build_left_panel(self):
        left = tk.Frame(self, bg=C["bg_secondary"],
                        highlightbackground=C["border"], highlightthickness=1)
        left.grid(row=1, column=0, sticky="nsew")

        canvas = tk.Canvas(left, bg=C["bg_secondary"], highlightthickness=0,
                           width=scaling.px(320))
        self._sidebar_canvas = canvas
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview,
                                  style=self._scrollbar_style())
        canvas.configure(yscrollcommand=scrollbar.set)
        inner = tk.Frame(canvas, bg=C["bg_secondary"])
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Auto-hide: the scrollbar only appears while the sections overflow
        # the canvas (re-checked whenever either side changes height).
        def _sync_scrollbar():
            bbox = canvas.bbox("all")
            needed = bbox is not None and bbox[3] - bbox[1] > canvas.winfo_height()
            if needed and not scrollbar.winfo_ismapped():
                scrollbar.pack(side="right", fill="y")
            elif not needed and scrollbar.winfo_ismapped():
                scrollbar.pack_forget()
                canvas.yview_moveto(0.0)

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            _sync_scrollbar()

        def _on_canvas_configure(event):
            canvas.itemconfigure(inner_id, width=event.width)
            _sync_scrollbar()

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # One global wheel binding that only acts when the pointer is over the
        # left panel (or a child of it) — no Enter/Leave toggling edge cases.
        def _wheel(event):
            widget = event.widget if isinstance(event.widget, tk.Misc) else None
            while widget is not None:
                if widget is left:
                    canvas.yview_scroll(int(-event.delta / 120), "units")
                    return
                widget = widget.master
        self.bind_all("<MouseWheel>", _wheel, add="+")

        canvas.pack(side="left", fill="both", expand=True)

        self._build_monitor_section(inner)
        self._build_controls_section(inner)
        self._build_executor_section(inner)
        self._build_counters_section(inner)
        self._build_game_info_section(inner)
        self._build_sidebets_section(inner)

    def _scrollbar_style(self):
        """Arrowless flat scrollbar matching the sidebar; returns the style
        name. The native Windows theme ignores ttk color options, so the
        trough/thumb elements are borrowed from clam, which honors them."""
        style = ttk.Style(self)
        for element in ("trough", "thumb"):
            try:
                style.element_create(f"Dark.Vertical.Scrollbar.{element}",
                                     "from", "clam",
                                     f"Vertical.Scrollbar.{element}")
            except tk.TclError:
                pass  # second window in one interpreter: elements persist
        style.layout("Dark.Vertical.TScrollbar", [
            ("Dark.Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
                ("Dark.Vertical.Scrollbar.thumb", {"expand": "1"})]})])
        style.configure("Dark.Vertical.TScrollbar",
                        troughcolor=C["bg_secondary"], background=C["border"],
                        bordercolor=C["bg_secondary"],
                        lightcolor=C["border"], darkcolor=C["border"],
                        relief="flat", width=scaling.px(10))
        style.map("Dark.Vertical.TScrollbar",
                  background=[("active", C["text_secondary"]),
                              ("pressed", C["text_secondary"])])
        return "Dark.Vertical.TScrollbar"

    def _section(self, parent, title):
        frame = tk.Frame(parent, bg=C["bg_secondary"])
        frame.pack(fill=tk.X, padx=18, pady=(14, 4))
        tk.Label(frame, text=title, font=constants.FONT_SECTION,
                 bg=C["bg_secondary"], fg=C["text_primary"]).pack(anchor="w", pady=(0, 8))
        return frame

    def _button(self, parent, text, command, bg, fg="white", hover=None, tooltip=None):
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                        font=constants.FONT_BODY, relief="flat", cursor="hand2",
                        borderwidth=0, padx=16, pady=9, activebackground=hover or bg)
        if hover:
            btn.bind("<Enter>", lambda e: btn["state"] == "normal" and btn.config(bg=hover))
            btn.bind("<Leave>", lambda e: btn.config(bg=btn._base_bg))
        btn._base_bg = bg
        if tooltip:
            ToolTip(btn, tooltip)
        return btn

    def _build_monitor_section(self, parent):
        section = self._section(parent, "Monitor")
        self.monitor_var = tk.StringVar()
        self.monitor_combo = ttk.Combobox(section, textvariable=self.monitor_var,
                                          state="readonly", font=constants.FONT_BODY)
        self._monitors = list(screeninfo.get_monitors())
        self.monitor_combo["values"] = [
            f"Monitor {i + 1}: {m.width}x{m.height}" + (" (primary)" if getattr(m, "is_primary", False) else "")
            for i, m in enumerate(self._monitors)
        ]
        self.monitor_combo.pack(fill=tk.X, pady=(0, 8))
        self.monitor_combo.bind("<<ComboboxSelected>>", lambda e: self._confirm_monitor())

        self.monitor_status = tk.Label(section, text="No monitor confirmed", font=constants.FONT_SMALL,
                                       bg=C["bg_secondary"], fg=C["text_secondary"])
        self.monitor_status.pack(anchor="w", pady=(0, 4))

    def _build_controls_section(self, parent):
        section = self._section(parent, "Controls")

        # Table profile: named region calibrations (seat/dealer + OCR rects)
        # per casino table. Selecting one swaps every region live.
        row = tk.Frame(section, bg=C["bg_secondary"])
        row.pack(fill=tk.X)
        tk.Label(row, text="Table profile", font=constants.FONT_BODY,
                 bg=C["bg_secondary"], fg=C["text_secondary"]
                 ).pack(anchor="w")
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(section, textvariable=self.profile_var,
                                          state="readonly", font=constants.FONT_BODY)
        self.profile_combo.pack(fill=tk.X, pady=(0, 8))
        self.profile_combo.bind("<<ComboboxSelected>>",
                                lambda e: self._select_profile())
        ToolTip(self.profile_combo,
                "Saved region calibrations per casino table — switching swaps "
                "the seat, dealer and OCR regions. Save a new one from "
                "Calibrate Regions.")
        self._profile_by_label = {}
        self._refresh_profiles()

        self.start_btn = self._button(
            section, "▶  Start Detection", self._toggle_detection,
            C["success"], hover=C["success_hover"],
            tooltip="Start/stop watching the selected monitor for cards")
        self.start_btn.pack(fill=tk.X, pady=4)
        self.start_btn.config(state="disabled")

        self._button(section, "↺  New Round", self._new_round, C["warning"],
                     fg=C["text_primary"],
                     tooltip="Clear all seats and the dealer card (shoe count is kept)"
                     ).pack(fill=tk.X, pady=4)
        self._button(section, "🂠  New Shoe", self._new_shoe, C["accent"], hover=C["accent_hover"],
                     tooltip="Reset the running count and per-card totals after a shuffle"
                     ).pack(fill=tk.X, pady=4)
        self.regions_btn = self._button(
            section, "⬚  Preview Regions", self._preview_regions, C["bg_canvas_soft"],
            tooltip="Capture the table and show the detected cards + seat regions")
        self.regions_btn.pack(fill=tk.X, pady=4)
        self.regions_btn.config(state="disabled")
        self.calibrate_btn = self._button(
            section, "✥  Calibrate Regions", self._calibrate_regions, C["bg_canvas_soft"],
            tooltip="Drag the seat polygons and dealer area over a live screenshot")
        self.calibrate_btn.pack(fill=tk.X, pady=4)
        self.calibrate_btn.config(state="disabled")
        self.ocr_btn = self._button(
            section, "🔡  OCR Regions", self._calibrate_ocr, C["bg_canvas_soft"],
            tooltip="Mark the balance / bet / result / timer areas to read "
                    "from the screen")
        self.ocr_btn.pack(fill=tk.X, pady=4)
        self.ocr_btn.config(state="disabled")
        self.controls_btn = self._button(
            section, "🎛  Capture Controls", self._calibrate_controls,
            C["bg_canvas_soft"],
            tooltip="Capture the Hit/Stand/Double/Split buttons and the bet "
                    "spot as templates — powers phase & turn detection")
        self.controls_btn.pack(fill=tk.X, pady=4)
        self.controls_btn.config(state="disabled")
        self.anchors_btn = self._button(
            section, "⚓  Capture Anchors", self._calibrate_anchors,
            C["bg_canvas_soft"],
            tooltip="Capture 2-3 stable UI elements (logo, chip tray, menu) "
                    "— maps this calibration to any resolution or window "
                    "position")
        self.anchors_btn.pack(fill=tk.X, pady=4)
        self.anchors_btn.config(state="disabled")

    def _build_executor_section(self, parent):
        """Ghost-mode executor controls (V4 Feature 2). Ghost draws where it
        WOULD click and logs the would-click match rate; Assist additionally
        fires on the F8 confirm key — after the §6 ghost gate, never before."""
        section = self._section(parent, "Executor (ghost / assist)")
        self.exec_mode_var = tk.StringVar()
        self._exec_mode_by_label = {"Off": "off", "Ghost (plan only)": "ghost",
                                    "Assist (F8 fires)": "assist"}
        combo = ttk.Combobox(section, textvariable=self.exec_mode_var,
                             state="readonly", font=constants.FONT_BODY,
                             values=list(self._exec_mode_by_label))
        labels = {v: k for k, v in self._exec_mode_by_label.items()}
        combo.set(labels.get(self.executor.mode, "Off"))
        combo.pack(fill=tk.X, pady=(0, 4))
        combo.bind("<<ComboboxSelected>>", lambda e: self._set_executor_mode())
        ToolTip(combo,
                "Ghost: computes and draws the click it WOULD make, clicks "
                "nothing, logs the match rate. Assist: a single F8 press "
                "fires the planned click — arm only after a long, "
                "near-perfect ghost sample.")
        self.arm_btn = self._button(
            section, "⚪  ARM", self._toggle_arm, C["text_secondary"],
            tooltip="Master safety. Off every session start; auto-disarms "
                    "on any guard breach. F9 = kill switch, mouse in a "
                    "screen corner = abort.")
        self.arm_btn.pack(fill=tk.X, pady=4)
        self.arm_btn.config(state="disabled")
        self.exec_status_var = tk.StringVar(value="Off — advisory only.")
        tk.Label(section, textvariable=self.exec_status_var,
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"], wraplength=240, justify="left"
                 ).pack(anchor="w")
        tk.Label(section, text="F8 confirm · F9 kill · corner = abort",
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"]).pack(anchor="w", pady=(2, 0))
        if self.executor.mode == "assist":
            # Restored from settings: hotkeys live, but NEVER armed at start.
            self.arm_btn.config(state="normal")
            self._start_hotkeys()

    def _set_executor_mode(self):
        mode = self._exec_mode_by_label.get(self.exec_mode_var.get(), "off")
        self.executor.set_mode(mode)
        self.controller.engine.persist_settings_async()
        self.arm_btn.config(state="normal" if mode == "assist" else "disabled")
        if mode == "assist":
            self._start_hotkeys()
        else:
            self._stop_hotkeys()
            self._hide_marker()
        if mode == "off":
            self.exec_status_var.set("Off — advisory only.")

    def _refresh_executor_ui(self):
        """Immediate executor display sync — the master safety indicator
        must reflect arm/disarm THE MOMENT it changes, not at the next
        snapshot poll."""
        self._render_executor(
            self.executor.step(self.controller.engine.get_snapshot()))

    def _toggle_arm(self):
        if self.executor.armed:
            self.executor.disarm("manual disarm")
            self._refresh_executor_ui()
            return
        if not self.controller.running:
            self.exec_status_var.set("Cannot arm — start detection first.")
            return
        stats = self.executor.stats() or {}
        rate = stats.get("match_rate")
        sample = (f"{stats.get('judged', 0)} judged plans, "
                  + (f"{rate:.0%} verified" if rate is not None
                     else "no verified sample yet"))
        if not messagebox.askyesno(
                "ARM executor",
                "Assist mode clicks REAL buttons with REAL money when you "
                f"press F8.\n\nGhost telemetry this session: {sample}.\n"
                "AUTONOMY_PLAN §6: arm only after a long ghost sample with "
                "a near-perfect match rate.\n\nArm now?",
                icon="warning", parent=self):
            return
        self.executor.arm()
        self._refresh_executor_ui()

    # Hotkeys fire on a watcher thread; marshal to the Tk thread.

    def _start_hotkeys(self):
        if self._hotkeys is not None and self._hotkeys.running:
            return
        from ..logic.hotkeys import GlobalHotkeys
        self._hotkeys = GlobalHotkeys({
            int(constants.EXECUTOR.get("confirm_vk", 0x77)):
                lambda: self.after(0, self._hotkey_confirm),
            int(constants.EXECUTOR.get("kill_vk", 0x78)):
                lambda: self.after(0, self._hotkey_kill),
        })
        self._hotkeys.start()

    def _stop_hotkeys(self):
        if self._hotkeys is not None:
            self._hotkeys.stop()
            self._hotkeys = None

    def _hotkey_confirm(self):
        result = self.executor.confirm()
        self._refresh_executor_ui()
        self.exec_status_var.set(result)

    def _hotkey_kill(self):
        self.executor.kill("kill switch (F9)")
        self._refresh_executor_ui()
        self.exec_status_var.set("KILLED — disarmed (F9).")

    def _hide_marker(self):
        if self.ghost_marker is not None and self.ghost_marker.winfo_exists():
            self.ghost_marker.hide()

    def _render_executor(self, state):
        """Per-snapshot executor display sync (Tk thread)."""
        armed = state["armed"]
        text = "🔴  ARMED — F8 fires" if armed else "⚪  ARM"
        bg = C["danger"] if armed else C["text_secondary"]
        if self.arm_btn.cget("text") != text:
            self.arm_btn.config(text=text, bg=bg, activebackground=bg)
            self.arm_btn._base_bg = bg
        if state["mode"] != "off":
            status = state["status"] or "watching…"
            session = state["session"]
            if session["plans"]:
                status += (f"  ·  {session['plans']} plans, "
                           f"{session['fired']} fired, "
                           f"{session['verified']} verified")
            self.exec_status_var.set(status)
        marker = state.get("marker")
        if (marker is not None and self.monitor is not None
                and self.controller.running):
            if self.ghost_marker is None or not self.ghost_marker.winfo_exists():
                from .ghost_overlay import GhostMarker
                self.ghost_marker = GhostMarker(self)
            self.ghost_marker.show(self.monitor.x + marker["x"],
                                   self.monitor.y + marker["y"],
                                   marker["text"], marker["color"])
        else:
            self._hide_marker()

    def _build_counters_section(self, parent):
        section = self._section(parent, "Cards Seen (this shoe)")
        self.counter_vars = {}
        for key in COUNTER_KEYS:
            row = tk.Frame(section, bg=C["bg_primary"])
            row.pack(fill=tk.X, pady=1)
            label = "10 / J / Q / K" if key == "10" else key
            tk.Label(row, text=label, font=constants.FONT_BODY, width=11, anchor="w",
                     bg=C["bg_primary"], fg=C["text_primary"]).pack(side=tk.LEFT, padx=(8, 0))
            var = tk.StringVar(value="0")
            tk.Label(row, textvariable=var, font=constants.FONT_BODY_BOLD, width=4,
                     bg=C["bg_primary"], fg=C["success"]).pack(side=tk.LEFT)
            for text, delta, color in (("−", -1, C["danger"]), ("+", 1, C["success"])):
                tk.Button(row, text=text, width=2, relief="flat", cursor="hand2",
                          bg=color, fg="white", font=constants.FONT_BODY_BOLD,
                          command=lambda k=key, d=delta: self._adjust_counter(k, d)
                          ).pack(side=tk.LEFT, padx=2, pady=2)
            self.counter_vars[key] = var

    def _build_game_info_section(self, parent):
        section = self._section(parent, "Game Info")
        self.info_vars = {}
        for key, label in [("round", "Round"), ("phase", "Phase"),
                           ("running", "Running count"),
                           ("true", "True count"), ("decks", "Decks remaining"),
                           ("seen", "Cards seen"), ("bet", "Bet hint"),
                           ("behind", "Bet behind"), ("pnl", "Session P&L")]:
            row = tk.Frame(section, bg=C["bg_secondary"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=constants.FONT_BODY, width=14, anchor="w",
                     bg=C["bg_secondary"], fg=C["text_secondary"]).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            tk.Label(row, textvariable=var, font=constants.FONT_BODY_BOLD, anchor="w",
                     bg=C["bg_secondary"], fg=C["text_primary"], wraplength=150,
                     justify="left").pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.info_vars[key] = var

        # Editable bankroll feeds the fractional-Kelly bet suggestion.
        row = tk.Frame(section, bg=C["bg_secondary"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Bankroll (€)", font=constants.FONT_BODY, width=14,
                 anchor="w", bg=C["bg_secondary"], fg=C["text_secondary"]
                 ).pack(side=tk.LEFT)
        self.bankroll_var = tk.StringVar(value=f"{constants.BETTING['bankroll']:.10g}")
        entry = tk.Entry(row, textvariable=self.bankroll_var, width=10,
                         font=constants.FONT_BODY_BOLD)
        entry.pack(side=tk.LEFT)
        attach_numeric_entry(entry)
        entry.bind("<Return>", lambda e: self._set_bankroll())
        entry.bind("<FocusOut>", lambda e: self._set_bankroll())
        self.bankroll_entry = entry
        self._synced_bankroll = constants.BETTING["bankroll"]

        # The bet actually placed per owned seat — settlement converts the
        # round's units into euros with this (defaults to the base bet).
        row = tk.Frame(section, bg=C["bg_secondary"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Bet placed (€)", font=constants.FONT_BODY, width=14,
                 anchor="w", bg=C["bg_secondary"], fg=C["text_secondary"]
                 ).pack(side=tk.LEFT)
        self.bet_placed_var = tk.StringVar(value=f"{constants.BASE_BET:.10g}")
        bet_entry = tk.Entry(row, textvariable=self.bet_placed_var, width=10,
                             font=constants.FONT_BODY_BOLD)
        bet_entry.pack(side=tk.LEFT)
        attach_numeric_entry(bet_entry)
        bet_entry.bind("<Return>", lambda e: self._set_bet_placed())
        bet_entry.bind("<FocusOut>", lambda e: self._set_bet_placed())
        self.bet_entry = bet_entry
        self._synced_bet = float(constants.BASE_BET)
        tk.Label(section, text="Click a seat's name on the table to mark it as"
                               " yours — only owned seats settle into the P&L.",
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"], wraplength=240, justify="left"
                 ).pack(anchor="w", pady=(4, 0))

    def _build_sidebets_section(self, parent):
        """Live pre-deal EV per enabled side bet; green when the bet is +EV
        (with a Kelly-sized stake suggestion). The € entry is the stake you
        actually place per owned seat — settlement books it into the P&L.
        Rows exist for every known bet and show/hide with the settings."""
        section = self._section(parent, "Side Bets (EV per unit · € stake)")
        self.sidebet_rows = {}
        self.sidebet_stake_vars = {}
        for key, cfg in constants.SIDE_BETS.items():
            row = tk.Frame(section, bg=C["bg_secondary"])
            if cfg.get("enabled"):
                row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=cfg.get("label", key), font=constants.FONT_BODY,
                     width=13, anchor="w", bg=C["bg_secondary"],
                     fg=C["text_secondary"]).pack(side=tk.LEFT)
            stake_var = tk.StringVar(value=f"{cfg.get('stake') or 0:.10g}")
            stake = tk.Entry(row, textvariable=stake_var, width=5,
                             font=constants.FONT_BODY)
            stake.pack(side=tk.RIGHT)
            attach_numeric_entry(stake)
            stake.bind("<Return>", lambda e, k=key: self._set_side_bet_stake(k))
            stake.bind("<FocusOut>", lambda e, k=key: self._set_side_bet_stake(k))
            ToolTip(stake, "Your stake on this bet per owned seat "
                           "(0 = not playing it)")
            self.sidebet_stake_vars[key] = stake_var
            var = tk.StringVar(value="—")
            lbl = tk.Label(row, textvariable=var, font=constants.FONT_BODY_BOLD,
                           anchor="w", bg=C["bg_secondary"], fg=C["text_primary"])
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sidebet_rows[key] = (row, var, lbl)

    def _set_side_bet_stake(self, key):
        var = self.sidebet_stake_vars[key]
        try:
            value = float(var.get().replace(",", "."))
        except ValueError:
            var.set(f"{constants.SIDE_BETS[key].get('stake') or 0:.10g}")
            return
        if value > 10_000_000:
            value = 10_000_000.0
            var.set(f"{value:.10g}")
        if value >= 0:
            self.controller.engine.set_side_bet_stake(key, value)

    # ------------------------------------------------------------ table/status

    def _build_table(self):
        wrapper = tk.Frame(self, bg=C["bg_primary"])
        wrapper.grid(row=1, column=1, sticky="nsew", padx=12, pady=12)
        wrapper.rowconfigure(0, weight=1)
        wrapper.columnconfigure(0, weight=1)
        self.table = TableView(wrapper, self._on_card_click, self._on_dealer_click,
                               self._on_split_click, self._on_seat_name_click,
                               self._on_dealer_extra_click,
                               on_advice_click=self._on_advice_click)
        self.table.canvas.grid(row=0, column=0, sticky="nsew")
        # Clicking the felt parks keyboard focus back on the window — an
        # Entry otherwise keeps focus for the whole session after one click
        # (labels/canvas never take it), which froze the money-entry sync.
        self.table.canvas.bind("<Button-1>", lambda e: self.focus_set(), add="+")

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=C["bg_secondary"],
                       highlightbackground=C["border"], highlightthickness=1, height=34)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.status_var, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"], anchor="w", padx=12
                 ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.metrics_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.metrics_var, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"], anchor="e", padx=12
                 ).pack(side=tk.RIGHT)

        # Reshuffle badge + mid-shoe paytable warning: created once here,
        # shown/hidden by _render on every snapshot poll (idempotent).
        self.paytable_warning = tk.Label(
            bar, text="⚠ Paytable changed mid-shoe", font=constants.FONT_SMALL,
            bg=C["bg_secondary"], fg=C["danger"], padx=8)
        self.reshuffle_badge = tk.Frame(bar, bg=C["warning"])
        tk.Label(self.reshuffle_badge, text="♻ Reshuffle detected",
                 font=constants.FONT_SMALL, bg=C["warning"], fg=C["bg_primary"],
                 padx=6).pack(side=tk.LEFT)
        tk.Button(self.reshuffle_badge, text="×", command=self._dismiss_reshuffle,
                  bg=C["warning"], fg=C["bg_primary"], relief="flat", bd=0,
                  cursor="hand2", padx=4, font=constants.FONT_BODY_BOLD
                  ).pack(side=tk.LEFT)
        # Anchor-drift badge (V3 E3): calibration may be misaligned; the
        # button re-solves the transform on the next worker cycle.
        self.anchor_badge = tk.Frame(bar, bg=C["danger"])
        tk.Label(self.anchor_badge, text="⚓ Anchor drift",
                 font=constants.FONT_SMALL, bg=C["danger"], fg="white",
                 padx=6).pack(side=tk.LEFT)
        tk.Button(self.anchor_badge, text="Re-anchor",
                  command=self._reanchor, bg=C["danger"], fg="white",
                  relief="flat", bd=0, cursor="hand2", padx=6,
                  font=constants.FONT_SMALL).pack(side=tk.LEFT)

    def _dismiss_reshuffle(self):
        self.controller.engine.dismiss_reshuffle_badge()

    def _reanchor(self):
        self.controller.engine.request_anchor_resolve()
        self.set_status("Re-anchoring on the next detection cycle…")

    def set_status(self, message, error=False):
        self.status_var.set(message)

    # --------------------------------------------------------------- actions

    def _auto_select_monitor(self):
        if self._monitors:
            primary = next((i for i, m in enumerate(self._monitors)
                            if getattr(m, "is_primary", False)), 0)
            self.monitor_combo.current(primary)
            self._confirm_monitor()

    def _confirm_monitor(self):
        idx = self.monitor_combo.current()
        if idx < 0 or idx >= len(self._monitors):
            return
        self.monitor = self._monitors[idx]
        self.controller.set_monitor(self.monitor)
        status = f"Using monitor {idx + 1} — {self.monitor.width}x{self.monitor.height}"
        from ..logic.ocr import OCR_AVAILABLE
        if (OCR_AVAILABLE and constants.OCR.get("enabled")
                and not self.controller.engine.ocr_calibrated):
            # The region file is keyed by exact resolution — without this
            # hint a monitor switch silently disables balance/bet OCR.
            status += " · OCR not calibrated for this resolution"
        self.monitor_status.config(text=status, fg=C["success"])
        self.start_btn.config(state="normal")
        self.regions_btn.config(state="normal")
        self.calibrate_btn.config(state="normal")
        self.ocr_btn.config(state="normal")
        self.controls_btn.config(state="normal")
        self.anchors_btn.config(state="normal")
        self.set_status(f"Monitor {idx + 1} confirmed ({self.monitor.width}x{self.monitor.height}).")

    def _refresh_profiles(self):
        """Rebuild the table-profile dropdown — '<name> — <saved date>'."""
        from ..logic import region_profiles
        active = region_profiles.active_name()
        labels, self._profile_by_label, current = [], {}, 0
        for i, item in enumerate(region_profiles.list_profiles()):
            date = (item["saved"] or "")[:10]
            label = f"{item['name']} — {date}" if date else item["name"]
            labels.append(label)
            self._profile_by_label[label] = item["name"]
            if item["name"] == active:
                current = i
        self.profile_combo["values"] = labels
        self.profile_combo.current(current)

    def _select_profile(self):
        from ..logic import region_profiles
        name = self._profile_by_label.get(self.profile_var.get())
        if not name or not region_profiles.set_active(name):
            return
        if self.monitor is not None:
            # Reloads seat/dealer/OCR regions and refreshes the OCR hint.
            self._confirm_monitor()
        self.set_status(f"Table profile \"{name}\" active — regions reloaded.")

    def _toggle_detection(self):
        if self.controller.running:
            self.controller.stop()
            # No fresh snapshots while stopped — acting on a stale phase
            # would be a blind click.
            self.executor.disarm("detection stopped")
            self._hide_marker()
            self.start_btn.config(text="▶  Start Detection", bg=C["success"],
                                  activebackground=C["success_hover"])
            self.start_btn._base_bg = C["success"]
            self.monitor_combo.config(state="readonly")
            self.profile_combo.config(state="readonly")
            self.calibrate_btn.config(state="normal")
            self.ocr_btn.config(state="normal")
            self.controls_btn.config(state="normal")
            self.anchors_btn.config(state="normal")
            self.set_status("Detection stopped.")
            return
        if self.monitor is None:
            messagebox.showerror("No monitor", "Confirm a monitor first.")
            return
        self.controller.start()
        self.start_btn.config(text="■  Stop Detection", bg=C["danger"],
                              activebackground=C["danger_hover"])
        self.start_btn._base_bg = C["danger"]
        # Switching monitors or table profiles mid-run would race the
        # in-flight capture cycle — and the calibrate editors can switch
        # the profile too (Save as profile), so they lock with it.
        self.monitor_combo.config(state="disabled")
        self.profile_combo.config(state="disabled")
        self.calibrate_btn.config(state="disabled")
        self.ocr_btn.config(state="disabled")
        self.controls_btn.config(state="disabled")
        self.anchors_btn.config(state="disabled")
        self.set_status("Detection starting — initializing models...")

    def _new_round(self):
        self.controller.engine.new_round()
        self.set_status("Round reset.")

    def _new_shoe(self):
        if messagebox.askyesno("New shoe", "Reset the running count and all per-card totals?"):
            self.controller.engine.reset_shoe()
            self.set_status("Shoe counts reset.")

    def _adjust_counter(self, key, delta):
        self.controller.engine.adjust_counter(key, delta)

    def _on_card_click(self, seat_idx, slot):
        snapshot = self.controller.engine.get_snapshot()
        seat_snap = snapshot["seats"][seat_idx] if snapshot else {"cards": []}
        n_cards = len(seat_snap["cards"])
        round_at_click = snapshot["round"] if snapshot else None
        slot = min(slot, n_cards)  # 99 from the "+" button -> append
        verb = "Replace" if slot < n_cards else "Add"
        split_info = None
        if slot >= n_cards and seat_snap.get("split"):
            # Adding to a split seat: let the user pick the target hand
            # instead of the engine's count-balancing fallback. Replacing
            # (slot < n_cards) keeps the card's existing hand tag, and
            # removal is slot-based — neither needs a selector.
            hand_of = seat_snap.get("hand_of", [])
            split_info = {"is_split": True,
                          "hands": [[n for n, h in zip(seat_snap["cards"], hand_of)
                                     if h == target] for target in (0, 1)]}
        CardPicker(self, f"Player {seat_idx + 1} — {verb} card {slot + 1}",
                   self.table.card_image,
                   lambda name, hand_idx=None: self.controller.engine.replace_card(
                       seat_idx, slot, name, expected_round=round_at_click,
                       hand_index=hand_idx),
                   allow_remove=slot < n_cards, split_info=split_info)

    def _on_dealer_click(self):
        CardPicker(self, "Dealer up-card", self.table.card_image,
                   lambda name: self.controller.engine.replace_dealer(name))

    def _on_dealer_extra_click(self, idx):
        """Correct a tracked dealer draw, or add one the detector missed
        (idx 99 comes from the '+' button). The round captured at picker
        open guards against a stale pick landing after an auto reset."""
        snap = self.controller.engine.get_snapshot()
        extras = (snap["dealer"].get("extras") or []) if snap else []
        round_at_click = snap["round"] if snap else None
        if idx >= len(extras):
            CardPicker(self, "Dealer draw — add missed card",
                       self.table.card_image,
                       lambda name: self.controller.engine.add_dealer_extra(
                           name, expected_round=round_at_click),
                       allow_remove=False)
        else:
            CardPicker(self, f"Dealer draw {idx + 1} — correct card",
                       self.table.card_image,
                       lambda name: self.controller.engine.replace_dealer_extra(
                           idx, name, expected_round=round_at_click),
                       allow_remove=True)

    def _preview_regions(self):
        if self._preview_busy or self.monitor is None:
            return
        self._preview_busy = True
        self.regions_btn.config(state="disabled")
        self.set_status("Generating region preview...")

        def work():
            try:
                image = build_region_preview(self.controller.engine)
                self.after(0, lambda: self._show_preview(image))
            except Exception as e:
                self.after(0, lambda e=e: self._preview_failed(e))

        threading.Thread(target=work, daemon=True).start()

    def _show_preview(self, image):
        self._preview_busy = False
        self.regions_btn.config(state="normal")
        self.table.show_preview(image, on_close=lambda: self.set_status("Preview closed."))
        self.set_status("Region preview ready — cards and seat regions are outlined.")

    def _preview_failed(self, error):
        self._preview_busy = False
        self.regions_btn.config(state="normal")
        self.set_status(f"Region preview failed: {error}", error=True)

    def _open_logs(self):
        if getattr(self, "logging_window", None) is not None and self.logging_window.winfo_exists():
            self.logging_window.lift()
            return
        from .logging_window import LoggingWindow
        if not self.log_manager:
            messagebox.showerror("Logs", "Logging system not initialized.")
            return
        self.logging_window = LoggingWindow(self, self.log_manager)

    # --------------------------------------------------------------- polling

    def _poll_snapshot(self):
        try:
            snapshot = self.controller.engine.get_snapshot()
            if snapshot is not None and snapshot["seq"] != self._last_seq:
                self._last_seq = snapshot["seq"]
                self._render(snapshot)
            # The executor ticks EVERY poll, not just on new snapshots —
            # its staleness disarm and verification deadlines must keep
            # running when the worker stalls and snapshots stop flowing.
            self._render_executor(self.executor.step(snapshot))
            self._sync_start_button()
        except Exception as e:
            # A render error must never kill the update loop.
            print(f"Error rendering snapshot: {e}")
        self.after(constants.SNAPSHOT_POLL_MS, self._poll_snapshot)

    def _sync_start_button(self):
        """Keep the Start/Stop button truthful if the worker dies (model error)."""
        showing_stop = self.start_btn.cget("text").startswith("■")
        if showing_stop and not self.controller.running:
            self.executor.disarm("detection worker stopped")
            self._hide_marker()
            self.start_btn.config(text="▶  Start Detection", bg=C["success"],
                                  activebackground=C["success_hover"])
            self.start_btn._base_bg = C["success"]
            self.monitor_combo.config(state="readonly")
            self.profile_combo.config(state="readonly")
            self.calibrate_btn.config(state="normal")
            self.ocr_btn.config(state="normal")
            self.controls_btn.config(state="normal")
            self.anchors_btn.config(state="normal")

    def _render(self, snap):
        self._sync_money_entries(snap)
        self.table.remember_snapshot(snap)
        self.table.update(snap)
        hud = getattr(self, "hud", None)
        if hud is not None and hud.winfo_exists():
            hud.update_from_snapshot(snap)

        count = snap["count"]
        for key, var in self.counter_vars.items():
            var.set(str(count["per_rank"].get(key, 0)))

        tc = count["true"]
        self.info_vars["round"].set(str(snap["round"]))
        phase_state = snap.get("phase") or {}
        phase_text = PHASE_LABELS.get(phase_state.get("phase"),
                                      phase_state.get("phase") or "—")
        if phase_state.get("timer_s") is not None:
            phase_text += f" · {phase_state['timer_s']}s"
        disc = phase_state.get("discipline") or {}
        if disc.get("checked"):
            phase_text += f" · played book {disc['matched']}/{disc['checked']}"
        if phase_state.get("triage"):
            phase_text += " ⚠"
        self.info_vars["phase"].set(phase_text)
        self.info_vars["running"].set(f"{count['running']:+d}")
        self.info_vars["true"].set(f"{tc:+.1f}")
        self.info_vars["decks"].set(f"{count['decks_remaining']:.1f}")
        self.info_vars["seen"].set(str(count["cards_seen"]))
        self.info_vars["bet"].set(snap["bet"])
        self.info_vars["behind"].set(snap.get("bet_behind", "—"))
        pnl = snap.get("session_pnl") or {}
        if pnl.get("rounds"):
            text = f"€{pnl['eur']:+.2f} ({pnl['units']:+g}u, {pnl['rounds']} rounds)"
            if pnl.get("side_eur"):
                text += f" · side €{pnl['side_eur']:+.2f}"
            self.info_vars["pnl"].set(text)
        else:
            self.info_vars["pnl"].set("— mark a seat as yours")

        live = {item["key"]: item for item in snap.get("side_bets", [])}
        for key, (row, var, lbl) in self.sidebet_rows.items():
            item = live.get(key)
            if item is None:
                row.pack_forget()
                continue
            if not row.winfo_ismapped():
                row.pack(fill=tk.X, pady=2)
            ev = item["ev"]
            if ev is None:
                var.set("—")
                lbl.config(fg=C["text_secondary"])
            else:
                text = f"{ev * 100:+.2f}%"
                if ev > 0:
                    text += " ●"
                    suggested = item.get("stake_suggested") or 0
                    if suggested:  # Kelly-sized: tiny by construction
                        text += f" €{suggested:g}"
                var.set(text)
                lbl.config(fg=C["success"] if ev > 0 else C["text_secondary"])

        if (snap.get("anchors") or {}).get("drift"):
            if not self.anchor_badge.winfo_ismapped():
                self.anchor_badge.pack(side=tk.RIGHT, padx=(0, 10))
        else:
            self.anchor_badge.pack_forget()
        if snap.get("reshuffle_badge"):
            if not self.reshuffle_badge.winfo_ismapped():
                self.reshuffle_badge.pack(side=tk.RIGHT, padx=(0, 10))
        else:
            self.reshuffle_badge.pack_forget()
        if snap.get("paytable_changed_midshoe"):
            if not self.paytable_warning.winfo_ismapped():
                self.paytable_warning.pack(side=tk.RIGHT)
        else:
            self.paytable_warning.pack_forget()

        if snap["error"]:
            self.set_status(f"⚠ {snap['error']}", error=True)
        elif self.controller.running:
            activity = {"dealing": "Dealing — watching for cards",
                        "complete": "Round complete",
                        "waiting": "Waiting for cards"}.get(snap["activity"], "")
            extra = " · cutting card seen" if snap["cutting_card_seen"] else ""
            self.set_status(f"{activity}{extra}")

        m = snap["metrics"]
        if self.controller.running and m["cycle_ms"]:
            skipped = " · idle (frame unchanged)" if m["skipped"] else ""
            text = (f"{snap['backend']} · cycle {m['cycle_ms']:.0f} ms"
                    f" · inference {m['inference_ms']:.0f} ms{skipped}")
            # OCR liveness: makes "the bankroll isn't updating" diagnosable
            # at a glance — off (no regions for this monitor), waiting
            # (enabled but nothing read yet), or seconds since last read.
            if constants.OCR.get("enabled"):
                ocr_info = snap.get("ocr")
                if ocr_info is None:
                    text += " · OCR off"
                elif ocr_info.get("ts"):
                    text += f" · OCR {max(0.0, time.time() - ocr_info['ts']):.0f}s"
                else:
                    text += " · OCR waiting"
            self.metrics_var.set(text)
        elif not self.controller.running:
            self.metrics_var.set("")

    def _sync_money_entries(self, snap):
        """Reflect engine-side bankroll / bet-placed (OCR sync, auto-settle)
        into the sidebar entries without fighting the user: skip an entry
        only while it is ACTIVELY being edited, and write only when the
        engine value moved since the last sync so a manual edit is never
        clobbered by an unchanged engine value. Setting a StringVar on an
        Entry fires no FocusOut, so this can't loop the commit handlers."""
        bankroll = snap.get("bankroll")
        if (bankroll is not None and bankroll != self._synced_bankroll
                and not self._entry_editing(self.bankroll_entry,
                                            self.bankroll_var,
                                            self._synced_bankroll)):
            self._synced_bankroll = bankroll
            self.bankroll_var.set(f"{bankroll:.10g}")
        bet = snap.get("bet_placed")
        if (bet is not None and bet != self._synced_bet
                and not self._entry_editing(self.bet_entry,
                                            self.bet_placed_var,
                                            self._synced_bet)):
            self._synced_bet = bet
            self.bet_placed_var.set(f"{bet:.10g}")

    def _entry_editing(self, entry, var, committed):
        """True only while the user is actively typing in the entry: it has
        keyboard focus AND its text no longer parses to the committed value.
        Focus alone must not block the sync — one click into an entry parks
        keyboard focus there for the rest of the session (canvas/labels
        never reclaim it), which froze the OCR sync the moment the user
        ever touched the field."""
        try:
            if self.focus_get() is not entry:
                return False
        except (KeyError, tk.TclError):  # combobox popdowns confuse focus_get
            return False
        try:
            return float(var.get().replace(",", ".")) != committed
        except ValueError:
            return True  # mid-edit text ("", "12.") — definitely typing

    def _open_settings(self):
        from .settings_dialog import SettingsDialog
        SettingsDialog(self, on_apply=self.controller.engine.refresh_settings)

    def _on_split_click(self, seat_idx, currently_split):
        self.controller.engine.set_split(seat_idx, not currently_split)

    def _on_seat_name_click(self, seat_idx):
        self.controller.engine.set_my_seat(seat_idx)

    def _set_bet_placed(self):
        try:
            value = float(self.bet_placed_var.get().replace(",", "."))
        except ValueError:
            self.bet_placed_var.set(f"{self.controller.engine.bet_placed:.10g}")
            self._synced_bet = self.controller.engine.bet_placed
            return
        # Same bound as the bankroll: keeps every later :.10g reset free of
        # scientific notation, which the keystroke validator would reject.
        if value > 10_000_000:
            value = 10_000_000.0
            self.bet_placed_var.set(f"{value:.10g}")
        if value >= 0:
            self._synced_bet = value
            self.controller.engine.set_bet_placed(value)

    def _open_stats(self):
        store = self.controller.engine.store
        if store is None:
            self.set_status("Session store unavailable.", error=True)
            return
        from .stats_window import StatsWindow
        StatsWindow(self, store, engine=self.controller.engine)

    def _open_bankroll(self):
        from .bankroll_window import BankrollWindow
        BankrollWindow(self, self.controller.engine.store,
                       engine=self.controller.engine)

    def _toggle_hud(self):
        hud = getattr(self, "hud", None)
        if hud is not None and hud.winfo_exists():
            hud.close()
            return
        from .hud import OverlayHUD
        self.hud = OverlayHUD(self, on_close=self._on_hud_closed)
        self.hud_nav_btn.config(fg=C["accent"])
        snap = self.controller.engine.get_snapshot()
        if snap:
            self.hud.update_from_snapshot(snap)

    def _on_hud_closed(self):
        """Runs however the HUD goes away — drop the ref and the active tint."""
        self.hud = None
        self.hud_nav_btn.config(fg=C["text_primary"])

    def _open_trainer(self, deck=None):
        existing = getattr(self, "trainer_window", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
        else:
            from .trainer_window import TrainerWindow
            self.trainer_window = TrainerWindow(self,
                                                self.controller.engine.store)
        if deck:
            self.trainer_window.load_replay_deck(deck, label="leak drill")

    def _on_advice_click(self, seat_index):
        """Open the EV inspector for a seat's advice line (V3 F8)."""
        snap = self.controller.engine.get_snapshot()
        if not snap:
            return
        seat = next((s for s in snap["seats"] if s["index"] == seat_index),
                    None)
        if seat is None \
                or len([c for c in seat["cards"] if c and c != "-"]) < 2 \
                or not snap["dealer"]["card"]:
            self.set_status("Nothing to inspect yet — the seat needs two "
                            "cards and the dealer an up-card.")
            return
        from .inspector_window import InspectorWindow
        InspectorWindow(self, seat=seat, dealer=snap["dealer"]["card"],
                        per_rank=snap["count"].get("per_rank") or {},
                        deck_count=self.controller.engine.counter.deck_count)

    def _open_leaks(self):
        store = self.controller.engine.store
        if store is None:
            self.set_status("Session store unavailable.", error=True)
            return
        existing = getattr(self, "leaks_window", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return
        from .leaks_window import LeaksWindow
        self.leaks_window = LeaksWindow(
            self, store, open_trainer=lambda deck: self._open_trainer(deck))

    def _maybe_restore_shoe(self):
        """Offer to restore a recent mid-shoe count after a restart."""
        engine = self.controller.engine
        if engine.store is None:
            return
        try:
            info = engine.store.load_recent_shoe_state()
        except Exception:
            return
        if info is None:
            return
        state = info["state"]
        minutes = int(info["age_s"] / 60)
        if messagebox.askyesno(
                "Restore shoe?",
                f"A shoe from {minutes} min ago was found:\n"
                f"round {info['round_number']}, {state.get('cards_seen', 0)} cards seen, "
                f"running count {state.get('running_count', 0):+d}.\n\n"
                "Restore it? (Choose No after a shuffle.)"):
            engine.apply_shoe_state(info)
            self.set_status("Previous shoe restored — counts are live again.")

    def _set_bankroll(self):
        try:
            value = float(self.bankroll_var.get().replace(",", "."))
        except ValueError:
            self.bankroll_var.set(f"{constants.BETTING['bankroll']:.10g}")
            self._synced_bankroll = constants.BETTING["bankroll"]
            return
        # Keystrokes only filter syntax; the bounds clamp happens here.
        if value > 10_000_000:
            value = 10_000_000.0
            self.bankroll_var.set(f"{value:.10g}")
        if value > 0 and value != constants.BETTING["bankroll"]:
            constants.BETTING["bankroll"] = value
            self._synced_bankroll = value
            # Off the Tk thread: an AV-scanned profile dir can make even a
            # small JSON write hitch the UI when it fires on focus changes.
            self.controller.engine.persist_settings_async()
            self.controller.engine.publish_snapshot()
            self.set_status(f"Bankroll set to €{value:g} — bet ramp updated.")

    def _calibrate_regions(self):
        if self.monitor is None:
            return
        from .region_editor import RegionEditor

        def reload_regions(saved=True):
            # _confirm_monitor reloads all regions AND refreshes the
            # sidebar's "OCR not calibrated" hint (a save into a NEW
            # profile has no OCR rects yet).
            self._confirm_monitor()
            self._refresh_profiles()  # a new/updated profile (and its date)
            from ..logic import region_profiles
            name = region_profiles.active_name()
            if saved:
                self.set_status(f"Regions saved to profile \"{name}\" — "
                                "detection now uses the calibrated layout.")
            else:
                self.set_status(f"Calibration removed from profile \"{name}\""
                                " — using the shipped default layout.")

        try:
            RegionEditor(self, self.controller.engine.capture, on_save=reload_regions)
        except Exception as e:
            self.set_status(f"Calibration failed: {e}", error=True)

    def _calibrate_controls(self):
        if self.monitor is None:
            return
        from .control_capture import ControlCaptureEditor

        def reload_controls():
            # _confirm_monitor reloads EVERY calibration kind (seat/dealer
            # regions, OCR rects, phase templates) — a "Save as profile"
            # switches the active profile, and reloading only the templates
            # would leave the engine running mixed-profile calibration.
            self._confirm_monitor()
            self._refresh_profiles()  # control saves restamp the profile date
            self.set_status("Control templates saved — phase & turn "
                            "detection now watches the buttons.")

        try:
            ControlCaptureEditor(self, self.controller.engine.capture,
                                 on_save=reload_controls)
        except Exception as e:
            self.set_status(f"Control capture failed: {e}", error=True)

    def _calibrate_anchors(self):
        if self.monitor is None:
            return
        from .anchor_editor import AnchorEditor

        def reload_anchors():
            # Same full-reload chain as the other editors: set_monitor
            # queues an anchor re-solve for the next worker cycle.
            self._confirm_monitor()
            self._refresh_profiles()
            self.set_status("Anchors saved — the transform is solved on the "
                            "next detection cycle.")

        AnchorEditor(self, self.controller.engine.capture,
                     on_save=reload_anchors)

    def _calibrate_ocr(self):
        if self.monitor is None:
            return
        from ..logic.ocr import OCR_AVAILABLE
        if not OCR_AVAILABLE:
            self.set_status("OCR needs the winocr package: pip install winocr",
                            error=True)
            return
        from .ocr_region_editor import OcrRegionEditor

        def reload_regions():
            self._confirm_monitor()   # reload + refresh the OCR hint
            self._refresh_profiles()  # OCR saves restamp the profile date
            self.set_status("OCR regions saved — balance/bet/result now read "
                            "from the screen.")

        try:
            OcrRegionEditor(self, self.controller.engine.capture, on_save=reload_regions)
        except Exception as e:
            self.set_status(f"OCR calibration failed: {e}", error=True)

    def _on_close(self):
        self._stop_hotkeys()
        self.executor.disarm("app closing")
        self.controller.stop()
        # OCR bankroll syncs persist at round end; a close mid-round must
        # not lose the latest value.
        try:
            from ..common import settings
            settings.save()
        except Exception:
            pass
        self.destroy()

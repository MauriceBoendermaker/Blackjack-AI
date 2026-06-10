"""Main application window.

Architecture: the window never talks to models or the screen directly. A
DetectionController runs the engine on a worker thread; this GUI polls
`engine.get_snapshot()` from a Tk `after()` loop and renders the result.
Every user action calls a thread-safe engine method. No Tkinter object is
ever touched from the worker thread.
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import screeninfo

from ..common import constants
from ..logic.background import DetectionController
from ..logic.counting import COUNTER_KEYS
from ..logic.region_preview import build_region_preview
from .card_picker import CardPicker
from .table_view import TableView
from .tooltip import ToolTip

C = constants.COLORS


class ModernBlackjackGUI(tk.Tk):
    def __init__(self, log_manager=None):
        super().__init__()
        self.title(constants.TITLE)
        self.geometry("1500x950")
        self.minsize(1150, 760)
        self.configure(bg=C["bg_primary"])

        self.log_manager = log_manager
        self.controller = DetectionController(log=print)
        self.monitor = None
        self._last_seq = -1
        self._preview_busy = False

        self.columnconfigure(0, weight=0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_table()
        self._build_status_bar()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(constants.SNAPSHOT_POLL_MS, self._poll_snapshot)
        self.after(300, lambda: self.table.preload_images())
        self.after(150, self._auto_select_monitor)
        self.after(600, self._maybe_restore_shoe)
        self.set_status("Ready — confirm a monitor, then press Start Detection.")

    # ------------------------------------------------------------ left panel

    def _build_left_panel(self):
        left = tk.Frame(self, bg=C["bg_secondary"],
                        highlightbackground=C["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew")

        tk.Label(left, text="Blackjack AI", font=constants.FONT_TITLE,
                 bg=C["bg_secondary"], fg=C["text_primary"]).pack(pady=(18, 10))

        canvas = tk.Canvas(left, bg=C["bg_secondary"], highlightthickness=0, width=280)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C["bg_secondary"])
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(inner_id, width=e.width))

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
        scrollbar.pack(side="right", fill="y")

        self._build_monitor_section(inner)
        self._build_controls_section(inner)
        self._build_counters_section(inner)
        self._build_game_info_section(inner)
        self._build_sidebets_section(inner)

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
            tooltip="Mark the balance / bet / result areas to read from the screen")
        self.ocr_btn.pack(fill=tk.X, pady=4)
        self.ocr_btn.config(state="disabled")
        self._button(section, "🗒  View Logs", self._open_logs, C["text_secondary"],
                     tooltip="Open the live log window").pack(fill=tk.X, pady=4)
        self._button(section, "⚙  Settings", self._open_settings, C["text_secondary"],
                     tooltip="Table rules, deck count, side-bet paytables"
                     ).pack(fill=tk.X, pady=4)
        self._button(section, "📊  Session Stats", self._open_stats, C["text_secondary"],
                     tooltip="Round history, count distribution, CSV export"
                     ).pack(fill=tk.X, pady=4)
        self._button(section, "🛡  Bankroll & Risk", self._open_bankroll, C["text_secondary"],
                     tooltip="Risk of ruin, Kelly risk table, Monte Carlo simulation"
                     ).pack(fill=tk.X, pady=4)
        self._button(section, "🎯  Overlay HUD", self._toggle_hud, C["text_secondary"],
                     tooltip="Compact always-on-top panel to park next to the stream"
                     ).pack(fill=tk.X, pady=4)

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
        for key, label in [("round", "Round"), ("running", "Running count"),
                           ("true", "True count"), ("decks", "Decks remaining"),
                           ("seen", "Cards seen"), ("bet", "Bet hint"),
                           ("pnl", "Session P&L")]:
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
        self.bankroll_var = tk.StringVar(value=f"{constants.BETTING['bankroll']:g}")
        entry = tk.Entry(row, textvariable=self.bankroll_var, width=10,
                         font=constants.FONT_BODY_BOLD)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", lambda e: self._set_bankroll())
        entry.bind("<FocusOut>", lambda e: self._set_bankroll())

        # The bet actually placed per owned seat — settlement converts the
        # round's units into euros with this (defaults to the base bet).
        row = tk.Frame(section, bg=C["bg_secondary"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Bet placed (€)", font=constants.FONT_BODY, width=14,
                 anchor="w", bg=C["bg_secondary"], fg=C["text_secondary"]
                 ).pack(side=tk.LEFT)
        self.bet_placed_var = tk.StringVar(value=f"{constants.BASE_BET:g}")
        bet_entry = tk.Entry(row, textvariable=self.bet_placed_var, width=10,
                             font=constants.FONT_BODY_BOLD)
        bet_entry.pack(side=tk.LEFT)
        bet_entry.bind("<Return>", lambda e: self._set_bet_placed())
        bet_entry.bind("<FocusOut>", lambda e: self._set_bet_placed())
        tk.Label(section, text="Click a seat's name on the table to mark it as"
                               " yours — only owned seats settle into the P&L.",
                 font=constants.FONT_SMALL, bg=C["bg_secondary"],
                 fg=C["text_secondary"], wraplength=240, justify="left"
                 ).pack(anchor="w", pady=(4, 0))

    def _build_sidebets_section(self, parent):
        """Live pre-deal EV per enabled side bet; green when the bet is +EV.
        Rows exist for every known bet and show/hide with the settings."""
        section = self._section(parent, "Side Bets (EV per unit)")
        self.sidebet_rows = {}
        for key, cfg in constants.SIDE_BETS.items():
            row = tk.Frame(section, bg=C["bg_secondary"])
            if cfg.get("enabled"):
                row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=cfg.get("label", key), font=constants.FONT_BODY,
                     width=14, anchor="w", bg=C["bg_secondary"],
                     fg=C["text_secondary"]).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            lbl = tk.Label(row, textvariable=var, font=constants.FONT_BODY_BOLD,
                           anchor="w", bg=C["bg_secondary"], fg=C["text_primary"])
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sidebet_rows[key] = (row, var, lbl)

    # ------------------------------------------------------------ table/status

    def _build_table(self):
        wrapper = tk.Frame(self, bg=C["bg_primary"])
        wrapper.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        wrapper.rowconfigure(0, weight=1)
        wrapper.columnconfigure(0, weight=1)
        self.table = TableView(wrapper, self._on_card_click, self._on_dealer_click,
                               self._on_split_click, self._on_seat_name_click)
        self.table.canvas.grid(row=0, column=0, sticky="nsew")

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=C["bg_secondary"],
                       highlightbackground=C["border"], highlightthickness=1, height=34)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.status_var, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"], anchor="w", padx=12
                 ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.metrics_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.metrics_var, font=constants.FONT_SMALL,
                 bg=C["bg_secondary"], fg=C["text_secondary"], anchor="e", padx=12
                 ).pack(side=tk.RIGHT)

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
        self.monitor_status.config(
            text=f"Using monitor {idx + 1} — {self.monitor.width}x{self.monitor.height}",
            fg=C["success"])
        self.start_btn.config(state="normal")
        self.regions_btn.config(state="normal")
        self.calibrate_btn.config(state="normal")
        self.ocr_btn.config(state="normal")
        self.set_status(f"Monitor {idx + 1} confirmed ({self.monitor.width}x{self.monitor.height}).")

    def _toggle_detection(self):
        if self.controller.running:
            self.controller.stop()
            self.start_btn.config(text="▶  Start Detection", bg=C["success"],
                                  activebackground=C["success_hover"])
            self.start_btn._base_bg = C["success"]
            self.monitor_combo.config(state="readonly")
            self.set_status("Detection stopped.")
            return
        if self.monitor is None:
            messagebox.showerror("No monitor", "Confirm a monitor first.")
            return
        self.controller.start()
        self.start_btn.config(text="■  Stop Detection", bg=C["danger"],
                              activebackground=C["danger_hover"])
        self.start_btn._base_bg = C["danger"]
        # Switching monitors mid-run would race the in-flight capture cycle.
        self.monitor_combo.config(state="disabled")
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
        n_cards = len(snapshot["seats"][seat_idx]["cards"]) if snapshot else 0
        round_at_click = snapshot["round"] if snapshot else None
        slot = min(slot, n_cards)  # 99 from the "+" button -> append
        verb = "Replace" if slot < n_cards else "Add"
        CardPicker(self, f"Player {seat_idx + 1} — {verb} card {slot + 1}",
                   self.table.card_image,
                   lambda name: self.controller.engine.replace_card(
                       seat_idx, slot, name, expected_round=round_at_click),
                   allow_remove=slot < n_cards)

    def _on_dealer_click(self):
        CardPicker(self, "Dealer up-card", self.table.card_image,
                   lambda name: self.controller.engine.replace_dealer(name))

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
            self._sync_start_button()
        except Exception as e:
            # A render error must never kill the update loop.
            print(f"Error rendering snapshot: {e}")
        self.after(constants.SNAPSHOT_POLL_MS, self._poll_snapshot)

    def _sync_start_button(self):
        """Keep the Start/Stop button truthful if the worker dies (model error)."""
        showing_stop = self.start_btn.cget("text").startswith("■")
        if showing_stop and not self.controller.running:
            self.start_btn.config(text="▶  Start Detection", bg=C["success"],
                                  activebackground=C["success_hover"])
            self.start_btn._base_bg = C["success"]
            self.monitor_combo.config(state="readonly")

    def _render(self, snap):
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
        self.info_vars["running"].set(f"{count['running']:+d}")
        self.info_vars["true"].set(f"{tc:+.1f}")
        self.info_vars["decks"].set(f"{count['decks_remaining']:.1f}")
        self.info_vars["seen"].set(str(count["cards_seen"]))
        self.info_vars["bet"].set(snap["bet"])
        pnl = snap.get("session_pnl") or {}
        if pnl.get("rounds"):
            self.info_vars["pnl"].set(
                f"€{pnl['eur']:+.2f} ({pnl['units']:+g}u, {pnl['rounds']} rounds)")
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
                var.set(f"{ev * 100:+.2f}%" + ("  ● BET" if ev > 0 else ""))
                lbl.config(fg=C["success"] if ev > 0 else C["text_secondary"])

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
            self.metrics_var.set(
                f"{snap['backend']} · cycle {m['cycle_ms']:.0f} ms"
                f" · inference {m['inference_ms']:.0f} ms{skipped}")
        elif not self.controller.running:
            self.metrics_var.set("")

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
            self.bet_placed_var.set(f"{self.controller.engine.bet_placed:g}")
            return
        if value >= 0:
            self.controller.engine.set_bet_placed(value)

    def _open_stats(self):
        store = self.controller.engine.store
        if store is None:
            self.set_status("Session store unavailable.", error=True)
            return
        from .stats_window import StatsWindow
        StatsWindow(self, store)

    def _open_bankroll(self):
        from .bankroll_window import BankrollWindow
        BankrollWindow(self, self.controller.engine.store)

    def _toggle_hud(self):
        hud = getattr(self, "hud", None)
        if hud is not None and hud.winfo_exists():
            hud.close()
            return
        from .hud import OverlayHUD
        self.hud = OverlayHUD(self, on_close=lambda: setattr(self, "hud", None))
        snap = self.controller.engine.get_snapshot()
        if snap:
            self.hud.update_from_snapshot(snap)

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
        from ..common import settings
        try:
            value = float(self.bankroll_var.get().replace(",", "."))
        except ValueError:
            self.bankroll_var.set(f"{constants.BETTING['bankroll']:g}")
            return
        if value > 0 and value != constants.BETTING["bankroll"]:
            constants.BETTING["bankroll"] = value
            settings.save()
            self.controller.engine.publish_snapshot()
            self.set_status(f"Bankroll set to €{value:g} — bet ramp updated.")

    def _calibrate_regions(self):
        if self.monitor is None:
            return
        from .region_editor import RegionEditor

        def reload_regions():
            self.controller.set_monitor(self.monitor)
            self.set_status("Regions saved — detection now uses the calibrated layout.")

        try:
            RegionEditor(self, self.controller.engine.capture, on_save=reload_regions)
        except Exception as e:
            self.set_status(f"Calibration failed: {e}", error=True)

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
            self.controller.set_monitor(self.monitor)
            self.set_status("OCR regions saved — balance/bet/result now read "
                            "from the screen.")

        try:
            OcrRegionEditor(self, self.controller.engine.capture, on_save=reload_regions)
        except Exception as e:
            self.set_status(f"OCR calibration failed: {e}", error=True)

    def _on_close(self):
        self.controller.stop()
        self.destroy()

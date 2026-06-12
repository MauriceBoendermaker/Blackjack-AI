"""Settings dialog: table rules / game & betting / side bets / app tabs.

Every EV in the app is conditional on these values, so they are editable per
table and persisted to output/settings.json. Saving applies the profile to
the live constants and tells the engine to drop its advice caches. The App
tab covers app-level tunables (UI scale, detection pacing, OCR sync) that
the engine/worker/GUI read at call time, so they too apply live.
"""

import tkinter as tk
from tkinter import ttk

from ..common import constants, settings
from . import scaling
from .validation import attach_numeric_entry

C = constants.COLORS


def notebook_style(widget) -> str:
    """Themed notebook tabs matching the dark dialogs; returns the style
    name. Like the sidebar scrollbar, the native Windows theme ignores ttk
    color options, so the tab/client elements come from clam. Style names
    are interpreter-global — every tabbed window (settings, bankroll)
    shares this one definition."""
    style = ttk.Style(widget)
    for element in ("client", "tab"):
        try:
            style.element_create(f"Settings.Notebook.{element}",
                                 "from", "clam", f"Notebook.{element}")
        except tk.TclError:
            pass  # dialog reopened in one interpreter: elements persist
    style.layout("Settings.TNotebook",
                 [("Settings.Notebook.client", {"sticky": "nswe"})])
    style.layout("Settings.TNotebook.Tab", [
        ("Settings.Notebook.tab", {"sticky": "nswe", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nswe",
                                  "children": [("Notebook.label",
                                                {"side": "top",
                                                 "sticky": ""})]})]})])
    style.configure("Settings.TNotebook", background=C["bg_secondary"],
                    bordercolor=C["border"], lightcolor=C["bg_secondary"],
                    darkcolor=C["bg_secondary"], tabmargins=(2, 4, 2, 0))
    style.configure("Settings.TNotebook.Tab", font=constants.FONT_BODY,
                    background=C["bg_primary"], foreground=C["text_secondary"],
                    bordercolor=C["border"], lightcolor=C["bg_primary"],
                    padding=(scaling.px(14), scaling.px(6)),
                    focuscolor=C["bg_secondary"])
    style.map("Settings.TNotebook.Tab",
              background=[("selected", C["bg_secondary"])],
              foreground=[("selected", C["text_primary"])])
    return "Settings.TNotebook"


_RULE_FIELDS = [
    # (key, label, kind, options/None, tooltip-ish hint shown after the field)
    ("s17", "Dealer soft 17", "choice", [("Stands (S17)", True), ("Hits (H17)", False)]),
    ("peek", "Hole card", "choice", [("No hole card (ENHC)", False), ("US peek", True)]),
    ("dealer_bj_takes", "Dealer BJ takes", "choice", [("All bets", "all"), ("Original bet only", "obo")]),
    ("das", "Double after split", "bool", None),
    ("double_on", "Double allowed on", "choice", [("Any two cards", "any"), ("9–11 only", "9-11"), ("10–11 only", "10-11")]),
    ("hit_split_aces", "Hit split aces", "bool", None),
    ("surrender", "Late surrender", "bool", None),
    ("bj_pays", "Blackjack pays", "choice", [("3:2", 1.5), ("6:5", 1.2)]),
]


class SettingsDialog(tk.Toplevel):
    """Modal editor for the active table profile."""

    def __init__(self, parent, on_apply=None):
        super().__init__(parent)
        self.title("Settings")
        self.configure(bg=C["bg_secondary"], padx=scaling.px(18),
                       pady=scaling.px(14))
        self.resizable(True, True)
        self.transient(parent)
        self.on_apply = on_apply
        self._vars = {}
        self._wrapped = []  # (label, 96-dpi wraplength) re-resolved on rescale

        # Bottom bar packs FIRST: if the window ever gets smaller than its
        # content, the notebook clips — never the Save/Cancel row.
        buttons = tk.Frame(self, bg=C["bg_secondary"])
        buttons.pack(side=tk.BOTTOM, fill=tk.X, pady=(scaling.px(8), 0))
        tk.Button(buttons, text="Save & Apply", command=self._save,
                  bg=C["success"], fg="white", relief="flat",
                  padx=scaling.px(16), pady=scaling.px(7),
                  font=constants.FONT_BODY, cursor="hand2"
                  ).pack(side=tk.RIGHT, padx=scaling.px(4))
        tk.Button(buttons, text="Cancel", command=self.destroy,
                  bg=C["text_secondary"], fg="white", relief="flat",
                  padx=scaling.px(16), pady=scaling.px(7),
                  font=constants.FONT_BODY, cursor="hand2"
                  ).pack(side=tk.RIGHT, padx=scaling.px(4))

        # Inline validation message — _save refuses instead of guessing.
        self.error_var = tk.StringVar(value="")
        error = tk.Label(self, textvariable=self.error_var, font=constants.FONT_SMALL,
                         bg=C["bg_secondary"], fg=C["danger"],
                         wraplength=scaling.px(360), justify="left")
        error.pack(side=tk.BOTTOM, anchor="w", pady=(scaling.px(8), 0))
        self._wrapped.append((error, 360))

        notebook = ttk.Notebook(self, style=self._notebook_style())
        notebook.pack(fill=tk.BOTH, expand=True)
        self._build_rules_tab(self._tab(notebook, "Table rules"))
        self._build_game_tab(self._tab(notebook, "Game & betting"))
        self._build_sidebets_tab(self._tab(notebook, "Side bets"))
        self._build_app_tab(self._tab(notebook, "App"))

        self._fit_to_monitor()
        # A DPI rescale while the dialog is open (the root moved monitors)
        # regrows the shared fonts; re-wrap and re-clamp so the Save row
        # can't end up off-screen. Detached on destroy — scaling keeps the
        # callback list for the process lifetime.
        scaling.on_change(self._on_rescale)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.grab_set()

    def _fit_to_monitor(self):
        """Open near the parent (the monitor the fonts were resolved for)
        and clamp to that monitor's work area only when the natural size
        would clip — an unconditional geometry() would freeze shrink-wrap.
        The position clamps keep the whole dialog (the bottom Save row
        included) inside the work area; ~45 px of headroom covers the WM
        caption that geometry's outer coordinates sit above."""
        self.update_idletasks()
        req_w, req_h = self.winfo_reqwidth(), self.winfo_reqheight()
        work_x, work_y, work_w, work_h = scaling.workarea(self.master)
        width = min(req_w, int(work_w * 0.9))
        height = min(req_h, int(work_h * 0.9))
        x = self.master.winfo_rootx() + scaling.px(60)
        y = self.master.winfo_rooty() + scaling.px(40)
        x = max(work_x, min(x, work_x + work_w - width))
        y = max(work_y, min(y, work_y + work_h - height - scaling.px(45)))
        if req_w > work_w * 0.9 or req_h > work_h * 0.9:
            self.geometry(f"{width}x{height}+{x}+{y}")
        else:
            self.geometry(f"+{x}+{y}")  # position only: keep shrink-wrap
        self.minsize(scaling.px(420), scaling.px(320))

    def _on_rescale(self):
        if not self.winfo_exists():
            return
        for label, base in self._wrapped:
            label.config(wraplength=scaling.px(base))
        self.geometry("")  # back to natural size under the new font scale
        self._fit_to_monitor()

    def _on_destroy(self, event):
        if event.widget is self:
            scaling.off_change(self._on_rescale)

    # --------------------------------------------------------------- tabs

    def _notebook_style(self):
        return notebook_style(self)

    def _tab(self, notebook, title):
        frame = tk.Frame(notebook, bg=C["bg_secondary"],
                         padx=scaling.px(14), pady=scaling.px(10))
        notebook.add(frame, text=title)
        return frame

    def _build_rules_tab(self, tab):
        row = self._heading(tab, 0, "Match the game help EXACTLY; the rules change every EV")
        for key, label, kind, options in _RULE_FIELDS:
            row = self._field(tab, row, key, label, kind, options, constants.RULES[key])

    def _build_game_tab(self, tab):
        row = self._heading(tab, 0, "Game")
        row = self._spin(tab, row, "deck_count", "Decks in shoe", 1, 8, constants.DECK_COUNT)
        row = self._spin(tab, row, "base_bet", "Base bet (€)", 1, 10_000, int(constants.BASE_BET))

        row = self._heading(tab, row, "Bet sizing (fractional Kelly)")
        row = self._field(tab, row, "betting:kelly_fraction", "Kelly fraction", "choice",
                          [("1/4 Kelly (safest)", 0.25), ("1/2 Kelly", 0.5),
                           ("Full Kelly", 1.0)],
                          constants.BETTING["kelly_fraction"])
        row = self._spin(tab, row, "betting:table_min", "Table minimum (€)", 1, 100_000,
                         int(constants.BETTING["table_min"]))
        row = self._spin(tab, row, "betting:table_max", "Table maximum (€, 0 = none)", 0,
                         1_000_000, int(constants.BETTING["table_max"]))
        row = self._field(tab, row, "betting:use_exact_edge", "Edge model", "choice",
                          [("Exact pre-deal EV (slower, honest)", 1),
                           ("Linear true-count estimate", 0)],
                          int(constants.BETTING.get("use_exact_edge", 1)))
        row = self._field(tab, row, "betting:auto_bankroll", "Bankroll updates", "choice",
                          [("Auto-settle owned seats", 1), ("Manual only", 0)],
                          int(constants.BETTING.get("auto_bankroll", 1)))

        note = tk.Label(tab, text="Deck count applies to new EV calculations immediately;\n"
                                  "reset the shoe after changing it mid-session.",
                        font=constants.FONT_SMALL, bg=C["bg_secondary"],
                        fg=C["text_secondary"], justify="left")
        note.grid(row=row, column=0, columnspan=2, sticky="w",
                  pady=(scaling.px(10), scaling.px(4)))

    def _build_sidebets_tab(self, tab):
        row = self._heading(tab, 0, "Side bets offered")
        for key, cfg in constants.SIDE_BETS.items():
            var = tk.BooleanVar(value=bool(cfg.get("enabled")))
            self._vars[f"sidebet:{key}"] = var
            tk.Checkbutton(tab, text=cfg.get("label", key), variable=var,
                           bg=C["bg_secondary"], fg=C["text_primary"],
                           font=constants.FONT_BODY, anchor="w",
                           activebackground=C["bg_secondary"]
                           ).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1

    def _build_app_tab(self, tab):
        row = self._heading(tab, 0, "Interface")
        row = self._field(tab, row, "ui:scale", "UI scale", "choice",
                          [("Auto (per-monitor)", 0), ("100%", 100),
                           ("125%", 125), ("150%", 150), ("175%", 175),
                           ("200%", 200)],
                          constants.UI["scale"])
        row = self._spin(tab, row, "app:SNAPSHOT_POLL_MS",
                         "Snapshot poll interval (ms)", 60, 500,
                         int(constants.SNAPSHOT_POLL_MS))

        row = self._heading(tab, row, "Detection & advice")
        row = self._field(tab, row, "app:DEALER_USE_PLAYER_MODEL",
                          "Dealer detection reads suits (52-class card model)",
                          "bool", None, bool(constants.DEALER_USE_PLAYER_MODEL))
        row = self._spin(tab, row, "app:EMPTY_FRAMES_FOR_RESET",
                         "Auto new-round after N empty frames", 3, 10,
                         int(constants.EMPTY_FRAMES_FOR_RESET))
        row = self._spin(tab, row, "app:CUTTING_CARD_CONFIRM_FRAMES",
                         "Cutting-card sightings to confirm", 1, 5,
                         int(constants.CUTTING_CARD_CONFIRM_FRAMES))
        row = self._spin(tab, row, "app:EV_ADVICE_TIMEOUT_S",
                         "EV advice timeout (s)", 1, 10,
                         int(constants.EV_ADVICE_TIMEOUT_S))
        row = self._spin(tab, row, "app:IDLE_REFRESH_GAP_S",
                         "Model refresh after idle (s)", 10, 600,
                         int(constants.IDLE_REFRESH_GAP_S))

        row = self._heading(tab, row, "Casino-UI OCR")
        row = self._field(tab, row, "ocr:enabled",
                          "Read balance/bet/result from screen", "bool", None,
                          constants.OCR["enabled"])
        row = self._field(tab, row, "ocr:sync_bankroll",
                          "Sync bankroll from screen", "bool", None,
                          constants.OCR["sync_bankroll"])
        row = self._field(tab, row, "ocr:sync_bet", "Sync bet from screen",
                          "bool", None, constants.OCR["sync_bet"])

        row = self._heading(tab, row, "Phase & turn detection")
        row = self._field(tab, row, "phase:enabled",
                          "Detect game phase (bets open / your turn)", "bool",
                          None, constants.PHASE["enabled"])

        row = self._heading(tab, row, "Executor limits (ghost / assist)")
        row = self._spin(tab, row, "executor:max_bet_eur",
                         "Max bet per action (€)", 0, 100_000,
                         int(constants.EXECUTOR["max_bet_eur"]))
        row = self._spin(tab, row, "executor:stop_loss_eur",
                         "Stop-loss: disarm beyond session −€", 0, 1_000_000,
                         int(constants.EXECUTOR["stop_loss_eur"]))
        row = self._spin(tab, row, "executor:stop_win_eur",
                         "Stop-win: disarm beyond session +€ (0 = off)", 0,
                         1_000_000, int(constants.EXECUTOR["stop_win_eur"]))
        row = self._entry(tab, row, "executor:chips",
                          "Table chips (comma-separated €)",
                          ", ".join(f"{c:g}" for c in
                                    constants.EXECUTOR["chips"]))
        row = self._field(tab, row, "executor:use_cdp",
                          "Click via CDP (needs playwright + Chrome "
                          "--remote-debugging-port)", "bool", None,
                          constants.EXECUTOR["use_cdp"])

        row = self._heading(tab, row, "Claude vision assist (optional)")
        row = self._field(tab, row, "vision:enabled",
                          "Enable (calibration suggestions + screen triage)",
                          "bool", None, constants.VISION["enabled"])
        row = self._field(tab, row, "vision:triage",
                          "Label unknown screens (modals, disconnects)",
                          "bool", None, constants.VISION["triage"])
        row = self._entry(tab, row, "vision:api_key", "Anthropic API key",
                          constants.VISION["api_key"], secret=True)
        self._entry(tab, row, "vision:model", "Vision model",
                    constants.VISION["model"])

    # ------------------------------------------------------------ widgets

    def _heading(self, parent, row, text):
        pad = (scaling.px(12), scaling.px(4)) if row else (0, scaling.px(4))
        label = tk.Label(parent, text=text, font=constants.FONT_SECTION,
                         bg=C["bg_secondary"], fg=C["text_primary"],
                         wraplength=scaling.px(360), justify="left")
        label.grid(row=row, column=0, columnspan=2, sticky="w", pady=pad)
        self._wrapped.append((label, 360))
        return row + 1

    def _field(self, parent, row, key, label, kind, options, current):
        if kind == "bool":
            var = tk.BooleanVar(value=bool(current))
            self._vars[key] = ("bool", var, None)
            tk.Checkbutton(parent, text=label, variable=var, bg=C["bg_secondary"],
                           fg=C["text_primary"], font=constants.FONT_BODY,
                           activebackground=C["bg_secondary"], anchor="w"
                           ).grid(row=row, column=0, columnspan=2, sticky="w")
            return row + 1
        labels = [text for text, _ in options]
        values = [value for _, value in options]
        var = tk.StringVar(value=labels[values.index(current)] if current in values else labels[0])
        self._vars[key] = ("choice", var, dict(zip(labels, values)))
        tk.Label(parent, text=label, font=constants.FONT_BODY, bg=C["bg_secondary"],
                 fg=C["text_secondary"], anchor="w").grid(row=row, column=0, sticky="w")
        ttk.Combobox(parent, textvariable=var, values=labels, state="readonly",
                     width=22).grid(row=row, column=1, sticky="w",
                                    padx=(scaling.px(10), 0), pady=scaling.px(2))
        return row + 1

    def _entry(self, parent, row, key, label, current, secret=False):
        """Free-text field (API key, model id). `secret` masks the input."""
        var = tk.StringVar(value=str(current or ""))
        self._vars[key] = ("str", var, None)
        tk.Label(parent, text=label, font=constants.FONT_BODY,
                 bg=C["bg_secondary"], fg=C["text_secondary"], anchor="w"
                 ).grid(row=row, column=0, sticky="w")
        tk.Entry(parent, textvariable=var, width=28, font=constants.FONT_BODY,
                 show="•" if secret else "").grid(
            row=row, column=1, sticky="w", padx=(scaling.px(10), 0),
            pady=scaling.px(2))
        return row + 1

    def _spin(self, parent, row, key, label, lo, hi, current):
        var = tk.IntVar(value=current)
        self._vars[key] = ("int", var, None)
        tk.Label(parent, text=label, font=constants.FONT_BODY, bg=C["bg_secondary"],
                 fg=C["text_secondary"], anchor="w").grid(row=row, column=0, sticky="w")
        spin = tk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=8)
        spin.grid(row=row, column=1, sticky="w",
                  padx=(scaling.px(10), 0), pady=scaling.px(2))
        attach_numeric_entry(spin, integer=True)
        return row + 1

    # -------------------------------------------------------------- save

    def _save(self):
        data = {"rules": {}, "side_bets": {}, "betting": {}, "ui": {},
                "app": {}, "ocr": {}, "phase": {}, "vision": {},
                "executor": {}}
        for key, spec in self._vars.items():
            if key.startswith("sidebet:"):
                data["side_bets"][key.split(":", 1)[1]] = {"enabled": bool(spec.get())}
                continue
            kind, var, mapping = spec
            try:
                value = mapping[var.get()] if kind == "choice" else var.get()
            except tk.TclError:  # numeric field left empty mid-edit
                self.error_var.set("Every numeric field needs a whole number.")
                return
            if key.startswith("betting:"):
                data["betting"][key.split(":", 1)[1]] = value
            elif key.startswith("app:"):
                data["app"][key.split(":", 1)[1]] = value
            elif key.startswith("ocr:"):
                data["ocr"][key.split(":", 1)[1]] = int(value)
            elif key.startswith("phase:"):
                data["phase"][key.split(":", 1)[1]] = int(value)
            elif key.startswith("vision:"):
                name = key.split(":", 1)[1]
                data["vision"][name] = (value if kind == "str"
                                        else int(value))
            elif key.startswith("executor:"):
                name = key.split(":", 1)[1]
                data["executor"][name] = (value if kind == "str"
                                          else int(value))
            elif key == "ui:scale":
                data["ui"]["scale"] = value
            elif key in ("deck_count", "base_bet"):
                data[key] = value
            else:
                data["rules"][key] = value
        if (data["betting"]["table_max"]  # 0 = no maximum
                and data["betting"]["table_min"] > data["betting"]["table_max"]):
            self.error_var.set("Table minimum exceeds table maximum — "
                               "fix the limits before saving.")
            return
        self.error_var.set("")
        old_pct = constants.UI["scale"]
        settings.apply(data)
        settings.save()
        if constants.UI["scale"] != old_pct:
            # Live rescale of the whole UI; the dialog's master is the root.
            scaling.apply_override(self.master, constants.UI["scale"])
        if self.on_apply:
            self.on_apply()
        self.destroy()

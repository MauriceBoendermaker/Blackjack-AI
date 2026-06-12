"""The game-table canvas: dealer card, seven seats in an arc, advice labels.

All widgets are created once and updated in place only when their content
changes (no per-cycle rebuild/re-bind churn). Geometry recomputes on resize
with a small debounce. Card images are cached per (name, size).
"""

import math
import tkinter as tk

from PIL import Image, ImageTk

from . import scaling
from ..common import constants
from ..logic import cards as cardlib
from ..logic.sidebet_outcomes import tier_label

C = constants.COLORS


def seat_positions(w, h, n, card_w, card_h):
    """Card-top anchor points for `n` seats fanned around the dealer.

    Pure (no Tk). Seat i sits at angle theta evenly spaced over
    [-span/2, +span/2] on the LOWER arc of a circle anchored at the dealer
    card (top centre): x = cx + R*sin(theta), y = anchor + R*cos(theta) —
    so the middle seat is lowest and the edge seats curve up toward the
    dealer. i=0 is the leftmost seat (Player 1). Tuning constants are
    96-dpi pixels scaled by the card size, so DPI-scaled cards scale the
    whole fan with them.
    """
    s = card_h / float(constants.CARD_RENDER_SIZE[1])
    cx = w / 2.0
    anchor_y = 110.0 * s  # just below the dealer card's centre
    span = math.radians(110.0)
    label_stack = (168.0 + 12.0) * s  # six label lines + bottom breathing room
    # R from whichever constraint is tighter: the middle seat's label stack
    # above the bottom edge, or the edge seats inside the side margins
    # (40 px edge gap + half the 118 px optimal-label wraplength). Never so
    # small that the edge seats climb into the dealer card / insurance row.
    fit_h = h - anchor_y - card_h - label_stack
    fit_w = (w / 2.0 - 100.0 * s) / math.sin(span / 2)
    radius = max(min(fit_h, fit_w), 150.0 * s)
    step = span / (n - 1) if n > 1 else 0.0
    out = []
    for i in range(n):
        theta = -span / 2 + i * step
        x = cx + radius * math.sin(theta)
        y = anchor_y + radius * math.cos(theta)
        y = min(y, h - card_h - label_stack)  # tiny-window safety clamp
        out.append((x, y))
    return out


class TableView:
    def __init__(self, parent, on_card_click, on_dealer_click, on_split_click=None,
                 on_seat_name_click=None, on_dealer_extra_click=None,
                 on_advice_click=None):
        self.on_card_click = on_card_click
        self.on_dealer_click = on_dealer_click
        self.on_split_click = on_split_click or (lambda *a: None)
        self.on_seat_name_click = on_seat_name_click or (lambda *a: None)
        self.on_dealer_extra_click = on_dealer_extra_click or (lambda *a: None)
        self.on_advice_click = on_advice_click  # V3 F8: "why this play?"

        self.canvas = tk.Canvas(parent, bg=C["bg_canvas"], highlightthickness=0)
        self._image_cache = {}
        self._master_cache = {}  # name -> decoded PIL image, survives rescales
        self._preload_gen = 0    # invalidates an in-flight preload chain
        self._last_snapshot = None
        self._resize_job = None
        self._preview_item = None
        self._preview_photo = None
        self._preview_close_btn = None

        self.dealer_title = tk.Label(self.canvas, text="DEALER", font=constants.FONT_SECTION,
                                     bg=C["bg_canvas"], fg=C["text_on_felt"])
        self.dealer_card_lbl = tk.Label(self.canvas, bg=C["bg_canvas"], bd=0, cursor="hand2")
        self.dealer_card_lbl.bind("<Button-1>", lambda e: self.on_dealer_click())
        self.dealer_insurance = tk.Label(self.canvas, text="", font=constants.FONT_BODY_BOLD,
                                         bg=C["bg_canvas"], fg=C["text_on_felt"],
                                         wraplength=scaling.px(170), justify="left")
        self._dealer_rendered = "__none__"
        # The dealer's playout draws as card images fanned left of the
        # up-card (was a "Draws: ..." text line). Click corrects a draw;
        # "+" adds one the detector missed.
        self.dealer_extra_lbls = []
        self._extras_rendered = []
        self.dealer_add = None

        self.seats = []
        self._make_seats()
        self.canvas.bind("<Configure>", self._on_resize)
        scaling.on_change(self._rescale)

    def _make_seats(self):
        for i in range(constants.NUM_SEATS):
            self.seats.append({
                "cards": [],          # list of tk.Label, created on demand
                "rendered": [],       # last rendered card names per label
                "name": tk.Label(self.canvas, text=f"Player {i + 1}", font=constants.FONT_SMALL,
                                 bg=C["bg_canvas"], fg=C["text_on_felt"], cursor="hand2"),
                "total": tk.Label(self.canvas, text="", font=constants.FONT_BODY_BOLD,
                                  bg=C["bg_canvas"], fg=C["text_on_felt"]),
                "advice": tk.Label(self.canvas, text="", font=constants.FONT_BODY_BOLD,
                                   bg=C["bg_canvas"], fg=C["text_on_felt"]),
                "optimal": tk.Label(self.canvas, text="", font=constants.FONT_SMALL,
                                    bg=C["bg_canvas"], fg=C["text_on_felt"],
                                    wraplength=scaling.px(118), justify="center"),
                "index": tk.Label(self.canvas, text="", font=constants.FONT_SMALL,
                                  bg=C["bg_canvas"], fg=C["text_on_felt"],
                                  wraplength=scaling.px(118), justify="center"),
                "sidebet": tk.Label(self.canvas, text="", font=constants.FONT_SMALL,
                                    bg=C["bg_canvas"], fg=C["text_on_felt"],
                                    wraplength=scaling.px(118), justify="center"),
                "add": None,          # "+" button, created lazily
                "split_btn": None,    # Split / Undo split badge, created lazily
                "hand_of": [],        # per-card hand tag from the snapshot
                "is_split": False,
                "pos": (0, 0),
            })
            self.seats[i]["name"].bind(
                "<Button-1>", lambda e, s=i: self.on_seat_name_click(s))
            if self.on_advice_click is not None:
                # Clicking any advice line opens the EV inspector (V3 F8).
                for key in ("advice", "optimal", "index"):
                    self.seats[i][key].config(cursor="hand2")
                    self.seats[i][key].bind(
                        "<Button-1>", lambda e, s=i: self.on_advice_click(s))

    # ----------------------------------------------------------------- images

    def card_image(self, card_name, size=None):
        if size is None:  # resolved at call time so DPI rescales take effect
            size = scaling.size(constants.CARD_RENDER_SIZE)
        key = (card_name, size)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached
        photo = ImageTk.PhotoImage(self._master(card_name).resize(size, Image.LANCZOS))
        self._image_cache[key] = photo
        return photo

    def _master(self, card_name):
        """Decoded source image at the largest size any render can ask for
        (the dealer card at scaling's max factor). Decoding the 500x726 PNG
        from disk costs ~5 ms; resizing this in-memory master costs <1 ms —
        so a DPI rescale (which drops every PhotoImage) re-renders the
        table without 40+ synchronous disk decodes on the Tk thread."""
        master = self._master_cache.get(card_name)
        if master is None:
            path = constants.card_image_path(card_name) if card_name not in (None, "back") \
                else constants.CARD_BACK_IMAGE_PATH
            if not path.exists():
                path = constants.CARD_BACK_IMAGE_PATH
            max_size = (round(constants.DEALER_CARD_RENDER_SIZE[0] * scaling.MAX_SCALE),
                        round(constants.DEALER_CARD_RENDER_SIZE[1] * scaling.MAX_SCALE))
            master = Image.open(path).resize(max_size, Image.LANCZOS)
            self._master_cache[card_name] = master
        return master

    def preload_images(self, names=None, _index=0, _gen=None):
        """Load card images a few per idle tick so the UI never freezes."""
        if names is None:
            names = cardlib.all_card_names()
            self._preload_gen += 1
            _gen = self._preload_gen
        elif _gen != self._preload_gen:
            return  # superseded by a newer chain (rescale mid-preload)
        end = min(_index + 4, len(names))
        for name in names[_index:end]:
            self.card_image(name)
            self.card_image(name, scaling.size(constants.PICKER_CARD_SIZE))
        if end < len(names):
            self.canvas.after(15, lambda: self.preload_images(names, end, _gen))

    # --------------------------------------------------------------- geometry

    def _on_resize(self, _event):
        if self._resize_job is not None:
            self.canvas.after_cancel(self._resize_job)
        self._resize_job = self.canvas.after(120, self._relayout)

    def _relayout(self):
        self._resize_job = None
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50 or h < 50:
            return

        if self._preview_item is not None:
            # While a preview is shown the table widgets stay hidden;
            # just keep the screenshot centered.
            self.canvas.coords(self._preview_item, w / 2, h / 2)
            return

        self.dealer_title.place(x=w / 2, y=scaling.px(18), anchor="n")
        self.dealer_card_lbl.place(x=w / 2, y=scaling.px(46), anchor="n")
        # Beside the card, not below it — below collides with the middle
        # seat's cards on short windows.
        card_w_d, card_h_d = scaling.size(constants.DEALER_CARD_RENDER_SIZE)
        self.dealer_insurance.place(
            x=w / 2 + card_w_d / 2 + scaling.px(14), y=scaling.px(46) + card_h_d / 2,
            anchor="w")
        self._place_dealer_extras()

        card_w, card_h = scaling.size(constants.CARD_RENDER_SIZE)
        positions = seat_positions(w, h, constants.NUM_SEATS, card_w, card_h)
        for i, seat in enumerate(self.seats):
            seat["pos"] = positions[i]
            self._place_seat(i)

    def _place_seat(self, i):
        seat = self.seats[i]
        x, top = seat["pos"]
        card_w, card_h = scaling.size(constants.CARD_RENDER_SIZE)
        hand_of = seat["hand_of"]
        split = seat["is_split"] and any(h == 1 for h in hand_of)
        depth = [0, 0]  # cards placed so far per hand (split layout)
        for j, lbl in enumerate(seat["cards"]):
            if split:
                h = hand_of[j] if j < len(hand_of) else 0
                col_x = x - card_w / 2 + scaling.px(-40 if h == 0 else 40)
                lbl.place(x=col_x + min(depth[h], 3) * scaling.px(12),
                          y=top + depth[h] * scaling.px(26), anchor="nw")
                depth[h] += 1
            else:
                lbl.place(x=x - card_w / 2 + min(j, 3) * scaling.px(16),
                          y=top + j * scaling.px(26), anchor="nw")
            lbl.lift()
        n_cards = max(max(depth) if split else len(seat["cards"]), 1)
        base_y = top + card_h + (n_cards - 1) * scaling.px(26) + scaling.px(6)
        seat["total"].place(x=x, y=base_y, anchor="n")
        seat["advice"].place(x=x, y=base_y + scaling.px(22), anchor="n")
        offset = scaling.px(44) + (scaling.px(18) if "\n" in seat["advice"].cget("text") else 0)
        seat["optimal"].place(x=x, y=base_y + offset, anchor="n")
        seat["index"].place(x=x, y=base_y + offset + scaling.px(18), anchor="n")
        seat["sidebet"].place(x=x, y=base_y + offset + scaling.px(36), anchor="n")
        seat["name"].place(x=x, y=base_y + offset + scaling.px(54), anchor="n")
        if seat["add"] is not None:
            seat["add"].place(x=x + card_w / 2 + scaling.px(54 if split else 14),
                              y=top + card_h / 2, anchor="w")
        if seat["split_btn"] is not None:
            seat["split_btn"].place(x=x, y=top - scaling.px(22), anchor="n")

    def _rescale(self):
        """scaling.on_change hook: the fonts are already updated — drop the
        cached photos (masters survive), re-render every card at the new
        size from the last snapshot, re-place the table, then re-warm the
        cache so the next CardPicker open doesn't pay 52 cold renders."""
        self._image_cache.clear()
        self._dealer_rendered = "__none__"
        self._extras_rendered = ["__none__"] * len(self._extras_rendered)
        self.dealer_insurance.config(wraplength=scaling.px(170))
        for seat in self.seats:
            seat["rendered"] = ["__none__"] * len(seat["rendered"])
            seat["optimal"].config(wraplength=scaling.px(118))
            seat["index"].config(wraplength=scaling.px(118))
            seat["sidebet"].config(wraplength=scaling.px(118))
        if self._last_snapshot is not None:
            self.update(self._last_snapshot)
        self._relayout()
        self.preload_images()

    # ----------------------------------------------------------------- update

    def update(self, snapshot):
        if snapshot is None or self._preview_item is not None:
            return

        dealer = snapshot["dealer"]["card"]
        if dealer != self._dealer_rendered:
            self._dealer_rendered = dealer
            if dealer:
                # The dealer model reports only ranks; render a representative card.
                full = dealer if " of " in str(dealer) else f"{dealer} of Spades"
                photo = self.card_image(full, scaling.size(constants.DEALER_CARD_RENDER_SIZE))
            else:
                photo = self.card_image("back", scaling.size(constants.DEALER_CARD_RENDER_SIZE))
            self.dealer_card_lbl.config(image=photo)
            self.dealer_card_lbl.image = photo

        ins = snapshot.get("insurance")
        ins_text = ins["text"] if ins else ""
        ins_color = ins["color"] if ins else C["text_on_felt"]
        if (self.dealer_insurance.cget("text") != ins_text
                or self.dealer_insurance.cget("fg") != ins_color):
            self.dealer_insurance.config(text=ins_text, fg=ins_color)

        self._update_dealer_extras(snapshot)

        for seat_snap in snapshot["seats"]:
            self._update_seat(seat_snap)

    def _update_dealer_extras(self, snapshot):
        """Render the dealer's playout draws as clickable card images."""
        extras = snapshot["dealer"].get("extras") or []
        layout_dirty = len(extras) != len(self.dealer_extra_lbls)
        while len(self.dealer_extra_lbls) < len(extras):
            lbl = tk.Label(self.canvas, bg=C["bg_canvas"], bd=0, cursor="hand2")
            idx = len(self.dealer_extra_lbls)
            lbl.bind("<Button-1>",
                     lambda e, i=idx: self.on_dealer_extra_click(i))
            self.dealer_extra_lbls.append(lbl)
            self._extras_rendered.append("__none__")
        for surplus in self.dealer_extra_lbls[len(extras):]:
            surplus.destroy()
        del self.dealer_extra_lbls[len(extras):]
        del self._extras_rendered[len(extras):]

        for i, name in enumerate(extras):
            if self._extras_rendered[i] != name:
                self._extras_rendered[i] = name
                # Rank-only draws render a representative card, like the
                # up-card does.
                full = name if " of " in str(name) else f"{name} of Spades"
                photo = self.card_image(full)
                self.dealer_extra_lbls[i].config(image=photo)
                self.dealer_extra_lbls[i].image = photo

        want_add = bool(snapshot["dealer"].get("locked"))
        if want_add and self.dealer_add is None:
            btn = tk.Label(self.canvas, text="+", font=constants.FONT_BODY_BOLD,
                           bg=C["bg_canvas_soft"], fg=C["text_on_felt"],
                           width=2, cursor="hand2")
            btn.bind("<Button-1>",
                     lambda e: self.on_dealer_extra_click(99))
            self.dealer_add = btn
            layout_dirty = True
        elif not want_add and self.dealer_add is not None:
            self.dealer_add.destroy()
            self.dealer_add = None
        if layout_dirty:
            self._place_dealer_extras()

    def _place_dealer_extras(self):
        """Fan the playout cards leftward from the up-card; later draws sit
        UNDER earlier ones so each card's top-left rank corner stays
        visible. The "+" add button trails the fan."""
        if self._preview_item is not None:
            return
        w = self.canvas.winfo_width()
        if w < 50:
            return
        card_w_d, card_h_d = scaling.size(constants.DEALER_CARD_RENDER_SIZE)
        card_w, card_h = scaling.size(constants.CARD_RENDER_SIZE)
        x = w / 2 - card_w_d / 2 - scaling.px(14)
        y = scaling.px(46) + card_h_d - card_h  # bottom-aligned with up-card
        for lbl in self.dealer_extra_lbls:
            lbl.place(x=x, y=y, anchor="ne")
            x -= scaling.px(22)
        for lbl in reversed(self.dealer_extra_lbls):
            lbl.lift()  # first draw on top, nearest the up-card
        if self.dealer_add is not None:
            self.dealer_add.place(x=x - scaling.px(6) - (card_w if self.dealer_extra_lbls else 0),
                                  y=y + card_h / 2, anchor="e")

    def _update_seat(self, snap):
        i = snap["index"]
        seat = self.seats[i]
        names = snap["cards"] if snap["cards"] else [None, None]  # two placeholders
        if len(names) == 1:
            names = names + [None]

        layout_dirty = len(names) != len(seat["cards"])
        while len(seat["cards"]) < len(names):
            lbl = tk.Label(self.canvas, bg=C["bg_canvas"], bd=0, cursor="hand2")
            slot = len(seat["cards"])
            lbl.bind("<Button-1>", lambda e, s=i, j=slot: self.on_card_click(s, j))
            seat["cards"].append(lbl)
            seat["rendered"].append("__none__")
        for extra in seat["cards"][len(names):]:
            extra.destroy()
        del seat["cards"][len(names):]
        del seat["rendered"][len(names):]

        for j, name in enumerate(names):
            if seat["rendered"][j] != name:
                seat["rendered"][j] = name
                photo = self.card_image(name or "back")
                seat["cards"][j].config(image=photo)
                seat["cards"][j].image = photo

        if seat["total"].cget("text") != snap["total"]:
            seat["total"].config(text=snap["total"])
        name_text = f"Player {i + 1}" + (" ★" if snap.get("mine") else "")
        if snap.get("book_pct") is not None and snap.get("book_n", 0) >= 10:
            name_text += f" · {snap['book_pct']:.0%}"
        if seat["name"].cget("text") != name_text:
            seat["name"].config(text=name_text,
                                fg=C["warning"] if snap.get("mine") else C["text_on_felt"])
        advice, color = snap["advice"], snap["advice_color"]
        if seat["advice"].cget("text") != advice or seat["advice"].cget("fg") != color:
            seat["advice"].config(text=advice, fg=color)
        optimal = snap.get("optimal", "")
        opt_color = snap.get("optimal_color", C["text_on_felt"])
        if seat["optimal"].cget("text") != optimal or seat["optimal"].cget("fg") != opt_color:
            seat["optimal"].config(text=optimal, fg=opt_color)
        idx_text = snap.get("index_advice", "")
        idx_color = snap.get("index_color", C["text_on_felt"])
        if seat["index"].cget("text") != idx_text or seat["index"].cget("fg") != idx_color:
            seat["index"].config(text=idx_text, fg=idx_color)

        # Side-bet outcomes: wins (green) and undecidable ones ("?"); losses
        # stay silent to keep the table calm.
        parts, any_win = [], False
        for o in (snap.get("side_outcomes") or {}).values():
            if o["result"] == "win":
                any_win = True
                parts.append(f"{o['label']}: {tier_label(o['tier'])} {o['pays']:g}x")
            elif o["result"] == "unknown":
                parts.append(f"{o['label']}: ?")
        sb_text = " · ".join(parts)
        sb_color = C["warning"] if any_win else C["text_on_felt"]
        if seat["sidebet"].cget("text") != sb_text or seat["sidebet"].cget("fg") != sb_color:
            seat["sidebet"].config(text=sb_text, fg=sb_color)

        want_add = 2 <= len(snap["cards"]) < constants.MAX_CARDS_PER_SEAT
        if want_add and seat["add"] is None:
            btn = tk.Label(self.canvas, text="+", font=constants.FONT_BODY_BOLD,
                           bg=C["bg_canvas_soft"], fg=C["text_on_felt"],
                           width=2, cursor="hand2")
            btn.bind("<Button-1>", lambda e, s=i: self.on_card_click(s, 99))
            seat["add"] = btn
            layout_dirty = True
        elif not want_add and seat["add"] is not None:
            seat["add"].destroy()
            seat["add"] = None

        hand_of = snap.get("hand_of", [0] * len(snap["cards"]))
        is_split = snap.get("split", False)
        if hand_of != seat["hand_of"] or is_split != seat["is_split"]:
            seat["hand_of"] = list(hand_of)
            seat["is_split"] = is_split
            layout_dirty = True

        want_split_btn = snap.get("can_split", False) or is_split
        if want_split_btn and seat["split_btn"] is None:
            btn = tk.Label(self.canvas, text="", font=constants.FONT_SMALL,
                           bg=C["badge_bg"], fg=C["text_on_felt"],
                           padx=6, cursor="hand2")
            btn.bind("<Button-1>",
                     lambda e, s=i: self.on_split_click(s, self.seats[s]["is_split"]))
            seat["split_btn"] = btn
            layout_dirty = True
        elif not want_split_btn and seat["split_btn"] is not None:
            seat["split_btn"].destroy()
            seat["split_btn"] = None
        if seat["split_btn"] is not None:
            label = "Undo split" if is_split else "Split ▸ two hands"
            if seat["split_btn"].cget("text") != label:
                seat["split_btn"].config(text=label)

        if layout_dirty:
            self._place_seat(i)

    # ---------------------------------------------------------------- preview

    def show_preview(self, pil_image, on_close):
        """Show a region-debug screenshot over the table (hides seat widgets)."""
        self.clear_preview()
        w = max(self.canvas.winfo_width(), 200)
        h = max(self.canvas.winfo_height(), 200)
        img = pil_image.copy()
        img.thumbnail((w - 30, h - 60))
        self._preview_photo = ImageTk.PhotoImage(img)
        for seat in self.seats:
            for lbl in seat["cards"]:
                lbl.place_forget()
            for key in ("name", "total", "advice", "optimal", "index", "sidebet"):
                seat[key].place_forget()
            if seat["add"] is not None:
                seat["add"].place_forget()
            if seat["split_btn"] is not None:
                seat["split_btn"].place_forget()
        self.dealer_title.place_forget()
        self.dealer_card_lbl.place_forget()
        self.dealer_insurance.place_forget()
        for lbl in self.dealer_extra_lbls:
            lbl.place_forget()
        if self.dealer_add is not None:
            self.dealer_add.place_forget()
        self._preview_item = self.canvas.create_image(w / 2, h / 2, image=self._preview_photo)
        self._preview_close_btn = tk.Button(
            self.canvas, text="Close preview", command=lambda: (self.clear_preview(), on_close()),
            bg=C["danger"], fg="white", font=constants.FONT_BODY, relief="flat",
            cursor="hand2", padx=14, pady=6)
        self._preview_close_btn.place(relx=0.5, rely=1.0, y=-14, anchor="s")

    def clear_preview(self):
        if self._preview_item is not None:
            self.canvas.delete(self._preview_item)
            self._preview_item = None
            self._preview_photo = None
        if self._preview_close_btn is not None:
            self._preview_close_btn.destroy()
            self._preview_close_btn = None
        self._relayout()
        # Force a re-render of everything that was hidden.
        self._dealer_rendered = "__none__"
        self._extras_rendered = ["__none__"] * len(self._extras_rendered)
        for seat in self.seats:
            seat["rendered"] = ["__none__"] * len(seat["rendered"])
        if self._last_snapshot is not None:
            self.update(self._last_snapshot)

    def remember_snapshot(self, snapshot):
        self._last_snapshot = snapshot

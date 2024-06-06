import cv2
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk
from tkinter import messagebox

from ..common import constants
from ..logic.monitor_utils import MonitorUtils
from ..logic.utils import Utils


class PlayerBoxGenerator:
    def __init__(self, gui):
        self.gui = gui
        self.model_players = Utils.initialize_player_model(self)
        self.current_image_path = None
        self.monitor_utils = MonitorUtils()

    def set_monitor(self, monitor):
        self.monitor_utils.set_monitor(monitor)

    def generate(self):
        if not self.monitor_utils.monitor:
            messagebox.showerror("Error", "Monitor not selected.")
            return

        current_resolution = self._capture_screen_and_predict()
        scale_x, scale_y = self.monitor_utils.get_scaling_factors(constants.BASE_RESOLUTION, current_resolution)
        player_regions = self.monitor_utils.scale_player_regions(constants.BASE_PLAYER_REGIONS, scale_x, scale_y)
        self._draw_player_regions_on_image(constants.OUTPUT_PREDICTION_PATH, player_regions)
        self.current_image_path = constants.OUTPUT_FINAL_PREDICTION_PATH
        self._display_image(self.current_image_path)

    def _capture_screen_and_predict(self):
        img = self.monitor_utils.capture_screen()

        img.save(constants.INPUT_SCREENSHOT_PATH)  # Save screenshot for prediction

        # Use model to predict on the saved screenshot and save the prediction result
        self.model_players.predict(
            constants.INPUT_SCREENSHOT_PATH, confidence=constants.PREDICTION_CONFIDENCE_PLAYERS,
            overlap=constants.PREDICTION_OVERLAP_PLAYERS
        ).save(constants.OUTPUT_PREDICTION_PATH)

        return self.monitor_utils.capture_screen().width, self.monitor_utils.capture_screen().height

    def _draw_player_regions_on_image(self, image_path, player_regions):
        img = cv2.imread(image_path)
        for region in player_regions:
            vertices = np.array([region.vertices], np.int32)
            cv2.polylines(img, vertices, isClosed=True, color=(255, 0, 0), thickness=2)

        final_image_path = constants.OUTPUT_FINAL_PREDICTION_PATH
        cv2.imwrite(final_image_path, img)
        print(f"Saved final image with player regions to '{final_image_path}'")

    def _display_image(self, image_path):
        if image_path is None:
            return

        img = Image.open(image_path)

        # Resize the image to fit 60% of the window dimensions while maintaining the aspect ratio
        window_width = self.gui.winfo_width()
        window_height = self.gui.winfo_height()
        target_width = int(window_width * 0.6)
        target_height = int(window_height * 0.6)
        img.thumbnail((target_width, target_height))

        img_tk = ImageTk.PhotoImage(img)
        self.gui.canvas.delete("all")
        self.gui.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.gui.img_tk = img_tk  # Keep a reference to avoid garbage collection

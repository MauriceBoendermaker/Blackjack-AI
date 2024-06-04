import cv2
import numpy as np
from PIL import Image, ImageTk
from roboflow import Roboflow
from mss import mss
from matplotlib.path import Path
import tkinter as tk
from tkinter import messagebox

from ..common import constants


class PlayerBoxGenerator:
    def __init__(self, gui):
        self.gui = gui
        self.monitor = None
        self.roboflow_model = self._initialize_roboflow_model()
        self.current_image_path = None

    def _initialize_roboflow_model(self):
        rf = Roboflow(api_key=constants.ROBOFLOW_API_KEY)
        project = rf.workspace().project(constants.ROBOFLOW_PROJECT_ID)
        return project.version(constants.ROBOFLOW_MODEL_VERSION).model

    def set_monitor(self, monitor):
        self.monitor = monitor

    def generate(self):
        if not self.monitor:
            messagebox.showerror("Error", "Monitor not selected.")
            return

        current_resolution = self._capture_screen_and_predict()
        scale_x, scale_y = self._get_scaling_factors(constants.BASE_RESOLUTION, current_resolution)
        player_regions = self._scale_player_regions(constants.BASE_PLAYER_REGIONS, scale_x, scale_y)
        self._draw_player_regions_on_image(constants.OUTPUT_PREDICTION_PATH, player_regions)
        self.current_image_path = constants.OUTPUT_FINAL_PREDICTION_PATH
        self._display_image(self.current_image_path)

    def _capture_screen_and_predict(self):
        monitor_info = self.monitor
        monitor = {
            "left": monitor_info.x,
            "top": monitor_info.y,
            "width": monitor_info.width,
            "height": monitor_info.height
        }
        sct = mss()
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.save(constants.INPUT_SCREENSHOT_PATH)  # Save screenshot for prediction

        # Use model to predict on the saved screenshot and save the prediction result
        self.roboflow_model.predict(
            constants.INPUT_SCREENSHOT_PATH, confidence=constants.PREDICTION_CONFIDENCE,
            overlap=constants.PREDICTION_OVERLAP
        ).save(constants.OUTPUT_PREDICTION_PATH)

        return (monitor_info.width, monitor_info.height)

    def _get_scaling_factors(self, base_resolution, current_resolution):
        base_width, base_height = base_resolution
        current_width, current_height = current_resolution
        scale_x = current_width / base_width
        scale_y = current_height / base_height
        return scale_x, scale_y

    def _scale_player_regions(self, base_player_regions, scale_x, scale_y):
        scaled_player_regions = []
        for region in base_player_regions:
            scaled_vertices = []
            for vertex in region:
                scaled_vertex = [vertex[0] * scale_x, vertex[1] * scale_y]
                scaled_vertices.append(scaled_vertex)
            scaled_player_regions.append(Path(np.array(scaled_vertices)))
        return scaled_player_regions

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

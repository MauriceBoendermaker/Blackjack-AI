import numpy as np
from mss import mss
from PIL import Image
from matplotlib.path import Path


class MonitorUtils:
    def __init__(self):
        self.monitor = None

    def set_monitor(self, monitor):
        if monitor is None:
            raise ValueError("Invalid monitor selected.")
        self.monitor = monitor

    def capture_screen(self):
        if self.monitor is None:
            raise ValueError("Monitor not set. Please select a monitor before capturing the screen.")

        monitor_info = self.monitor
        monitor = {
            "left": monitor_info.x,
            "top": monitor_info.y,
            "width": monitor_info.width,
            "height": monitor_info.height
        }
        screenshot = mss()
        screenshot_image = screenshot.grab(monitor)
        captured_screenshot = Image.frombytes("RGB", screenshot_image.size, screenshot_image.bgra, "raw", "BGRX")
        return captured_screenshot

    def get_current_resolution(self):
        if self.monitor is None:
            raise ValueError("Monitor not set. Please select a monitor before getting the resolution.")
        return self.monitor.width, self.monitor.height

    def get_scaling_factors(self, base_resolution, current_resolution):
        base_width, base_height = base_resolution
        current_width, current_height = current_resolution
        scale_x = current_width / base_width
        scale_y = current_height / base_height
        return scale_x, scale_y

    def scale_player_regions(self, base_player_regions, scale_x, scale_y):
        scaled_player_regions = []
        for region in base_player_regions:
            scaled_vertices = []
            for vertex in region:
                scaled_vertex = [vertex[0] * scale_x, vertex[1] * scale_y]
                scaled_vertices.append(scaled_vertex)
            scaled_player_regions.append(Path(np.array(scaled_vertices)))
        return scaled_player_regions

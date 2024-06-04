import screeninfo
from mss import mss
from PIL import Image

# import lib.common.constants as constants


class PlayerBoxGenerator:
    def __init__(self):
        self.monitor = None

    def generate(self):
        self._capture_screen()
        print("I am generating playerboxes!!!")

    def _capture_screen(self) -> Image:
        monitors = screeninfo.get_monitors()
        print("Available monitors:")
        for monitor in monitors:
            print(monitor)

        # TODO: Make selectable in tkinter (GUI)
        monitor = monitors[0]
        monitor_info = {
            "left": monitor.x,
            "top": monitor.y,
            "width": monitor.width,
            "height": monitor.height,
        }
        sct = mss()
        screenshot = sct.grab(monitor_info)

        # TODO: Show image in tkinter (GUI)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        img.save("INPUT_current_screen.jpg")
        return img

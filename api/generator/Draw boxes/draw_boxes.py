import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
from mss import mss
from screeninfo import get_monitors
from matplotlib.path import Path

# TODO: Convert player_regions to 1920x1080 resolution (now 2560x1440).
# Define player regions with numpy arrays or convert from your provided points

RESOLUTION = "1920x1080"

player_regions = {
    "1920x1080": [
        Path(
            np.array([[1074, 830], [478, 1100], [600, 1230], [1070, 947], [1074, 830]])
        ),
        Path(
            np.array([[604, 1226], [822, 1330], [1213, 943], [1070, 943], [604, 1226]])
        ),
        Path(
            np.array([[822, 1326], [1113, 1391], [1309, 943], [1213, 943], [822, 1334]])
        ),
        Path(
            np.array(
                [[1304, 939], [1370, 943], [1422, 1391], [1109, 1391], [1309, 939]]
            )
        ),
        Path(
            np.array(
                [[1370, 939], [1457, 943], [1713, 1339], [1422, 1395], [1365, 943]]
            )
        ),
        Path(
            np.array(
                [[1452, 943], [1713, 1339], [1939, 1239], [1570, 939], [1452, 943]]
            )
        ),
        Path(
            np.array(
                [[1570, 839], [2070, 1095], [1939, 1239], [1565, 934], [1574, 839]]
            )
        ),
    ],
    "2560x1440": [
        Path(
            np.array([[476, 1096], [604, 1228], [1072, 948], [1076, 832], [476, 1096]])
        ),
        Path(
            np.array([[604, 1228], [820, 1332], [1216, 944], [1072, 948], [604, 1228]])
        ),
        Path(
            np.array([[820, 1332], [1112, 1392], [1308, 944], [1216, 944], [820, 1332]])
        ),
        Path(
            np.array(
                [[1112, 1392], [1424, 1392], [1372, 944], [1308, 944], [1112, 1392]]
            )
        ),
        Path(
            np.array(
                [[1424, 1392], [1716, 1340], [1456, 940], [1372, 944], [1424, 1392]]
            )
        ),
        Path(
            np.array(
                [[1716, 1336], [1940, 1236], [1572, 940], [1456, 940], [1716, 1336]]
            )
        ),
        Path(
            np.array(
                [[1940, 1236], [2072, 1096], [1572, 840], [1572, 940], [1940, 1236]]
            )
        ),
    ],
}

# Initialize Roboflow
api_key = "WBy7jG6AiiqjzifOfiNH"
project_id = "dey022"
rf = Roboflow(api_key=api_key)
project = rf.workspace().project(project_id)
model = project.version(1).model


# Capture screen and save to a file
def capture_screen_and_predict(monitor_number=0):
    monitor_info = get_monitors()[monitor_number]
    monitor = {
        "left": monitor_info.x,
        "top": monitor_info.y,
        "width": monitor_info.width,
        "height": monitor_info.height,
    }
    sct = mss()
    sct_img = sct.grab(monitor)
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    img.save("INPUT_current_screen.jpg")  # Save screenshot for prediction

    # Use model to predict on the saved screenshot and save the prediction result
    model.predict("INPUT_current_screen.jpg", confidence=50, overlap=100).save(
        "OUTPUT_prediction.jpg"
    )


# Draw player regions on the prediction image
def draw_player_regions_on_image(image_path):
    img = cv2.imread(image_path)
    for region in player_regions.get(RESOLUTION, []):
        if not region:
            raise ValueError("Size not found")
        vertices = np.array([region.vertices], np.int32)
        cv2.polylines(img, vertices, isClosed=True, color=(255, 0, 0), thickness=2)

    final_image_path = "OUTPUT_final_prediction_with_players.jpg"
    cv2.imwrite(final_image_path, img)
    print(f"Saved final image with player regions to '{final_image_path}'")


if __name__ == "__main__":
    # Capture the screen and predict
    capture_screen_and_predict()

    # Draw player regions on the saved prediction image
    draw_player_regions_on_image("OUTPUT_prediction.jpg")

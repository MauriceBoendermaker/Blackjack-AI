"""Screen capture (mss, reused per thread) and player-region geometry.

matplotlib is gone: point-in-polygon is a few lines of numpy ray casting.

Region sources, in priority order: the ACTIVE table profile's calibration
for this resolution (lib/logic/region_profiles.py, written by the region
editor), falling back to BASE_PLAYER_REGIONS scaled from the base
resolution.
"""

import threading

import numpy as np

try:  # mss >= 10 renamed the factory; keep both spellings working
    from mss import MSS as _MSS
except ImportError:
    from mss import mss as _MSS

from ..common import constants
from . import region_profiles


class Polygon:
    """Immutable 2D polygon with a fast contains() test and a bounding box."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        xs, ys = self.vertices[:, 0], self.vertices[:, 1]
        self.bounds = (xs.min(), ys.min(), xs.max(), ys.max())

    def contains(self, x, y) -> bool:
        left, top, right, bottom = self.bounds
        if not (left <= x <= right and top <= y <= bottom):
            return False
        # Even-odd ray casting.
        inside = False
        v = self.vertices
        j = len(v) - 1
        for i in range(len(v)):
            xi, yi = v[i]
            xj, yj = v[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    def scaled(self, sx, sy) -> "Polygon":
        return Polygon(self.vertices * np.array([sx, sy]))


class ScreenCapture:
    """Captures a chosen monitor. One mss instance per thread (mss is not
    cross-thread safe and re-creating it per grab leaks GDI handles)."""

    def __init__(self):
        self.monitor = None  # screeninfo monitor object
        self._local = threading.local()

    def set_monitor(self, monitor):
        if monitor is None:
            raise ValueError("Invalid monitor selected.")
        self.monitor = monitor

    @property
    def resolution(self):
        if self.monitor is None:
            raise ValueError("Monitor not set.")
        return self.monitor.width, self.monitor.height

    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = _MSS()
            self._local.sct = sct
        return sct

    def close_local(self):
        """Release this thread's mss instance (GDI handles + DIB memory).
        Call before a capture thread exits; harmless if nothing was created."""
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            self._local.sct = None
            try:
                sct.close()
            except Exception:
                pass

    def grab_bgr(self) -> np.ndarray:
        """Full-monitor frame as a BGR numpy array (what cv2/models expect)."""
        if self.monitor is None:
            raise ValueError("Monitor not set. Select a monitor before capturing.")
        shot = self._sct().grab({
            "left": self.monitor.x,
            "top": self.monitor.y,
            "width": self.monitor.width,
            "height": self.monitor.height,
        })
        frame = np.asarray(shot)  # BGRA
        return frame[:, :, :3].copy()  # BGR, contiguous


def scaling_factors(current_resolution):
    bw, bh = constants.BASE_RESOLUTION
    cw, ch = current_resolution
    return cw / bw, ch / bh


# ----------------------------------------------- calibrated region profiles

def load_custom_regions(resolution):
    """{"players": [[[x,y],...], ...], "dealer": [l,t,r,b]} or None — the
    ACTIVE table profile's calibration. Coordinates are native to
    `resolution` (no scaling applied)."""
    try:
        data = region_profiles.get_regions(resolution)
        if data is None:
            return None
        players = data["players"]
        dealer = data["dealer"]
        if (len(players) != constants.NUM_SEATS or len(dealer) != 4
                or any(len(poly) < 3 for poly in players)):
            raise ValueError("wrong shape")
        return {"players": players, "dealer": [int(v) for v in dealer]}
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"Calibrated regions unreadable ({e}); using defaults.")
        return None


def save_custom_regions(resolution, players, dealer, profile=None):
    """Save into the named table profile (the active one when None);
    saving stamps the profile's date and makes it active."""
    region_profiles.set_regions(
        resolution, {"players": players, "dealer": list(dealer)},
        profile=profile)


def delete_custom_regions(resolution):
    region_profiles.delete_regions(resolution)


def default_player_regions(current_resolution):
    """The shipped seat polygons scaled to the live resolution (point lists)."""
    sx, sy = scaling_factors(current_resolution)
    return [[[x * sx, y * sy] for x, y in region]
            for region in constants.BASE_PLAYER_REGIONS]


def default_dealer_rect(current_resolution):
    sx, sy = scaling_factors(current_resolution)
    left = int(constants.DEALER_AREA_LEFT * sx)
    top = int(constants.DEALER_AREA_UPPER * sy)
    right = int((constants.DEALER_AREA_LEFT + constants.DEALER_AREA_WIDTH) * sx)
    bottom = int((constants.DEALER_AREA_UPPER + constants.DEALER_AREA_HEIGHT) * sy)
    return left, top, right, bottom


def scaled_player_regions(current_resolution):
    """The 7 seat polygons at the live resolution — calibrated profile when
    one exists, else the defaults scaled from base resolution."""
    custom = load_custom_regions(current_resolution)
    if custom is not None:
        return [Polygon(poly) for poly in custom["players"]]
    return [Polygon(region) for region in default_player_regions(current_resolution)]


def dealer_area_rect(current_resolution):
    """Dealer crop rectangle (left, top, right, bottom) at the live resolution."""
    custom = load_custom_regions(current_resolution)
    if custom is not None:
        return tuple(custom["dealer"])
    return default_dealer_rect(current_resolution)

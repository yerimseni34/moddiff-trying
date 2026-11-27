"""Camera views. Convention: Y up, model front faces +Z, orthographic cameras."""
from dataclasses import dataclass, asdict, replace
import numpy as np


@dataclass
class View:
    name: str
    az: float  # degrees around up axis; 0 = front, 90 = right (camera at +X)
    el: float  # degrees elevation; 90 = top
    # orthographic frame in view-plane world units: centre (cx, cy), half width / height.
    # hw == 0 means "not framed yet" (see fit_view / uniform_view)
    cx: float = 0.0
    cy: float = 0.0
    hw: float = 0.0
    hh: float = 0.0

    def to_dict(self):
        return asdict(self)


_RING4 = [("front", 0), ("right", 90), ("back", 180), ("left", 270)]
_RING8 = [("front", 0), ("front_right", 45), ("right", 90), ("back_right", 135),
          ("back", 180), ("back_left", 225), ("left", 270), ("front_left", 315)]
_CAPS = [View("top", 0, 90), View("bottom", 0, -90)]

PRESETS = {
    "4": [View(n, a, 0) for n, a in _RING4],
    "6": [View(n, a, 0) for n, a in _RING4] + _CAPS,
    "8": [View(n, a, 0) for n, a in _RING8],
    "10": [View(n, a, 0) for n, a in _RING8] + _CAPS,
}


def parse_views(spec):
    """'6' | '8' | ... or custom 'name:az:el,name:az:el'."""
    if spec in PRESETS:
        return list(PRESETS[spec])
    out = []
    for part in spec.split(","):
        name, az, el = part.split(":")
        out.append(View(name.strip(), float(az), float(el)))
    return out


def basis(view):
    """Returns (right, up, toward_camera) unit vectors."""
    az, el = np.radians(view.az), np.radians(view.el)
    c = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    if abs(view.el) > 89.0:
        s = 1.0 if view.el > 0 else -1.0
        hint = -s * np.array([np.sin(az), 0.0, np.cos(az)])
    else:
        hint = np.array([0.0, 1.0, 0.0])
    r = np.cross(hint, c)
    r /= np.linalg.norm(r)
    u = np.cross(c, r)
    return r, u, c


def fit_scale(p, views, fill=0.92):
    """One common scale that fits the model in every view (used for previews / old jobs)."""
    m = 1e-9
    for v in views:
        r, u, _ = basis(v)
        m = max(m, np.abs(p @ r).max(), np.abs(p @ u).max())
    return float(fill / m)


def uniform_view(view, scale):
    """Frame a view with a fixed scale, centred on the origin (all views same size)."""
    return replace(view, cx=0.0, cy=0.0, hw=1.0 / scale, hh=1.0 / scale)


ASPECTS = (1.0, 2.0 / 3.0, 3.0 / 2.0)  # width / height ratios image models handle natively


def fit_view(p, view, fill=0.9, aspects=ASPECTS):
    """Frame a view tightly around the model as seen from that view ("zoomed render"),
    choosing the image aspect ratio (square / portrait / landscape) that wastes least space."""
    r, u, _ = basis(view)
    a, b = p @ r, p @ u
    return frame_box(view, a.min(), a.max(), b.min(), b.max(), fill, aspects)


def frame_box(view, a0, a1, b0, b1, fill=0.9, aspects=ASPECTS):
    """Frame a view around a box [a0,a1] x [b0,b1] given in view-plane coordinates."""
    cx, cy = (a0 + a1) / 2, (b0 + b1) / 2
    ex, ey = max((a1 - a0) / 2, 1e-6), max((b1 - b0) / 2, 1e-6)
    best = None
    for A in aspects:
        hw = max(ex, ey * A) / fill
        hh = hw / A
        used = (ex * ey) / (hw * hh)  # fraction of the frame the model's bbox fills
        if best is None or used > best[0] + 1e-9:
            best = (used, hw, hh)
    return replace(view, cx=float(cx), cy=float(cy), hw=float(best[1]), hh=float(best[2]))


def image_size(view, short=1024):
    """Pixel size (W, H) of a framed view whose shorter side is `short`."""
    A = view.hw / view.hh
    if A >= 1:
        return int(round(short * A)), short
    return short, int(round(short / A))


def long_size(view, long):
    """Pixel size (W, H) of a framed view whose longer side is `long`."""
    A = view.hw / view.hh
    if A >= 1:
        return long, max(1, int(round(long / A)))
    return max(1, int(round(long * A))), long


def project_points(p, view, W, H):
    """World points -> (x, y) pixel coords in a W x H image of the view, depth (smaller = closer)."""
    r, u, c = basis(view)
    x = ((p @ r - view.cx) / view.hw * 0.5 + 0.5) * W
    y = (0.5 - (p @ u - view.cy) / view.hh * 0.5) * H
    return x, y, -(p @ c)


def footprint(view, W):
    """World-space size of one pixel."""
    return 2.0 * view.hw / W

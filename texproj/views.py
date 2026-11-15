"""Camera views. Convention: Y up, model front faces +Z, orthographic cameras."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class View:
    name: str
    az: float  # degrees around up axis; 0 = front, 90 = right (camera at +X)
    el: float  # degrees elevation; 90 = top

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
    m = 1e-9
    for v in views:
        r, u, _ = basis(v)
        m = max(m, np.abs(p @ r).max(), np.abs(p @ u).max())
    return float(fill / m)


def project_points(p, view, scale, R):
    """World points -> (x, y) pixel coords at resolution R, depth (smaller = closer)."""
    r, u, c = basis(view)
    x = (p @ r * scale * 0.5 + 0.5) * R
    y = (0.5 - p @ u * scale * 0.5) * R
    return x, y, -(p @ c)


def footprint(scale, R):
    """World-space size of one pixel."""
    return 2.0 / (scale * R)

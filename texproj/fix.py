"""Pick extra camera angles that look straight at the texels no view could paint."""
import numpy as np
from .views import View, basis, project_points, footprint, fit_view, frame_box, long_size
from .render import raster_view


def candidates():
    out = [View("fix_top", 0, 90), View("fix_bottom", 0, -90)]
    for el in (-60, -30, 0, 30, 60):
        for az in range(0, 360, 30):
            out.append(View(f"fix_az{az:03d}_el{el:+03d}", az, el))
    return out


def zoom_on(view, pts, margin=0.25, min_frac=0.3):
    """Reframe a (whole-model) view onto the given points, with some context around them,
    but never tighter than min_frac of the whole-model frame (the AI needs surrounding paint)."""
    r, u, _ = basis(view)
    a, b = pts @ r, pts @ u
    a0, a1 = np.percentile(a, [1, 99])
    b0, b1 = np.percentile(b, [1, 99])
    m = margin * max(a1 - a0, b1 - b0) + 1e-6
    a0, a1, b0, b1 = a0 - m, a1 + m, b0 - m, b1 + m
    lim = min_frac * max(view.hw, view.hh)
    if a1 - a0 < 2 * lim:
        c = (a0 + a1) / 2
        a0, a1 = c - lim, c + lim
    if b1 - b0 < 2 * lim:
        c = (b0 + b1) / 2
        b0, b1 = c - lim, c + lim
    # stay inside the whole-model frame
    a0, a1 = max(a0, view.cx - view.hw), min(a1, view.cx + view.hw)
    b0, b1 = max(b0, view.cy - view.hh), min(b1, view.cy + view.hh)
    return frame_box(view, a0, a1, b0, b1, fill=1.0)


def select_fix_views(baker, hole, max_views=3, min_gain=0.0015, res=384, log=print):
    """Greedy set cover of hole texels. min_gain is a fraction of all texels."""
    idx = np.flatnonzero(hole)
    if len(idx) > 150_000:
        idx = np.random.default_rng(0).choice(idx, 150_000, replace=False)
    if len(idx) == 0:
        return []
    pos, nrm = baker.pos[idx], baker.nrm[idx]
    cands, seen = [fit_view(baker.mesh.p, v) for v in candidates()], []
    for v in cands:
        W, H = long_size(v, res)
        fp = footprint(v, W)
        _, _, zb = raster_view(baker.mesh, v, W, H)
        x, y, z = project_points(pos, v, W, H)
        ix = np.clip(np.floor(x).astype(int), 0, W - 1)
        iy = np.clip(np.floor(y).astype(int), 0, H - 1)
        cos = nrm @ basis(v)[2]
        seen.append((cos > 0.4) & (z <= zb[iy, ix] + 4 * fp))
    left = np.ones(len(idx), bool)
    hole_frac = hole.mean()
    chosen = []
    for _ in range(max_views):
        gains = [(s & left).sum() for s in seen]
        k = int(np.argmax(gains))
        if gains[k] / len(idx) * hole_frac < min_gain:
            break
        newly = seen[k] & left
        chosen.append(zoom_on(cands[k], pos[newly]))
        log(f"  fix view {cands[k].name}: covers {gains[k] / len(idx):.0%} of the regions to fix")
        left &= ~seen[k]
    return chosen

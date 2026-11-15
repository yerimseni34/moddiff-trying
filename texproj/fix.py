"""Pick extra camera angles that look straight at the texels no view could paint."""
import numpy as np
from .views import View, basis, project_points, footprint
from .render import raster_view


def candidates():
    out = [View("fix_top", 0, 90), View("fix_bottom", 0, -90)]
    for el in (-60, -30, 0, 30, 60):
        for az in range(0, 360, 30):
            out.append(View(f"fix_az{az:03d}_el{el:+03d}", az, el))
    return out


def select_fix_views(baker, hole, max_views=3, min_gain=0.0015, res=384, log=print):
    """Greedy set cover of hole texels. min_gain is a fraction of all texels."""
    idx = np.flatnonzero(hole)
    if len(idx) > 150_000:
        idx = np.random.default_rng(0).choice(idx, 150_000, replace=False)
    if len(idx) == 0:
        return []
    pos, nrm = baker.pos[idx], baker.nrm[idx]
    fp = footprint(baker.scale, res)
    cands, seen = candidates(), []
    for v in cands:
        _, _, zb = raster_view(baker.mesh, v, baker.scale, res)
        x, y, z = project_points(pos, v, baker.scale, res)
        ix = np.clip(np.floor(x).astype(int), 0, res - 1)
        iy = np.clip(np.floor(y).astype(int), 0, res - 1)
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
        chosen.append(cands[k])
        log(f"  fix view {cands[k].name}: covers {gains[k] / len(idx):.0%} of remaining-hole sample")
        left &= ~seen[k]
    return chosen

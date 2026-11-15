"""Small vectorised numpy triangle rasteriser (no GPU / OpenGL needed).

Used both for screen-space renders (orthographic views) and for UV-space
rasterisation (texel -> surface point lookup).
"""
import numpy as np


def rasterize(xy, z, faces, W, H, max_frag=2_000_000):
    """Rasterise triangles.

    xy    : (V,2) pixel coordinates (x right, y down; pixel centres at i+0.5)
    z     : (V,) depth, smaller = closer
    faces : (F,3) int
    Returns tri (H,W) int32 (-1 = empty), bary (H,W,3) float32, zbuf (H,W) float32.
    """
    xy = np.asarray(xy, np.float64)
    z = np.asarray(z, np.float64)
    faces = np.asarray(faces)
    tri = np.full(H * W, -1, np.int32)
    bary = np.zeros((H * W, 3), np.float32)
    zbuf = np.full(H * W, np.inf, np.float32)

    p = xy[faces]
    x0, y0 = p[:, 0, 0], p[:, 0, 1]
    x1, y1 = p[:, 1, 0], p[:, 1, 1]
    x2, y2 = p[:, 2, 0], p[:, 2, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    bx0 = np.maximum(np.ceil(p[..., 0].min(1) - 0.5), 0).astype(np.int64)
    bx1 = np.minimum(np.floor(p[..., 0].max(1) - 0.5), W - 1).astype(np.int64)
    by0 = np.maximum(np.ceil(p[..., 1].min(1) - 0.5), 0).astype(np.int64)
    by1 = np.minimum(np.floor(p[..., 1].max(1) - 0.5), H - 1).astype(np.int64)
    ok = (np.abs(area) > 1e-12) & (bx1 >= bx0) & (by1 >= by0)
    zf = z[faces]
    # bucket triangles by (quantised) bbox width/height so each bucket can be
    # evaluated as one dense (n_tris, KW*KH) block without much wasted work
    lv = np.array(sorted({int(round(2 ** (i / 2))) for i in range(0, 2 * int(np.ceil(np.log2(max(W, H)))) + 3)}))
    qw = lv[np.searchsorted(lv, bx1 - bx0 + 1)]
    qh = lv[np.searchsorted(lv, by1 - by0 + 1)]
    key = np.where(ok, qw * (lv[-1] + 1) + qh, -1)
    order_all = np.argsort(key, kind="stable")
    keys, starts = np.unique(key[order_all], return_index=True)
    ends = np.append(starts[1:], len(order_all))

    for kk, s0, s1 in zip(keys, starts, ends):
        if kk < 0:
            continue
        KW, KH = int(kk // (lv[-1] + 1)), int(kk % (lv[-1] + 1))
        ids = order_all[s0:s1]
        oy, ox = np.divmod(np.arange(KW * KH), KW)
        step = max(1, max_frag // (KW * KH))
        for s in range(0, len(ids), step):
            f = ids[s:s + step]
            px = bx0[f, None] + ox[None]
            py = by0[f, None] + oy[None]
            cx = px + 0.5
            cy = py + 0.5
            a = area[f, None]
            w0 = ((x1[f, None] - cx) * (y2[f, None] - cy) - (x2[f, None] - cx) * (y1[f, None] - cy)) / a
            w1 = ((x2[f, None] - cx) * (y0[f, None] - cy) - (x0[f, None] - cx) * (y2[f, None] - cy)) / a
            w2 = 1.0 - w0 - w1
            eps = -1e-6
            inside = (px <= bx1[f, None]) & (py <= by1[f, None]) & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            fi, pi = np.nonzero(inside)
            if len(fi) == 0:
                continue
            w0, w1, w2 = w0[fi, pi], w1[fi, pi], w2[fi, pi]
            pix = py[fi, pi] * W + px[fi, pi]
            fid = f[fi]
            zz = (w0 * zf[fid, 0] + w1 * zf[fid, 1] + w2 * zf[fid, 2]).astype(np.float32)
            keep = zz < zbuf[pix]
            if not keep.any():
                continue
            pix, fid, zz = pix[keep], fid[keep], zz[keep]
            b = np.stack([w0[keep], w1[keep], w2[keep]], 1).astype(np.float32)
            order = np.argsort(-zz, kind="stable")  # farthest first; nearest written last wins
            pix, fid, zz, b = pix[order], fid[order], zz[order], b[order]
            zbuf[pix] = zz
            tri[pix] = fid
            bary[pix] = b
    return tri.reshape(H, W), bary.reshape(H, W, 3), zbuf.reshape(H, W)


def interp(attr, faces, tri, bary):
    """Interpolate a per-vertex attribute for the (flat) covered pixels tri>=0."""
    a = attr[faces[tri]]  # (N,3,C)
    return (a * bary[:, :, None]).sum(1)

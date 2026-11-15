"""Projection baking: stylised views -> UV texture.

For every texel we know its 3D position and normal (UV-space rasterisation).
Each view contributes its colour where the texel is visible (depth test),
weighted by facing angle and by distance to silhouette / occlusion edges, so
the unreliable pixels of every view (grazing angles, borders) are never used
when a better view exists.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .raster import rasterize, interp
from .render import bilinear, raster_view
from .views import basis, project_points, footprint
from .align import align_to_mask


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Baker:
    def __init__(self, mesh, scale, tex=2048, depth_res=2048, cos_min=0.15,
                 feather=0.012, sharp=3.0, bg=(255, 255, 255), log=print):
        self.mesh, self.scale, self.T, self.R = mesh, scale, tex, depth_res
        self.cos_min, self.sharp, self.bg, self.log = cos_min, sharp, bg, log
        self.feather_px = max(2.0, feather * depth_res)
        uvpx = np.stack([mesh.uv[:, 0] * tex, (1 - mesh.uv[:, 1]) * tex], 1)
        tri, bary, _ = rasterize(uvpx, np.zeros(len(uvpx)), mesh.f, tex, tex)
        self.valid = tri >= 0
        t, b = tri[self.valid], bary[self.valid]
        self.pos = interp(mesh.p, mesh.f, t, b)
        n = interp(mesh.n, mesh.f, t, b)
        self.nrm = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        self.N = len(self.pos)
        uvt = uvpx[mesh.f]
        e1, e2 = uvt[:, 1] - uvt[:, 0], uvt[:, 2] - uvt[:, 0]
        uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
        if mesh.uv.min() < -1e-4 or mesh.uv.max() > 1 + 1e-4:
            log("[warn] UVs fall outside 0..1 (tiling?) - consider --unwrap")
        if uv_area > 1.25 * max(1, self.N):
            log(f"[warn] UV islands overlap (~{uv_area / self.N:.2f}x coverage). Mirrored/stacked UVs "
                "cannot hold a projected texture correctly - rerun with --unwrap")

    # ------------------------------------------------------------------ views
    def view_layer(self, view, image, align=True, name=None):
        """Returns (weight (N,), colour (N,3) float32 0..1, info) for one stylised view."""
        R = self.R
        tri, _, zb = raster_view(self.mesh, view, self.scale, R)
        mask = tri >= 0
        S = max(image.size)
        img = image.convert("RGB").resize((S, S), Image.LANCZOS) if image.size != (S, S) else image.convert("RGB")
        mask_s = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((S, S), Image.BILINEAR)) > 127
        if align:
            img, fg, info = align_to_mask(img, mask_s, self.bg)
        else:
            from .align import foreground_mask
            fg = foreground_mask(img)
            info = {"iou": float((fg & mask_s).sum() / max(1, (fg | mask_s).sum())), "aligned": False}
        fg_r = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((R, R), Image.NEAREST)) > 127

        # unreliable pixels: silhouette, occlusion (depth) edges, and places near the
        # silhouette where the stylised image shows background instead of the object
        fp = footprint(self.scale, R)
        zf = np.where(mask, zb, 1e3)
        dz = np.zeros_like(zf)
        for ax in (0, 1):
            d = np.abs(np.diff(zf, axis=ax))
            pad = [(0, 0), (0, 0)]
            pad[ax] = (0, 1)
            dz = np.maximum(dz, np.pad(d, pad))
            pad[ax] = (1, 0)
            dz = np.maximum(dz, np.pad(d, pad))
        inside = ndi.distance_transform_edt(mask)
        bad = (~mask) | (dz > 8 * fp) | (mask & ~fg_r & (inside <= 2.5 * self.feather_px))
        dist = ndi.distance_transform_edt(~bad)
        feath = smoothstep((dist - 1.0) / self.feather_px).astype(np.float32)

        x, y, z = project_points(self.pos, view, self.scale, R)
        _, _, c = basis(view)
        cos = self.nrm @ c
        ix = np.clip(np.floor(x).astype(np.int64), 0, R - 1)
        iy = np.clip(np.floor(y).astype(np.int64), 0, R - 1)
        tan = np.sqrt(np.clip(1 - cos * cos, 0, 1)) / np.clip(cos, 0.1, 1)
        vis = (cos > self.cos_min) & (z <= zb[iy, ix] + fp * (1.5 * tan + 1.0) + 1e-4)
        w = np.zeros(self.N, np.float32)
        w[vis] = ((cos[vis] - self.cos_min) / (1 - self.cos_min)) * feath[iy[vis], ix[vis]]
        col = np.zeros((self.N, 3), np.float32)
        nz = w > 0
        arr = np.asarray(img, np.float32) / 255.0
        col[nz] = bilinear(arr, x[nz] / R * S, y[nz] / R * S)
        info["coverage"] = float(nz.mean())
        return w, col, info, img

    # ------------------------------------------------------------------ blend
    def blend(self, layers, gain=True, hole_thresh=0.03):
        """layers: list of (w, col). Returns colour (N,3), hole (N,) bool, confidence (N,)."""
        ws = [l[0] for l in layers]
        cs = [l[1] for l in layers]
        if gain and len(layers) > 1:
            cs = self._gain(ws, cs)
        num = np.zeros((self.N, 3), np.float32)
        den = np.zeros(self.N, np.float32)
        best = np.zeros(self.N, np.float32)
        for w, c in zip(ws, cs):
            ww = w ** self.sharp
            num += ww[:, None] * c
            den += ww
            best = np.maximum(best, w)
        hole = best < hole_thresh
        col = num / np.maximum(den, 1e-12)[:, None]
        col[hole] = 0
        return col, hole, best

    def _gain(self, ws, cs, iters=2):
        """Per-view colour gain so overlapping views agree in brightness/tint."""
        cs = [c.copy() for c in cs]
        for _ in range(iters):
            num = sum(w[:, None] * c for w, c in zip(ws, cs))
            den = sum(ws)
            for i, (w, c) in enumerate(zip(ws, cs)):
                other = den - w
                ov = (w > 0.05) & (other > 0.05)
                if ov.sum() < 500:
                    continue
                cons = (num[ov] - w[ov, None] * c[ov]) / other[ov, None]
                wt = np.minimum(w[ov], other[ov])[:, None]
                g = (wt * cons).sum(0) / np.maximum((wt * c[ov]).sum(0), 1e-6)
                g = np.clip(np.sqrt(g), 0.85, 1.18)  # sqrt: meet in the middle
                cs[i] = np.clip(c * g, 0, 1)
        return cs

    def fill3d(self, col, hole, k=8):
        """Fill hole texels from the nearest painted texels in 3D (not in UV space,
        so colour never bleeds across unrelated UV islands)."""
        if not hole.any() or hole.all():
            return col
        src = np.flatnonzero(~hole)
        if len(src) > 400_000:
            src = np.random.default_rng(0).choice(src, 400_000, replace=False)
        tree = cKDTree(self.pos[src])
        d, j = tree.query(self.pos[hole], k=min(k, len(src)), workers=-1)
        if d.ndim == 1:
            d, j = d[:, None], j[:, None]
        w = 1.0 / (d + 1e-4) ** 2
        out = col.copy()
        out[hole] = (col[src][j] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        return out

    def to_texture(self, col):
        tex = np.zeros((self.T, self.T, 3), np.float32)
        tex[self.valid] = col
        _, (iy, ix) = ndi.distance_transform_edt(~self.valid, return_indices=True)
        tex = tex[iy, ix]  # gutter padding: nearest valid texel
        return (np.clip(tex, 0, 1) * 255 + 0.5).astype(np.uint8)

    def flat_to_map(self, values):
        m = np.zeros((self.T, self.T), np.float32)
        m[self.valid] = values
        return m

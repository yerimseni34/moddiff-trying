"""Screen-space rendering of views, sheets (grids) and previews."""
import numpy as np
from PIL import Image
from .raster import rasterize, interp
from .views import basis, project_points, View, uniform_view, fit_scale


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def bilinear(img, x, y):
    """Sample img (H,W[,C]) at continuous pixel coords (pixel centres at i+0.5)."""
    H, W = img.shape[:2]
    x = np.clip(x - 0.5, 0, W - 1)
    y = np.clip(y - 0.5, 0, H - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return a * (1 - fy) + b * fy


def raster_view(mesh, view, W, H):
    x, y, z = project_points(mesh.p, view, W, H)
    return rasterize(np.stack([x, y], 1), z, mesh.f, W, H)


def render_view(mesh, view, W, H, bg=(255, 255, 255), texture=None,
                shading=True, ss=2, color_fn=None):
    """Returns (rgb uint8 (H,W,3), mask bool (H,W)) for a framed view.

    texture : optional (H,W,3) uint8 overriding the mesh's own colours
    color_fn: optional f(world_pos (N,3)) -> rgb 0..1 (used by the mock stylizer)
    """
    tri, bary, _ = raster_view(mesh, view, W * ss, H * ss)
    m = tri >= 0
    t, b = tri[m], bary[m]
    tex = texture if texture is not None else mesh.base_tex
    if color_fn is not None:
        col = color_fn(interp(mesh.p, mesh.f, t, b))
    elif tex is not None:
        uv = interp(mesh.uv, mesh.f, t, b)
        th, tw = tex.shape[:2]
        col = bilinear(tex.astype(np.float32) / 255.0, uv[:, 0] * tw, (1 - uv[:, 1]) * th)
    elif mesh.vcol is not None:
        col = interp(mesh.vcol, mesh.f, t, b)
    else:
        col = np.full((len(t), 3), 0.82)
    if shading and color_fn is None:
        n = interp(mesh.n, mesh.f, t, b)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        _, u, c = basis(view)
        key = c * 0.8 + u * 0.6
        key /= np.linalg.norm(key)
        lam = 0.55 + 0.35 * np.clip(n @ key, 0, 1) + 0.10 * np.clip(n @ c, 0, 1)
        lam = np.where(n @ c < -0.05, 0.3, lam)  # back faces (holes in the mesh) shown dark
        if texture is not None:
            lam = 0.85 + 0.15 * np.clip(n @ c, 0, 1)  # near-unlit for baked previews
        col = col * lam[:, None]
    img = np.empty((H * ss, W * ss, 3), np.float32)
    img[:] = np.array(bg, np.float32) / 255.0
    img[m] = np.clip(col, 0, 1)
    img = img.reshape(H, ss, W, ss, 3).mean((1, 3))
    cov = m.reshape(H, ss, W, ss).mean((1, 3))
    return (img * 255 + 0.5).astype(np.uint8), cov > 0.5


def grid_shape(n):
    table = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2), 6: (3, 2),
             8: (4, 2), 9: (3, 3), 10: (5, 2), 12: (4, 3)}
    if n in table:
        return table[n]
    cols = int(np.ceil(np.sqrt(n)))
    return cols, int(np.ceil(n / cols))


def pad_to_aspect(img, aspect, bg=(255, 255, 255)):
    """Pad a PIL image (centred, never cropped or stretched) to width/height == aspect."""
    w, h = img.size
    if abs(w / h - aspect) < 1e-3:
        return img
    W, H = (w, round(w / aspect)) if w / h > aspect else (round(h * aspect), h)
    out = Image.new("RGB", (W, H), tuple(bg))
    out.paste(img.convert("RGB"), ((W - w) // 2, (H - h) // 2))
    return out


def make_sheet(images, cell, bg=(255, 255, 255)):
    """Grid of square cells; non-square images are fitted inside their cell (letterboxed)."""
    cols, rows = grid_shape(len(images))
    sheet = Image.new("RGB", (cols * cell, rows * cell), bg)
    for i, im in enumerate(images):
        sq = pad_to_aspect(im, 1.0, bg).resize((cell, cell), Image.LANCZOS)
        sheet.paste(sq, ((i % cols) * cell, (i // cols) * cell))
    return sheet


def split_sheet(sheet, n, sizes=None):
    """Inverse of make_sheet. sizes: optional list of original (W, H) to undo the letterbox."""
    cols, rows = grid_shape(n)
    W, H = sheet.size
    out = []
    for i in range(n):
        cx, cy = i % cols, i // cols
        box = (round(cx * W / cols), round(cy * H / rows),
               round((cx + 1) * W / cols), round((cy + 1) * H / rows))
        cell = sheet.crop(box)
        if sizes is not None:
            w, h = sizes[i]
            cw, ch = cell.size
            iw, ih = (cw, round(cw * h / w)) if w >= h else (round(ch * w / h), ch)
            l, t = (cw - iw) // 2, (ch - ih) // 2
            cell = cell.crop((l, t, l + iw, t + ih)).resize((w, h), Image.LANCZOS)
        out.append(cell)
    return out


def preview_sheet(mesh, texture, size=512, n=8, el=18.0):
    views = [View(f"p{i}", 360.0 * i / n + 22.5, el if i % 2 == 0 else -el) for i in range(n)]
    scale = fit_scale(mesh.p, views, fill=0.92)
    ims = [Image.fromarray(render_view(mesh, uniform_view(v, scale), size, size, bg=(46, 48, 54),
                                       texture=texture)[0]) for v in views]
    return make_sheet(ims, size, bg=(46, 48, 54))

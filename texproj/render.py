"""Screen-space rendering of views, sheets (grids) and previews."""
import numpy as np
from PIL import Image
from .raster import rasterize, interp
from .views import basis, project_points, View


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


def raster_view(mesh, view, scale, R):
    x, y, z = project_points(mesh.p, view, scale, R)
    return rasterize(np.stack([x, y], 1), z, mesh.f, R, R)


def render_view(mesh, view, scale, size, bg=(255, 255, 255), texture=None,
                shading=True, ss=2, color_fn=None):
    """Returns (rgb uint8 (size,size,3), mask bool (size,size)).

    texture : optional (H,W,3) uint8 overriding the mesh's own colours
    color_fn: optional f(world_pos (N,3)) -> rgb 0..1 (used by the mock stylizer)
    """
    R = size * ss
    tri, bary, _ = raster_view(mesh, view, scale, R)
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
        if texture is not None:
            lam = 0.85 + 0.15 * np.clip(n @ c, 0, 1)  # near-unlit for baked previews
        col = col * lam[:, None]
    img = np.empty((R, R, 3), np.float32)
    img[:] = np.array(bg, np.float32) / 255.0
    img[m] = np.clip(col, 0, 1)
    img = img.reshape(size, ss, size, ss, 3).mean((1, 3))
    cov = m.reshape(size, ss, size, ss).mean((1, 3))
    return (img * 255 + 0.5).astype(np.uint8), cov > 0.5


def grid_shape(n):
    table = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2), 6: (3, 2),
             8: (4, 2), 9: (3, 3), 10: (5, 2), 12: (4, 3)}
    if n in table:
        return table[n]
    cols = int(np.ceil(np.sqrt(n)))
    return cols, int(np.ceil(n / cols))


def make_sheet(images, cell, bg=(255, 255, 255)):
    cols, rows = grid_shape(len(images))
    sheet = Image.new("RGB", (cols * cell, rows * cell), bg)
    for i, im in enumerate(images):
        sheet.paste(im.resize((cell, cell), Image.LANCZOS), ((i % cols) * cell, (i // cols) * cell))
    return sheet


def split_sheet(sheet, n):
    cols, rows = grid_shape(n)
    W, H = sheet.size
    out = []
    for i in range(n):
        cx, cy = i % cols, i // cols
        box = (round(cx * W / cols), round(cy * H / rows),
               round((cx + 1) * W / cols), round((cy + 1) * H / rows))
        out.append(sheet.crop(box))
    return out


def preview_sheet(mesh, scale, texture, size=512, n=8, el=18.0):
    views = [View(f"p{i}", 360.0 * i / n + 22.5, el if i % 2 == 0 else -el) for i in range(n)]
    ims = [Image.fromarray(render_view(mesh, v, scale * 0.9, size, bg=(46, 48, 54), texture=texture)[0])
           for v in views]
    return make_sheet(ims, size, bg=(46, 48, 54))

"""Stylisation backends: OpenAI image API, or an offline mock for testing.

(For the manual workflow no backend is needed: paint views/*.png or sheet.png
in ChatGPT yourself and drop the results into <work>/stylized/.)
"""
import base64
import io
import math
import time
import numpy as np
from PIL import Image
from .render import make_sheet, split_sheet, grid_shape, render_view
from .views import image_size

RULES = (
    "HARD RULES: This is a texture-projection job, so geometry must be preserved exactly. "
    "Keep the object at exactly the same position, scale, orientation, silhouette and proportions as in "
    "the input render - output must be pixel-aligned with it. Do not move, rotate, crop, zoom, redesign, "
    "add or remove any part. Paint only inside the existing silhouette. "
    "Use flat, even, neutral lighting: no cast shadows, no ground shadow, no rim light, no strong "
    "highlights (the result is used as a diffuse game texture). "
    "Keep the plain solid background colour exactly as in the input. No text, labels, borders or frames."
)


def mock_color(p):
    """A deterministic 3D pattern used as fake 'style' (lets us measure projection error)."""
    base = 0.5 + 0.5 * np.sin(p * 9.0 + np.array([0.0, 2.0, 4.0]))
    chk = (np.floor(p * 6.0 + 100).sum(1) % 2)[:, None]
    return 0.15 + 0.6 * base * (0.55 + 0.45 * chk)


def manual_grid_prompt(views, extra=""):
    """Prompt to paste into ChatGPT together with sheet.png + style reference(s)."""
    from .render import grid_shape as _gs
    cols, rows = _gs(len(views))
    names = ", ".join(v.name.replace("_", " ") for v in views)
    return (f"The first image is a sheet of {len(views)} orthographic renders of the SAME single 3D model in a "
            f"{cols}x{rows} grid of equal square cells (left to right, top to bottom: {names}). The other "
            "image(s) are STYLE REFERENCES. Paint/texture the model in every cell in exactly the art style, "
            "colour palette and material rendering of the style references. All cells show the same object, "
            "so colours and details must be fully consistent between views. Keep every cell exactly where it "
            f"is and output the image in the same aspect ratio as the sheet. {RULES} {extra}").strip()


def manual_fix_prompt(extra=""):
    return ("This is a render of a textured 3D model. Most of the texture is finished, but some regions are "
            "only a blurry / smeared placeholder. Repaint those blurry regions with clean, detailed texture "
            "that continues the surrounding painted surface seamlessly, in the same art style. Leave the "
            f"already-detailed regions unchanged. {RULES} {extra}").strip()


class MockStylizer:
    """Renders the ground-truth 3D pattern, with optional random drift/zoom per view
    to imitate an image model that does not keep things pixel-exact."""

    def __init__(self, ctx, jitter=0.0):
        self.ctx, self.jitter = ctx, jitter
        self.rng = np.random.default_rng(1)

    def _one(self, view):
        c = self.ctx
        W, H = image_size(view, c["size"])
        img = Image.fromarray(render_view(c["mesh"], view, W, H, bg=c["bg"], color_fn=mock_color)[0])
        if self.jitter:
            s = 1 + self.rng.uniform(-self.jitter, self.jitter)
            t = self.rng.uniform(-self.jitter, self.jitter, 2) * min(W, H)
            img = img.transform(img.size, Image.AFFINE,
                                (s, 0, W / 2 - s * W / 2 + t[0], 0, s, H / 2 - s * H / 2 + t[1]),
                                resample=Image.BICUBIC, fillcolor=tuple(c["bg"]))
        return img

    def stylize_views(self, views, renders, refs, prompt, mode):
        return [self._one(v) for v in views]

    def fix_views(self, views, renders, masks, refs, prompt):
        return [self._one(v) for v in views]


def api_size(model, w, h):
    if model.startswith("gpt-image-1") or model.startswith("dall"):
        r = w / h
        return (1536, 1024) if r > 1.2 else (1024, 1536) if r < 0.83 else (1024, 1024)
    k = min(1.0, 3840 / max(w, h), math.sqrt(8_000_000 / (w * h)))
    return max(16, int(w * k) // 16 * 16), max(16, int(h * k) // 16 * 16)


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class OpenAIStylizer:
    def __init__(self, ctx, model="gpt-image-2", quality="high", sheet_cell=768, log=print):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("pip install openai  (and set OPENAI_API_KEY)") from e
        self.client = OpenAI()
        self.ctx, self.model, self.quality, self.cell, self.log = ctx, model, quality, sheet_cell, log
        self.fidelity = model.startswith("gpt-image-1") and "mini" not in model

    # -- low level ---------------------------------------------------------
    def edit(self, image, extra_images, prompt, mask=None):
        """Edit `image` (PIL RGB); returns PIL RGB of the same size."""
        w, h = image.size
        aw, ah = api_size(self.model, w, h)
        ar = aw / ah
        cw, ch = (w, round(w / ar)) if w / h >= ar else (round(h * ar), h)
        ox, oy = (cw - w) // 2, (ch - h) // 2
        canvas = Image.new("RGB", (cw, ch), tuple(self.ctx["bg"]))
        canvas.paste(image, (ox, oy))
        files = [("image.png", _png(canvas.resize((aw, ah), Image.LANCZOS)), "image/png")]
        for i, r in enumerate(extra_images):
            r = r.convert("RGB")
            r.thumbnail((1024, 1024), Image.LANCZOS)
            files.append((f"ref{i}.png", _png(r), "image/png"))
        kw = dict(model=self.model, image=files, prompt=prompt, size=f"{aw}x{ah}", quality=self.quality)
        if self.fidelity:
            kw["input_fidelity"] = "high"
        if mask is not None:
            mc = Image.new("RGBA", (cw, ch), (0, 0, 0, 255))
            mc.paste(mask, (ox, oy))
            kw["mask"] = ("mask.png", _png(mc.resize((aw, ah), Image.NEAREST)), "image/png")
        for attempt in range(4):
            try:
                resp = self.client.images.edit(**kw)
                break
            except Exception as e:
                msg = str(e)
                if "input_fidelity" in msg and "input_fidelity" in kw:
                    kw.pop("input_fidelity")
                    continue
                if attempt == 3 or any(s in msg for s in ("invalid", "Invalid", "401", "403", "safety")):
                    raise
                self.log(f"  API error ({msg[:120]}), retrying ...")
                time.sleep(5 * (attempt + 1))
        out = Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json))).convert("RGB")
        return out.resize((cw, ch), Image.LANCZOS).crop((ox, oy, ox + w, oy + h))

    # -- modes -------------------------------------------------------------
    def _grid(self, views, renders, refs, prompt):
        cols, rows = grid_shape(len(views))
        names = ", ".join(v.name.replace("_", " ") for v in views)
        sheet = make_sheet(renders, self.cell, tuple(self.ctx["bg"]))
        p = (f"Image 1 is a sheet of {len(views)} orthographic renders of the SAME single 3D model, laid out "
             f"in a {cols}x{rows} grid of equal square cells (reading order, left to right, top to bottom: "
             f"{names}). The other image(s) are STYLE REFERENCES. "
             "Task: paint/texture the model in every cell in exactly the art style, colour palette, "
             "material rendering and level of detail of the style references. "
             "All cells show one and the same object, so colours, materials and details must be fully "
             "consistent between views (what is painted on the front must match where it is seen from the "
             f"sides, top and back). Keep every cell exactly where it is. {RULES} {prompt}")
        self.log(f"  grid call: {len(views)} views, sheet {sheet.size[0]}x{sheet.size[1]}")
        out = self.edit(sheet, refs, p)
        return split_sheet(out, len(views), [r.size for r in renders]), out

    def _single(self, view, render, anchors, refs, prompt, guide=None):
        p = (f"Image 1 is an orthographic render of a 3D model seen from the {view.name.replace('_', ' ')}. ")
        extra = []
        if guide is not None:
            extra.append(guide)
            p += ("Image 2 is a rough, low-resolution painted version of this exact view: the final colours, "
                  "materials and design MUST match it - only make it sharper, cleaner and more detailed. ")
        if anchors:
            extra += anchors
            p += (f"The next {len(anchors)} image(s) show the same model, already painted, from other "
                  "angles: stay fully consistent with them (same colours, materials, details). ")
        extra += refs
        p += ("The remaining image(s) are STYLE REFERENCES. Task: paint/texture the model in image 1 in "
              f"exactly the art style, palette and material rendering of the style references. {RULES} {prompt}")
        self.log(f"  view call: {view.name}")
        return self.edit(render, extra, p)

    def stylize_views(self, views, renders, refs, prompt, mode="grid"):
        if mode == "grid":
            cells, self.last_sheet = self._grid(views, renders, refs, prompt)
            return cells
        if mode == "hybrid":  # consistency from the grid, resolution from per-view passes
            cells, self.last_sheet = self._grid(views, renders, refs, prompt)
            return [self._single(v, r, [], refs, prompt, guide=c.resize(r.size, Image.LANCZOS))
                    for v, r, c in zip(views, renders, cells)]
        out = []  # per-view: earlier results are passed on as consistency anchors
        for v, r in zip(views, renders):
            anchors = out[:1] + [o for o in out[-2:] if o is not out[0]]
            out.append(self._single(v, r, anchors, refs, prompt))
        return out

    def fix_views(self, views, renders, masks, refs, prompt):
        p = ("Image 1 is a render of a textured 3D model. The texture is finished everywhere except in the "
             "masked regions, which currently contain only a blurry placeholder. The other image(s) are "
             "style references. Task: repaint the masked regions with clean, detailed texture that "
             "continues the surrounding painted surface seamlessly, in the same art style. Leave everything "
             f"outside the masked regions unchanged. {RULES} {prompt}")
        out = []
        for v, r, m in zip(views, renders, masks):
            self.log(f"  fix call: {v.name}")
            out.append(self.edit(r, refs, p, mask=m))
        return out

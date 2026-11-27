"""Pipeline stages operating on a work directory."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from . import mesh as meshio
from .views import View, parse_views, fit_scale, fit_view, uniform_view, image_size
from .render import render_view, raster_view, make_sheet, split_sheet, preview_sheet, hex_rgb, bilinear
from .raster import interp
from .bake import Baker, smoothstep
from .fix import select_fix_views

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _find(folder, stem):
    for e in IMG_EXT:
        p = Path(folder) / f"{stem}{e}"
        if p.exists():
            return p
    return None


def _framed(v, scale):
    """Views of jobs made before per-view framing have no frame: use the old common scale."""
    return v if v.hw > 0 else uniform_view(v, scale)


def load_job(work):
    job = json.loads((Path(work) / "job.json").read_text())
    job["views"] = [_framed(View(**v), job["scale"]) for v in job["views"]]
    return job


def view_sizes(ctx):
    return [image_size(v, ctx["size"]) for v in ctx["views"]]


def context(work):
    job = load_job(work)
    return dict(job, mesh=meshio.load_work(work), work=Path(work))


# ------------------------------------------------------------------- render
def prepare(model, work, views="6", size=1024, bg="#ffffff", up="y", yaw=0.0,
            unwrap=False, sheet_cell=768, strip=False, log=print):
    work = Path(work)
    (work / "views").mkdir(parents=True, exist_ok=True)
    m = meshio.load_mesh(model, up=up, yaw=yaw, unwrap=unwrap, strip=strip, log=log)
    meshio.save_work(m, work)
    base = parse_views(views)
    scale = fit_scale(m.p, base)
    vs = [fit_view(m.p, v) for v in base]  # each view framed tightly ("zoomed") on the model
    job = dict(source=str(model), size=size, bg=list(hex_rgb(bg)), scale=scale,
               views=[v.to_dict() for v in vs], sheet_cell=sheet_cell)
    (work / "job.json").write_text(json.dumps(job, indent=2))
    ims = []
    for v in vs:
        rgb, _ = render_view(m, v, *image_size(v, size), bg=job["bg"])
        im = Image.fromarray(rgb)
        im.save(work / "views" / f"{v.name}.png")
        ims.append(im)
    make_sheet(ims, sheet_cell, tuple(job["bg"])).save(work / "sheet.png")
    log(f"rendered {len(vs)} views ({len(m.f)} faces) -> {work / 'views'}  + sheet.png "
        f"(order: {', '.join(v.name for v in vs)})")
    for v, im in zip(vs, ims):
        log(f"  {v.name:12s} {im.size[0]}x{im.size[1]}  detail x{(1 / scale) / min(v.hw, v.hh):.2f} vs. unzoomed")
    return work


# ------------------------------------------------------------------ stylize
def make_stylizer(ctx, backend, model, quality, jitter=0.0, log=print, local_opts=None):
    from . import stylize as st
    if backend == "mock":
        return st.MockStylizer(ctx, jitter=jitter)
    if backend == "local":
        from .localai import LocalInpainter
        return LocalInpainter(ctx, log=log, **(local_opts or {}))
    return st.OpenAIStylizer(ctx, model=model, quality=quality, sheet_cell=ctx["sheet_cell"], log=log)


def stylize(work, refs, prompt="", mode="grid", backend="openai", model="gpt-image-2",
            quality="high", jitter=0.0, log=print):
    ctx = context(work)
    sty = make_stylizer(ctx, backend, model, quality, jitter, log)
    renders = [Image.open(ctx["work"] / "views" / f"{v.name}.png").convert("RGB") for v in ctx["views"]]
    ref_imgs = [Image.open(r).convert("RGB") for r in refs]
    out = sty.stylize_views(ctx["views"], renders, ref_imgs, prompt, mode)
    d = ctx["work"] / "stylized"
    d.mkdir(exist_ok=True)
    for old in d.iterdir():
        old.unlink()
    for v, im in zip(ctx["views"], out):
        im.save(d / f"{v.name}.png")
    if getattr(sty, "last_sheet", None) is not None:
        sty.last_sheet.save(ctx["work"] / "stylized_sheet.png")
    log(f"stylized {len(out)} views -> {d}")


# ------------------------------------------------------------------ project
def _stylized_images(ctx, log):
    d = ctx["work"] / "stylized"
    views = ctx["views"]
    paths = [_find(d, v.name) for v in views]
    sheet = _find(d, "sheet")
    if any(paths) or sheet is None:
        if not any(paths):
            raise SystemExit(f"no stylized images in {d} (expected {[v.name for v in views]} or sheet.png)")
        missing = [v.name for v, p in zip(views, paths) if p is None]
        if missing:
            log(f"  [note] no stylized image for {missing} - skipping those views")
        return [Image.open(p).convert("RGB") if p else None for p in paths]
    log(f"splitting {sheet.name} into {len(views)} cells")
    return split_sheet(Image.open(sheet).convert("RGB"), len(views), view_sizes(ctx))


def _bake(ctx, opts, use_fix=True, log=print):
    work = ctx["work"]
    baker = Baker(ctx["mesh"], tex=opts.get("tex", 2048), depth_res=opts.get("depth_res", 2048),
                  sharp=opts.get("sharp", 3.0), feather=opts.get("feather", 0.012),
                  bg=ctx["bg"], log=log)
    (work / "aligned").mkdir(exist_ok=True)
    layers = []
    for v, im in zip(ctx["views"], _stylized_images(ctx, log)):
        if im is None:
            continue
        w, c, info, aligned = baker.view_layer(v, im, align=opts.get("align", True))
        aligned.save(work / "aligned" / f"{v.name}.png")
        note = f" (auto-aligned, was {info['iou_before']:.3f})" if info.get("aligned") else ""
        flag = "  [warn] shape drifted - consider regenerating this view" if info["iou"] < 0.9 else ""
        log(f"  {v.name:12s} silhouette IoU {info['iou']:.3f}{note}{flag}")
        layers.append((w, c))
    col, hole, conf, dis = baker.blend(layers, gain=opts.get("gain", True))
    log(f"  primary views paint {1 - hole.mean():.1%} of the surface")
    fixdir = work / "fix"
    if use_fix and (fixdir / "cameras.json").exists() and (fixdir / "stylized").exists():
        fl = []
        for vd in json.loads((fixdir / "cameras.json").read_text()):
            v = _framed(View(**vd), ctx["scale"])
            p = _find(fixdir / "stylized", v.name)
            if p is None:
                continue
            mp = fixdir / "masks" / f"{v.name}.png"
            pm = None
            if mp.exists():  # only the repainted (masked) pixels of a fix view are used
                pm = Image.fromarray(np.where(np.asarray(Image.open(mp))[..., 3] < 128, 255, 0).astype(np.uint8))
            w, c, info, _ = baker.view_layer(v, Image.open(p).convert("RGB"), align=False, paint_mask=pm)
            fl.append((w, c))
        if fl:
            colf, holef, conff, _ = baker.blend(fl, gain=False)
            a = smoothstep(conff / 0.3)  # fix paint replaces, feathered at the mask border
            a = np.where(hole, 1.0, a)[:, None]  # unpainted texels have no colour to fade from
            ok = ~holef
            col[ok] = (1 - a[ok]) * col[ok] + a[ok] * colf[ok]
            hole &= holef
            log(f"  with fix views: {1 - hole.mean():.1%} painted")
    return baker, col, hole, conf, dis


def regional(pos, val, cell=0.025, sigma=1.0):
    """Average a per-texel value over its 3D neighbourhood (voxel grid + gaussian blur).
    Thin scattered features (crease lines, edges) average out, whole regions stay."""
    lo = pos.min(0)
    idx = np.floor((pos - lo) / cell).astype(np.int64)
    shape = tuple(idx.max(0) + 1)
    flat = np.ravel_multi_index(idx.T, shape)
    n = np.prod(shape)
    s = ndi.gaussian_filter(np.bincount(flat, weights=val, minlength=n).reshape(shape), sigma)
    c = ndi.gaussian_filter(np.bincount(flat, minlength=n).astype(np.float64).reshape(shape), sigma)
    return (s / np.maximum(c, 1e-9)).ravel()[flat].astype(np.float32)


def need_map(hole, conf, dis, mix, pos):
    """Per-texel 0..1: how much a texel needs fixing. A texel counts if it is unpainted, only
    weakly seen, or the views that get mixed there disagree (= visible seam / ghosting) -
    and only when it is part of a whole region like that: thin lines along creases and edges
    are already handled well by the 3D fill and are left alone."""
    n_conf = 1.0 - smoothstep(conf / 0.2)
    n_dis = smoothstep((dis * mix - 0.02) / 0.03)
    raw = np.maximum.reduce([hole.astype(np.float32), n_conf, n_dis]).astype(np.float32)
    region = regional(pos, raw)
    return (raw * smoothstep((region - 0.4) / 0.2)).astype(np.float32)


def project(work, opts=None, log=print):
    opts = opts or {}
    ctx = context(work)
    baker, col, hole, conf, dis = _bake(ctx, opts, log=log)
    tex = baker.to_texture(baker.fill3d(col, hole))
    out = ctx["work"] / "out"
    meshio.export(ctx["mesh"], tex, out)
    preview_sheet(ctx["mesh"], tex).save(out / "preview.png")
    dbg = out / "debug"
    dbg.mkdir(exist_ok=True)
    Image.fromarray((baker.flat_to_map(hole) * 255).astype(np.uint8)).save(dbg / "holes.png")
    Image.fromarray((baker.flat_to_map(conf) * 255).astype(np.uint8)).save(dbg / "confidence.png")
    need = need_map(hole, conf, dis, baker.last_mix, baker.pos)
    Image.fromarray((baker.flat_to_map(need) * 255).astype(np.uint8)).save(dbg / "need.png")
    log(f"baked {baker.T}px texture -> {out}/texture.png, model.obj, model.glb, preview.png "
        f"({hole.mean():.1%} of surface filled by 3D inpaint)")
    return dict(baker=baker, texture=tex, hole=hole, need=need, ctx=ctx)


# ---------------------------------------------------------------------- fix
def fix_prepare(work, max_views=3, opts=None, log=print):
    """Pick extra (zoomed) views aimed at the regions that need fixing, render them with the
    current texture and write repaint masks. Returns the chosen views."""
    import shutil
    opts = opts or {}
    ctx = context(work)
    fixdir = ctx["work"] / "fix"
    if fixdir.exists():
        shutil.rmtree(fixdir)
    baker, col, hole, conf, dis = _bake(ctx, opts, use_fix=False, log=log)
    need = need_map(hole, conf, dis, baker.last_mix, baker.pos)
    log(f"  needs fixing: {(need > 0.5).mean():.1%} of the surface "
        f"(unpainted {hole.mean():.1%}, rest = weakly seen or views disagree)")
    views = select_fix_views(baker, need > 0.5, max_views=max_views, log=log)
    if not views:
        log("nothing significant to fix")
        return []
    (fixdir / "views").mkdir(parents=True)
    (fixdir / "masks").mkdir()
    tex = baker.to_texture(baker.fill3d(col, hole))
    needmap = ndi.maximum_filter(baker.flat_to_map(need), 3)
    Image.fromarray((needmap * 255).astype(np.uint8)).save(fixdir / "need.png")
    mesh = ctx["mesh"]
    for v in views:
        W, H = image_size(v, ctx["size"])
        rgb, obj = render_view(mesh, v, W, H, bg=ctx["bg"], texture=tex, shading=False)
        Image.fromarray(rgb).save(fixdir / "views" / f"{v.name}.png")
        tri, bary, _ = raster_view(mesh, v, W, H)
        m = tri >= 0
        uv = interp(mesh.uv, mesh.f, tri[m], bary[m])
        hm = np.zeros((H, W), np.float32)
        hm[m] = bilinear(needmap, uv[:, 0] * baker.T, (1 - uv[:, 1]) * baker.T)
        k = max(2, min(W, H) // 200)
        hpx = ndi.binary_opening(hm > 0.3, iterations=k)  # drop thin lines (eye / mouth creases ...)
        hpx = ndi.binary_dilation(hpx, iterations=2 * k)  # some overlap with the good paint around
        hpx &= ndi.binary_dilation(m, iterations=3)  # never ask to paint the background
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 3] = np.where(hpx, 0, 255)  # transparent = region to repaint
        Image.fromarray(rgba).save(fixdir / "masks" / f"{v.name}.png")
    (fixdir / "cameras.json").write_text(json.dumps([v.to_dict() for v in views], indent=2))
    log(f"{len(views)} fix views -> {fixdir / 'views'} (masks in fix/masks, put results in fix/stylized)")
    return views


def fix_stylize(work, refs, prompt="", backend="openai", model="gpt-image-2", quality="high", log=print,
                local_opts=None):
    ctx = context(work)
    fixdir = ctx["work"] / "fix"
    views = [_framed(View(**v), ctx["scale"]) for v in json.loads((fixdir / "cameras.json").read_text())]
    if not refs:  # no explicit style reference: use the painted primary views as style anchor
        refs = [p for p in (_find(ctx["work"] / "stylized", v.name) for v in ctx["views"]) if p][:2]
    sty = make_stylizer(ctx, backend, model, quality, log=log, local_opts=local_opts)
    renders = [Image.open(fixdir / "views" / f"{v.name}.png").convert("RGB") for v in views]
    masks = [Image.open(fixdir / "masks" / f"{v.name}.png") for v in views]
    out = sty.fix_views(views, renders, masks, [Image.open(r).convert("RGB") for r in refs], prompt)
    (fixdir / "stylized").mkdir(exist_ok=True)
    for v, im in zip(views, out):
        im.save(fixdir / "stylized" / f"{v.name}.png")


def ai_fix(work, prompt="", max_views=4, opts=None, local_opts=None, log=print):
    """One-click local AI repair: find what needs fixing, render zoomed views of it, let the local
    model repaint only those regions (in the style of your painted views), project back."""
    views = fix_prepare(work, max_views, opts, log)
    if not views:
        return project(work, opts, log)
    fix_stylize(work, [], prompt, backend="local", log=log, local_opts=local_opts)
    return project(work, opts, log)


def need_preview(work, size=384, log=print):
    """Preview of the current result with the regions that need fixing tinted red."""
    ctx = context(work)
    out = ctx["work"] / "out"
    tex = np.asarray(Image.open(out / "texture.png").convert("RGB")).astype(np.float32)
    need = np.asarray(Image.open(out / "debug" / "need.png").convert("L").resize(
        (tex.shape[1], tex.shape[0]), Image.BILINEAR)).astype(np.float32)[..., None] / 255.0
    red = np.array([235, 40, 40], np.float32)
    mix = (tex * (1 - 0.7 * need) + red * 0.7 * need).astype(np.uint8)
    path = out / "need_preview.png"
    preview_sheet(ctx["mesh"], mix, size=size).save(path)
    return path

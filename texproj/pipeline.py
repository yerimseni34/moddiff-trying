"""Pipeline stages operating on a work directory."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from . import mesh as meshio
from .views import View, parse_views, fit_scale
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


def load_job(work):
    job = json.loads((Path(work) / "job.json").read_text())
    job["views"] = [View(**v) for v in job["views"]]
    return job


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
    vs = parse_views(views)
    scale = fit_scale(m.p, vs)
    job = dict(source=str(model), size=size, bg=list(hex_rgb(bg)), scale=scale,
               views=[v.to_dict() for v in vs], sheet_cell=sheet_cell)
    (work / "job.json").write_text(json.dumps(job, indent=2))
    ims = []
    for v in vs:
        rgb, _ = render_view(m, v, scale, size, bg=job["bg"])
        im = Image.fromarray(rgb)
        im.save(work / "views" / f"{v.name}.png")
        ims.append(im)
    make_sheet(ims, sheet_cell, tuple(job["bg"])).save(work / "sheet.png")
    log(f"rendered {len(vs)} views ({len(m.f)} faces) -> {work / 'views'}  + sheet.png "
        f"(order: {', '.join(v.name for v in vs)})")
    return work


# ------------------------------------------------------------------ stylize
def make_stylizer(ctx, backend, model, quality, jitter=0.0, log=print):
    from . import stylize as st
    if backend == "mock":
        return st.MockStylizer(ctx, jitter=jitter)
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
    return split_sheet(Image.open(sheet).convert("RGB"), len(views))


def _bake(ctx, opts, use_fix=True, log=print):
    work = ctx["work"]
    baker = Baker(ctx["mesh"], ctx["scale"], tex=opts.get("tex", 2048), depth_res=opts.get("depth_res", 2048),
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
    col, hole, conf = baker.blend(layers, gain=opts.get("gain", True))
    log(f"  primary views paint {1 - hole.mean():.1%} of the surface")
    fixdir = work / "fix"
    if use_fix and (fixdir / "cameras.json").exists() and (fixdir / "stylized").exists():
        fl = []
        for vd in json.loads((fixdir / "cameras.json").read_text()):
            v = View(**vd)
            p = _find(fixdir / "stylized", v.name)
            if p is None:
                continue
            w, c, info, _ = baker.view_layer(v, Image.open(p).convert("RGB"), align=opts.get("align", True))
            fl.append((w, c))
        if fl:
            colf, holef, _ = baker.blend(fl, gain=False)
            a = smoothstep(conf / 0.3)[:, None]  # keep confident primary paint, fade into fix paint
            ok = ~holef
            col[ok] = a[ok] * col[ok] + (1 - a[ok]) * colf[ok]
            hole &= holef
            log(f"  with fix views: {1 - hole.mean():.1%} painted")
    return baker, col, hole, conf


def project(work, opts=None, log=print):
    opts = opts or {}
    ctx = context(work)
    baker, col, hole, conf = _bake(ctx, opts, log=log)
    tex = baker.to_texture(baker.fill3d(col, hole))
    out = ctx["work"] / "out"
    meshio.export(ctx["mesh"], tex, out)
    preview_sheet(ctx["mesh"], ctx["scale"], tex).save(out / "preview.png")
    dbg = out / "debug"
    dbg.mkdir(exist_ok=True)
    Image.fromarray((baker.flat_to_map(hole) * 255).astype(np.uint8)).save(dbg / "holes.png")
    Image.fromarray((baker.flat_to_map(conf) * 255).astype(np.uint8)).save(dbg / "confidence.png")
    log(f"baked {baker.T}px texture -> {out}/texture.png, model.obj, model.glb, preview.png "
        f"({hole.mean():.1%} of surface filled by 3D inpaint)")
    return dict(baker=baker, texture=tex, hole=hole, ctx=ctx)


# ---------------------------------------------------------------------- fix
def fix_prepare(work, max_views=3, opts=None, log=print):
    """Render extra views aimed at unpainted regions; returns the chosen views."""
    import shutil
    opts = opts or {}
    ctx = context(work)
    fixdir = ctx["work"] / "fix"
    if fixdir.exists():
        shutil.rmtree(fixdir)
    baker, col, hole, _ = _bake(ctx, opts, use_fix=False, log=log)
    views = select_fix_views(baker, hole, max_views=max_views, log=log)
    if not views:
        log("no significant holes - nothing to fix")
        return []
    (fixdir / "views").mkdir(parents=True)
    (fixdir / "masks").mkdir()
    tex = baker.to_texture(baker.fill3d(col, hole))
    holemap = ndi.maximum_filter(baker.flat_to_map(hole), 3)
    size, mesh = ctx["size"], ctx["mesh"]
    for v in views:
        rgb, _ = render_view(mesh, v, ctx["scale"], size, bg=ctx["bg"], texture=tex, shading=False)
        Image.fromarray(rgb).save(fixdir / "views" / f"{v.name}.png")
        tri, bary, _ = raster_view(mesh, v, ctx["scale"], size)
        m = tri >= 0
        uv = interp(mesh.uv, mesh.f, tri[m], bary[m])
        hm = np.zeros((size, size), np.float32)
        hm[m] = bilinear(holemap, uv[:, 0] * baker.T, (1 - uv[:, 1]) * baker.T)
        hpx = ndi.binary_dilation(hm > 0.3, iterations=max(2, size // 128))
        rgba = np.zeros((size, size, 4), np.uint8)
        rgba[..., 3] = np.where(hpx, 0, 255)  # transparent = region to repaint
        Image.fromarray(rgba).save(fixdir / "masks" / f"{v.name}.png")
    (fixdir / "cameras.json").write_text(json.dumps([v.to_dict() for v in views], indent=2))
    log(f"{len(views)} fix views -> {fixdir / 'views'} (masks in fix/masks, put results in fix/stylized)")
    return views


def fix_stylize(work, refs, prompt="", backend="openai", model="gpt-image-2", quality="high", log=print):
    ctx = context(work)
    fixdir = ctx["work"] / "fix"
    views = [View(**v) for v in json.loads((fixdir / "cameras.json").read_text())]
    sty = make_stylizer(ctx, backend, model, quality, log=log)
    renders = [Image.open(fixdir / "views" / f"{v.name}.png").convert("RGB") for v in views]
    masks = [Image.open(fixdir / "masks" / f"{v.name}.png") for v in views]
    out = sty.fix_views(views, renders, masks, [Image.open(r).convert("RGB") for r in refs], prompt)
    (fixdir / "stylized").mkdir(exist_ok=True)
    for v, im in zip(views, out):
        im.save(fixdir / "stylized" / f"{v.name}.png")

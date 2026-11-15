"""texproj command line."""
import argparse
import time
from . import pipeline as pl


def _add_render(p):
    p.add_argument("model", help="OBJ / GLB / PLY / STL model")
    p.add_argument("-o", "--work", default="work", help="work directory (default: work)")
    p.add_argument("--views", default="6", help="4 | 6 | 8 | 10 or custom 'name:az:el,...' (default 6)")
    p.add_argument("--size", type=int, default=1024, help="render size per view")
    p.add_argument("--bg", default="#ffffff", help="background colour of renders")
    p.add_argument("--up", choices=["y", "z"], default="y", help="model up axis (z = raw Blender coords)")
    p.add_argument("--yaw", type=float, default=0.0, help="rotate model around up axis (deg) so its front faces the 'front' view")
    p.add_argument("--unwrap", action="store_true", help="force a fresh xatlas UV unwrap")
    p.add_argument("--sheet-cell", type=int, default=768, help="cell size of the grid sheet")


def _add_stylize(p):
    p.add_argument("--ref", nargs="+", default=[], help="style reference image(s)")
    p.add_argument("--prompt", default="", help="extra instructions (what the object is, materials, ...)")
    p.add_argument("--mode", choices=["grid", "per-view", "hybrid"], default="grid")
    p.add_argument("--backend", choices=["openai", "mock"], default="openai")
    p.add_argument("--image-model", default="gpt-image-2")
    p.add_argument("--quality", default="high")
    p.add_argument("--mock-jitter", type=float, default=0.0, help=argparse.SUPPRESS)


def _add_project(p):
    p.add_argument("--tex", type=int, default=2048, help="output texture size")
    p.add_argument("--sharp", type=float, default=3.0, help="blend sharpness (higher = less cross-fading/ghosting)")
    p.add_argument("--feather", type=float, default=0.012, help="edge feather as fraction of view size")
    p.add_argument("--no-align", action="store_true", help="disable automatic silhouette alignment")
    p.add_argument("--no-gain", action="store_true", help="disable per-view colour equalisation")


def _opts(a):
    return dict(tex=a.tex, sharp=a.sharp, feather=a.feather, align=not a.no_align, gain=not a.no_gain,
                depth_res=max(2048, a.tex))


def _sty(a):
    return dict(backend=a.backend, model=a.image_model, quality=a.quality)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="texproj", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render", help="1) render the views (+ sheet.png) of a model")
    _add_render(p)
    p = sub.add_parser("stylize", help="2) paint the views with the image API")
    p.add_argument("work")
    _add_stylize(p)
    p = sub.add_parser("project", help="3) bake <work>/stylized/* onto the model")
    p.add_argument("work")
    _add_project(p)
    p = sub.add_parser("fix", help="4) AI-repaint regions no view could see, then re-bake")
    p.add_argument("work")
    p.add_argument("--max-views", type=int, default=3)
    p.add_argument("--render-only", action="store_true", help="only write fix/views + fix/masks (manual workflow)")
    _add_stylize(p)
    _add_project(p)
    p = sub.add_parser("run", help="everything: render -> stylize -> project -> fix")
    _add_render(p)
    _add_stylize(p)
    _add_project(p)
    p.add_argument("--fix-views", type=int, default=3, help="max extra AI fix views (0 = off)")
    a = ap.parse_args(argv)

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    if a.cmd == "render":
        pl.prepare(a.model, a.work, a.views, a.size, a.bg, a.up, a.yaw, a.unwrap, a.sheet_cell, log)
    elif a.cmd == "stylize":
        pl.stylize(a.work, a.ref, a.prompt, a.mode, jitter=a.mock_jitter, log=log, **_sty(a))
    elif a.cmd == "project":
        pl.project(a.work, _opts(a), log)
    elif a.cmd == "fix":
        if pl.fix_prepare(a.work, a.max_views, _opts(a), log) and not a.render_only:
            pl.fix_stylize(a.work, a.ref, a.prompt, log=log, **_sty(a))
            pl.project(a.work, _opts(a), log)
    elif a.cmd == "run":
        pl.prepare(a.model, a.work, a.views, a.size, a.bg, a.up, a.yaw, a.unwrap, a.sheet_cell, log)
        pl.stylize(a.work, a.ref, a.prompt, a.mode, jitter=a.mock_jitter, log=log, **_sty(a))
        if a.fix_views > 0 and pl.fix_prepare(a.work, a.fix_views, _opts(a), log):
            pl.fix_stylize(a.work, a.ref, a.prompt, log=log, **_sty(a))
        pl.project(a.work, _opts(a), log)


if __name__ == "__main__":
    main()

"""Mesh loading / normalisation / UV unwrapping / export."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class Mesh:
    v: np.ndarray            # (V,3) original coordinates (used for export)
    f: np.ndarray            # (F,3)
    uv: np.ndarray           # (V,2), v up
    p: np.ndarray = None     # (V,3) processing frame: Y up, front +Z, centred, ~unit size
    n: np.ndarray = None     # (V,3) smooth normals in processing frame
    vcol: np.ndarray = None  # (V,3) float 0..1 or None
    base_tex: np.ndarray = None  # (H,W,3) uint8 or None


def smooth_normals(p, f):
    fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])  # area weighted
    key = np.round(p / (np.abs(p).max() + 1e-12) * 1e5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[f[:, k]], fn)
    n = acc[inv]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def _finish(mesh, up="y", yaw=0.0):
    p = mesh.v.astype(np.float64).copy()
    if up == "z":  # Blender style: Z up, front -Y  ->  Y up, front +Z
        p = np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)
    if yaw:
        a = np.radians(yaw)
        rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        p = p @ rot.T
    lo, hi = p.min(0), p.max(0)
    p = (p - (lo + hi) / 2) / (np.linalg.norm(hi - lo) / 2 + 1e-12)
    mesh.p = p
    mesh.n = smooth_normals(p, mesh.f)
    return mesh


def load_mesh(path, up="y", yaw=0.0, unwrap=False, strip=False, log=print):
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=False)
    v = np.asarray(m.vertices, np.float64)
    f = np.asarray(m.faces, np.int64)
    uv = vcol = base = None
    vis = m.visual
    if getattr(vis, "kind", None) == "texture":
        if vis.uv is not None and len(vis.uv) == len(v):
            uv = np.asarray(vis.uv, np.float64)
        mat = vis.material
        img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if img is not None:
            base = np.asarray(img.convert("RGB"))
    elif getattr(vis, "kind", None) == "vertex":
        vcol = np.asarray(vis.vertex_colors)[:, :3] / 255.0
    if strip:  # ignore existing texture / vertex colours: render as plain clay
        base = vcol = None
    if uv is None or unwrap:
        try:
            import xatlas
        except ImportError as e:
            raise SystemExit("Model has no UVs (or --unwrap given): pip install xatlas") from e
        log(f"UV unwrap (xatlas) for {len(f)} faces ...")
        vmap, f, uv = xatlas.parametrize(v.astype(np.float32), f.astype(np.uint32))
        v, f, uv = v[vmap], f.astype(np.int64), uv.astype(np.float64)
        if vcol is not None:
            vcol = vcol[vmap]
        base = None  # old texture no longer matches new UVs
    return _finish(Mesh(v=v, f=f, uv=uv, vcol=vcol, base_tex=base), up, yaw)


def save_work(mesh, workdir):
    d = dict(v=mesh.v, f=mesh.f, uv=mesh.uv, p=mesh.p, n=mesh.n)
    if mesh.vcol is not None:
        d["vcol"] = mesh.vcol
    np.savez_compressed(Path(workdir) / "mesh.npz", **d)
    if mesh.base_tex is not None:
        Image.fromarray(mesh.base_tex).save(Path(workdir) / "base.png")


def load_work(workdir):
    z = np.load(Path(workdir) / "mesh.npz")
    m = Mesh(v=z["v"], f=z["f"], uv=z["uv"], p=z["p"], n=z["n"],
             vcol=z["vcol"] if "vcol" in z else None)
    b = Path(workdir) / "base.png"
    if b.exists():
        m.base_tex = np.asarray(Image.open(b).convert("RGB"))
    return m


def export_clay(mesh, path):
    """Untextured (plain grey) GLB, for side-by-side comparison."""
    import trimesh
    tm = trimesh.Trimesh(mesh.v, mesh.f, process=False)
    tm.visual = trimesh.visual.ColorVisuals(tm, vertex_colors=[150, 150, 155, 255])
    tm.export(path)


def export(mesh, texture, outdir, name="model"):
    """Write OBJ+MTL+PNG and GLB with the baked texture."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(texture)
    img.save(outdir / "texture.png")
    (outdir / f"{name}.mtl").write_text(
        "newmtl baked\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd texture.png\n")
    with open(outdir / f"{name}.obj", "w") as fh:
        fh.write(f"mtllib {name}.mtl\nusemtl baked\n")
        np.savetxt(fh, mesh.v, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, mesh.uv, fmt="vt %.6f %.6f")
        i = mesh.f + 1
        np.savetxt(fh, np.stack([i[:, 0], i[:, 0], i[:, 1], i[:, 1], i[:, 2], i[:, 2]], 1),
                   fmt="f %d/%d %d/%d %d/%d")
    try:
        import trimesh
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=img, metallicFactor=0.0, roughnessFactor=1.0)
        tm = trimesh.Trimesh(mesh.v, mesh.f, process=False,
                             visual=trimesh.visual.TextureVisuals(uv=mesh.uv, material=mat))
        tm.export(outdir / f"{name}.glb")
    except Exception as e:  # GLB is a convenience; OBJ is already written
        print(f"[warn] GLB export failed: {e}")

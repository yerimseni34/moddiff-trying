"""Browser UI (gradio): render -> paint in ChatGPT (or API) -> project -> fix."""
import os
import shutil
from pathlib import Path
import gradio as gr
from . import pipeline as pl
from .stylize import manual_grid_prompt, manual_fix_prompt

ROOT = Path("work_ui")


def _run(fn, *a, **k):
    """Run a pipeline stage, collecting its log; turn failures into UI errors."""
    lines = []
    try:
        out = fn(*a, log=lines.append, **k)
    except BaseException as e:  # pipeline uses SystemExit for user-facing errors
        raise gr.Error(f"{type(e).__name__}: {e}")
    return out, "\n".join(lines)


def _need(work):
    if not work:
        raise gr.Error("Önce 1. adımda modeli yükleyip render al.")
    return Path(work)


def _outputs(work):
    out = Path(work) / "out"
    files = [str(out / n) for n in ("texture.png", "model.glb", "model.obj", "model.mtl") if (out / n).exists()]
    return str(out / "model.glb"), str(out / "preview.png"), files


def do_render(model, views, up, yaw, unwrap, size):
    if not model:
        raise gr.Error("Bir model dosyası seç (OBJ / GLB / PLY / STL).")
    work = ROOT / Path(model).stem
    if work.exists():
        shutil.rmtree(work)
    _, log = _run(pl.prepare, model, work, views=views, size=int(size), up=up, yaw=float(yaw), unwrap=unwrap)
    job = pl.load_job(work)
    return (str(work), str(work / "sheet.png"), [str(work / "views" / f"{v.name}.png") for v in job["views"]],
            manual_grid_prompt(job["views"]), log)


def _store(folder, names, files):
    """Copy uploaded files into folder, matching them to view names (by file name, else by order)."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        old.unlink()
    files = sorted(Path(f) for f in files)
    left = list(files)
    for n in sorted(names, key=len, reverse=True):
        hit = next((f for f in left if n in f.stem.lower()), None)
        if hit:
            left.remove(hit)
            shutil.copy(hit, folder / f"{n}{hit.suffix.lower()}")
    if len(left) == len(files) == len(names):  # no names matched: assume same order
        for n, f in zip(names, files):
            shutil.copy(f, folder / f"{n}{f.suffix.lower()}")


def do_project(work, sheet, view_files, tex):
    work = _need(work)
    names = [v.name for v in pl.load_job(work)["views"]]
    d = work / "stylized"
    if view_files:
        _store(d, names, view_files)
    elif sheet:
        d.mkdir(exist_ok=True)
        for old in d.iterdir():
            old.unlink()
        shutil.copy(sheet, d / f"sheet{Path(sheet).suffix.lower() or '.png'}")
    elif not d.exists():
        raise gr.Error("Boyanmış sheet'i (veya açı görsellerini) yükle.")
    _, log = _run(pl.project, work, dict(tex=int(tex)))
    return (*_outputs(work), log)


def do_api(work, refs, prompt, mode, model, tex):
    work = _need(work)
    if not os.environ.get("OPENAI_API_KEY"):
        raise gr.Error("OPENAI_API_KEY tanımlı değil (API anahtarı ChatGPT aboneliğinden ayrıdır).")
    if not refs:
        raise gr.Error("En az bir stil referansı yükle.")
    _, log1 = _run(pl.stylize, work, list(refs), prompt or "", mode, model=model)
    _, log2 = _run(pl.project, work, dict(tex=int(tex)))
    return (*_outputs(work), log1 + "\n" + log2)


def do_fix_render(work, tex):
    work = _need(work)
    views, log = _run(pl.fix_prepare, work, 3, dict(tex=int(tex)))
    return [str(work / "fix" / "views" / f"{v.name}.png") for v in views], manual_fix_prompt(), log


def do_fix_apply(work, files, tex):
    work = _need(work)
    cams = work / "fix" / "cameras.json"
    if not cams.exists() or not files:
        raise gr.Error("Önce düzeltme açılarını üret, boyat ve sonuçları buraya yükle.")
    import json
    _store(work / "fix" / "stylized", [c["name"].lower() for c in json.loads(cams.read_text())], files)
    _, log = _run(pl.project, work, dict(tex=int(tex)))
    return (*_outputs(work), log)


def build():
    with gr.Blocks(title="texproj") as app:
        work = gr.State("")
        gr.Markdown("# texproj\nModeli render et → ChatGPT'de referans stilde boyat → modele geri projekte et.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1 · Model")
                model = gr.File(label="Model (OBJ / GLB / PLY / STL)", type="filepath",
                                file_types=[".obj", ".glb", ".gltf", ".ply", ".stl"])
                with gr.Row():
                    views = gr.Dropdown(["4", "6", "8", "10"], value="6", label="Açı sayısı",
                                        info="Elle ChatGPT için 6 önerilir (3:2 sheet)")
                    size = gr.Dropdown([768, 1024, 1536], value=1024, label="Render boyutu")
                with gr.Accordion("Yön / UV ayarları", open=False):
                    up = gr.Radio(["y", "z"], value="y", label="Yukarı ekseni", info="z = ham Blender koordinatı")
                    yaw = gr.Slider(0, 270, value=0, step=90, label="Modeli döndür (ön yüz 'front' açısına baksın)")
                    unwrap = gr.Checkbox(label="UV'yi yeniden aç (mirror / üst üste binen UV varsa)")
                b_render = gr.Button("Render al", variant="primary")
            with gr.Column(scale=2):
                sheet = gr.Image(label="sheet.png — bunu indir, stil referansınla birlikte ChatGPT'ye ver",
                                 type="filepath", interactive=False, height=380)
                prompt_out = gr.Textbox(label="ChatGPT'ye yapıştırılacak prompt (sonuna objeyi/malzemeleri ekle)",
                                        lines=4, interactive=False)
                with gr.Accordion("Tek tek açılar", open=False):
                    gallery = gr.Gallery(columns=6, height=180, show_label=False)
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 2 · Boyanmış görseli yükle ve projekte et")
                with gr.Tab("ChatGPT'den (elle)"):
                    sty_sheet = gr.Image(label="Boyanmış sheet", type="filepath", height=220)
                    sty_files = gr.File(label="…veya açıları tek tek (front.png, right.png, …)",
                                        file_count="multiple", type="filepath")
                    tex = gr.Dropdown([1024, 2048, 4096], value=2048, label="Texture boyutu")
                    b_project = gr.Button("Projekte et", variant="primary")
                with gr.Tab("API ile (otomatik)"):
                    refs = gr.File(label="Stil referansları", file_count="multiple", type="filepath")
                    api_prompt = gr.Textbox(label="Ek açıklama (obje, malzemeler)")
                    with gr.Row():
                        mode = gr.Dropdown(["grid", "hybrid", "per-view"], value="hybrid", label="Mod")
                        img_model = gr.Textbox(value="gpt-image-2", label="Model")
                    b_api = gr.Button("Boyat + projekte et")
                gr.Markdown("### 3 · Görülmeyen yerleri düzelt (isteğe bağlı)")
                b_fix = gr.Button("Düzeltme açılarını üret")
                fix_gallery = gr.Gallery(columns=3, height=160, label="Bunları indirip ChatGPT'de boyat")
                fix_prompt = gr.Textbox(label="Düzeltme prompt'u", lines=3, interactive=False)
                fix_files = gr.File(label="Boyanmış düzeltme görselleri (aynı sırayla/isimle)",
                                    file_count="multiple", type="filepath")
                b_fix_apply = gr.Button("Düzeltmeleri uygula")
            with gr.Column(scale=2):
                viewer = gr.Model3D(label="Sonuç", height=460, clear_color=[0.18, 0.19, 0.21, 1.0])
                preview = gr.Image(label="Önizleme (8 açı)", interactive=False)
                files = gr.File(label="Çıktılar", file_count="multiple", interactive=False)
        log = gr.Textbox(label="Log (IoU < 0.9 olan açıyı yeniden ürettir)", lines=8, interactive=False)

        res = [viewer, preview, files, log]
        b_render.click(do_render, [model, views, up, yaw, unwrap, size], [work, sheet, gallery, prompt_out, log])
        b_project.click(do_project, [work, sty_sheet, sty_files, tex], res)
        b_api.click(do_api, [work, refs, api_prompt, mode, img_model, tex], res)
        b_fix.click(do_fix_render, [work, tex], [fix_gallery, fix_prompt, log])
        b_fix_apply.click(do_fix_apply, [work, fix_files, tex], res)
    return app


def launch(root="work_ui", port=7860):
    global ROOT
    ROOT = Path(root).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    build().launch(server_port=port, inbrowser=True, allowed_paths=[str(ROOT)])

"""Browser UI: model -> view images -> (paint them in ChatGPT) -> upload -> result."""
import shutil
import zipfile
from pathlib import Path
import gradio as gr
from PIL import Image
from . import pipeline as pl
from . import mesh as meshio
from .align import match_to_views
from .stylize import RULES

ROOT = Path("work_ui")
MAX_VIEWS = 10
VIEW_TR = {"front": "Ön", "right": "Sağ", "back": "Arka", "left": "Sol", "top": "Üst", "bottom": "Alt",
           "front_right": "Ön-sağ", "back_right": "Arka-sağ", "back_left": "Arka-sol", "front_left": "Ön-sol"}

CSS = """
.step-title {margin-bottom: 0 !important}
.slot-row {align-items: stretch}
"""


def _tr(name):
    return VIEW_TR.get(name, name)


def _run(fn, *a, **k):
    lines = []
    try:
        out = fn(*a, log=lines.append, **k)
    except BaseException as e:  # pipeline uses SystemExit for user-facing errors
        raise gr.Error(f"{e}")
    return out, "\n".join(lines)


def _prompt(view_name):
    return (f"This image is an untextured grey orthographic render of a 3D model, seen from the "
            f"{view_name.replace('_', ' ').upper()}. Paint/texture it [in the style of the attached reference "
            "image]. For every view after the first one, also attach your first painted result and keep "
            f"exactly the same colours, materials and details so all views match. {RULES}")


# ---------------------------------------------------------------- step 1
def step1(model, strip, views, up, yaw):
    if not model:
        raise gr.Error("Önce bir model dosyası seç (GLB / OBJ / PLY / STL).")
    work = ROOT / Path(model).stem
    if work.exists():
        shutil.rmtree(work)
    _, log = _run(pl.prepare, model, work, views=views, size=1024, up=up, yaw=float(yaw), strip=strip)
    job = pl.load_job(work)
    names = [v.name for v in job["views"]]
    meshio.export_clay(meshio.load_work(work), work / "clay.glb")
    zp = work / "gorseller.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for n in names:
            z.write(work / "views" / f"{n}.png", f"{n}.png")
    renders = [str(work / "views" / f"{n}.png") for n in names]
    gallery = [(p, _tr(n)) for p, n in zip(renders, names)]
    slots = _slot_updates(names, renders, [None] * len(names))
    return [str(work), names, gallery, str(zp), _prompt("front"),
            gr.update(visible=True), gr.update(visible=True), None, None, *slots, log]


def _slot_updates(names, renders, painted):
    """Updates for MAX_VIEWS slot groups: (group, render image, painted image) each."""
    out = []
    for i in range(MAX_VIEWS):
        if i < len(names):
            out += [gr.update(visible=True),
                    gr.update(value=renders[i], label=f"{_tr(names[i])} — render"),
                    gr.update(value=painted[i], label=f"{_tr(names[i])} — boyanmış")]
        else:
            out += [gr.update(visible=False), gr.update(value=None), gr.update(value=None)]
    return out


# ---------------------------------------------------------------- step 2
def auto_fill(files, work, names):
    """Bulk upload: put each painted image into the slot whose silhouette it matches."""
    if not files:
        return [gr.update()] * (3 * MAX_VIEWS) + [""]
    if not work:
        raise gr.Error("Önce 1. adımı yap.")
    work = Path(work)
    renders = [str(work / "views" / f"{n}.png") for n in names]
    imgs = [Image.open(f).convert("RGB") for f in files]
    assign, scores = match_to_views(imgs, [Image.open(r) for r in renders])
    painted, notes = [], []
    for n, a, s in zip(names, assign, scores):
        if a is None or s < 0.5:
            painted.append(None)
            notes.append(f"{_tr(n)}: görsel yok")
        else:
            painted.append(str(files[a]))
            notes.append(f"{_tr(n)}: eşleşti (benzerlik {s:.2f})")
    extra = len(files) - sum(p is not None for p in painted)
    if extra > 0:
        notes.append(f"{extra} görsel hiçbir açıya uymadı, kullanılmadı")
    return _slot_updates(names, renders, painted) + ["\n".join(notes)]


def step2(work, names, tex, *painted):
    if not work:
        raise gr.Error("Önce 1. adımı yap.")
    work = Path(work)
    d = work / "stylized"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    n_ok = 0
    for n, p in zip(names, painted):
        if p is not None:
            Image.open(p).convert("RGB").save(d / f"{n}.png")
            n_ok += 1
    if n_ok == 0:
        raise gr.Error("Hiç boyanmış görsel yüklenmedi.")
    if (work / "fix").exists():  # an earlier AI repair belongs to the old paint - drop it
        shutil.rmtree(work / "fix")
    _, log = _run(pl.project, work, dict(tex=int(tex)))
    out = work / "out"
    needp, _ = _run(pl.need_preview, work)
    return (str(work / "clay.glb"), str(out / "model.glb"), str(out / "preview.png"),
            _files(out), _summary(log, n_ok, len(names)), str(needp), [], log)


def _files(out):
    return [str(out / f) for f in ("model.glb", "texture.png", "model.obj", "model.mtl")]


def step3_ai(work, tex, prompt, strength, nviews, model, ip):
    if not work or not (Path(work) / "out" / "texture.png").exists():
        raise gr.Error("Önce 2. adımda Çalıştır'a bas.")
    from . import localai
    ok, msg = localai.available()
    if not ok:
        raise gr.Error(msg)
    work = Path(work)
    local = dict(model=(model or "").strip() or None, strength=float(strength), ip_scale=float(ip))
    _, log = _run(pl.ai_fix, work, prompt or "", int(nviews), dict(tex=int(tex)), local)
    out = work / "out"
    pairs = []
    fx = work / "fix"
    if (fx / "cameras.json").exists():
        import json
        for i, c in enumerate(json.loads((fx / "cameras.json").read_text()), 1):
            pairs += [(str(fx / "views" / f"{c['name']}.png"), f"{i} · önce"),
                      (str(fx / "stylized" / f"{c['name']}.png"), f"{i} · AI sonrası")]
    lines = [msg]
    for l in log.splitlines():
        if "needs fixing" in l:
            lines.append("Düzeltilmesi gereken: yüzeyin %" + l.split("fixing:")[1].split("%")[0].strip() + "'i.")
        if "with fix views" in l:
            lines.append("AI düzeltmesinden sonra gerçek boyalı yüzey: %" + l.split(":")[-1].split("%")[0].strip())
        if "nothing significant" in l:
            lines.append("Düzeltilecek önemli bir yer bulunamadı.")
    needp, _ = _run(pl.need_preview, work)
    return (str(out / "model.glb"), str(out / "preview.png"), _files(out), "\n".join(lines), str(needp),
            pairs, log)


def _summary(log, n_ok, n_all):
    """Short, human-readable result summary from the pipeline log."""
    lines = [f"{n_ok}/{n_all} açı kullanıldı."]
    weak = [l.split()[0] for l in log.splitlines() if "IoU" in l and "[warn]" in l]
    for l in log.splitlines():
        if "primary views paint" in l:
            pct = l.split("paint")[1].split("%")[0].strip()
            lines.append(f"Yüzeyin %{pct}'i görsellerden boyandı, gerisi (hiçbir açının görmediği yerler) "
                         "çevresindeki renklerle dolduruldu.")
    if weak:
        lines.append("Şekli kaymış açılar (yeniden boyatman iyi olur): " + ", ".join(_tr(w) for w in weak))
    return "\n".join(lines)


# ---------------------------------------------------------------- layout
def build():
    with gr.Blocks(title="texproj") as app:
        work = gr.State("")
        names = gr.State([])
        gr.Markdown("# texproj\n**1.** Modeli yükle → **2.** Görselleri indir, ChatGPT'de boyat → "
                    "**3.** Boyanmış görselleri geri yükle, çalıştır → sonucu burada gör.")

        # ---- 1
        gr.Markdown("## 1 · Model", elem_classes="step-title")
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.File(label="Model (GLB / OBJ / PLY / STL)", type="filepath",
                                file_types=[".obj", ".glb", ".gltf", ".ply", ".stl"])
                strip = gr.Checkbox(value=True, label="Modelin kendi texture'ını yok say (gri render)")
                with gr.Accordion("Ayarlar", open=False):
                    views = gr.Dropdown(["4", "6", "8", "10"], value="6", label="Açı sayısı")
                    up = gr.Radio(["y", "z"], value="y", label="Yukarı ekseni (z = ham Blender koordinatı)")
                    yaw = gr.Slider(0, 270, value=0, step=90, label="Modeli döndür (ön yüz öne baksın)")
                b1 = gr.Button("Görselleri hazırla", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Boyatılacak görseller", columns=6, height=230,
                                     object_fit="contain")
                with gr.Row():
                    zipf = gr.File(label="Hepsini indir (zip)", interactive=False)
                    prompt = gr.Textbox(label="ChatGPT prompt'u", lines=4, interactive=False, buttons=["copy"])

        # ---- 2
        with gr.Column(visible=False) as s2:
            gr.Markdown("## 2 · Boyanmış görselleri yükle", elem_classes="step-title")
            gr.Markdown("Hepsini birden sürükle; isimleri ve sırası önemli değil, program şekillerine bakıp "
                        "hangi açıya ait olduklarını kendisi buluyor. Yanlış eşleşen olursa aşağıdaki "
                        "kutulardan tek tek değiştirebilirsin.")
            with gr.Row():
                bulk = gr.File(label="Boyanmış görseller (toplu)", file_count="multiple", type="filepath",
                               file_types=["image"])
                match_info = gr.Textbox(label="Eşleşme", lines=6, interactive=False)
            slot_groups, slot_r, slot_p = [], [], []
            with gr.Row():
                for i in range(MAX_VIEWS):
                    with gr.Column(visible=False, min_width=150) as g:
                        r = gr.Image(type="filepath", interactive=False, height=150, show_label=True,
                                     buttons=[])
                        p = gr.Image(type="filepath", height=150, show_label=True,
                                     sources=["upload", "clipboard"])
                    slot_groups.append(g)
                    slot_r.append(r)
                    slot_p.append(p)
            with gr.Row():
                tex = gr.Dropdown([1024, 2048, 4096], value=2048, label="Texture boyutu", scale=1)
                b2 = gr.Button("Çalıştır", variant="primary", scale=3)

        # ---- 3
        with gr.Column(visible=False) as s3:
            gr.Markdown("## 3 · Sonuç", elem_classes="step-title")
            summary = gr.Textbox(label="Özet", lines=3, interactive=False)
            with gr.Row():
                clay = gr.Model3D(label="Texture'sız", height=520, clear_color=[0.18, 0.19, 0.21, 1.0])
                result = gr.Model3D(label="Texture'lı", height=520, clear_color=[0.18, 0.19, 0.21, 1.0])
            with gr.Row():
                preview = gr.Image(label="8 açıdan önizleme", interactive=False, scale=2)
                files = gr.File(label="İndir", file_count="multiple", interactive=False, scale=1)

            gr.Markdown("## 4 · AI ile düzelt (yerel, kendi ekran kartında)", elem_classes="step-title")
            gr.Markdown("Kırmızı yerler: hiçbir açının görmediği, zayıf gördüğü ya da açıların birbiriyle "
                        "çeliştiği (dikiş izi) yerler. **AI ile düzelt** bu yerlere yakından bakan ekstra "
                        "açılar çeker, senin boyattığın görselin stilinde sadece bu yerleri yeniden boyar ve "
                        "modele geri yansıtır. İlk çalıştırmada model indirilir (~10 GB).")
            with gr.Row():
                needimg = gr.Image(label="Düzeltilecek yerler (kırmızı)", interactive=False, scale=2)
                with gr.Column(scale=1):
                    ai_prompt = gr.Textbox(label="Model ne? (kısa, İngilizce)",
                                           placeholder="anime girl, blue hair, white bikini, tanned skin")
                    with gr.Accordion("AI ayarları", open=False):
                        ai_views = gr.Slider(1, 8, value=4, step=1, label="Düzeltme açısı sayısı")
                        ai_strength = gr.Slider(0.3, 1.0, value=0.9, step=0.05,
                                                label="Ne kadar yeniden boyasın (düşük = mevcut renge sadık)")
                        ai_ip = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                          label="Stil referansı gücü (0 = kapalı)")
                        ai_model = gr.Textbox(label="Model (boş = SDXL inpainting; HF id veya .safetensors yolu)")
                    b3 = gr.Button("AI ile düzelt", variant="primary")
            fixgal = gr.Gallery(label="AI'ın düzelttiği görüntüler (önce / sonra)", columns=4, height=260,
                                object_fit="contain")
        with gr.Accordion("Teknik log", open=False):
            log = gr.Textbox(lines=12, show_label=False, interactive=False)

        slot_outs = [x for trio in zip(slot_groups, slot_r, slot_p) for x in trio]
        b1.click(step1, [model, strip, views, up, yaw],
                 [work, names, gallery, zipf, prompt, s2, s3, clay, result, *slot_outs, log])
        bulk.upload(auto_fill, [bulk, work, names], [*slot_outs, match_info])
        b2.click(step2, [work, names, tex, *slot_p], [clay, result, preview, files, summary, needimg, fixgal, log])
        b3.click(step3_ai, [work, tex, ai_prompt, ai_strength, ai_views, ai_model, ai_ip],
                 [result, preview, files, summary, needimg, fixgal, log])
    return app


def launch(root="work_ui", port=7860):
    global ROOT
    ROOT = Path(root).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    build().launch(server_port=port, inbrowser=True, allowed_paths=[str(ROOT)], css=CSS)

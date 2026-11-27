"""Local AI repair (no API, no subscription): SDXL inpainting on your own GPU.

Used for the "fix" stage: extra zoomed views look at the regions that need fixing
(unpainted, only weakly seen, or where the painted views disagree = seams), the
regions are masked, and SDXL repaints only the masked pixels so that they continue
the surrounding paint. An IP-Adapter feeds one of your painted views in as a style
reference so the repaint stays in the same style. Results are projected back onto
the model exactly like the ChatGPT-painted views.

Needs:  pip install torch (CUDA build) diffusers transformers accelerate safetensors
First run downloads the models (~7 GB SDXL inpainting + ~3.5 GB IP-Adapter) into the
Hugging Face cache; later runs reuse them.
"""
import time
import numpy as np
from PIL import Image, ImageChops, ImageFilter

DEFAULT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
IP_REPO, IP_SUB, IP_WEIGHT, IP_ENCODER = ("h94/IP-Adapter", "sdxl_models",
                                          "ip-adapter-plus_sdxl_vit-h.safetensors", "models/image_encoder")
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"

POSITIVE = "seamless continuation of the surrounding texture, same art style and colours, clean detail, flat lighting"
NEGATIVE = "blurry, smeared, low detail, seam, visible border, text, watermark, cast shadow, harsh light, deformed"

_CACHE = {}  # keep the loaded pipeline between runs (loading takes a while)


def available():
    """(ok, message) - whether the local AI dependencies are installed and a GPU is usable."""
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, f"Yerel AI paketleri kurulu değil ({e.name}). kur_ai.bat'ı çalıştır."
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        return True, f"GPU: {name} ({vram:.0f} GB)"
    return True, "GPU bulunamadı - CPU'da çalışır ama çok yavaştır (görsel başına dakikalar)."


def _load(model, use_ip, device, log):
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
    key = (model, use_ip, device)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()  # only one pipeline in memory at a time
    # half precision everywhere: fp16 on the GPU, bf16 on CPU (fp32 SDXL needs >15 GB RAM)
    dtype = torch.float16 if device in ("cuda", "mps") else torch.bfloat16
    t0 = time.time()
    log(f"  loading {model} (first run downloads it, this can take a while) ...")
    if str(model).lower().endswith((".safetensors", ".ckpt")):
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(model, torch_dtype=dtype)
    else:
        try:  # fp16 files are half the download; they are upcast when running in fp32
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model, torch_dtype=dtype, variant="fp16")
        except (OSError, ValueError):  # repo has no fp16 variant
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model, torch_dtype=dtype)
    if device == "cuda":
        try:  # the stock SDXL VAE overflows in fp16; this drop-in replacement does not
            pipe.vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=dtype)
        except Exception as e:
            log(f"  [note] fp16 VAE not loaded ({str(e)[:80]}), using the model's own VAE")
    ip = False
    if use_ip:
        try:
            pipe.load_ip_adapter(IP_REPO, subfolder=IP_SUB, weight_name=IP_WEIGHT, image_encoder_folder=IP_ENCODER)
            ip = True
        except Exception as e:
            log(f"  [warn] IP-Adapter (style reference) could not be loaded, continuing without it: "
                f"{str(e)[:120]}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        if vram < 16:  # e.g. RTX 5070 12 GB: keep only the active part on the GPU
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    log(f"  model ready in {time.time() - t0:.0f}s")
    _CACHE[key] = (pipe, ip)
    return pipe, ip


def _to_mask(m, size):
    """Accepts the fix-stage RGBA mask (transparent = repaint) or an L mask (white = repaint)."""
    if m.mode == "RGBA":
        a = np.asarray(m)[..., 3]
        m = Image.fromarray(np.where(a < 128, 255, 0).astype(np.uint8))
    return m.convert("L").resize(size, Image.NEAREST)


class LocalInpainter:
    def __init__(self, ctx, model=DEFAULT_MODEL, strength=0.9, steps=30, guidance=6.0, ip_scale=0.5,
                 seed=0, blur=12, device=None, log=print):
        import torch
        self.ctx, self.log = ctx, log
        self.model, self.strength, self.steps, self.guidance = model or DEFAULT_MODEL, strength, steps, guidance
        self.ip_scale, self.seed, self.blur = ip_scale, seed, blur
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                 else "cpu")
        if self.device == "cpu":
            log("  [warn] no GPU found - running on CPU, this will be very slow")

    def fix_views(self, views, renders, masks, refs, prompt):
        import torch
        pipe, ip = _load(self.model, self.ip_scale > 0, self.device, self.log)
        if ip:
            pipe.set_ip_adapter_scale(self.ip_scale)
        ref = refs[0].convert("RGB") if refs else None
        text = f"{prompt.strip()}, {POSITIVE}" if prompt and prompt.strip() else POSITIVE
        out = []
        for i, (v, img, m) in enumerate(zip(views, renders, masks)):
            t0 = time.time()
            img = img.convert("RGB")
            W, H = img.size
            w8, h8 = W // 8 * 8, H // 8 * 8  # SDXL needs multiples of 8
            src = img.resize((w8, h8), Image.LANCZOS) if (w8, h8) != (W, H) else img
            mask = _to_mask(m, (w8, h8))
            if not np.asarray(mask).any():
                out.append(img)
                continue
            # feather only inwards, so every pixel outside the mask stays exactly as it was
            soft = ImageChops.multiply(mask.filter(ImageFilter.GaussianBlur(self.blur)), mask)
            kw = dict(prompt=text, negative_prompt=NEGATIVE, image=src, mask_image=mask, width=w8, height=h8,
                      strength=self.strength, num_inference_steps=self.steps, guidance_scale=self.guidance,
                      generator=torch.Generator("cpu").manual_seed(self.seed + i))
            if ip and ref is not None:
                kw["ip_adapter_image"] = ref
            gen = pipe(**kw).images[0].resize((w8, h8), Image.LANCZOS)
            res = Image.composite(gen, src, soft)  # (the VAE round trip alters every pixel otherwise)
            if (w8, h8) != (W, H):
                res = res.resize((W, H), Image.LANCZOS)
            out.append(res)
            self.log(f"  AI repaint {v.name} ({W}x{H}) in {time.time() - t0:.0f}s")
        return out

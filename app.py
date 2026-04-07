"""
Gurukul AI — Kids Educational Video Generator
Gradio web app for the full pipeline:
  Topic → Gemma script → FLUX images → TTS narration → Animation → Final video

Start:
    cd /Volumes/bujji1/sravya/ai_edu
    python app.py
"""

import json, os, re, shutil, subprocess, sys, textwrap, time, threading, urllib.request
from pathlib import Path

import gradio as gr

# Patch gradio_client bug: schema can be bool in JSON Schema draft 2019+
# Causes "TypeError: argument of type 'bool' is not iterable" in Gradio 4.44.1
try:
    import gradio_client.utils as _gc_utils
    _orig_schema_fn = _gc_utils._json_schema_to_python_type
    def _patched_schema_fn(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        return _orig_schema_fn(schema, defs)
    _gc_utils._json_schema_to_python_type = _patched_schema_fn
except Exception:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
COMFYUI_DIR  = Path("/Volumes/bujji1/sravya/ComfyUI")
SCENES_DIR   = AI_EDU_DIR / "output" / "island_scenes"
AUDIO_DIR    = AI_EDU_DIR / "output" / "island_audio"
CLIPS_DIR    = AI_EDU_DIR / "output" / "island_clips"
FINAL_OUT    = AI_EDU_DIR / "output" / "animated.mp4"
COMFYUI_URL  = "http://127.0.0.1:8288"
MFLUX        = Path("/Volumes/bujji1/sravya/ai_vidgen/venv/bin/mflux-generate")
MLX_PYTHON   = "/Volumes/bujji1/sravya/ai_vidgen/venv/bin/python"
GEMMA_MODEL  = "mlx-community/gemma-3-4b-it-4bit"

for d in [SCENES_DIR, AUDIO_DIR, CLIPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "ken-burns": {
        "label":   "Ken Burns (instant)",
        "engine":  "ffmpeg",
        "time":    "< 5s/scene",
        "quality": "★★☆☆☆",
        "best_for": "Quick previews, no GPU needed",
        "desc":    "Cinematic zoom-and-pan using ffmpeg. Zero GPU, instant. Great for previewing the full pipeline end-to-end.",
        "badge":   "INSTANT",
    },
    "ltx-2b": {
        "label":   "LTX Video 2B (fast)",
        "engine":  "ComfyUI",
        "time":    "~40s/scene",
        "quality": "★★★☆☆",
        "best_for": "Fast iteration and batch rendering",
        "desc":    "LTX Video 2B distilled FP8. 4 denoising steps. 768×512, 3.9s clips. Excellent speed/quality tradeoff for educational content.",
        "badge":   "FAST",
    },
    "ltx-13b": {
        "label":   "LTX Video 13B (quality)",
        "engine":  "ComfyUI",
        "time":    "~11min/scene",
        "quality": "★★★★☆",
        "best_for": "Final renders, landscapes, aerial shots",
        "desc":    "LTX Video 13B distilled FP8. 8 steps. Significantly more detailed motion and scene understanding. Recommended for final YouTube uploads.",
        "badge":   "QUALITY",
    },
    "wan22-fun-5b": {
        "label":   "Wan 2.2 Fun 5B (object motion)",
        "engine":  "ComfyUI",
        "time":    "~12min/scene",
        "quality": "★★★★☆",
        "best_for": "Coin flips, dice rolls, object animations",
        "desc":    "Wan 2.2 Fun InP 5B BF16. Start-frame anchored — the first frame is locked to your image. Best for scenes with specific objects like coins or dice.",
        "badge":   "BEST FOR OBJECTS",
    },
    "wan-fun-1b": {
        "label":   "Wan Fun 1.3B (lightweight)",
        "engine":  "ComfyUI",
        "time":    "~44min/scene",
        "quality": "★★★☆☆",
        "best_for": "Start-frame fidelity on slower machines",
        "desc":    "Wan 2.1 Fun InP 1.3B BF16. Same start-frame anchor approach as 5B but much lighter. Good if the 5B model isn't downloaded.",
        "badge":   "LIGHTWEIGHT",
    },
    "mlx-ltx2": {
        "label":   "MLX LTX-2 (Apple Silicon native)",
        "engine":  "MLX",
        "time":    "~30min/scene",
        "quality": "★★★☆☆",
        "best_for": "Research / no ComfyUI",
        "desc":    "prince-canuma/LTX-2-distilled running natively on Apple Silicon via mlx-video. Doesn't need ComfyUI. Uses ~36 GB RAM — close to the 36 GB limit.",
        "badge":   "NO COMFYUI",
    },
    # ── NEW GGUF models ──────────────────────────────────────────────────────
    "wan22-fun-5b-gguf": {
        "label":   "Wan 2.2 Fun 5B GGUF (faster, stabler)",
        "engine":  "ComfyUI",
        "time":    "~8-10min/scene",
        "quality": "★★★★☆",
        "best_for": "Object motion — faster & more stable than BF16",
        "desc":    "Wan 2.2 Fun InP 5B Q8_0 GGUF. Same WanFunInpaintToVideo workflow as BF16 but avoids MPS float8 errors. Loads faster, uses less peak RAM. Recommended over the BF16 version.",
        "badge":   "GGUF UPGRADE",
    },
    "wan22-i2v-14b-gguf": {
        "label":   "Wan 2.2 I2V-A14B GGUF (best quality)",
        "engine":  "ComfyUI",
        "time":    "~15-20min/scene",
        "quality": "★★★★★",
        "best_for": "Hero scenes, maximum quality finals",
        "desc":    "Wan 2.2 I2V 14B dual-model (HighNoise Q5_0 + LowNoise Q4_0) via GGUF loader. Previously failed with safetensors channel mismatch — GGUF bypasses this. Best-in-class motion quality.",
        "badge":   "BEST QUALITY",
    },
    "ltx23-gguf": {
        "label":   "LTX-2.3 22B GGUF (newest LTX)",
        "engine":  "ComfyUI",
        "time":    "~4-6min/scene",
        "quality": "★★★★☆",
        "best_for": "Speed + quality balance, latest model",
        "desc":    "LTX-2.3 22B distilled Q4_0 GGUF. Newest generation from Lightricks — better motion than LTX 0.9.8, less freezing. Requires MPS float16 patch. Fastest quality option.",
        "badge":   "NEWEST",
    },
}

STYLE = (
    "Pixar animated movie landscape, vibrant saturated colors, "
    "cinematic wide shot, golden warm light, magical atmosphere, "
    "stunning environmental storytelling, ultra detailed, "
    "beautiful Pixar 3D render, no text, no people, no characters"
)

NEG_PROMPT = "blurry, static, ugly, distorted, low quality, watermark, jitter"

# ── GGUF model filenames ───────────────────────────────────────────────────────
WAN22_FUN5B_GGUF = "Wan2.2-Fun-5B-InP-Q8_0.gguf"
WAN22_14B_HIGH   = "Wan2.2-I2V-A14B-HighNoise-Q5_0.gguf"
WAN22_14B_LOW    = "Wan2.2-I2V-A14B-LowNoise-Q4_0.gguf"
LTX23_DISTILLED  = "ltx-2.3-22b-distilled-Q4_0.gguf"
LTX23_TE         = "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
LTX23_VAE        = "ltx-2.3-22b-distilled_video_vae.safetensors"

# ── Session state (mutable dicts, safe because only 1 user at a time) ─────────
_session = {
    "topic":       None,
    "island_name": None,
    "scene_defs":  [],    # list of [id, desc]
    "scenes":      [],    # list of {id, narration}
    "scene_files": {},    # {id: Path}
    "audio_files": {},    # {id: Path}
    "clip_files":  {},    # {id: Path}
}

# ── ComfyUI helpers ────────────────────────────────────────────────────────────

def _comfy_get(path):
    with urllib.request.urlopen(f"{COMFYUI_URL}{path}", timeout=10) as r:
        return json.loads(r.read())

def _comfy_post(path, data):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}{path}", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def comfyui_running() -> bool:
    try:
        _comfy_get("/system_stats")
        return True
    except Exception:
        return False

def _upload_to_comfy(img_path: Path) -> str:
    dest = COMFYUI_DIR / "input" / img_path.name
    shutil.copy2(str(img_path), str(dest))
    return img_path.name

def _frames_to_mp4(frames: list, out: Path, fps: int = 25):
    lst = out.parent / f"_fl_{out.stem}.txt"
    with open(lst, "w") as f:
        for fr in frames:
            f.write(f"file '{fr}'\nduration {1/fps}\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(lst), "-vf", f"fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    ], check=True, capture_output=True)
    lst.unlink(missing_ok=True)

def _wait_comfy_job(prompt_id: str, prefix: str, out: Path,
                    fps: int = 25, max_wait: int = 7200):
    for _ in range(max_wait // 3):
        time.sleep(3)
        try:
            history = _comfy_get(f"/history/{prompt_id}")
        except Exception:
            continue
        if prompt_id not in history:
            continue
        entry  = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            return None, f"ComfyUI error: {status.get('messages', [])[-1:]}"
        for _, node_out in entry.get("outputs", {}).items():
            if node_out.get("images"):
                frames = sorted((COMFYUI_DIR / "output").glob(f"{prefix}*.png"))
                if frames:
                    _frames_to_mp4(frames, out, fps)
                    for f in frames:
                        f.unlink(missing_ok=True)
                    return out, None
        if status.get("completed"):
            break
    return None, "Timed out"

# ── ComfyUI workflows ──────────────────────────────────────────────────────────

def _wf_ltx(img_file, prompt, model, steps, num_frames, scene_id, prefix):
    num_frames = max(9, ((num_frames - 9) // 8) * 8 + 9)
    return {
        "1":  {"class_type": "LoadImage",            "inputs": {"image": img_file}},
        "2":  {"class_type": "CheckpointLoaderSimple","inputs": {"ckpt_name": model}},
        "3":  {"class_type": "CLIPLoader",            "inputs": {"clip_name": "t5xxl_fp8_e4m3fn.safetensors", "type": "ltxv"}},
        "5":  {"class_type": "CLIPTextEncode",        "inputs": {"text": prompt + ", smooth cinematic motion, high quality, beautiful", "clip": ["3", 0]}},
        "6":  {"class_type": "CLIPTextEncode",        "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
        "7":  {"class_type": "LTXVImgToVideo",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["2", 2],
                          "image": ["1", 0], "width": 768, "height": 512,
                          "length": num_frames, "batch_size": 1, "strength": 1.0}},
        "8":  {"class_type": "LTXVConditioning",     "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": 25.0}},
        "9":  {"class_type": "LTXVScheduler",
               "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95,
                          "stretch": True, "terminal": 0.1, "latent": ["7", 2]}},
        "10": {"class_type": "RandomNoise",           "inputs": {"noise_seed": 42 + scene_id}},
        "11": {"class_type": "BasicGuider",           "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "KSamplerSelect",        "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10", 0], "guider": ["11", 0],
                          "sampler": ["13", 0], "sigmas": ["9", 0], "latent_image": ["7", 2]}},
        "15": {"class_type": "VAEDecode",             "inputs": {"samples": ["14", 0], "vae": ["2", 2]}},
        "12": {"class_type": "SaveImage",             "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }

def _wf_wan_fun(img_file, prompt, model_name, vae_name, num_frames, scene_id, prefix):
    num_frames = max(5, ((num_frames - 1) // 4) * 4 + 1)
    return {
        "1":  {"class_type": "LoadImage",       "inputs": {"image": img_file}},
        "2":  {"class_type": "UNETLoader",      "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "3":  {"class_type": "CLIPLoader",      "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "4":  {"class_type": "CLIPVisionLoader","inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "5":  {"class_type": "VAELoader",       "inputs": {"vae_name": vae_name}},
        "6":  {"class_type": "CLIPVisionEncode","inputs": {"clip_vision": ["4", 0], "image": ["1", 0], "crop": "center"}},
        "7":  {"class_type": "CLIPTextEncode",  "inputs": {"text": prompt + ", smooth motion, high quality, cinematic", "clip": ["3", 0]}},
        "8":  {"class_type": "CLIPTextEncode",  "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
        "9":  {"class_type": "WanFunInpaintToVideo",
               "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["5", 0],
                          "width": 832, "height": 480, "length": num_frames,
                          "batch_size": 1, "clip_vision_output": ["6", 0], "start_image": ["1", 0]}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "positive": ["9", 0], "negative": ["9", 1],
                          "latent_image": ["9", 2], "seed": 42 + scene_id,
                          "steps": 10, "cfg": 5.0, "sampler_name": "euler",
                          "scheduler": "linear_quadratic", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode",       "inputs": {"samples": ["10", 0], "vae": ["5", 0]}},
        "12": {"class_type": "SaveImage",       "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }

# ── GGUF workflow builders ─────────────────────────────────────────────────────

def _wf_wan_fun_gguf(img_file, prompt, model_name, vae_name, num_frames, scene_id, prefix):
    """Wan Fun InP via UnetLoaderGGUF — same nodes as BF16 but GGUF loader."""
    num_frames = max(5, ((num_frames - 1) // 4) * 4 + 1)
    return {
        "1":  {"class_type": "LoadImage",         "inputs": {"image": img_file}},
        "2":  {"class_type": "UnetLoaderGGUF",    "inputs": {"unet_name": model_name}},
        "3":  {"class_type": "CLIPLoader",        "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "4":  {"class_type": "CLIPVisionLoader",  "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "5":  {"class_type": "VAELoader",         "inputs": {"vae_name": vae_name}},
        "6":  {"class_type": "CLIPVisionEncode",  "inputs": {"clip_vision": ["4", 0], "image": ["1", 0], "crop": "center"}},
        "7":  {"class_type": "CLIPTextEncode",    "inputs": {"text": prompt + ", smooth motion, high quality, cinematic", "clip": ["3", 0]}},
        "8":  {"class_type": "CLIPTextEncode",    "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
        "9":  {"class_type": "WanFunInpaintToVideo",
               "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["5", 0],
                          "width": 832, "height": 480, "length": num_frames,
                          "batch_size": 1, "clip_vision_output": ["6", 0], "start_image": ["1", 0]}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "positive": ["9", 0], "negative": ["9", 1],
                          "latent_image": ["9", 2], "seed": 42 + scene_id,
                          "steps": 10, "cfg": 5.0, "sampler_name": "euler",
                          "scheduler": "linear_quadratic", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode",         "inputs": {"samples": ["10", 0], "vae": ["5", 0]}},
        "12": {"class_type": "SaveImage",         "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


def _wf_wan22_i2v_gguf(img_file, prompt, high_name, low_name, num_frames, scene_id, prefix):
    """Wan 2.2 I2V-A14B dual-model GGUF: HighNoise → LowNoise refinement."""
    num_frames = max(5, ((num_frames - 1) // 4) * 4 + 1)
    return {
        "1":  {"class_type": "LoadImage",         "inputs": {"image": img_file}},
        "3":  {"class_type": "CLIPLoader",        "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "4":  {"class_type": "CLIPVisionLoader",  "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
        "5":  {"class_type": "VAELoader",         "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "6":  {"class_type": "CLIPVisionEncode",  "inputs": {"clip_vision": ["4", 0], "image": ["1", 0], "crop": "center"}},
        "7":  {"class_type": "CLIPTextEncode",    "inputs": {"text": prompt + ", smooth cinematic motion, high quality", "clip": ["3", 0]}},
        "8":  {"class_type": "CLIPTextEncode",    "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
        # Stage 1: HighNoise
        "20": {"class_type": "UnetLoaderGGUF",    "inputs": {"unet_name": high_name}},
        "21": {"class_type": "WanAnimateToVideo",
               "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["5", 0],
                          "width": 832, "height": 480, "length": num_frames, "batch_size": 1,
                          "continue_motion_max_frames": 5, "video_frame_offset": 0,
                          "clip_vision_output": ["6", 0], "reference_image": ["1", 0]}},
        "22": {"class_type": "KSampler",
               "inputs": {"model": ["20", 0], "positive": ["21", 0], "negative": ["21", 1],
                          "latent_image": ["21", 2], "seed": 42 + scene_id,
                          "steps": 10, "cfg": 5.0, "sampler_name": "euler",
                          "scheduler": "linear_quadratic", "denoise": 1.0}},
        # Stage 2: LowNoise refinement
        "30": {"class_type": "UnetLoaderGGUF",    "inputs": {"unet_name": low_name}},
        "31": {"class_type": "KSampler",
               "inputs": {"model": ["30", 0], "positive": ["21", 0], "negative": ["21", 1],
                          "latent_image": ["22", 0], "seed": 42 + scene_id,
                          "steps": 10, "cfg": 5.0, "sampler_name": "euler",
                          "scheduler": "linear_quadratic", "denoise": 0.5}},
        "40": {"class_type": "VAEDecode",         "inputs": {"samples": ["31", 0], "vae": ["5", 0]}},
        "41": {"class_type": "SaveImage",         "inputs": {"images": ["40", 0], "filename_prefix": prefix}},
    }


def _wf_ltx23_gguf(img_file, prompt, model_name, te_name, vae_name, num_frames, scene_id, prefix):
    """LTX-2.3 22B distilled GGUF I2V workflow."""
    num_frames = max(9, ((num_frames - 9) // 8) * 8 + 9)
    return {
        "1":  {"class_type": "LoadImage",            "inputs": {"image": img_file}},
        "2":  {"class_type": "UnetLoaderGGUF",       "inputs": {"unet_name": model_name}},
        "3":  {"class_type": "CLIPLoader",           "inputs": {"clip_name": te_name, "type": "ltxv"}},
        "4":  {"class_type": "VAELoader",            "inputs": {"vae_name": vae_name}},
        "5":  {"class_type": "CLIPTextEncode",       "inputs": {"text": prompt + ", smooth cinematic motion, high quality, beautiful", "clip": ["3", 0]}},
        "6":  {"class_type": "CLIPTextEncode",       "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
        "7":  {"class_type": "LTXVImgToVideo",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["4", 0],
                          "image": ["1", 0], "width": 768, "height": 512,
                          "length": num_frames, "batch_size": 1, "strength": 1.0}},
        "8":  {"class_type": "LTXVConditioning",    "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": 25.0}},
        "9":  {"class_type": "LTXVScheduler",
               "inputs": {"steps": 8, "max_shift": 2.05, "base_shift": 0.95,
                          "stretch": True, "terminal": 0.1, "latent": ["7", 2]}},
        "10": {"class_type": "RandomNoise",          "inputs": {"noise_seed": 42 + scene_id}},
        "11": {"class_type": "BasicGuider",          "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "KSamplerSelect",       "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10", 0], "guider": ["11", 0],
                          "sampler": ["13", 0], "sigmas": ["9", 0], "latent_image": ["7", 2]}},
        "15": {"class_type": "VAEDecode",            "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "12": {"class_type": "SaveImage",            "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


# ── Ken Burns (ffmpeg only, no GPU) ───────────────────────────────────────────

def _kenburns(img_path: Path, audio_path: Path, out: Path, scene_id: int) -> Path:
    try:
        import soundfile as sf
        duration = sf.info(str(audio_path)).duration
    except Exception:
        duration = 8.0
    fps = 25
    total_frames = int(duration * fps)
    W, H = 1280, 720
    nf = str(max(total_frames - 1, 1))

    SCENE_MOTION = {
        3: (1.0, 1.30, "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"),
        4: (1.05, 1.10, f"(iw-iw/zoom)/2-60+120*n/{nf}", "(ih-ih/zoom)/2"),
        5: (1.0, 1.25, "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"),
        9: (1.0, 1.20, "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"),
    }
    if scene_id in SCENE_MOTION:
        zs, ze, xexpr, yexpr = SCENE_MOTION[scene_id]
    else:
        zoom_in = (scene_id % 2 == 0)
        zs, ze = (1.08, 1.0) if zoom_in else (1.0, 1.08)
        pan_end = 40 if (scene_id % 3 != 0) else -40
        xexpr = f"(iw-iw/zoom)/2+{pan_end}*n/{nf}"
        yexpr = "(ih-ih/zoom)/2"

    z_expr = f"{zs}+({ze}-{zs})*n/{nf}"
    zp = f"scale=8000:-1,zoompan=z='{z_expr}':x='{xexpr}':y='{yexpr}':d={total_frames}:s={W}x{H}:fps={fps},setsar=1"
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
        "-i", str(audio_path),
        "-vf", zp, "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return out

# ── Animation dispatch ─────────────────────────────────────────────────────────

ANIMATION_PROMPTS = {
    1: "slow majestic aerial camera pan over a magical fantasy island, golden light shifts softly, trees sway gently, river shimmers",
    2: "gentle ocean waves crash rhythmically against tall golden coin cliffs, coins gleam in warm sunlight, camera slowly pulls back",
    3: "gold coin slowly flipping in the air in smooth slow motion, one face shows H clearly, rotates to reveal T, warm golden light, coin stays centered",
    4: "six large red dice gently rocking and slowly rolling in place, each dice shows its dots face clearly, subtle gentle rocking motion, warm golden desert light",
    5: "close-up of one giant red dice, the four-dot face glows bright gold, spotlight beam pulses brighter, dice stays still, four dots glow",
    6: "magical glowing trees sway gently, red and blue light orbs float softly upward, fireflies drift through golden air",
    7: "glowing red fruit falls slowly in golden light, leaves flutter gently, rays stream down through canopy, magical sparkles",
    8: "breathtaking golden sunrise rises over the horizon, warm light floods the island, god rays sweep across the landscape",
    9: "dark mist swirls around impossible shore, red vines on the X slowly pulse, bright island visible behind",
    10: "epic slow aerial camera pull back revealing the entire island, golden light floods all zones, river sparkles as camera rises",
}

def _default_anim_prompt(scene_id: int) -> str:
    return ANIMATION_PROMPTS.get(scene_id, "smooth cinematic motion, beautiful scene, golden light")

def animate_scene(scene_id: int, model_key: str, img_path: Path, audio_path: Path) -> tuple:
    """Returns (out_path, error_msg)."""
    out = CLIPS_DIR / f"scene_{scene_id:02d}.mp4"

    if model_key == "ken-burns":
        result = _kenburns(img_path, audio_path, out, scene_id)
        return (result, None) if result else (None, "ffmpeg Ken Burns failed")

    if model_key == "mlx-ltx2":
        return _animate_mlx(scene_id, img_path, out)

    # ComfyUI models
    if not comfyui_running():
        return None, "ComfyUI not running on port 8288"

    prompt = _default_anim_prompt(scene_id)
    img_file = _upload_to_comfy(img_path)
    prefix = f"app_s{scene_id:02d}_"

    if model_key == "ltx-2b":
        wf = _wf_ltx(img_file, prompt, "ltxv-2b-0.9.8-distilled-fp8.safetensors",
                     steps=4, num_frames=97, scene_id=scene_id, prefix=prefix)
    elif model_key == "ltx-13b":
        ck = COMFYUI_DIR / "models" / "checkpoints" / "ltxv-13b-0.9.8-distilled-fp8.safetensors"
        if not ck.exists():
            return None, "LTX 13B model not downloaded"
        wf = _wf_ltx(img_file, prompt, "ltxv-13b-0.9.8-distilled-fp8.safetensors",
                     steps=8, num_frames=97, scene_id=scene_id, prefix=prefix)
    elif model_key == "wan22-fun-5b":
        mp = COMFYUI_DIR / "models" / "diffusion_models" / "wan2.2_fun_inpaint_5B_bf16.safetensors"
        if not mp.exists():
            return None, "Wan 2.2 Fun 5B model not downloaded"
        wf = _wf_wan_fun(img_file, prompt, mp.name, "wan2.2_vae.safetensors",
                         num_frames=81, scene_id=scene_id, prefix=prefix)
    elif model_key == "wan-fun-1b":
        mp = COMFYUI_DIR / "models" / "diffusion_models" / "wan2.1_fun_inp_1.3B_bf16.safetensors"
        if not mp.exists():
            return None, "Wan Fun 1.3B model not downloaded"
        wf = _wf_wan_fun(img_file, prompt, mp.name, "wan_2.1_vae.safetensors",
                         num_frames=81, scene_id=scene_id, prefix=prefix)
    elif model_key == "wan22-fun-5b-gguf":
        mp = COMFYUI_DIR / "models" / "diffusion_models" / WAN22_FUN5B_GGUF
        if not mp.exists():
            return None, f"{WAN22_FUN5B_GGUF} not downloaded yet"
        wf = _wf_wan_fun_gguf(img_file, prompt, mp.name, "wan2.2_vae.safetensors",
                              num_frames=81, scene_id=scene_id, prefix=prefix)
    elif model_key == "wan22-i2v-14b-gguf":
        high = COMFYUI_DIR / "models" / "diffusion_models" / WAN22_14B_HIGH
        low  = COMFYUI_DIR / "models" / "diffusion_models" / WAN22_14B_LOW
        if not high.exists() or not low.exists():
            return None, "Wan2.2 I2V-A14B GGUF files not downloaded yet"
        wf = _wf_wan22_i2v_gguf(img_file, prompt, high.name, low.name,
                                 num_frames=33, scene_id=scene_id, prefix=prefix)
    elif model_key == "ltx23-gguf":
        mp  = COMFYUI_DIR / "models" / "diffusion_models" / LTX23_DISTILLED
        te  = COMFYUI_DIR / "models" / "text_encoders" / LTX23_TE
        if not mp.exists():
            return None, f"{LTX23_DISTILLED} not downloaded yet"
        if not te.exists():
            return None, f"{LTX23_TE} not downloaded yet"
        wf = _wf_ltx23_gguf(img_file, prompt, mp.name, LTX23_TE, LTX23_VAE,
                             num_frames=97, scene_id=scene_id, prefix=prefix)
    else:
        return None, f"Unknown model: {model_key}"

    try:
        r = _comfy_post("/prompt", {"prompt": wf})
        return _wait_comfy_job(r["prompt_id"], prefix, out)
    except Exception as e:
        return None, str(e)

def _animate_mlx(scene_id: int, img_path: Path, out: Path) -> tuple:
    prompt = _default_anim_prompt(scene_id) + ", smooth cinematic motion, high quality, cinematic"
    script = f"""
import sys
try:
    from mlx_video.models.ltx_2.generate import generate_video, PipelineType
except ImportError as e:
    print(f"Import error: {{e}}", flush=True)
    sys.exit(1)
generate_video(
    model_repo="prince-canuma/LTX-2-distilled",
    text_encoder_repo=None,
    prompt={prompt!r},
    pipeline=PipelineType.DISTILLED,
    negative_prompt={NEG_PROMPT!r},
    height=512, width=768,
    num_frames=25, num_inference_steps=4,
    image_path={str(img_path)!r},
    image_frame_strength=1.0,
    output_path={str(out)!r},
)
"""
    r = subprocess.run(
        [MLX_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=7200,
    )
    if r.returncode == 0 and out.exists():
        return out, None
    return None, r.stderr[-500:]

# ── Gemma script generation ────────────────────────────────────────────────────

GEMMA_SYSTEM = textwrap.dedent("""
You are a creative director for a kids' educational YouTube channel.
You produce Pixar-style animated educational videos where every scene is a
LANDSCAPE (no human characters — avoids consistency problems with AI image generation).

The visual style:
- Pixar animated movie landscape, vibrant saturated colors
- Cinematic wide shots, golden warm light, magical atmosphere
- Environmental storytelling: the LANDSCAPE itself teaches the concept
- No text, no people, no characters — only magical environments and objects

Output format: valid JSON only, no markdown, no explanation.
""").strip()

GEMMA_PROMPT_TPL = textwrap.dedent("""
Create a 10-scene Gurukul Island episode teaching "{topic}" to kids aged 6-10.

Rules:
1. Each scene is a pure LANDSCAPE — the environment itself visualises the concept.
   No human characters, no text in images, no narrators shown.
2. Scene 1: introduce the island / world for this topic.
3. Scenes 2-9: each scene introduces one key concept about {topic}, building up.
4. Scene 10: triumphant aerial wide shot showing the whole island — everything learned.
5. Narration is warm, simple, wonder-filled. 2-4 short sentences per scene.
   Age 6-10 level. No jargon without explanation.
6. Image descriptions must be vivid and specific.

Return a JSON object:
{{
  "topic": "{topic}",
  "island_name": "...",
  "scene_defs": [[1, "desc1"], [2, "desc2"], ..., [10, "desc10"]],
  "scenes": [{{"id": 1, "narration": "..."}}, ..., {{"id": 10, "narration": "..."}}}]
}}
""").strip()

def run_gemma(topic: str, progress_fn=None) -> dict:
    if progress_fn: progress_fn("Loading Gemma 3 4B model...")
    messages = json.dumps([
        {"role": "system", "content": GEMMA_SYSTEM},
        {"role": "user",   "content": GEMMA_PROMPT_TPL.format(topic=topic)},
    ])
    script = f"""
import sys
try:
    from mlx_lm import load, generate
except ImportError:
    print("INSTALL_NEEDED", flush=True); sys.exit(1)
model, tokenizer = load({GEMMA_MODEL!r})
messages = {messages}
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=4096, verbose=False)
print(response, flush=True)
"""
    result = subprocess.run(
        [MLX_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Gemma failed: {result.stderr[-500:]}")
    raw = result.stdout.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1:
        raise ValueError(f"No JSON in Gemma output:\n{raw[:500]}")
    return json.loads(raw[start:end])

# ── TTS ────────────────────────────────────────────────────────────────────────

def _generate_elevenlabs(text: str, out_path: Path) -> bool:
    import numpy as np
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        return False
    try:
        from elevenlabs.client import ElevenLabs
        import soundfile as sf
        client = ElevenLabs(api_key=api_key)
        audio_gen = client.text_to_speech.convert(
            voice_id="onwK4e9ZLuTAKqWW03F9",
            text=text, model_id="eleven_multilingual_v2", output_format="pcm_24000",
        )
        pcm = b"".join(audio_gen)
        sf.write(str(out_path), np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0, 24000)
        return True
    except Exception:
        return False

def generate_tts(text: str, out_path: Path):
    if _generate_elevenlabs(text, out_path):
        return
    from kokoro import KPipeline
    import numpy as np, soundfile as sf
    pipeline = KPipeline(lang_code='a')
    segments, silence = [], np.zeros(int(0.3 * 24000), dtype=np.float32)
    for _, _, audio in pipeline(text, voice='am_adam', speed=0.88):
        segments.append(audio)
    segments.append(silence)
    sf.write(str(out_path), np.concatenate(segments), 24000)

# ── FLUX image generation ──────────────────────────────────────────────────────

def generate_image(scene_id: int, description: str, out_path: Path,
                   model: str = "dev", steps: int = 20):
    if out_path.exists():
        return out_path
    prompt = f"{description}, {STYLE}"
    cmd = [
        str(MFLUX), "--model", model,
        "--prompt", prompt,
        "--width", "1360", "--height", "768",
        "--steps", str(steps),
        "--seed", str(3000 + scene_id),
        "--output", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mflux failed scene {scene_id}: {r.stderr[-300:]}")
    return out_path

# ── Assemble final video ───────────────────────────────────────────────────────

def assemble_video(scenes: list, out_path: Path) -> Path:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip, vfx
    clips = []
    for scene in scenes:
        sid = scene["id"]
        audio_path = AUDIO_DIR / f"scene_{sid:02d}.wav"
        clip_path  = CLIPS_DIR / f"scene_{sid:02d}.mp4"
        img_path   = SCENES_DIR / f"scene_{sid:02d}.png"
        if not audio_path.exists():
            continue
        audio = AudioFileClip(str(audio_path))
        if clip_path.exists():
            pp = CLIPS_DIR / f"scene_{sid:02d}_pp.mp4"
            if not pp.exists():
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(clip_path),
                    "-vf", "reverse", "-af", "areverse",
                    str(pp),
                ], capture_output=True)
                if pp.exists():
                    # concat original + reversed = ping-pong
                    lst = CLIPS_DIR / f"_pp_{sid}.txt"
                    lst.write_text(f"file '{clip_path}'\nfile '{pp}'\n")
                    final_pp = CLIPS_DIR / f"scene_{sid:02d}_loop.mp4"
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", str(final_pp),
                    ], capture_output=True)
                    lst.unlink(missing_ok=True)
                    pp.unlink(missing_ok=True)
                    clip_path = final_pp if final_pp.exists() else clip_path
            video = VideoFileClip(str(clip_path))
            if video.duration < audio.duration:
                video = video.fx(vfx.loop, duration=audio.duration)
            else:
                video = video.subclip(0, audio.duration)
            clips.append(video.set_audio(audio))
        elif img_path.exists():
            clips.append(ImageClip(str(img_path)).set_duration(audio.duration + 0.5).set_audio(audio))
    if not clips:
        raise RuntimeError("No clips to assemble")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(str(out_path), fps=24, codec="libx264", audio_codec="aac", logger=None)
    return out_path

# ── Model availability check ───────────────────────────────────────────────────

def check_model_availability():
    status = {}
    status["ken-burns"] = True

    comfy_up = comfyui_running()
    dm = COMFYUI_DIR / "models" / "diffusion_models"
    ck = COMFYUI_DIR / "models" / "checkpoints"
    te = COMFYUI_DIR / "models" / "text_encoders"

    status["ltx-2b"]            = comfy_up and (ck / "ltxv-2b-0.9.8-distilled-fp8.safetensors").exists()
    status["ltx-13b"]           = comfy_up and (ck / "ltxv-13b-0.9.8-distilled-fp8.safetensors").exists()
    status["wan22-fun-5b"]      = comfy_up and (dm / "wan2.2_fun_inpaint_5B_bf16.safetensors").exists()
    status["wan-fun-1b"]        = comfy_up and (dm / "wan2.1_fun_inp_1.3B_bf16.safetensors").exists()
    status["mlx-ltx2"]          = Path(MLX_PYTHON).exists()
    # GGUF models
    status["wan22-fun-5b-gguf"] = comfy_up and (dm / WAN22_FUN5B_GGUF).exists()
    status["wan22-i2v-14b-gguf"]= comfy_up and (dm / WAN22_14B_HIGH).exists() and (dm / WAN22_14B_LOW).exists()
    status["ltx23-gguf"]        = comfy_up and (dm / LTX23_DISTILLED).exists() and (te / LTX23_TE).exists()

    return status, comfy_up

# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_model_table(status: dict) -> str:
    rows = []
    for key, info in MODELS.items():
        avail = status.get(key, False)
        avail_badge = "✅ Ready" if avail else "⚠️ Unavailable"
        rows.append(
            f"| **{info['label']}** | {info['badge']} | {info['time']} | {info['quality']} | {info['best_for']} | {avail_badge} |"
        )
    header = "| Model | Badge | Speed | Quality | Best For | Status |\n|---|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)

def refresh_status():
    status, comfy_up = check_model_availability()
    comfy_msg = "ComfyUI: ✅ Running on port 8288" if comfy_up else "ComfyUI: ❌ Not running"
    return build_model_table(status), comfy_msg

# ── Step 1: Generate Script ────────────────────────────────────────────────────

def step_generate_script(topic: str):
    if not topic.strip():
        yield "Please enter a topic.", ""
        return

    log = [f"Generating script for: **{topic}**\n"]
    yield "\n".join(log), ""

    try:
        def progress(msg):
            log.append(msg)
        data = run_gemma(topic.strip(), progress_fn=progress)
    except Exception as e:
        log.append(f"Error: {e}")
        yield "\n".join(log), ""
        return

    _session["topic"]       = topic.strip()
    _session["island_name"] = data.get("island_name", f"{topic.title()} Island")
    _session["scene_defs"]  = data.get("scene_defs", [])
    _session["scenes"]      = data.get("scenes", [])

    log.append(f"Island: **{_session['island_name']}**")
    log.append(f"Generated {len(_session['scene_defs'])} scenes\n")

    script_lines = [f"# {_session['island_name']}\n"]
    for (sid, desc), scene in zip(_session["scene_defs"], _session["scenes"]):
        script_lines.append(f"### Scene {sid}")
        script_lines.append(f"**Image:** {desc[:120]}...")
        script_lines.append(f"**Narration:** {scene['narration']}\n")

    yield "\n".join(log), "\n".join(script_lines)

# ── Step 2: Generate Images ────────────────────────────────────────────────────

def step_generate_images(img_model: str, start_scene: int, end_scene: int):
    if not _session["scene_defs"]:
        yield "Generate a script first.", []
        return

    steps = 20 if img_model == "dev" else 4
    log   = [f"Generating images with FLUX {img_model} ({steps} steps)...\n"]
    yield "\n".join(log), []

    image_paths = []
    for scene_id, desc in _session["scene_defs"]:
        if scene_id < start_scene or scene_id > end_scene:
            # Check if image already exists
            p = SCENES_DIR / f"scene_{scene_id:02d}.png"
            if p.exists():
                image_paths.append(str(p))
            continue

        out = SCENES_DIR / f"scene_{scene_id:02d}.png"
        log.append(f"Scene {scene_id:02d}: {desc[:60]}...")
        yield "\n".join(log), [str(p) for p in image_paths]

        try:
            generate_image(scene_id, desc, out, model=img_model, steps=steps)
            image_paths.append(str(out))
            log.append(f"  ✓ scene_{scene_id:02d}.png")
        except Exception as e:
            log.append(f"  ✗ ERROR: {e}")
        yield "\n".join(log), [str(p) for p in image_paths]

    _session["scene_files"] = {
        sid: SCENES_DIR / f"scene_{sid:02d}.png"
        for sid, _ in _session["scene_defs"]
        if (SCENES_DIR / f"scene_{sid:02d}.png").exists()
    }
    log.append(f"\nDone! {len(image_paths)} images generated.")
    yield "\n".join(log), [str(p) for p in image_paths]

# ── Step 3: Generate Audio ─────────────────────────────────────────────────────

def step_generate_audio():
    if not _session["scenes"]:
        yield "Generate a script first.", []
        return

    backend = "ElevenLabs" if os.environ.get("ELEVENLABS_API_KEY") else "Kokoro (local)"
    log = [f"Generating TTS narration — {backend}\n"]
    yield "\n".join(log), []

    audio_paths = []
    for scene in _session["scenes"]:
        sid = scene["id"]
        out = AUDIO_DIR / f"scene_{sid:02d}.wav"
        if out.exists():
            log.append(f"Scene {sid:02d}: cached")
            audio_paths.append(str(out))
            yield "\n".join(log), audio_paths
            continue
        log.append(f"Scene {sid:02d}: generating...")
        yield "\n".join(log), audio_paths
        try:
            generate_tts(scene["narration"], out)
            audio_paths.append(str(out))
            log.append(f"  ✓ scene_{sid:02d}.wav")
        except Exception as e:
            log.append(f"  ✗ ERROR: {e}")
        yield "\n".join(log), audio_paths

    _session["audio_files"] = {
        scene["id"]: AUDIO_DIR / f"scene_{scene['id']:02d}.wav"
        for scene in _session["scenes"]
        if (AUDIO_DIR / f"scene_{scene['id']:02d}.wav").exists()
    }
    log.append(f"\nDone! {len(audio_paths)} audio files.")
    yield "\n".join(log), audio_paths

# ── Step 4: Animate ────────────────────────────────────────────────────────────

def step_animate(model_key: str, scene_ids_str: str):
    if not _session["scene_files"]:
        yield "Generate images first.", None
        return
    if not _session["audio_files"]:
        yield "Generate audio first.", None
        return

    # Parse scene IDs
    if scene_ids_str.strip().lower() in ("all", ""):
        scene_ids = sorted(_session["scene_files"].keys())
    else:
        try:
            scene_ids = [int(x.strip()) for x in scene_ids_str.split(",")]
        except ValueError:
            yield "Invalid scene IDs. Use comma-separated numbers or 'all'.", None
            return

    m = MODELS.get(model_key, {})
    log = [f"Animating with **{m.get('label', model_key)}**\n", f"Scenes: {scene_ids}\n"]
    yield "\n".join(log), None

    last_clip = None
    for sid in scene_ids:
        img_path   = _session["scene_files"].get(sid)
        audio_path = _session["audio_files"].get(sid)
        if not img_path or not img_path.exists():
            log.append(f"Scene {sid:02d}: no image — skip")
            continue
        if not audio_path or not audio_path.exists():
            log.append(f"Scene {sid:02d}: no audio — skip")
            continue

        log.append(f"Scene {sid:02d}: animating...")
        yield "\n".join(log), last_clip

        clip_out, err = animate_scene(sid, model_key, img_path, audio_path)
        if clip_out:
            _session["clip_files"][sid] = clip_out
            last_clip = str(clip_out)
            log.append(f"  ✓ scene_{sid:02d}.mp4 ({clip_out.stat().st_size//1024}KB)")
        else:
            log.append(f"  ✗ ERROR: {err}")
        yield "\n".join(log), last_clip

    log.append(f"\nDone! {len(_session['clip_files'])} clips ready.")
    yield "\n".join(log), last_clip

# ── Step 5: Assemble ───────────────────────────────────────────────────────────

def step_assemble():
    if not _session["scenes"]:
        yield "Generate a script first.", None
        return

    log = ["Assembling final video...\n"]
    yield "\n".join(log), None

    have_clips  = len(_session["clip_files"])
    have_audio  = len(_session["audio_files"])
    have_images = len(_session["scene_files"])
    log.append(f"Clips: {have_clips} | Audio: {have_audio} | Images: {have_images}")
    yield "\n".join(log), None

    try:
        out = assemble_video(_session["scenes"], FINAL_OUT)
        log.append(f"\n✅ Saved: {out}")
        yield "\n".join(log), str(out)
    except Exception as e:
        log.append(f"\n✗ Error: {e}")
        yield "\n".join(log), None

# ── Load existing session if images/audio already present ─────────────────────

def _load_existing_session():
    """Populate session from disk if island_scenes + island_audio already exist."""
    files = sorted(SCENES_DIR.glob("scene_??.png"))
    for f in files:
        sid = int(f.stem.split("_")[1])
        _session["scene_files"][sid] = f
    afiles = sorted(AUDIO_DIR.glob("scene_??.wav"))
    for f in afiles:
        sid = int(f.stem.split("_")[1])
        _session["audio_files"][sid] = f
    cfiles = sorted(CLIPS_DIR.glob("scene_??.mp4"))
    for f in cfiles:
        sid = int(f.stem.split("_")[1])
        _session["clip_files"][sid] = f

_load_existing_session()

# ── Build UI ───────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="yellow",
    neutral_hue="slate",
)

with gr.Blocks(theme=THEME, title="Gurukul AI Video Generator") as demo:

    gr.Markdown("""
# 🏝️ Gurukul AI — Kids Educational Video Generator
**Full pipeline:** Topic → Gemma script → FLUX images → TTS narration → Animation → Final video

All models run locally on your Mac Studio M4 Max. No internet needed after first model downloads.
    """)

    # ── Status bar ──────────────────────────────────────────────────────────
    with gr.Row():
        comfy_status = gr.Markdown("ComfyUI: checking...")
        refresh_btn  = gr.Button("🔄 Refresh Status", size="sm", scale=0)

    model_table_md = gr.Markdown()

    # ── Model selector ───────────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("## 🎬 Choose Animation Model")
        with gr.Row():
            with gr.Column(scale=3):
                model_radio = gr.Radio(
                    choices=[(info["label"], key) for key, info in MODELS.items()],
                    value="ken-burns",
                    label="Animation Model",
                    info="Select the model for animating scenes",
                )
            with gr.Column(scale=2):
                model_desc = gr.Markdown()

        def update_model_desc(model_key):
            if model_key not in MODELS:
                return ""
            m = MODELS[model_key]
            return f"""**{m['label']}** — `{m['badge']}`

{m['desc']}

- **Speed:** {m['time']}
- **Quality:** {m['quality']}
- **Engine:** {m['engine']}
- **Best for:** {m['best_for']}
"""
        model_radio.change(update_model_desc, model_radio, model_desc)

    gr.Markdown("---")

    # ── Tab: Script ─────────────────────────────────────────────────────────
    with gr.Tab("1. Generate Script"):
        gr.Markdown("Use Gemma 3 4B to generate a 10-scene educational script.")
        with gr.Row():
            topic_input = gr.Textbox(
                placeholder="e.g. fractions, gravity, photosynthesis, volcanoes",
                label="Topic",
                scale=4,
            )
            script_btn = gr.Button("✨ Generate Script", variant="primary", scale=1)

        script_log    = gr.Markdown()
        script_output = gr.Markdown()

        script_btn.click(
            step_generate_script,
            inputs=topic_input,
            outputs=[script_log, script_output],
        )

    # ── Tab: Images ─────────────────────────────────────────────────────────
    with gr.Tab("2. Generate Images"):
        gr.Markdown("Generate FLUX scene images. **Dev** = best quality (20 steps). **Schnell** = fast preview (4 steps).")
        with gr.Row():
            img_model_radio = gr.Radio(
                choices=[("FLUX Dev (quality, ~3min/scene)", "dev"),
                         ("FLUX Schnell (fast, ~30s/scene)", "schnell")],
                value="dev", label="FLUX Model",
            )
        with gr.Row():
            img_start = gr.Slider(1, 10, value=1, step=1, label="From scene")
            img_end   = gr.Slider(1, 10, value=10, step=1, label="To scene")
        img_btn = gr.Button("🖼️ Generate Images", variant="primary")
        img_log = gr.Markdown()
        img_gallery = gr.Gallery(label="Scene Images", columns=5, height="auto")

        img_btn.click(
            step_generate_images,
            inputs=[img_model_radio, img_start, img_end],
            outputs=[img_log, img_gallery],
        )

    # ── Tab: Audio ──────────────────────────────────────────────────────────
    with gr.Tab("3. Generate Audio"):
        gr.Markdown("Generate TTS narration. Uses **ElevenLabs** (needs API key) → falls back to **Kokoro** (fully local).")
        audio_btn = gr.Button("🎙️ Generate Audio", variant="primary")
        audio_log = gr.Markdown()

        # Show a few audio players
        with gr.Row():
            audio_out1 = gr.Audio(label="Scene 1", interactive=False)
            audio_out2 = gr.Audio(label="Scene 2", interactive=False)
            audio_out3 = gr.Audio(label="Scene 3", interactive=False)

        def step_audio_ui():
            log_parts = []
            paths = []
            for msg, new_paths in step_generate_audio():
                log_parts = [msg]
                paths = new_paths
                # yield partial updates
            a1 = paths[0] if len(paths) > 0 else None
            a2 = paths[1] if len(paths) > 1 else None
            a3 = paths[2] if len(paths) > 2 else None
            return "\n".join(log_parts), a1, a2, a3

        # Use generator-based approach
        def stream_audio():
            for log_msg, paths in step_generate_audio():
                a1 = paths[0] if len(paths) > 0 else None
                a2 = paths[1] if len(paths) > 1 else None
                a3 = paths[2] if len(paths) > 2 else None
                yield log_msg, a1, a2, a3

        audio_btn.click(
            stream_audio,
            outputs=[audio_log, audio_out1, audio_out2, audio_out3],
        )

    # ── Tab: Animate ─────────────────────────────────────────────────────────
    with gr.Tab("4. Animate Scenes"):
        gr.Markdown("Animate scenes using your selected model. Uses the model chosen at the top.")
        with gr.Row():
            anim_scenes_input = gr.Textbox(
                value="all",
                label="Scenes to animate",
                placeholder="all  or  1,2,3  or  3",
                info="'all' for all scenes, or comma-separated IDs",
                scale=3,
            )
            anim_btn = gr.Button("🎬 Animate", variant="primary", scale=1)

        anim_log   = gr.Markdown()
        anim_video = gr.Video(label="Latest clip preview", height=360)

        anim_btn.click(
            step_animate,
            inputs=[model_radio, anim_scenes_input],
            outputs=[anim_log, anim_video],
        )

    # ── Tab: Final Video ─────────────────────────────────────────────────────
    with gr.Tab("5. Assemble Final Video"):
        gr.Markdown("Combine all animated clips + narration into a single MP4.\nIf a scene has no animated clip, uses the static image as fallback.")
        assemble_btn = gr.Button("🎞️ Assemble Final Video", variant="primary")
        assemble_log = gr.Markdown()
        final_video  = gr.Video(label="Final Video", height=480)

        assemble_btn.click(
            step_assemble,
            outputs=[assemble_log, final_video],
        )

    # ── Tab: Quick Test ───────────────────────────────────────────────────────
    with gr.Tab("Quick Test (1 Scene)"):
        gr.Markdown("""
Test a single scene through the full pipeline without generating a new script.
Useful for model comparison — animate the same scene with different models.
        """)
        with gr.Row():
            test_scene_id   = gr.Slider(1, 10, value=3, step=1, label="Scene ID")
            test_scene_btn  = gr.Button("🧪 Test This Scene", variant="primary")

        test_log   = gr.Markdown()
        test_video = gr.Video(label="Result", height=360)

        def quick_test(scene_id: int, model_key: str):
            sid = int(scene_id)
            img_path   = SCENES_DIR / f"scene_{sid:02d}.png"
            audio_path = AUDIO_DIR  / f"scene_{sid:02d}.wav"

            if not img_path.exists():
                yield f"No image for scene {sid}. Generate images first.", None
                return
            if not audio_path.exists():
                yield f"No audio for scene {sid}. Generate audio first.", None
                return

            _session["scene_files"][sid]  = img_path
            _session["audio_files"][sid]  = audio_path

            m = MODELS.get(model_key, {})
            yield f"Animating scene {sid} with **{m.get('label', model_key)}**...", None

            clip_out, err = animate_scene(sid, model_key, img_path, audio_path)
            if clip_out:
                yield f"✅ Done! {clip_out.name} ({clip_out.stat().st_size//1024}KB)", str(clip_out)
            else:
                yield f"✗ Error: {err}", None

        test_scene_btn.click(
            quick_test,
            inputs=[test_scene_id, model_radio],
            outputs=[test_log, test_video],
        )

    # ── Tab: About Models ─────────────────────────────────────────────────────
    with gr.Tab("About Models"):
        gr.Markdown("""
## Animation Model Guide

| Model | Speed | Quality | Best For |
|---|---|---|---|
| **Ken Burns** | Instant | ★★ | Previews, any machine, zero GPU |
| **LTX 2B** | ~40s | ★★★ | Fast iteration, batch rendering |
| **LTX 13B** | ~11min | ★★★★ | Final renders, aerial landscapes |
| **LTX-2.3 22B GGUF** ✨ | ~4-6min | ★★★★ | Newest LTX, speed+quality balance |
| **Wan 2.2 Fun 5B** | ~12min | ★★★★ | Objects in motion (BF16 original) |
| **Wan 2.2 Fun 5B GGUF** ✨ | ~8-10min | ★★★★ | Objects in motion, stabler than BF16 |
| **Wan 2.2 I2V-A14B GGUF** ✨ | ~15-20min | ★★★★★ | Best quality, hero scenes |
| **Wan Fun 1.3B** | ~44min | ★★★ | Start-frame fidelity on lighter load |
| **MLX LTX-2** | ~30min | ★★★ | Native Apple Silicon, no ComfyUI needed |

## Image Model Guide

| Model | Speed | Quality | When to Use |
|---|---|---|---|
| **FLUX Schnell** | ~30s | ★★★ | Quick previews, testing prompts |
| **FLUX Dev** | ~3min | ★★★★★ | Final images for YouTube upload |

## Requirements

- **ComfyUI** must be running on port 8288 for LTX and Wan models
  ```
  cd /Volumes/bujji1/sravya/ComfyUI
  venv/bin/python main.py --port 8288
  ```
- **ELEVENLABS_API_KEY** in `.env` for ElevenLabs TTS (optional, falls back to Kokoro)
- MLX LTX-2 uses ~36 GB RAM — close to the 36 GB M4 Max limit

## Pipeline Flow

```
Topic
  └─ Gemma 3 4B → script (scene descriptions + narration)
       └─ FLUX Dev/Schnell → scene images (sequential, never parallel)
            └─ ElevenLabs / Kokoro → narration WAVs
                 └─ Animation model → animated MP4 clips
                      └─ moviepy → final assembled video
```
        """)

    # ── On load: refresh status ───────────────────────────────────────────────
    def on_load():
        table, comfy_msg = refresh_status()
        desc = update_model_desc("ken-burns")
        return table, comfy_msg, desc

    demo.load(on_load, outputs=[model_table_md, comfy_status, model_desc])
    refresh_btn.click(refresh_status, outputs=[model_table_md, comfy_status])


if __name__ == "__main__":
    # Load .env for ELEVENLABS_API_KEY
    env_file = AI_EDU_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=False,
    )

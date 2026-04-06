"""
Wan 2.2 Animation Pipeline — Probability Island
Animates each scene image using Wan 2.2 Image-to-Video via ComfyUI API.
Then merges animated clips with ElevenLabs/Kokoro audio into final MP4.

Requirements:
  - ComfyUI running on port 8288 (start: cd /Volumes/bujji1/sravya/ComfyUI && python main.py --port 8288)
  - Wan 2.2 I2V models in ComfyUI/models/diffusion_models/
  - Scene images in output/island_scenes/
  - Audio in output/island_audio/

Run:
    python wan_animate.py --test          # animate scene 2 only (quick test)
    python wan_animate.py --all           # animate all 10 scenes
    python wan_animate.py --assemble      # merge clips + audio into final MP4
    python wan_animate.py --full          # --all + --assemble
"""

import json, time, subprocess, sys, urllib.request, urllib.parse, math, struct, zlib
from pathlib import Path

AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
COMFYUI_DIR  = Path("/Volumes/bujji1/sravya/ComfyUI")
SCENES_DIR   = AI_EDU_DIR / "output" / "island_scenes"
AUDIO_DIR    = AI_EDU_DIR / "output" / "island_audio"
CLIPS_DIR    = AI_EDU_DIR / "output" / "island_clips"   # animated video clips
FINAL_OUT    = AI_EDU_DIR / "output" / "animated.mp4"
COMFYUI_URL  = "http://127.0.0.1:8288"

# ── MLX-native video generation (Apple Silicon, no ComfyUI needed) ───────────
MLX_VENV_PYTHON = "/Volumes/bujji1/sravya/ai_vidgen/venv/bin/python"
# prince-canuma/LTX-2-distilled = pre-converted MLX weights (~14 GB, fits in 36 GB)
# Lightricks/LTX-2               = original PyTorch weights (mlx-video converts on load)
MLX_MODEL_REPO  = "prince-canuma/LTX-2-distilled"

CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# ── Animation prompts — describe the MOTION for each scene ───────────────────
ANIMATION_PROMPTS = {
    1:  "slow majestic aerial camera pan over a magical fantasy island, "
        "golden light shifts softly, trees sway gently, river shimmers",

    2:  "gentle ocean waves crash rhythmically against tall golden coin cliffs, "
        "coins gleam in warm sunlight, camera slowly pulls back revealing the full cliffs",

    3:  "gold coin slowly flipping in the air in smooth slow motion, "
        "one face shows the letter H clearly, then gently rotates to reveal the letter T on the other face, "
        "steady gentle flip, coin stays centered in frame, "
        "warm golden light, calm unhurried motion, both H and T faces readable",

    4:  "six large red dice gently rocking and slowly rolling in place on desert ground, "
        "each dice shows its dots face clearly, subtle gentle rocking motion, "
        "calm steady movement, warm golden desert light, dice stay in position",

    5:  "close-up of one giant red dice boulder, the face showing exactly four white dots arranged in a 2x2 square pattern glowing bright gold, "
        "spotlight beam locks onto the four-dot face and pulses brighter, "
        "four dots clearly visible and glowing, NOT three dots NOT five dots, exactly four dots, "
        "golden glow intensifies on the four white dots, dramatic cinematic reveal, "
        "dice boulder stays perfectly still, only the four dots glow and pulse",

    6:  "magical glowing trees sway gently in an enchanted forest, "
        "red and blue light orbs float softly upward, fireflies drift through golden air",

    7:  "glowing red fruit falls slowly in golden light, "
        "leaves flutter gently, rays of light stream down through canopy, magical sparkles",

    8:  "breathtaking golden sunrise slowly rises over the horizon, "
        "warm light floods the entire island, god rays sweep across the landscape",

    9:  "dark mist swirls around impossible shore, "
        "red vines on the X slowly pulse, contrast with bright island visible behind",

    10: "epic slow aerial camera pull back revealing the entire probability island, "
        "golden light floods all three zones simultaneously, "
        "river sparkles as camera rises higher, triumphant and beautiful",
}

# ── ComfyUI helpers ───────────────────────────────────────────────────────────

def _comfy_get(path):
    with urllib.request.urlopen(f"{COMFYUI_URL}{path}", timeout=10) as r:
        return json.loads(r.read())

def _comfy_post(path, data):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def check_comfyui():
    try:
        _comfy_get("/system_stats")
        return True
    except Exception:
        return False

def start_comfyui():
    print("Starting ComfyUI on port 8288...")
    subprocess.Popen(
        ["python", "main.py", "--port", "8288", "--preview-method", "none"],
        cwd=str(COMFYUI_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        time.sleep(2)
        if check_comfyui():
            print("ComfyUI ready.")
            return True
    print("ComfyUI failed to start.")
    return False

def ensure_comfyui():
    if check_comfyui():
        print("ComfyUI already running.")
        return True
    return start_comfyui()

# ── Model availability ────────────────────────────────────────────────────────

LTX_13B  = "ltxv-13b-0.9.8-distilled-fp8.safetensors"
LTX_2B   = "ltxv-2b-0.9.8-distilled-fp8.safetensors"
WAN_FUN  = "wan2.1_fun_inp_1.3B_bf16.safetensors"

def _ltxv_model_name() -> str:
    """Return the best available LTX model: 13B > 2B."""
    p13 = COMFYUI_DIR / "models" / "checkpoints" / LTX_13B
    return LTX_13B if p13.exists() else LTX_2B

def _wan_fun_available() -> bool:
    p = COMFYUI_DIR / "models" / "diffusion_models" / WAN_FUN
    return p.exists()


# ── LTX Video I2V workflow ────────────────────────────────────────────────────

def _build_ltxv_workflow(image_filename: str, prompt: str, num_frames: int = 49,
                         scene_id: int = 1, steps: int = None, strength: float = 1.0) -> dict:
    """
    LTX Video I2V workflow — works for both 2B distilled and 13B dev.
    13B dev: needs more steps (20-30). 2B distilled: 4 steps.
    MPS constraint for 13B: frames must satisfy (n-1) % 8 == 0.
    """
    model = _ltxv_model_name()
    is_13b = (model == LTX_13B)

    # Frame count: must satisfy (n-9) % 8 == 0. 13B: cap at 97 frames (3.9s) for MPS stability.
    num_frames = max(9, ((num_frames - 9) // 8) * 8 + 9)
    if is_13b:
        num_frames = min(num_frames, 97)   # 97 frames = 3.9s, safe on MPS

    if steps is None:
        steps = 8 if is_13b else 4         # 13B distilled: 8 steps for quality; 2B distilled: 4

    prefix = f"ltxv_s{scene_id:02d}_"
    print(f"    Using {model} | {num_frames} frames | {steps} steps")

    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp8_e4m3fn.safetensors", "type": "ltxv"}},
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt + ", smooth cinematic motion, high quality, beautiful",
                "clip": ["3", 0]
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, static, ugly, distorted, low quality, watermark",
                "clip": ["3", 0]
            }
        },
        "7": {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["2", 2],
                "image": ["1", 0],
                "width": 768,
                "height": 512,
                "length": num_frames,
                "batch_size": 1,
                "strength": strength,
            }
        },
        "8": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": 25.0}
        },
        "9": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps, "max_shift": 2.05, "base_shift": 0.95,
                "stretch": True, "terminal": 0.1, "latent": ["7", 2],
            }
        },
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42 + scene_id}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0], "guider": ["11", 0],
                "sampler": ["13", 0], "sigmas": ["9", 0],
                "latent_image": ["7", 2],
            }
        },
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["2", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def _build_wan_fun_workflow(image_filename: str, prompt: str,
                             scene_id: int = 1, num_frames: int = 81) -> dict:
    """
    Wan 2.1 Fun InP 1.3B I2V workflow.
    Takes a start image and animates it. The 1.3B model runs on MPS with CPU fallback.
    num_frames: must satisfy (n-1) % 4 == 0. Default 81 = ~3.2s at 25fps.
    Uses UMT5 text encoder + Wan 2.1 VAE + CLIP Vision H.
    """
    num_frames = max(5, ((num_frames - 1) // 4) * 4 + 1)
    prefix = f"wan_fun_s{scene_id:02d}_"

    return {
        "1":  {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "2":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": WAN_FUN, "weight_dtype": "default"}},
        "3":  {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                          "type": "wan", "device": "default"}},
        "4":  {"class_type": "CLIPVisionLoader",
               "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "5":  {"class_type": "VAELoader",
               "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "6":  {"class_type": "CLIPVisionEncode",
               "inputs": {"clip_vision": ["4", 0], "image": ["1", 0], "crop": "center"}},
        "7":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt + ", smooth motion, high quality, cinematic",
                          "clip": ["3", 0]}},
        "8":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": "blurry, distorted, low quality, ugly, static",
                          "clip": ["3", 0]}},
        "9":  {"class_type": "WanFunInpaintToVideo",
               "inputs": {
                   "positive": ["7", 0],
                   "negative": ["8", 0],
                   "vae": ["5", 0],
                   "width": 832, "height": 480,
                   "length": num_frames,
                   "batch_size": 1,
                   "clip_vision_output": ["6", 0],
                   "start_image": ["1", 0],
               }},
        "10": {"class_type": "KSampler",
               "inputs": {
                   "model": ["2", 0],
                   "positive": ["9", 0],
                   "negative": ["9", 1],
                   "latent_image": ["9", 2],
                   "seed": 42 + scene_id,
                   "steps": 20,
                   "cfg": 5.0,
                   "sampler_name": "euler",
                   "scheduler": "linear_quadratic",
                   "denoise": 1.0,
               }},
        "11": {"class_type": "VAEDecode",
               "inputs": {"samples": ["10", 0], "vae": ["5", 0]}},
        "12": {"class_type": "SaveImage",
               "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


# ── Wan 2.2 I2V workflow ──────────────────────────────────────────────────────

def _build_wan_workflow(image_filename: str, prompt: str, num_frames: int = 49, scene_id: int = 1) -> dict:
    """
    Build ComfyUI API workflow for Wan 2.2 I2V.
    Flow: LoadImage → CLIPEncode → WanImageToVideo (conditioning+latent)
          → KSampler → VAEDecode → SaveImage (frames)
    Then ffmpeg assembles frames → mp4.
    """
    # num_frames must be: 1 + multiple of 4. Keep low for Apple Silicon MPS.
    num_frames = 17  # ~1s at 16fps — fast enough to test, scale up later
    prefix = f"wan_s{scene_id:02d}_"

    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename}
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default"
            }
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default"
            }
        },
        "4": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"}
        },
        # Encode the start image with CLIP Vision
        "12": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["4", 0],
                "image": ["1", 0],
                "crop": "center",
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt + ", smooth cinematic motion, beautiful animation, high quality",
                "clip": ["3", 0]
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, static, ugly, distorted, low quality, jerky",
                "clip": ["3", 0]
            }
        },
        # WanImageToVideo: start_image + clip_vision_output anchor to source image
        "8": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["5", 0],
                "width": 832,
                "height": 480,
                "length": num_frames,
                "batch_size": 1,
                "clip_vision_output": ["12", 0],
                "start_image": ["1", 0],
            }
        },
        # KSampler uses the model + latent from WanImageToVideo
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["8", 0],   # conditioning from WanImageToVideo
                "negative": ["8", 1],
                "latent_image": ["8", 2],
                "seed": 42,
                "steps": 20,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "linear_quadratic",
                "denoise": 1.0,
            }
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["5", 0],
            }
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": prefix,
            }
        }
    }


def upload_image_to_comfy(image_path: Path) -> str:
    """Upload image to ComfyUI input folder and return filename."""
    import shutil
    dest = COMFYUI_DIR / "input" / image_path.name
    shutil.copy2(str(image_path), str(dest))
    return image_path.name


def _find_gold_bbox(arr):
    """
    Find bounding box of the gold coin in the image using color.
    Gold = high red, medium green, low blue. Returns (x1, y1, x2, y2).
    Falls back to center 50% of frame if not found.
    """
    import numpy as np
    r = arr[:,:,0].astype(float)
    g = arr[:,:,1].astype(float)
    b = arr[:,:,2].astype(float)
    H, W = arr.shape[:2]
    # Gold/yellow: red>150, green>90, blue<130, red clearly dominant over blue
    mask = (r > 150) & (g > 90) & (b < 140) & (r > b * 1.3) & (r > g * 0.7)
    rows = _np_where_1d(mask.any(axis=1))
    cols = _np_where_1d(mask.any(axis=0))
    if len(rows) < 10 or len(cols) < 10:
        # fallback: center 55% of image
        return (int(W * 0.22), int(H * 0.15), int(W * 0.78), int(H * 0.85))
    pad = 30
    return (max(0, cols[0] - pad), max(0, rows[0] - pad),
            min(W, cols[-1] + pad), min(H, rows[-1] + pad))


def _find_red_bboxes(arr, n=6):
    """
    Find bounding boxes of n red dice using color clustering.
    Returns list of (cx, cy, half_w, half_h) per dice sorted left→right.
    """
    import numpy as np
    r = arr[:,:,0].astype(float)
    g = arr[:,:,1].astype(float)
    b = arr[:,:,2].astype(float)
    H, W = arr.shape[:2]
    # Red dice: strong red, low green, low blue
    mask = (r > 130) & (g < 120) & (b < 120) & (r > g * 1.2) & (r > b * 1.2)
    ys, xs = _np_where_2d(mask)
    if len(xs) < 50:
        # fallback: evenly divide lower 70% into n columns
        bboxes = []
        col_w = W // n
        for i in range(n):
            cx = col_w * i + col_w // 2
            cy = int(H * 0.65)
            bboxes.append((cx, cy, col_w // 2 - 5, int(H * 0.25)))
        return bboxes
    # Cluster into n groups by x-coordinate (sort then split)
    order = xs.argsort()
    xs_s, ys_s = xs[order], ys[order]
    size = len(xs_s) // n
    bboxes = []
    for i in range(n):
        chunk_x = xs_s[i*size:(i+1)*size]
        chunk_y = ys_s[i*size:(i+1)*size]
        cx = int(chunk_x.mean())
        cy = int(chunk_y.mean())
        hw = max(40, int((chunk_x.max() - chunk_x.min()) / 2) + 20)
        hh = max(40, int((chunk_y.max() - chunk_y.min()) / 2) + 20)
        bboxes.append((cx, cy, hw, hh))
    return sorted(bboxes, key=lambda b: b[0])


def _np_where_1d(bool_arr):
    """Return indices where bool_arr is True (avoids numpy import at module level)."""
    import numpy as np
    return np.where(bool_arr)[0]


def _np_where_2d(bool_arr):
    """Return (rows, cols) where bool_arr is True."""
    import numpy as np
    return np.where(bool_arr)


def _encode_frames(frames_dir: Path, audio_path: Path, out: Path, fps: int = 25):
    """Encode PNG frame sequence + audio to MP4."""
    list_file = frames_dir / "frames.txt"
    n = len(list(frames_dir.glob("f?????.png")))
    with open(list_file, "w") as f:
        for i in range(n):
            f.write(f"file 'f{i:05d}.png'\nduration {1/fps:.6f}\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(frames_dir))
    return r


def _coin_flip_animation(img_path: Path, audio_path: Path, out: Path):
    """
    Programmatic coin flip.
    - Background plate: coin region erased via per-scanline left↔right interpolation
      (natural sky/ocean fill, not a flat blob).
    - Front face: the coin as rendered in the image (H side).
    - Back face: plain smooth gold gradient (no markings).
    - No elliptical masks — straightforward bounding-box compositing.
    """
    try:
        from PIL import Image
        import numpy as np
        import soundfile as sf
        import shutil
    except ImportError:
        return None

    duration = sf.info(str(audio_path)).duration
    fps = 25
    W, H = 1280, 720
    flip_period = 1.5
    total_frames = int(duration * fps)

    src     = Image.open(str(img_path)).convert("RGB").resize((W, H), Image.LANCZOS)
    src_arr = np.array(src)

    # ── Coin location ─────────────────────────────────────────────────────────
    coin_cx = int(W * 0.50)
    coin_cy = int(H * 0.40)
    coin_rx = int(W * 0.22)
    coin_ry = int(H * 0.32)

    # ── Background plate: erase coin via per-row scanline interpolation ───────
    # For each row that intersects the coin ellipse, interpolate linearly between
    # the pixels just to the LEFT and RIGHT of the coin edge on that row.
    # This gives natural sky/ocean fill rather than a flat colour blob.
    bg_arr = src_arr.copy()
    for y in range(H):
        dy_norm = (y - coin_cy) / coin_ry
        if abs(dy_norm) >= 1.0:
            continue
        dx_half = int(coin_rx * math.sqrt(max(0.0, 1.0 - dy_norm ** 2)))
        xl = coin_cx - dx_half      # left edge of coin on this row
        xr = coin_cx + dx_half      # right edge
        # Sample just outside the coin edge (clipped to frame)
        left_col  = src_arr[y, max(0, xl - 2)].astype(float)
        right_col = src_arr[y, min(W - 1, xr + 2)].astype(float)
        x0 = max(0, xl);  x1 = min(W, xr)
        n  = x1 - x0
        if n <= 0:
            continue
        t = np.linspace(0.0, 1.0, n)
        bg_arr[y, x0:x1] = (left_col * (1 - t[:, None]) +
                             right_col * t[:, None]).astype(np.uint8)
    bg = Image.fromarray(bg_arr)

    # ── Coin bounding box (for patch operations) ──────────────────────────────
    ex1 = max(0, coin_cx - coin_rx)
    ex2 = min(W, coin_cx + coin_rx)
    ey1 = max(0, coin_cy - coin_ry)
    ey2 = min(H, coin_cy + coin_ry)
    patch_w = ex2 - ex1
    patch_h = ey2 - ey1

    # ── Front face: crop coin from source ─────────────────────────────────────
    coin_front = src.crop((ex1, ey1, ex2, ey2))

    # ── Back face: plain smooth gold radial gradient ──────────────────────────
    py_g, px_g = np.mgrid[0:patch_h, 0:patch_w]
    pd = np.sqrt(((px_g - patch_w / 2) / max(1, patch_w / 2)) ** 2 +
                 ((py_g - patch_h / 2) / max(1, patch_h / 2)) ** 2)
    r_ch = (240 - 60 * pd).clip(0, 255).astype(np.uint8)
    g_ch = (200 - 60 * pd).clip(0, 255).astype(np.uint8)
    b_ch = ( 60 - 30 * pd).clip(0, 255).astype(np.uint8)
    back_arr = np.stack([r_ch, g_ch, b_ch], axis=2)
    coin_back = Image.fromarray(back_arr)

    frames_dir = out.parent / f"_cf_frames_{out.stem}"
    frames_dir.mkdir(exist_ok=True)

    for i in range(total_frames):
        t_sec   = i / fps
        phase   = (t_sec % flip_period) / flip_period * 2 * math.pi
        cos_val = math.cos(phase)
        scale_x = max(0.01, abs(cos_val))
        use_back = (cos_val < 0)

        face  = coin_back if use_back else coin_front
        new_w = max(1, int(patch_w * scale_x))
        squeezed = face.resize((new_w, patch_h), Image.LANCZOS)

        # Always start from the clean bg (coin already erased)
        frame = bg.copy()
        # Paste squeezed face centred horizontally in the coin bounding box
        frame.paste(squeezed, (ex1 + (patch_w - new_w) // 2, ey1))
        frame.save(str(frames_dir / f"f{i:05d}.png"))

    r = _encode_frames(frames_dir, audio_path, out, fps)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    if r.returncode != 0:
        print(f"    Coin flip encode error: {r.stderr[-200:]}")
        return None
    print(f"    Coin flip animation: {out.name} ({out.stat().st_size//1024}KB)")
    return out


def _dice_roll_animation(img_path: Path, audio_path: Path, out: Path):
    """
    Programmatic dice rolling: each dice slides left↔right + bounces vertically.
    Dice shape is NEVER distorted — pure 2D translation only, no rotation.
    Background stays perfectly static. Each dice rolls at a staggered phase.
    """
    try:
        from PIL import Image
        import numpy as np
        import soundfile as sf
        import shutil
    except ImportError:
        return None

    duration = sf.info(str(audio_path)).duration
    fps = 25
    W, H = 1280, 720
    roll_period = 2.0   # seconds per left↔right roll cycle
    max_slide   = 12    # pixels horizontal travel
    max_bounce  = 6     # pixels vertical bounce (up only, twice per cycle)

    src     = Image.open(str(img_path)).convert("RGB").resize((W, H), Image.LANCZOS)
    src_arr = np.array(src)
    total_frames = int(duration * fps)

    bboxes = _find_red_bboxes(src_arr, n=6)
    print(f"    Found {len(bboxes)} dice regions")

    # Pre-extract each dice patch once — shape never changes
    pad = 4
    patches, rects = [], []
    for (cx, cy, hw, hh) in bboxes:
        rx1 = max(0, cx - hw - pad);  rx2 = min(W, cx + hw + pad)
        ry1 = max(0, cy - hh - pad);  ry2 = min(H, cy + hh + pad)
        patches.append(src_arr[ry1:ry2, rx1:rx2].copy())
        rects.append((rx1, ry1, rx2, ry2))

    frames_dir = out.parent / f"_dr_frames_{out.stem}"
    frames_dir.mkdir(exist_ok=True)

    for i in range(total_frames):
        t = i / fps
        frame_arr = src_arr.copy()   # fresh static background each frame

        for di, ((rx1, ry1, rx2, ry2), patch) in enumerate(zip(rects, patches)):
            phase = 2 * math.pi * t / roll_period + di * math.pi / 3

            # Horizontal slide (left↔right)
            dx = int(max_slide * math.sin(phase))
            # Vertical bounce: abs(sin(2*phase)) → bounces twice per roll cycle, always upward
            dy = -int(max_bounce * abs(math.sin(2 * phase)))

            ph, pw = patch.shape[:2]

            # Translate destination box
            dx1 = rx1 + dx;  dx2 = rx2 + dx
            dy1 = ry1 + dy;  dy2 = ry2 + dy

            # Clip to frame and adjust source slice accordingly
            cdx1 = max(0, dx1);  cdx2 = min(W, dx2)
            cdy1 = max(0, dy1);  cdy2 = min(H, dy2)
            sx1  = cdx1 - dx1;   sx2  = sx1 + (cdx2 - cdx1)
            sy1  = cdy1 - dy1;   sy2  = sy1 + (cdy2 - cdy1)

            if cdx2 > cdx1 and cdy2 > cdy1 and sx2 <= pw and sy2 <= ph and sx1 >= 0 and sy1 >= 0:
                frame_arr[cdy1:cdy2, cdx1:cdx2] = patch[sy1:sy2, sx1:sx2]

        Image.fromarray(frame_arr).save(str(frames_dir / f"f{i:05d}.png"))

    r = _encode_frames(frames_dir, audio_path, out, fps)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    if r.returncode != 0:
        print(f"    Dice roll encode error: {r.stderr[-200:]}")
        return None
    print(f"    Dice roll animation: {out.name} ({out.stat().st_size//1024}KB)")
    return out


def _kenburns_fallback(img_path: Path, audio_path: Path, out: Path, scene_id: int):
    """
    Ken Burns cinematic zoom/pan using ffmpeg only.
    No model, no GPU — instant. Always works.
    Scene-specific motion for maximum cinematic impact.
    """
    import soundfile as sf
    duration = sf.info(str(audio_path)).duration
    fps = 25
    total_frames = int(duration * fps)
    W, H = 1280, 720
    nf = str(max(total_frames - 1, 1))

    # Scene-specific cinematic motion
    # fmt: zoom_start, zoom_end, x_expr, y_expr
    SCENE_MOTION = {
        # Scene 3 (coin flip): dramatic push-in to centre of coin, no horizontal drift
        3: (1.0, 1.30,
            "(iw-iw/zoom)/2",
            "(ih-ih/zoom)/2"),
        # Scene 4 (dice plains): slow epic pan left→right across the plains, slight zoom
        4: (1.05, 1.10,
            f"(iw-iw/zoom)/2-60+120*n/{nf}",
            "(ih-ih/zoom)/2"),
        # Scene 5 (spotlight dice): zoom into the glowing face
        5: (1.0, 1.25,
            "(iw-iw/zoom)/2",
            "(ih-ih/zoom)/2"),
        # Scene 9 (impossible shore): slow ominous push-in
        9: (1.0, 1.20,
            "(iw-iw/zoom)/2",
            "(ih-ih/zoom)/2"),
    }

    if scene_id in SCENE_MOTION:
        zoom_start, zoom_end, x_expr, y_expr = SCENE_MOTION[scene_id]
    else:
        # Default: alternate zoom in/out + gentle drift
        zoom_in = (scene_id % 2 == 0)
        zoom_start, zoom_end = (1.08, 1.0) if zoom_in else (1.0, 1.08)
        pan_end = 40 if (scene_id % 3 != 0) else -40
        x_expr = f"(iw-iw/zoom)/2+{pan_end}*n/{nf}"
        y_expr = "(ih-ih/zoom)/2"

    z_expr = f"{zoom_start}+({zoom_end}-{zoom_start})*n/{nf}"
    zp_filter = (
        f"scale=8000:-1,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={total_frames}:s={W}x{H}:fps={fps},"
        f"setsar=1"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(img_path),
        "-i", str(audio_path),
        "-vf", zp_filter,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    Ken Burns error: {r.stderr[-200:]}")
        return None
    print(f"    Ken Burns: {out.name} ({out.stat().st_size//1024}KB)")
    return out


def _try_ltxv(scene_id: int, img_path: Path, out: Path):
    """Try LTX Video via ComfyUI. Returns output path or None on any failure."""
    prompt = ANIMATION_PROMPTS.get(scene_id, "smooth cinematic motion, beautiful scene")

    # LTX: 25fps, step=8. 201 frames = ~8s clip.
    num_frames = 201
    num_frames = max(9, ((num_frames - 9) // 8) * 8 + 9)

    # Scene 3: coin flip — more steps, lower strength to keep image stable
    # Scene 4/5: dice — more steps for quality; scene 5 lower strength to preserve 4-dot face
    steps    = 8 if scene_id in (3, 4, 5) else 4
    strength = 0.75 if scene_id in (3, 5) else 1.0

    try:
        img_filename = upload_image_to_comfy(img_path)
    except Exception as e:
        print(f"    Upload failed: {e}")
        return None

    prefix = f"ltxv_s{scene_id:02d}_"
    workflow = _build_ltxv_workflow(img_filename, prompt, num_frames, scene_id, steps=steps, strength=strength)

    try:
        result = _comfy_post("/prompt", {"prompt": workflow})
    except Exception as e:
        print(f"    ComfyUI submit failed: {e}")
        return None

    prompt_id = result["prompt_id"]
    print(f"    Queued: {prompt_id[:8]}... ({num_frames} frames = {num_frames/25:.1f}s)")

    for _ in range(600):   # max ~30 min
        time.sleep(3)
        try:
            history = _comfy_get(f"/history/{prompt_id}")
        except Exception:
            continue
        if prompt_id not in history:
            continue
        entry = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            print(f"    LTX error: {status.get('messages', [{}])[-1]}")
            return None
        outputs = entry.get("outputs", {})
        for node_id, node_out in outputs.items():
            if node_out.get("images"):
                frames_dir = COMFYUI_DIR / "output"
                frames = sorted(frames_dir.glob(f"{prefix}*.png"))
                if frames:
                    print(f"    Got {len(frames)} frames — assembling MP4...")
                    _frames_to_mp4(frames, out, fps=25)
                    for f in frames:
                        f.unlink(missing_ok=True)
                    return out
        if status.get("completed"):
            break

    print(f"    LTX timed out for scene {scene_id}")
    return None


def _try_wan_fun(scene_id: int, img_path: Path, out: Path):
    """
    Try Wan 2.1 Fun InP 1.3B for scenes that need controlled start-frame motion.
    Better than LTX for specific object animation (coin flip, dice).
    Falls back gracefully if model not downloaded or ComfyUI errors.
    """
    if not _wan_fun_available():
        return None

    prompt = ANIMATION_PROMPTS.get(scene_id, "smooth cinematic motion")
    num_frames = 81   # 3.2s at 25fps — safe for Wan 1.3B on MPS

    try:
        img_filename = upload_image_to_comfy(img_path)
    except Exception as e:
        print(f"    Upload failed: {e}")
        return None

    prefix = f"wan_fun_s{scene_id:02d}_"
    workflow = _build_wan_fun_workflow(img_filename, prompt, scene_id, num_frames)

    try:
        result = _comfy_post("/prompt", {"prompt": workflow})
    except Exception as e:
        print(f"    Wan Fun submit failed: {e}")
        return None

    prompt_id = result["prompt_id"]
    print(f"    Wan Fun queued: {prompt_id[:8]}... ({num_frames} frames = {num_frames/25:.1f}s)")

    for _ in range(600):
        time.sleep(3)
        try:
            history = _comfy_get(f"/history/{prompt_id}")
        except Exception:
            continue
        if prompt_id not in history:
            continue
        entry = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            print(f"    Wan Fun error: {status.get('messages', [{}])[-1]}")
            return None
        outputs = entry.get("outputs", {})
        for node_id, node_out in outputs.items():
            if node_out.get("images"):
                frames_dir = COMFYUI_DIR / "output"
                frames = sorted(frames_dir.glob(f"{prefix}*.png"))
                if frames:
                    print(f"    Got {len(frames)} frames — assembling MP4...")
                    _frames_to_mp4(frames, out, fps=25)
                    for f in frames:
                        f.unlink(missing_ok=True)
                    return out
        if status.get("completed"):
            break

    print(f"    Wan Fun timed out for scene {scene_id}")
    return None


def _mlx_available() -> bool:
    """True if mlx-video venv exists."""
    return Path(MLX_VENV_PYTHON).exists()


def _try_mlx_video(scene_id: int, img_path: Path, out: Path):
    """
    MLX-native LTX-2 distilled video generation via mlx-video library.
    Runs in the mflux venv (python 3.11 + mlx-video installed).
    No ComfyUI needed — pure Apple Silicon MLX execution.
    Returns output path on success, None on any failure.
    """
    if not _mlx_available():
        return None

    prompt = ANIMATION_PROMPTS.get(scene_id, "smooth cinematic motion, beautiful scene")
    full_prompt = prompt + ", smooth cinematic motion, high quality, cinematic, beautiful"
    neg_prompt  = "blurry, static, ugly, distorted, low quality, watermark, jitter"

    # 97 frames at 24fps = ~4s. LTX-2 distilled works well at 8 steps.
    num_frames = 97
    steps      = 8

    # Inline Python script executed in the mflux venv
    script = f"""
import sys
try:
    from mlx_video.models.ltx_2.generate import generate_video, PipelineType
except ImportError as e:
    print(f"mlx_video import error: {{e}}", flush=True)
    sys.exit(1)

try:
    generate_video(
        model_repo={MLX_MODEL_REPO!r},
        text_encoder_repo=None,
        prompt={full_prompt!r},
        pipeline=PipelineType.DISTILLED,
        negative_prompt={neg_prompt!r},
        height=480,
        width=832,
        num_frames={num_frames},
        num_inference_steps={steps},
        seed={42 + scene_id},
        fps=24,
        output_path={str(out)!r},
        image={str(img_path)!r},
        image_strength=1.0,
        image_frame_idx=0,
        verbose=True,
    )
    print("MLX generation complete", flush=True)
except Exception as e:
    print(f"MLX generation failed: {{e}}", flush=True)
    sys.exit(1)
"""

    print(f"    MLX LTX-2 distilled ({MLX_MODEL_REPO}): {num_frames} frames @ 24fps, {steps} steps...")
    try:
        result = subprocess.run(
            [MLX_VENV_PYTHON, "-c", script],
            timeout=3600,   # 60 min max
        )
        if result.returncode == 0 and out.exists() and out.stat().st_size > 10_000:
            print(f"    MLX done: {out.name} ({out.stat().st_size//1024}KB)")
            return out
        print(f"    MLX failed (returncode={result.returncode})")
        if out.exists():
            out.unlink(missing_ok=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"    MLX timed out for scene {scene_id}")
        return None
    except Exception as e:
        print(f"    MLX error: {e}")
        return None


def animate_scene(scene_id: int):
    out = CLIPS_DIR / f"scene_{scene_id:02d}.mp4"
    if out.exists():
        print(f"  Scene {scene_id:02d}: cached")
        return out

    img_path   = SCENES_DIR / f"scene_{scene_id:02d}.png"
    audio_path = AUDIO_DIR  / f"scene_{scene_id:02d}.wav"

    if not img_path.exists():
        print(f"  Scene {scene_id:02d}: no image, skip")
        return None

    # ── 1. MLX-native LTX-2 (best quality, no ComfyUI needed) ───────────────
    if _mlx_available():
        print(f"  Scene {scene_id:02d}: trying MLX LTX-2 distilled (native Apple Silicon)...")
        result = _try_mlx_video(scene_id, img_path, out)
        if result:
            return result
        print(f"  Scene {scene_id:02d}: MLX failed — falling back to ComfyUI")

    if not check_comfyui():
        print(f"  Scene {scene_id:02d}: ComfyUI not running — using Ken Burns fallback")
        if audio_path.exists():
            return _kenburns_fallback(img_path, audio_path, out, scene_id)
        return None

    # ── 2. Wan 2.1 Fun InP (ComfyUI, controlled object motion) ──────────────
    if scene_id in (3, 4) and _wan_fun_available():
        print(f"  Scene {scene_id:02d}: trying Wan 2.1 Fun InP 1.3B...")
        result = _try_wan_fun(scene_id, img_path, out)
        if result:
            return result
        print(f"  Scene {scene_id:02d}: Wan Fun failed — falling back to LTX Video")

    # ── 3. LTX Video via ComfyUI (13B if available, else 2B) ─────────────────
    model = _ltxv_model_name()
    print(f"  Scene {scene_id:02d}: trying LTX Video ({model})...")
    result = _try_ltxv(scene_id, img_path, out)
    if result:
        return result
    print(f"  Scene {scene_id:02d}: LTX failed — using Ken Burns fallback")

    # ── 4. Last resort: Ken Burns zoom/pan ───────────────────────────────────
    if audio_path.exists():
        return _kenburns_fallback(img_path, audio_path, out, scene_id)

    print(f"  Scene {scene_id:02d}: no audio, skip")
    return None


def _make_pingpong(src: Path, out: Path):
    """Create forward+reverse (ping-pong) MP4 via ffmpeg. No GIF jump at loop point."""
    rev = src.parent / f"_rev_{src.name}"
    # Reverse the clip
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "reverse", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(rev)
    ], check=True, capture_output=True)
    # Concatenate forward + reversed
    list_file = src.parent / "_pp_list.txt"
    with open(list_file, "w") as f:
        f.write(f"file '{src}'\nfile '{rev}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(out)
    ], check=True, capture_output=True)
    list_file.unlink(missing_ok=True)
    rev.unlink(missing_ok=True)


def _frames_to_mp4(frames: list, out: Path, fps: int = 16):
    """Assemble PNG frames into MP4 using ffmpeg."""
    # Write frame list
    list_file = out.parent / f"frames_{out.stem}.txt"
    with open(list_file, "w") as f:
        for frame in frames:
            f.write(f"file '{frame}'\nduration {1/fps}\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-vf", f"fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out)
    ], check=True, capture_output=True)
    list_file.unlink(missing_ok=True)
    print(f"    MP4: {out.name} ({out.stat().st_size//1024}KB)")


def animate_all():
    # Try to start ComfyUI, but continue even if it fails (Ken Burns fallback)
    if not check_comfyui():
        started = start_comfyui()
        if not started:
            print("ComfyUI unavailable — will use Ken Burns fallback for all scenes.")
    ltx = _ltxv_model_name()
    wan = "Wan 2.1 Fun InP" if _wan_fun_available() else "not available"
    print(f"Animating all scenes | LTX: {ltx} | Wan Fun InP: {wan} | fallback: Ken Burns")
    for scene_id in range(1, 11):
        animate_scene(scene_id)
    print("All animations done!")


# ── Final assembly: animated clips + audio → MP4 ─────────────────────────────

def assemble_final():
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip

    clips = []
    for scene_id in range(1, 11):
        clip_path  = CLIPS_DIR / f"scene_{scene_id:02d}.mp4"
        audio_path = AUDIO_DIR  / f"scene_{scene_id:02d}.wav"

        if not audio_path.exists():
            print(f"  SKIP {scene_id}: no audio")
            continue

        audio = AudioFileClip(str(audio_path))

        if clip_path.exists():
            # Build ping-pong MP4 via ffmpeg (more reliable than moviepy time_mirror)
            pingpong_path = CLIPS_DIR / f"scene_{scene_id:02d}_pp.mp4"
            if not pingpong_path.exists():
                _make_pingpong(clip_path, pingpong_path)
            video = VideoFileClip(str(pingpong_path))
            if video.duration < audio.duration:
                from moviepy.editor import vfx
                video = video.fx(vfx.loop, duration=audio.duration)
            else:
                video = video.subclip(0, audio.duration)
            clip = video.set_audio(audio)
        else:
            # Fall back to static image
            img_path = SCENES_DIR / f"scene_{scene_id:02d}.png"
            if not img_path.exists():
                print(f"  SKIP {scene_id}: no image or clip")
                continue
            clip = ImageClip(str(img_path)).set_duration(audio.duration + 0.5).set_audio(audio)
            print(f"  Scene {scene_id:02d}: static fallback")

        clips.append(clip)
        print(f"  Scene {scene_id:02d}: {clip.duration:.1f}s")

    if not clips:
        print("No clips.")
        return

    print(f"\nAssembling {len(clips)} clips...")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(str(FINAL_OUT), fps=24, codec="libx264", audio_codec="aac", logger="bar")
    print(f"\nSaved: {FINAL_OUT}")
    return FINAL_OUT


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test",     action="store_true", help="Animate scene 2 only (quick test)")
    p.add_argument("--all",      action="store_true", help="Animate all 10 scenes")
    p.add_argument("--assemble", action="store_true", help="Merge clips + audio into final MP4")
    p.add_argument("--full",     action="store_true", help="--all + --assemble")
    a = p.parse_args()

    if a.test:
        if not ensure_comfyui(): sys.exit(1)
        clip = animate_scene(2)
        if clip:
            # Quick preview: merge scene 2 clip + audio
            from moviepy.editor import VideoFileClip, AudioFileClip, vfx
            audio = AudioFileClip(str(AUDIO_DIR / "scene_02.wav"))
            video = VideoFileClip(str(clip))
            if video.duration < audio.duration:
                video = video.fx(vfx.loop, duration=audio.duration)
            preview = str(AI_EDU_DIR / "output" / "test_scene02_animated.mp4")
            video.set_audio(audio).write_videofile(preview, fps=24, codec="libx264", audio_codec="aac", logger="bar")
            subprocess.run(["open", preview])
            print(f"Preview: {preview}")

    if a.all or a.full:
        animate_all()

    if a.assemble or a.full:
        out = assemble_final()
        if out:
            subprocess.run(["open", str(out)])

    if not any(vars(a).values()):
        print(__doc__)

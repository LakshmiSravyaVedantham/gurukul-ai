"""
Video model benchmark — test every available animation model on the same scene image.

Outputs: output/model_tests/<model_name>.mp4  (one clip per model)
At the end prints a comparison table (size, duration, time taken).

Models tested:
  ComfyUI (needs ComfyUI on port 8288):
    1. ltx-2b        — LTX Video 2B distilled FP8   (4 steps,  fast)
    2. ltx-13b       — LTX Video 13B distilled FP8  (8 steps,  high quality)
    3. wan-fun-1b    — Wan 2.1 Fun InP 1.3B BF16    (20 steps, start-frame anchored)
    4. wan22-low     — Wan 2.2 I2V 14B low-noise    (20 steps)
    5. wan22-high    — Wan 2.2 I2V 14B high-noise   (20 steps)

  MLX-native (no ComfyUI, auto-downloads model first time ~14 GB):
    6. mlx-ltx2      — LTX-2 distilled (prince-canuma/LTX-2-distilled)

Usage:
    python test_models.py                        # test all available models on scene_01
    python test_models.py --scene 3              # test on scene_03 (coin flip)
    python test_models.py --models ltx-2b ltx-13b   # only specific models
    python test_models.py --no-comfyui           # skip ComfyUI tests (MLX only)
    python test_models.py --no-mlx               # skip MLX tests (ComfyUI only)
"""

import argparse, json, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
COMFYUI_DIR  = Path("/Volumes/bujji1/sravya/ComfyUI")
COMFYUI_URL  = "http://127.0.0.1:8288"
MLX_PYTHON   = "/Volumes/bujji1/sravya/ai_vidgen/venv/bin/python"
MLX_REPO     = "prince-canuma/LTX-2-distilled"
TEST_DIR     = AI_EDU_DIR / "output" / "model_tests"

TEST_DIR.mkdir(parents=True, exist_ok=True)

# ── Test prompt (scene 1 — safe for all models) ───────────────────────────────
TEST_PROMPTS = {
    1:  "slow majestic aerial camera pan over a magical fantasy island, "
        "golden light shifts softly, trees sway gently, river shimmers",
    3:  "gold coin slowly flipping in the air in smooth slow motion, "
        "one face shows the letter H clearly, rotates to reveal letter T, "
        "warm golden light, coin stays centered, both faces readable",
    4:  "six large red dice gently rocking and slowly rolling in place, "
        "each dice shows its dots clearly, subtle gentle rocking motion",
}

NEG_PROMPT = "blurry, static, ugly, distorted, low quality, watermark, jitter"

# ── ComfyUI helpers ───────────────────────────────────────────────────────────

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

def _comfy_up():
    try: _comfy_get("/system_stats"); return True
    except: return False

def _upload(img_path: Path) -> str:
    dest = COMFYUI_DIR / "input" / img_path.name
    shutil.copy2(str(img_path), str(dest))
    return img_path.name

def _wait_for_job(prompt_id: str, prefix: str, out: Path, fps: int = 25,
                  max_wait: int = 3600) -> Path | None:
    """Poll ComfyUI until job completes, then assemble MP4."""
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
            print(f"      ComfyUI error: {status.get('messages', [])[-1:]}")
            return None
        outputs = entry.get("outputs", {})
        for _, node_out in outputs.items():
            if node_out.get("images"):
                frames = sorted((COMFYUI_DIR / "output").glob(f"{prefix}*.png"))
                if frames:
                    _frames_to_mp4(frames, out, fps)
                    for f in frames: f.unlink(missing_ok=True)
                    return out
        if status.get("completed"):
            break
    return None

def _frames_to_mp4(frames: list, out: Path, fps: int):
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

# ── ComfyUI workflows ─────────────────────────────────────────────────────────

def _wf_ltx(img_file: str, prompt: str, model: str, steps: int,
             num_frames: int, scene_id: int, label: str) -> dict:
    """LTX Video I2V workflow (works for both 2B and 13B)."""
    num_frames = max(9, ((num_frames - 9) // 8) * 8 + 9)
    prefix = f"test_{label}_"
    return {
        "1":  {"class_type": "LoadImage",           "inputs": {"image": img_file}},
        "2":  {"class_type": "CheckpointLoaderSimple","inputs": {"ckpt_name": model}},
        "3":  {"class_type": "CLIPLoader",           "inputs": {"clip_name": "t5xxl_fp8_e4m3fn.safetensors", "type": "ltxv"}},
        "5":  {"class_type": "CLIPTextEncode",       "inputs": {"text": prompt + ", smooth cinematic motion, high quality", "clip": ["3", 0]}},
        "6":  {"class_type": "CLIPTextEncode",       "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
        "7":  {"class_type": "LTXVImgToVideo",
               "inputs": {"positive": ["5", 0], "negative": ["6", 0], "vae": ["2", 2],
                          "image": ["1", 0], "width": 768, "height": 512,
                          "length": num_frames, "batch_size": 1, "strength": 1.0}},
        "8":  {"class_type": "LTXVConditioning",    "inputs": {"positive": ["7", 0], "negative": ["7", 1], "frame_rate": 25.0}},
        "9":  {"class_type": "LTXVScheduler",
               "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95,
                          "stretch": True, "terminal": 0.1, "latent": ["7", 2]}},
        "10": {"class_type": "RandomNoise",          "inputs": {"noise_seed": 42 + scene_id}},
        "11": {"class_type": "BasicGuider",          "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "KSamplerSelect",       "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10", 0], "guider": ["11", 0],
                          "sampler": ["13", 0], "sigmas": ["9", 0], "latent_image": ["7", 2]}},
        "15": {"class_type": "VAEDecode",            "inputs": {"samples": ["14", 0], "vae": ["2", 2]}},
        "12": {"class_type": "SaveImage",            "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }

def _wf_wan_fun(img_file: str, prompt: str, num_frames: int, scene_id: int) -> dict:
    """Wan 2.1 Fun InP 1.3B workflow."""
    num_frames = max(5, ((num_frames - 1) // 4) * 4 + 1)
    return {
        "1":  {"class_type": "LoadImage",       "inputs": {"image": img_file}},
        "2":  {"class_type": "UNETLoader",      "inputs": {"unet_name": "wan2.1_fun_inp_1.3B_bf16.safetensors", "weight_dtype": "default"}},
        "3":  {"class_type": "CLIPLoader",      "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "4":  {"class_type": "CLIPVisionLoader","inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "5":  {"class_type": "VAELoader",       "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
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
                          "steps": 20, "cfg": 5.0, "sampler_name": "euler",
                          "scheduler": "linear_quadratic", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode",       "inputs": {"samples": ["10", 0], "vae": ["5", 0]}},
        "12": {"class_type": "SaveImage",       "inputs": {"images": ["11", 0], "filename_prefix": "test_wan-fun-1b_"}},
    }

def _wf_wan22(img_file: str, prompt: str, model_name: str, num_frames: int,
              scene_id: int, label: str) -> dict:
    """Wan 2.2 I2V 14B workflow (low-noise or high-noise)."""
    num_frames = max(5, ((num_frames - 1) // 4) * 4 + 1)
    return {
        "1":  {"class_type": "LoadImage",       "inputs": {"image": img_file}},
        "2":  {"class_type": "UNETLoader",      "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "3":  {"class_type": "CLIPLoader",      "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "4":  {"class_type": "CLIPVisionLoader","inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
        "5":  {"class_type": "VAELoader",       "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "12": {"class_type": "CLIPVisionEncode","inputs": {"clip_vision": ["4", 0], "image": ["1", 0], "crop": "center"}},
        "6":  {"class_type": "CLIPTextEncode",  "inputs": {"text": prompt + ", smooth cinematic motion, high quality", "clip": ["3", 0]}},
        "7":  {"class_type": "CLIPTextEncode",  "inputs": {"text": NEG_PROMPT, "clip": ["3", 0]}},
        "8":  {"class_type": "WanImageToVideo",
               "inputs": {"positive": ["6", 0], "negative": ["7", 0], "vae": ["5", 0],
                          "width": 832, "height": 480, "length": num_frames,
                          "batch_size": 1, "clip_vision_output": ["12", 0], "start_image": ["1", 0]}},
        "9":  {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "positive": ["8", 0], "negative": ["8", 1],
                          "latent_image": ["8", 2], "seed": 42 + scene_id,
                          "steps": 20, "cfg": 5.0, "sampler_name": "euler",
                          "scheduler": "linear_quadratic", "denoise": 1.0}},
        "10": {"class_type": "VAEDecode",       "inputs": {"samples": ["9", 0], "vae": ["5", 0]}},
        "11": {"class_type": "SaveImage",       "inputs": {"images": ["10", 0], "filename_prefix": f"test_{label}_"}},
    }

# ── Individual model test functions ───────────────────────────────────────────

def test_ltx_2b(img_path: Path, prompt: str, scene_id: int) -> Path | None:
    out = TEST_DIR / "ltx-2b.mp4"
    img_file = _upload(img_path)
    wf = _wf_ltx(img_file, prompt, "ltxv-2b-0.9.8-distilled-fp8.safetensors",
                 steps=4, num_frames=97, scene_id=scene_id, label="ltx-2b")
    r = _comfy_post("/prompt", {"prompt": wf})
    print(f"    Queued {r['prompt_id'][:8]}...")
    return _wait_for_job(r["prompt_id"], "test_ltx-2b_", out)

def test_ltx_13b(img_path: Path, prompt: str, scene_id: int) -> Path | None:
    out = TEST_DIR / "ltx-13b.mp4"
    ck = COMFYUI_DIR / "models" / "checkpoints" / "ltxv-13b-0.9.8-distilled-fp8.safetensors"
    if not ck.exists():
        print("    ltx-13b not downloaded — skip")
        return None
    img_file = _upload(img_path)
    wf = _wf_ltx(img_file, prompt, "ltxv-13b-0.9.8-distilled-fp8.safetensors",
                 steps=8, num_frames=97, scene_id=scene_id, label="ltx-13b")
    r = _comfy_post("/prompt", {"prompt": wf})
    print(f"    Queued {r['prompt_id'][:8]}...")
    return _wait_for_job(r["prompt_id"], "test_ltx-13b_", out)

def test_wan_fun_1b(img_path: Path, prompt: str, scene_id: int) -> Path | None:
    out = TEST_DIR / "wan-fun-1b.mp4"
    wf_path = COMFYUI_DIR / "models" / "diffusion_models" / "wan2.1_fun_inp_1.3B_bf16.safetensors"
    if not wf_path.exists():
        print("    wan-fun-1b not downloaded — skip")
        return None
    img_file = _upload(img_path)
    wf = _wf_wan_fun(img_file, prompt, num_frames=81, scene_id=scene_id)
    r = _comfy_post("/prompt", {"prompt": wf})
    print(f"    Queued {r['prompt_id'][:8]}...")
    return _wait_for_job(r["prompt_id"], "test_wan-fun-1b_", out, fps=25)

def test_wan22_low(img_path: Path, prompt: str, scene_id: int) -> Path | None:
    out = TEST_DIR / "wan22-low.mp4"
    mp = COMFYUI_DIR / "models" / "diffusion_models" / "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    if not mp.exists():
        print("    wan22-low not downloaded — skip")
        return None
    img_file = _upload(img_path)
    wf = _wf_wan22(img_file, prompt, mp.name, num_frames=81, scene_id=scene_id, label="wan22-low")
    r = _comfy_post("/prompt", {"prompt": wf})
    print(f"    Queued {r['prompt_id'][:8]}...")
    return _wait_for_job(r["prompt_id"], "test_wan22-low_", out, fps=25)

def test_wan22_high(img_path: Path, prompt: str, scene_id: int) -> Path | None:
    out = TEST_DIR / "wan22-high.mp4"
    mp = COMFYUI_DIR / "models" / "diffusion_models" / "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    if not mp.exists():
        print("    wan22-high not downloaded — skip")
        return None
    img_file = _upload(img_path)
    wf = _wf_wan22(img_file, prompt, mp.name, num_frames=81, scene_id=scene_id, label="wan22-high")
    r = _comfy_post("/prompt", {"prompt": wf})
    print(f"    Queued {r['prompt_id'][:8]}...")
    return _wait_for_job(r["prompt_id"], "test_wan22-high_", out, fps=25)

def test_mlx_ltx2(img_path: Path, prompt: str, scene_id: int) -> Path | None:
    out = TEST_DIR / "mlx-ltx2.mp4"
    if not Path(MLX_PYTHON).exists():
        print("    mlx-video venv not found — skip")
        return None

    full_prompt = prompt + ", smooth cinematic motion, high quality, cinematic, beautiful"
    script = f"""
import sys
try:
    from mlx_video.models.ltx_2.generate import generate_video, PipelineType
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)
generate_video(
    model_repo={MLX_REPO!r},
    text_encoder_repo=None,
    prompt={full_prompt!r},
    pipeline=PipelineType.DISTILLED,
    negative_prompt={NEG_PROMPT!r},
    height=480, width=832,
    num_frames=97,
    num_inference_steps=8,
    seed={42 + scene_id},
    fps=24,
    output_path={str(out)!r},
    image={str(img_path)!r},
    image_strength=1.0,
    image_frame_idx=0,
    verbose=True,
)
"""
    print(f"    MLX LTX-2 distilled (downloads {MLX_REPO} if not cached)...")
    result = subprocess.run([MLX_PYTHON, "-c", script], timeout=3600)
    if result.returncode == 0 and out.exists() and out.stat().st_size > 10_000:
        return out
    print(f"    MLX failed (exit {result.returncode})")
    if out.exists(): out.unlink(missing_ok=True)
    return None

# ── Model registry ────────────────────────────────────────────────────────────

ALL_MODELS = {
    "ltx-2b":     ("ComfyUI", test_ltx_2b),
    "ltx-13b":    ("ComfyUI", test_ltx_13b),
    "wan-fun-1b": ("ComfyUI", test_wan_fun_1b),
    "wan22-low":  ("ComfyUI", test_wan22_low),
    "wan22-high": ("ComfyUI", test_wan22_high),
    "mlx-ltx2":   ("MLX",    test_mlx_ltx2),
}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=1,
                    help="Which scene image to test on (default: 1)")
    ap.add_argument("--models", nargs="+", choices=list(ALL_MODELS),
                    help="Specific models to test (default: all)")
    ap.add_argument("--no-comfyui", action="store_true",
                    help="Skip all ComfyUI models")
    ap.add_argument("--no-mlx", action="store_true",
                    help="Skip all MLX models")
    args = ap.parse_args()

    scene_id  = args.scene
    img_path  = AI_EDU_DIR / "output" / "island_scenes" / f"scene_{scene_id:02d}.png"
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        print("Run: python gurukul_island.py --scenes")
        sys.exit(1)

    prompt = TEST_PROMPTS.get(scene_id,
        "slow cinematic camera pan, smooth motion, golden light, high quality")

    models_to_test = args.models or list(ALL_MODELS)
    if args.no_comfyui:
        models_to_test = [m for m in models_to_test if ALL_MODELS[m][0] != "ComfyUI"]
    if args.no_mlx:
        models_to_test = [m for m in models_to_test if ALL_MODELS[m][0] != "MLX"]

    # Check ComfyUI if needed
    comfyui_needed = any(ALL_MODELS[m][0] == "ComfyUI" for m in models_to_test)
    if comfyui_needed and not _comfy_up():
        print("ComfyUI not running on port 8288.")
        print(f"Start it: cd {COMFYUI_DIR} && python main.py --port 8288 --preview-method none")
        print("Re-run with --no-comfyui to test only MLX models.\n")
        # Don't exit — MLX models can still run
        models_to_test = [m for m in models_to_test if ALL_MODELS[m][0] != "ComfyUI"]
        if not models_to_test:
            sys.exit(1)

    print(f"\nTesting {len(models_to_test)} model(s) on scene_{scene_id:02d}.png")
    print(f"Results → {TEST_DIR}\n")

    results = []
    for name in models_to_test:
        engine, fn = ALL_MODELS[name]
        print(f"[{name}] ({engine})")
        t0 = time.time()
        out = fn(img_path, prompt, scene_id)
        elapsed = time.time() - t0
        if out and out.exists():
            size_mb = out.stat().st_size / 1_048_576
            # Get video duration via ffprobe
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
                    capture_output=True, text=True)
                dur = float(r.stdout.strip())
            except Exception:
                dur = 0.0
            print(f"    ✓ {out.name}  {size_mb:.1f}MB  {dur:.1f}s clip  (took {elapsed:.0f}s)")
            results.append((name, engine, size_mb, dur, elapsed, "OK"))
        else:
            print(f"    ✗ FAILED  (took {elapsed:.0f}s)")
            results.append((name, engine, 0, 0, elapsed, "FAIL"))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"{'Model':<15} {'Engine':<8} {'Size':>7} {'Clip':>6} {'Time':>7} {'Status'}")
    print("─" * 60)
    for name, engine, size_mb, dur, elapsed, status in results:
        clip_str = f"{dur:.1f}s"  if dur  else "—"
        size_str = f"{size_mb:.1f}MB" if size_mb else "—"
        time_str = f"{elapsed:.0f}s"
        print(f"{name:<15} {engine:<8} {size_str:>7} {clip_str:>6} {time_str:>7}  {status}")
    print("─" * 60)
    print(f"\nClips saved in: {TEST_DIR}")


if __name__ == "__main__":
    main()

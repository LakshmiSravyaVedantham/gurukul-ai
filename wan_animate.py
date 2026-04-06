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

import json, time, subprocess, sys, urllib.request, urllib.parse
from pathlib import Path

AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
COMFYUI_DIR  = Path("/Volumes/bujji1/sravya/ComfyUI")
SCENES_DIR   = AI_EDU_DIR / "output" / "island_scenes"
AUDIO_DIR    = AI_EDU_DIR / "output" / "island_audio"
CLIPS_DIR    = AI_EDU_DIR / "output" / "island_clips"   # animated video clips
FINAL_OUT    = AI_EDU_DIR / "output" / "animated.mp4"
COMFYUI_URL  = "http://127.0.0.1:8288"

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

    5:  "spotlight beam slowly sweeps across dice boulders and locks onto the four-dot face, "
        "golden glow pulses and brightens on the four dots, dramatic cinematic reveal",

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

# ── LTX Video I2V workflow ────────────────────────────────────────────────────

def _build_ltxv_workflow(image_filename: str, prompt: str, num_frames: int = 49, scene_id: int = 1, steps: int = 4, strength: float = 1.0) -> dict:
    """
    LTX Video 2B distilled I2V workflow.
    Distilled model = only 4 steps. Fast on Apple Silicon MPS.
    Flow: LoadImage → CLIPTextEncode → LTXVImgToVideo → LTXVConditioning
          → LTXVScheduler → KSampler → VAEDecode → SaveImage
    """
    # frames: min 9, step 8. 49 frames = ~2s at 25fps
    num_frames = max(9, ((num_frames - 9) // 8) * 8 + 9)
    prefix = f"ltxv_s{scene_id:02d}_"

    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        # CheckpointLoaderSimple loads model[0] + clip[1] + vae[2] from the bundled file
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltxv-2b-0.9.8-distilled-fp8.safetensors"}},
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
        # Steps also passed to KSamplerSelect via sigmas above
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["2", 2]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
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


def _kenburns_fallback(img_path: Path, audio_path: Path, out: Path, scene_id: int):
    """
    Fallback: Ken Burns cinematic zoom/pan using ffmpeg only.
    No model, no GPU — instant. Always works.
    """
    import soundfile as sf
    duration = sf.info(str(audio_path)).duration
    fps = 25
    total_frames = int(duration * fps)
    W, H = 1280, 720

    # Alternate zoom direction per scene
    zoom_in = (scene_id % 2 == 0)
    zoom_start, zoom_end = (1.08, 1.0) if zoom_in else (1.0, 1.08)
    # Gentle horizontal drift
    pan_end = 40 if (scene_id % 3 != 0) else -40

    nf = str(max(total_frames - 1, 1))
    z_expr = f"{zoom_start}+({zoom_end}-{zoom_start})*n/{nf}"
    x_expr = f"(iw-iw/zoom)/2+{pan_end}*n/{nf}"
    y_expr = "(ih-ih/zoom)/2"

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
    print(f"    Ken Burns fallback: {out.name} ({out.stat().st_size//1024}KB)")
    return out


def _try_ltxv(scene_id: int, img_path: Path, out: Path):
    """Try LTX Video via ComfyUI. Returns output path or None on any failure."""
    prompt = ANIMATION_PROMPTS.get(scene_id, "smooth cinematic motion, beautiful scene")

    # LTX: 25fps, step=8. 201 frames = ~8s clip.
    num_frames = 201
    num_frames = max(9, ((num_frames - 9) // 8) * 8 + 9)

    # Scene 3: coin flip — more steps, lower strength to keep image stable
    # Scene 4: dice — more steps for quality
    steps    = 8 if scene_id in (3, 4) else 4
    strength = 0.75 if scene_id == 3 else 1.0

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


def animate_scene(scene_id: int):
    out = CLIPS_DIR / f"scene_{scene_id:02d}.mp4"
    if out.exists():
        print(f"  Scene {scene_id:02d}: cached")
        return out

    img_path  = SCENES_DIR / f"scene_{scene_id:02d}.png"
    audio_path = AUDIO_DIR / f"scene_{scene_id:02d}.wav"

    if not img_path.exists():
        print(f"  Scene {scene_id:02d}: no image, skip")
        return None

    print(f"  Scene {scene_id:02d}: trying LTX Video...")

    # ── Attempt 1: LTX Video via ComfyUI ─────────────────────────────────────
    if check_comfyui():
        result = _try_ltxv(scene_id, img_path, out)
        if result:
            return result
        print(f"  Scene {scene_id:02d}: LTX failed — falling back to Ken Burns")
    else:
        print(f"  Scene {scene_id:02d}: ComfyUI not running — using Ken Burns fallback")

    # ── Attempt 2: Ken Burns zoom/pan (ffmpeg, no model needed) ──────────────
    if audio_path.exists():
        return _kenburns_fallback(img_path, audio_path, out, scene_id)

    print(f"  Scene {scene_id:02d}: no audio for Ken Burns fallback, skip")
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
    print("Animating all scenes (LTX Video → Ken Burns fallback)...")
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

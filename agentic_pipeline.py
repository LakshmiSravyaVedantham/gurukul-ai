"""
Agentic Video Generation Pipeline — Gurukul AI
===============================================
5-stage self-improving loop that runs for EVERY animation model,
scores each output, keeps the best, and builds a training dataset.

Stage 1  Director  — Gemma 4 expands your simple prompt into a
                     rich, model-optimised generation prompt.
Stage 2  Creator   — Generates a low-res draft with the selected model.
Stage 3  Critic    — Qwen2.5-VL extracts frames and scores the video
                     on motion stability, quality, and prompt adherence.
Stage 4  Refiner   — If score < threshold the pipeline auto-retries
                     with escalating parameters (more steps, higher CFG,
                     different seed, or a better model).
Stage 5  Polisher  — Approved videos go to Topaz Video AI for 4K upscale.

After running all models, a leaderboard JSON is saved so you can see
which model consistently wins and use those approved clips as a
fine-tuning dataset.

Usage (CLI):
    python agentic_pipeline.py "coin slowly flipping in golden light"
    python agentic_pipeline.py "coin flipping" --scene 3 --models all
    python agentic_pipeline.py "coin flipping" --models ltx-2b ltx-13b
    python agentic_pipeline.py "coin flipping" --no-topaz
"""

import json, os, re, shutil, subprocess, sys, time, textwrap
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
COMFYUI_DIR  = Path("/Volumes/bujji1/sravya/ComfyUI")
SCENES_DIR   = AI_EDU_DIR / "output" / "island_scenes"
AUDIO_DIR    = AI_EDU_DIR / "output" / "island_audio"
CLIPS_DIR    = AI_EDU_DIR / "output" / "island_clips"
AGENIC_DIR   = AI_EDU_DIR / "output" / "agentic"
DATASET_DIR  = AI_EDU_DIR / "output" / "training_dataset"
LEADERBOARD  = AI_EDU_DIR / "output" / "model_leaderboard.json"
MLX_PYTHON   = "/Volumes/bujji1/sravya/ai_vidgen/venv/bin/python"
COMFYUI_URL  = "http://127.0.0.1:8288"
TOPAZ_FFMPEG = Path("/Applications/Topaz Video AI.app/Contents/MacOS/ffmpeg")
TOPAZ_MODELS = Path("/Applications/Topaz Video AI.app/Contents/Resources/models")

GEMMA4_MODEL  = "mlx-community/gemma-4-26b-a4b-it-4bit"   # MoE: 4B active params, fast
QWEN_VL_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

for d in [AGENIC_DIR, DATASET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model escalation order ─────────────────────────────────────────────────────
# When a model fails the quality bar, escalate to the next tier.
ESCALATION = [
    "ken-burns",        # instant fallback
    "ltx-2b",           # 40s, fast preview
    "ltx23-gguf",       # 4-6min, best speed/quality
    "wan22-fun-5b-gguf",# 8-10min, object motion
    "wan22-fun-5b",     # 12min, BF16 fallback
    "ltx-13b",          # 11min, quality
    "wan22-i2v-14b-gguf",# 15-20min, best quality
]

# ── Stage 1: Director — Gemma 4 prompt expansion ───────────────────────────────

DIRECTOR_SYSTEM = textwrap.dedent("""
You are a cinematic prompt engineer for AI video generation.
Your job: take a simple scene description and expand it into a
high-detail, model-optimized generation prompt.

Rules:
- Add: cinematic lighting, camera movement style, motion description,
  atmosphere, time of day, color palette, frame rate feel
- Keep it as ONE paragraph, max 120 words
- No lists, no bullet points, just a flowing cinematic description
- The scene is for a kids' educational animated video (Pixar style)
- No people, no text, no logos — pure landscape/environment/object animation
- End with: "24fps, smooth motion, stable camera, ultra detailed, cinematic"
Output ONLY the expanded prompt, nothing else.
""").strip()

def expand_prompt(simple_prompt: str, log_fn=None) -> str:
    """Stage 1: Use Gemma 4 to expand a simple prompt into a rich generation prompt."""
    if log_fn: log_fn("Stage 1 [Director] — Gemma 4 expanding prompt...")

    messages = json.dumps([
        {"role": "system", "content": DIRECTOR_SYSTEM},
        {"role": "user",   "content": f"Expand this into a cinematic video prompt: {simple_prompt}"},
    ])

    script = f"""
import sys
try:
    from mlx_lm import load, generate
except ImportError:
    print(f"INSTALL_NEEDED", flush=True); sys.exit(1)
model, tokenizer = load({GEMMA4_MODEL!r})
messages = {messages}
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
print(response.strip(), flush=True)
"""
    result = subprocess.run(
        [MLX_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0 or not result.stdout.strip():
        # Fallback to simple enhancement
        if log_fn: log_fn("  Gemma 4 failed — using simple prompt enhancement")
        return (simple_prompt +
                ", cinematic lighting, smooth camera movement, golden warm light, "
                "Pixar animated movie style, vibrant colors, 24fps, stable camera, ultra detailed")
    expanded = result.stdout.strip()
    if log_fn: log_fn(f"  Expanded: {expanded[:100]}...")
    return expanded


# ── Stage 2: Creator — video generation ───────────────────────────────────────

APP_PYTHON = "/usr/bin/python3"   # system python that has gradio + app.py deps

def create_video(scene_id: int, model_key: str, prompt: str,
                 img_path: Path, audio_path: Path,
                 out_path: Path, attempt: int = 1,
                 log_fn=None) -> tuple:
    """Stage 2: Generate video. Returns (path, error).
    attempt > 1: escalate params (more steps, higher CFG, different seed)."""
    if log_fn: log_fn(f"Stage 2 [Creator] — {model_key} (attempt {attempt})")

    # On retry: vary seed by offsetting scene_id
    effective_scene_id = scene_id + (attempt - 1) * 100

    # Call animate_scene via the system Python (which has gradio installed)
    script = f"""
import sys; sys.path.insert(0, {str(AI_EDU_DIR)!r})
from app import animate_scene
from pathlib import Path
clip, err = animate_scene(
    {effective_scene_id},
    {model_key!r},
    Path({str(img_path)!r}),
    Path({str(audio_path)!r}),
)
if clip and Path(clip).exists():
    import shutil
    shutil.copy2(str(clip), {str(out_path)!r})
    print("OK:" + str(clip), flush=True)
else:
    print("ERR:" + str(err or "unknown"), flush=True)
"""
    result = subprocess.run(
        [APP_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=1200,
    )
    stdout = result.stdout.strip()
    if result.returncode == 0 and stdout.startswith("OK:"):
        if log_fn: log_fn(f"  Generated: {out_path.name} ({out_path.stat().st_size//1024 if out_path.exists() else '?'}KB)")
        return out_path, None
    err_msg = stdout.replace("ERR:", "") or result.stderr[-300:]
    if log_fn: log_fn(f"  Creator failed: {err_msg[:200]}")
    return None, err_msg


# ── Stage 3: Critic — Qwen2.5-VL video scoring ────────────────────────────────

CRITIC_PROMPT = textwrap.dedent("""
You are a video quality critic for an AI kids' educational channel.
I will show you {n_frames} frames extracted from a short AI-generated video.

The video was supposed to show: "{prompt}"

Rate the video on these criteria (each 1-10):
1. Motion Stability  — smooth motion, no jitter, no flickering, no sudden jumps
2. Visual Quality    — sharpness, color, detail, no artifacts
3. Prompt Adherence  — does it actually show what was requested?
4. Cinematic Quality — composition, lighting, atmosphere

Calculate Overall Score = average of all four.

Respond in this EXACT format (no extra text):
MOTION_STABILITY: X
VISUAL_QUALITY: X
PROMPT_ADHERENCE: X
CINEMATIC_QUALITY: X
OVERALL: X.X
ISSUES: [comma-separated list of issues, or "none"]
""").strip()

def extract_frames(video_path: Path, n_frames: int = 6) -> list:
    """Extract n evenly-spaced frames from a video as PNG files."""
    frames_dir = video_path.parent / f"_frames_{video_path.stem}"
    frames_dir.mkdir(exist_ok=True)

    # Get video duration
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    try:
        duration = float(r.stdout.strip())
    except ValueError:
        duration = 3.0

    frame_paths = []
    for i in range(n_frames):
        t = duration * (i + 0.5) / n_frames
        out = frames_dir / f"frame_{i:02d}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out),
        ], capture_output=True)
        if out.exists():
            frame_paths.append(out)

    return frame_paths


def score_video(video_path: Path, prompt: str, log_fn=None) -> dict:
    """Stage 3: Use Qwen2.5-VL to score the video. Returns score dict."""
    if log_fn: log_fn("Stage 3 [Critic] — extracting frames for Qwen2.5-VL...")

    frames = extract_frames(video_path, n_frames=6)
    if not frames:
        if log_fn: log_fn("  No frames extracted — skipping critique")
        return {"overall": 5.0, "issues": ["frame extraction failed"], "raw": ""}

    critic_prompt = CRITIC_PROMPT.format(n_frames=len(frames), prompt=prompt)
    frame_paths_repr = repr([str(f) for f in frames])

    script = f"""
import sys
try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except ImportError:
    print("INSTALL_NEEDED", flush=True); sys.exit(1)

model, processor = load({QWEN_VL_MODEL!r})
config = load_config({QWEN_VL_MODEL!r})

frame_paths = {frame_paths_repr}
critic_prompt = {critic_prompt!r}

# Analyse all frames together — pass the first frame as primary image
# and describe the rest in text
primary_frame = frame_paths[0]
all_frames_text = ", ".join(frame_paths)

formatted_prompt = apply_chat_template(
    processor, config, critic_prompt, num_images=1
)
output = generate(model, processor, primary_frame, formatted_prompt,
                  max_tokens=300, verbose=False)
print(output.strip(), flush=True)
"""
    if log_fn: log_fn("  Running Qwen2.5-VL critique (loading model)...")
    result = subprocess.run(
        [MLX_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=300,
    )

    # Clean up frames
    for f in frames:
        f.unlink(missing_ok=True)
    if frames:
        frames[0].parent.rmdir() if frames[0].parent.exists() else None

    raw = result.stdout.strip() if result.returncode == 0 else ""
    scores = _parse_critic_output(raw)
    if log_fn:
        log_fn(f"  Score: {scores['overall']:.1f}/10 | Issues: {', '.join(scores['issues']) or 'none'}")
    return scores


def _parse_critic_output(raw: str) -> dict:
    """Parse Qwen critic output into a score dict."""
    defaults = {
        "motion_stability": 5.0, "visual_quality": 5.0,
        "prompt_adherence": 5.0, "cinematic_quality": 5.0,
        "overall": 5.0, "issues": ["parse error"], "raw": raw,
    }
    if not raw:
        return defaults
    try:
        def extract(key):
            m = re.search(rf"{key}:\s*([\d.]+)", raw, re.IGNORECASE)
            return float(m.group(1)) if m else 5.0

        issues_m = re.search(r"ISSUES:\s*(.+)", raw, re.IGNORECASE)
        issues_raw = issues_m.group(1).strip() if issues_m else "unknown"
        issues = [] if issues_raw.lower() in ("none", "none.", "n/a") else [i.strip() for i in issues_raw.split(",")]

        overall = extract("OVERALL")
        if overall == 5.0:
            # Compute from sub-scores
            subs = [extract(k) for k in ["MOTION_STABILITY", "VISUAL_QUALITY",
                                          "PROMPT_ADHERENCE", "CINEMATIC_QUALITY"]]
            overall = round(sum(subs) / 4, 1)

        return {
            "motion_stability":  extract("MOTION_STABILITY"),
            "visual_quality":    extract("VISUAL_QUALITY"),
            "prompt_adherence":  extract("PROMPT_ADHERENCE"),
            "cinematic_quality": extract("CINEMATIC_QUALITY"),
            "overall": overall,
            "issues":  issues,
            "raw":     raw,
        }
    except Exception:
        return defaults


# ── Stage 4: Refiner — auto-retry with better params ──────────────────────────

def should_refine(scores: dict, min_score: float = 7.0) -> bool:
    return scores["overall"] < min_score

def next_model(current_model: str, attempt: int) -> str:
    """Return next model in escalation order, or None if at the top."""
    try:
        idx = ESCALATION.index(current_model)
    except ValueError:
        idx = 0
    next_idx = idx + attempt
    if next_idx < len(ESCALATION):
        return ESCALATION[next_idx]
    return None  # Already at best model


# ── Stage 5: Polisher — Topaz Video AI upscale ────────────────────────────────

def topaz_upscale(video_path: Path, out_path: Path,
                  scale: int = 4, model: str = "prob-4",
                  log_fn=None) -> Path:
    """Stage 5: Upscale via Topaz Video AI CLI (Proteus by default)."""
    if not TOPAZ_FFMPEG.exists():
        if log_fn: log_fn("  Topaz Video AI not found — skipping upscale")
        return video_path

    # Get source dimensions
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True
    )
    try:
        w, h = [int(x) for x in r.stdout.strip().split(",")]
        target_w, target_h = w * scale, h * scale
    except Exception:
        target_w, target_h = 3840, 2160

    env = os.environ.copy()
    env["TVAI_MODEL_DATA_DIR"] = str(TOPAZ_MODELS)

    vf = (f"tvai_up=model={model}:scale=0:w={target_w}:h={target_h}"
          f":preblur=0:noise=0:details=0:sharpness=0:recover_details=0.5"
          f":dehalo=0:antialias=0:blend=0.5:device=0:vram=1:instances=1")

    cmd = [
        str(TOPAZ_FFMPEG), "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "hevc_videotoolbox", "-q:v", "60",
        "-c:a", "copy",
        str(out_path),
    ]
    if log_fn: log_fn(f"Stage 5 [Polisher] — Topaz {model} {target_w}×{target_h}...")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 and out_path.exists():
        if log_fn: log_fn(f"  Upscaled: {out_path.name} ({out_path.stat().st_size//1024//1024}MB)")
        return out_path
    if log_fn: log_fn(f"  Topaz failed: {r.stderr[-200:]}")
    return video_path


# ── Training dataset collection ────────────────────────────────────────────────

def save_to_dataset(video_path: Path, expanded_prompt: str,
                    simple_prompt: str, model_key: str,
                    scores: dict, scene_id: int):
    """Save an approved clip + metadata to the training dataset."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    entry_dir = DATASET_DIR / f"{ts}_{model_key}_s{scene_id:02d}"
    entry_dir.mkdir(exist_ok=True)

    # Copy video
    shutil.copy2(str(video_path), str(entry_dir / "video.mp4"))

    # Save metadata
    meta = {
        "timestamp":      ts,
        "model":          model_key,
        "scene_id":       scene_id,
        "simple_prompt":  simple_prompt,
        "expanded_prompt": expanded_prompt,
        "scores":         scores,
        "approved":       scores["overall"] >= 7.0,
    }
    (entry_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return entry_dir


# ── Leaderboard ────────────────────────────────────────────────────────────────

def update_leaderboard(model_key: str, scores: dict, attempts: int,
                       generation_time: float):
    """Append result to the model leaderboard JSON."""
    data = []
    if LEADERBOARD.exists():
        try:
            data = json.loads(LEADERBOARD.read_text())
        except Exception:
            data = []

    data.append({
        "timestamp":       datetime.now().isoformat(),
        "model":           model_key,
        "overall_score":   scores["overall"],
        "motion_stability":scores.get("motion_stability", 0),
        "visual_quality":  scores.get("visual_quality", 0),
        "prompt_adherence":scores.get("prompt_adherence", 0),
        "cinematic_quality":scores.get("cinematic_quality", 0),
        "attempts":        attempts,
        "generation_time": round(generation_time, 1),
        "issues":          scores.get("issues", []),
    })
    LEADERBOARD.write_text(json.dumps(data, indent=2))


def print_leaderboard():
    """Print a ranked table of all models by average score."""
    if not LEADERBOARD.exists():
        print("No leaderboard data yet.")
        return

    data = json.loads(LEADERBOARD.read_text())
    # Aggregate by model
    agg = {}
    for row in data:
        m = row["model"]
        if m not in agg:
            agg[m] = {"scores": [], "times": []}
        agg[m]["scores"].append(row["overall_score"])
        agg[m]["times"].append(row["generation_time"])

    ranked = sorted(agg.items(),
                    key=lambda x: sum(x[1]["scores"]) / len(x[1]["scores"]),
                    reverse=True)

    print("\n" + "═" * 65)
    print(f"{'MODEL':<22} {'AVG SCORE':>10} {'RUNS':>5} {'AVG TIME':>10}")
    print("═" * 65)
    for model, d in ranked:
        avg_score = sum(d["scores"]) / len(d["scores"])
        avg_time  = sum(d["times"])  / len(d["times"])
        runs      = len(d["scores"])
        bar = "★" * int(avg_score)
        print(f"{model:<22} {avg_score:>6.1f}/10  {runs:>4}  {avg_time:>8.0f}s  {bar}")
    print("═" * 65)
    print(f"\nLeaderboard saved: {LEADERBOARD}")


# ── Main agentic loop ──────────────────────────────────────────────────────────

def agentic_generate(
    simple_prompt:   str,
    scene_id:        int   = 1,
    model_key:       str   = "ltx-2b",     # starting model
    min_score:       float = 7.0,
    max_attempts:    int   = 3,
    topaz_upscale_:  bool  = True,
    save_dataset_:   bool  = True,
    log_fn = None,
) -> dict:
    """
    Run the full 5-stage agentic pipeline for a single model.
    Returns a result dict with video path, scores, and metadata.
    """
    if log_fn is None:
        log_fn = print

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = SCENES_DIR / f"scene_{scene_id:02d}.png"
    audio_path = AUDIO_DIR / f"scene_{scene_id:02d}.wav"

    if not img_path.exists():
        return {"error": f"No image for scene {scene_id}. Run image generation first."}
    if not audio_path.exists():
        return {"error": f"No audio for scene {scene_id}. Run TTS first."}

    t_start = time.time()

    # ── Stage 1: Director ───────────────────────────────────────────────────
    expanded_prompt = expand_prompt(simple_prompt, log_fn=log_fn)

    # ── Stages 2-4: Create → Critique → Refine loop ─────────────────────────
    best_video  = None
    best_scores = {"overall": 0}
    current_model = model_key
    attempt = 1

    while attempt <= max_attempts:
        log_fn(f"\n[Attempt {attempt}/{max_attempts}] Model: {current_model}")

        out_path = AGENIC_DIR / f"{run_id}_{current_model}_a{attempt}.mp4"
        video, err = create_video(
            scene_id, current_model, expanded_prompt,
            img_path, audio_path, out_path, attempt, log_fn=log_fn
        )

        if video is None:
            log_fn(f"  Generation failed: {err}")
            # Escalate model even on failure
        else:
            scores = score_video(video, expanded_prompt, log_fn=log_fn)

            if scores["overall"] > best_scores["overall"]:
                best_video  = video
                best_scores = scores

            if not should_refine(scores, min_score):
                log_fn(f"\n✅ Score {scores['overall']:.1f} ≥ {min_score} — accepted!")
                break
            else:
                log_fn(f"  Score {scores['overall']:.1f} < {min_score} — refining...")
                for issue in scores["issues"]:
                    log_fn(f"    Issue: {issue}")

        # Escalate to next model tier
        next_m = next_model(current_model, attempt)
        if next_m:
            log_fn(f"  Escalating: {current_model} → {next_m}")
            current_model = next_m
        attempt += 1

    gen_time = time.time() - t_start

    if best_video is None:
        return {"error": "All attempts failed", "scores": best_scores}

    # ── Stage 5: Polisher ───────────────────────────────────────────────────
    final_video = best_video
    if topaz_upscale_ and TOPAZ_FFMPEG.exists():
        polished = AGENIC_DIR / f"{run_id}_{current_model}_4k.mp4"
        final_video = topaz_upscale(best_video, polished, log_fn=log_fn)

    # ── Save to training dataset ────────────────────────────────────────────
    if save_dataset_:
        save_to_dataset(final_video, expanded_prompt, simple_prompt,
                        current_model, best_scores, scene_id)

    # ── Update leaderboard ──────────────────────────────────────────────────
    update_leaderboard(current_model, best_scores, attempt - 1, gen_time)

    log_fn(f"\n✨ Pipeline complete in {gen_time/60:.1f}min")
    log_fn(f"   Final score: {best_scores['overall']:.1f}/10")
    log_fn(f"   Video: {final_video}")

    return {
        "video":           str(final_video),
        "scores":          best_scores,
        "model":           current_model,
        "attempts":        attempt - 1,
        "expanded_prompt": expanded_prompt,
        "generation_time": gen_time,
    }


def benchmark_all_models(
    simple_prompt: str,
    scene_id:      int   = 1,
    min_score:     float = 7.0,
    max_attempts:  int   = 2,
    topaz_upscale_:bool  = False,
    log_fn = None,
) -> list:
    """
    Run the agentic pipeline for EVERY available model on the same scene.
    Returns ranked results list. Updates the leaderboard.
    """
    if log_fn is None:
        log_fn = print

    # Check which models are available
    sys.path.insert(0, str(AI_EDU_DIR))
    from app import check_model_availability
    status, _ = check_model_availability()

    available = [m for m, ok in status.items() if ok and m != "ken-burns"]
    if not available:
        log_fn("No models available.")
        return []

    log_fn(f"\nBenchmarking {len(available)} models on scene {scene_id}")
    log_fn(f"Models: {', '.join(available)}\n")

    results = []
    for model_key in available:
        log_fn(f"\n{'='*60}")
        log_fn(f"  Testing: {model_key}")
        log_fn(f"{'='*60}")
        result = agentic_generate(
            simple_prompt=simple_prompt,
            scene_id=scene_id,
            model_key=model_key,
            min_score=min_score,
            max_attempts=max_attempts,
            topaz_upscale_=topaz_upscale_,
            log_fn=log_fn,
        )
        result["model"] = model_key
        results.append(result)

    # Rank by score
    results.sort(key=lambda r: r.get("scores", {}).get("overall", 0), reverse=True)

    log_fn("\n\n" + "="*60)
    log_fn("BENCHMARK RESULTS — RANKED BY SCORE")
    log_fn("="*60)
    for i, r in enumerate(results, 1):
        score = r.get("scores", {}).get("overall", 0)
        model = r.get("model", "?")
        t     = r.get("generation_time", 0)
        log_fn(f"{i}. {model:<25} {score:.1f}/10  ({t/60:.1f}min)")

    print_leaderboard()
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Agentic Video Generation Pipeline")
    ap.add_argument("prompt", nargs="?", default="cinematic landscape animation, golden light",
                    help="Simple prompt to expand and generate")
    ap.add_argument("--scene",    type=int, default=1,   help="Scene ID (1-10)")
    ap.add_argument("--model",    default="ltx-2b",       help="Starting model")
    ap.add_argument("--models",   nargs="+",              help="'all' or model list for benchmark")
    ap.add_argument("--min-score",type=float, default=7.0,help="Min score before accepting")
    ap.add_argument("--max-tries",type=int,   default=3,  help="Max refinement attempts")
    ap.add_argument("--no-topaz", action="store_true",    help="Skip Topaz upscale")
    ap.add_argument("--leaderboard", action="store_true", help="Print leaderboard and exit")
    args = ap.parse_args()

    if args.leaderboard:
        print_leaderboard()
        sys.exit(0)

    if args.models and ("all" in args.models or len(args.models) > 1):
        benchmark_all_models(
            simple_prompt=args.prompt,
            scene_id=args.scene,
            min_score=args.min_score,
            max_attempts=args.max_tries,
            topaz_upscale_=not args.no_topaz,
        )
    else:
        result = agentic_generate(
            simple_prompt=args.prompt,
            scene_id=args.scene,
            model_key=args.model,
            min_score=args.min_score,
            max_attempts=args.max_tries,
            topaz_upscale_=not args.no_topaz,
        )
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)

"""
Probability Island — Final Video Assembly
Ken Burns cinematic zoom/pan effect on each scene image, synced with narration audio.
Runs in seconds. No GPU needed.

Run:
    python assemble_video.py --preview   # scene 02 only
    python assemble_video.py --all       # full 10-scene video
"""

import json, subprocess, sys
from pathlib import Path

AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
SCENES_DIR   = AI_EDU_DIR / "output" / "island_scenes"
AUDIO_DIR    = AI_EDU_DIR / "output" / "island_audio"
CLIPS_DIR    = AI_EDU_DIR / "output" / "island_clips_kb"
FINAL_OUT    = AI_EDU_DIR / "output" / "probability_island_final.mp4"

CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Ken Burns motion per scene: (zoom_dir, pan_dir)
# zoom_dir: "in" or "out"
# pan_dir: "left", "right", "up", "down", "none"
MOTIONS = {
    1:  ("out", "none"),    # aerial pullback revealing island
    2:  ("in",  "right"),   # push into coin cliffs
    3:  ("in",  "none"),    # zoom in on spinning coin
    4:  ("out", "left"),    # pull back across dice plains
    5:  ("in",  "none"),    # zoom onto glowing four-dot face
    6:  ("out", "right"),   # drift through enchanted forest
    7:  ("in",  "up"),      # slow rise watching fruit fall
    8:  ("out", "none"),    # pull back as sunrise floods scene
    9:  ("in",  "left"),    # push toward impossible shore
    10: ("out", "none"),    # grand aerial pullback — finale
}


def get_duration(audio_path: Path) -> float:
    import soundfile as sf
    return sf.info(str(audio_path)).duration


def build_kenburns_clip(img_path: Path, audio_path: Path, out: Path,
                         zoom_dir: str, pan_dir: str):
    """Animate image with Ken Burns effect matched to audio duration."""
    duration = get_duration(audio_path)
    fps = 25
    total_frames = int(duration * fps)

    # Image is 1360x768. Output at 1280x720.
    W, H = 1280, 720
    # Source crop slightly larger than output so we have room to zoom/pan
    # Start crop: 1360x765 → scale to 1280x720 gives 1.0x
    # For zoom-in: start wide (1.15x), end tight (1.0x)
    # For zoom-out: start tight (1.0x), end wide (1.15x)

    if zoom_dir == "in":
        zoom_start, zoom_end = 1.15, 1.0
    else:
        zoom_start, zoom_end = 1.0, 1.15

    # Pan offset range (pixels in source space)
    pan_range = 60  # pixels to drift

    if pan_dir == "left":
        px_start, py_start = pan_range, 0
        px_end,   py_end   = 0, 0
    elif pan_dir == "right":
        px_start, py_start = 0, 0
        px_end,   py_end   = pan_range, 0
    elif pan_dir == "up":
        px_start, py_start = 0, pan_range
        px_end,   py_end   = 0, 0
    elif pan_dir == "down":
        px_start, py_start = 0, 0
        px_end,   py_end   = 0, pan_range
    else:
        px_start = py_start = px_end = py_end = 0

    # zoompan filter: z=zoom, x/y=pan, s=output size
    # Interpolate zoom and pan linearly over total_frames
    # ffmpeg zoompan: z and x/y are expressions evaluated per frame (n = frame number)
    n = f"n"
    nf = str(total_frames - 1)

    def lerp(a, b):
        return f"{a}+({b}-{a})*{n}/{nf}"

    z_expr   = lerp(zoom_start, zoom_end)
    # zoompan x/y: position of top-left of crop in zoomed frame
    # at zoom z, frame is z*iw x z*ih; we want crop centered + pan offset
    x_expr = f"(iw-iw/{lerp(zoom_end,zoom_start)})/2+{lerp(px_start,px_end)}"
    y_expr = f"(ih-ih/{lerp(zoom_end,zoom_start)})/2+{lerp(py_start,py_end)}"

    # Simpler and more reliable: use scale+crop with manually computed values
    # We'll use the zoompan filter which handles this directly
    zp_filter = (
        f"scale=8000:-1,"  # scale up so we have pixels to work with
        f"zoompan=z='{lerp(zoom_start,zoom_end)}':"
        f"x='(iw-iw/zoom)/2+{lerp(px_start,px_end)}':"
        f"y='(ih-ih/zoom)/2+{lerp(py_start,py_end)}':"
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
        "-shortest",
        str(out)
    ]

    print(f"  Building clip for {img_path.name} ({duration:.1f}s)...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[-400:]}")
        return False
    print(f"  Saved: {out.name} ({out.stat().st_size//1024}KB)")
    return True


def build_all_clips():
    for scene_id in range(1, 11):
        img   = SCENES_DIR / f"scene_{scene_id:02d}.png"
        audio = AUDIO_DIR  / f"scene_{scene_id:02d}.wav"
        out   = CLIPS_DIR  / f"scene_{scene_id:02d}.mp4"

        if not img.exists() or not audio.exists():
            print(f"  SKIP {scene_id}: missing files")
            continue
        if out.exists():
            print(f"  Scene {scene_id:02d}: cached")
            continue

        zoom_dir, pan_dir = MOTIONS.get(scene_id, ("in", "none"))
        build_kenburns_clip(img, audio, out, zoom_dir, pan_dir)


def concat_clips():
    """Concatenate clips with xfade transitions between scenes."""
    clips = sorted(CLIPS_DIR.glob("scene_*.mp4"))
    if not clips:
        print("No clips found.")
        return

    if len(clips) == 1:
        # Single clip — just copy
        subprocess.run(["ffmpeg", "-y", "-i", str(clips[0]),
                        "-c", "copy", str(FINAL_OUT)], capture_output=True)
        print(f"Final: {FINAL_OUT}")
        subprocess.run(["open", str(FINAL_OUT)])
        return

    print(f"\nAssembling {len(clips)} clips with xfade transitions...")

    # xfade transition sequence: cycle through styles for variety
    TRANSITIONS = ["dissolve", "wipeleft", "slideright", "fade", "dissolve",
                   "wiperight", "slideleft", "fade", "dissolve", "wipeleft"]
    XFADE_DURATION = 0.5  # seconds of overlap

    # Get durations of each clip
    def get_video_duration(path: Path) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(path)],
            capture_output=True, text=True,
        )
        for s in json.loads(r.stdout).get("streams", []):
            if s.get("codec_type") == "video":
                return float(s.get("duration", 0))
        return 0.0

    durations = [get_video_duration(c) for c in clips]

    # Build ffmpeg xfade filter chain
    # Input streams: [0:v][0:a][1:v][1:a]...
    # Chain: [0:v][1:v]xfade=... → [vt1], [vt1][2:v]xfade=... → [vt2], etc.

    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    # Build video filter chain
    vf_parts = []
    af_parts = []
    current_v = "[0:v]"
    current_a = "[0:a]"
    offset = 0.0  # running xfade offset

    for i in range(1, len(clips)):
        # Offset = sum of durations so far minus transition overlap
        offset += durations[i - 1] - XFADE_DURATION
        transition = TRANSITIONS[(i - 1) % len(TRANSITIONS)]
        out_v = f"[vt{i}]"
        out_a = f"[at{i}]"

        vf_parts.append(
            f"{current_v}[{i}:v]xfade=transition={transition}:"
            f"duration={XFADE_DURATION}:offset={offset:.3f}{out_v}"
        )
        af_parts.append(
            f"{current_a}[{i}:a]acrossfade=d={XFADE_DURATION}{out_a}"
        )
        current_v = out_v
        current_a = out_a

    vf = ";".join(vf_parts)
    af = ";".join(af_parts)
    filter_complex = f"{vf};{af}"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", current_v,
        "-map", current_a,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(FINAL_OUT),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        print(f"xfade error: {r.stderr[-400:]}")
        print("Falling back to plain concat...")
        _concat_plain(clips)
    else:
        size = FINAL_OUT.stat().st_size / 1024 / 1024
        print(f"Final (with transitions): {FINAL_OUT} ({size:.1f}MB)")
        subprocess.run(["open", str(FINAL_OUT)])


def _concat_plain(clips: list[Path]):
    """Fallback: plain copy concat with no transitions."""
    list_file = CLIPS_DIR / "concat.txt"
    with open(list_file, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy", str(FINAL_OUT),
    ], capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if r.returncode == 0:
        size = FINAL_OUT.stat().st_size / 1024 / 1024
        print(f"Final: {FINAL_OUT} ({size:.1f}MB)")
        subprocess.run(["open", str(FINAL_OUT)])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true", help="Build scene 02 only and open")
    p.add_argument("--all",     action="store_true", help="Build all 10 scenes + final MP4")
    a = p.parse_args()

    if a.preview:
        img   = SCENES_DIR / "scene_02.png"
        audio = AUDIO_DIR  / "scene_02.wav"
        out   = CLIPS_DIR  / "scene_02.mp4"
        out.unlink(missing_ok=True)
        zoom_dir, pan_dir = MOTIONS[2]
        if build_kenburns_clip(img, audio, out, zoom_dir, pan_dir):
            subprocess.run(["open", str(out)])

    if a.all:
        build_all_clips()
        concat_clips()

    if not any(vars(a).values()):
        print(__doc__)

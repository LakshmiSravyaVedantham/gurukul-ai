"""
subtitles.py — Free WhisperX-style word-level subtitles for Gurukul videos.

Uses mlx-whisper (Apple Silicon native) for word-level transcription,
generates styled ASS subtitle file with current-word highlight,
then burns into video via ffmpeg.

Usage:
    python subtitles.py video.mp4                     # burn subtitles in-place
    python subtitles.py video.mp4 --out video_sub.mp4  # new file
    python subtitles.py video.mp4 --ass-only           # just write the .ass file
    python subtitles.py --srt video.mp4               # generate .srt instead
"""

import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path

MLX_PYTHON = "/Volumes/bujji1/sravya/ai_vidgen/venv/bin/python"

# Whisper model — "small" is fast and accurate enough for clean narration
WHISPER_MODEL = "mlx-community/whisper-small-mlx"

# ── ASS Style ─────────────────────────────────────────────────────────────────
# Bold yellow text at bottom center, shadow for readability on any background

ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Rounded MT Bold,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,40,40,60,1
Style: Highlight,Arial Rounded MT Bold,52,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,40,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    """Convert float seconds to ASS timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _srt_ts(seconds: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(video_path: Path) -> list[dict]:
    """
    Run mlx-whisper on the video and return a list of word-level segments:
      [{"word": "Hello", "start": 0.0, "end": 0.4}, ...]
    Falls back to segment-level if word timestamps unavailable.
    """
    script = f"""
import sys, json
try:
    import mlx_whisper
except ImportError:
    print(json.dumps({{"error": "mlx_whisper not installed"}}))
    sys.exit(0)

result = mlx_whisper.transcribe(
    {str(video_path)!r},
    path_or_hf_repo={WHISPER_MODEL!r},
    word_timestamps=True,
    language="en",
    verbose=False,
)

words = []
for seg in result.get("segments", []):
    if "words" in seg and seg["words"]:
        for w in seg["words"]:
            words.append({{
                "word": w.get("word", "").strip(),
                "start": float(w.get("start", seg["start"])),
                "end": float(w.get("end", seg["end"])),
            }})
    else:
        # Segment-level fallback
        words.append({{
            "word": seg.get("text", "").strip(),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "segment": True,
        }})

print(json.dumps(words))
"""
    result = subprocess.run(
        [MLX_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"Whisper error:\n{result.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError("Transcription failed")

    data = json.loads(result.stdout.strip())
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    return data


# ── Grouping into lines ───────────────────────────────────────────────────────

def _group_into_lines(words: list[dict], max_words: int = 6) -> list[dict]:
    """
    Group individual words into subtitle lines (max_words per line).
    Returns list of: {"text": str, "start": float, "end": float, "words": list}
    """
    if not words:
        return []

    lines = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        line_text = " ".join(w["word"] for w in chunk if w["word"])
        lines.append({
            "text": line_text,
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "words": chunk,
        })
        i += max_words
    return lines


# ── ASS generation ────────────────────────────────────────────────────────────

def _escape_ass(text: str) -> str:
    """Escape special ASS characters."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_ass(words: list[dict]) -> str:
    """
    Build an ASS subtitle string with karaoke-style word highlighting.
    Each line shows all words; the current word is yellow, rest are white.
    """
    lines = _group_into_lines(words)
    events = []

    for line in lines:
        line_start = line["start"]
        line_end = line["end"]
        line_words = line["words"]

        # For each word, emit a Dialogue event covering that word's duration
        # showing the full line with that word highlighted yellow
        for wi, word in enumerate(line_words):
            w_start = word["start"]
            w_end = word["end"]

            # Build the styled text: white words, yellow for current word
            parts = []
            for j, w in enumerate(line_words):
                escaped = _escape_ass(w["word"])
                if j == wi:
                    parts.append(f"{{\\c&H00FFFF&}}{escaped}{{\\c&HFFFFFF&}}")
                else:
                    parts.append(escaped)
            styled = " ".join(parts)

            events.append(
                f"Dialogue: 0,{_ts(w_start)},{_ts(w_end)},Default,,0,0,0,,{styled}"
            )

    return ASS_HEADER + "\n".join(events) + "\n"


def build_srt(words: list[dict]) -> str:
    """Build a simple SRT file from word groups."""
    lines = _group_into_lines(words, max_words=8)
    parts = []
    for i, line in enumerate(lines, 1):
        parts.append(
            f"{i}\n"
            f"{_srt_ts(line['start'])} --> {_srt_ts(line['end'])}\n"
            f"{line['text']}\n"
        )
    return "\n".join(parts)


# ── Burn into video ───────────────────────────────────────────────────────────

def burn_subtitles(video_path: Path, ass_path: Path, out_path: Path):
    """Use ffmpeg to burn ASS subtitles into video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        str(out_path),
    ]
    print(f"Burning subtitles → {out_path.name}...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ffmpeg error:\n{r.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError("Subtitle burn failed")
    size = out_path.stat().st_size / 1024 / 1024
    print(f"Done: {out_path} ({size:.1f} MB)")


# ── Public API ────────────────────────────────────────────────────────────────

def add_subtitles(
    video_path: str | Path,
    out_path: str | Path | None = None,
    ass_only: bool = False,
    srt: bool = False,
) -> Path:
    """
    Main entry point. Transcribes video and burns in word-level subtitles.

    Args:
        video_path: Input video file.
        out_path:   Output video path. If None, overwrites input (via temp file).
        ass_only:   If True, only write the .ass file, don't burn video.
        srt:        If True, write .srt instead of .ass.

    Returns:
        Path to output video (or .ass/.srt if ass_only/srt).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print(f"Transcribing {video_path.name} with mlx-whisper ({WHISPER_MODEL})...")
    words = transcribe(video_path)
    print(f"  Got {len(words)} word timestamps")

    if srt:
        srt_path = video_path.with_suffix(".srt")
        srt_path.write_text(build_srt(words))
        print(f"SRT: {srt_path}")
        return srt_path

    ass_path = video_path.with_suffix(".ass")
    ass_path.write_text(build_ass(words))
    print(f"ASS: {ass_path}")

    if ass_only:
        return ass_path

    # Burn in
    if out_path is None:
        # Overwrite: write to temp then replace
        tmp = video_path.with_suffix(".tmp.mp4")
        burn_subtitles(video_path, ass_path, tmp)
        tmp.replace(video_path)
        return video_path
    else:
        out_path = Path(out_path)
        burn_subtitles(video_path, ass_path, out_path)
        return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Burn word-level subtitles into video (free, Apple Silicon)")
    ap.add_argument("video", help="Input video file")
    ap.add_argument("--out", default=None, help="Output path (default: overwrite input)")
    ap.add_argument("--ass-only", action="store_true", help="Only generate .ass file, skip burn")
    ap.add_argument("--srt", action="store_true", help="Generate .srt file instead of .ass")
    ap.add_argument("--model", default=None, help=f"Whisper model (default: {WHISPER_MODEL})")
    args = ap.parse_args()

    if args.model:
        WHISPER_MODEL = args.model

    add_subtitles(
        video_path=args.video,
        out_path=args.out,
        ass_only=args.ass_only,
        srt=args.srt,
    )

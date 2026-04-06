"""
Generate a new Gurukul Island episode for any math/science topic using Gemma.

Uses Gemma-3 4B (4-bit quantized, ~2.5 GB) via mlx-lm running fully locally on
Apple Silicon. No internet needed after first model download.

Usage:
    python generate_topic.py "fractions"
    python generate_topic.py "geometry" --out gurukul_geometry.py
    python generate_topic.py "multiplication" --preview   # print scenes, don't write

After generating:
    python gurukul_fractions.py --scenes   # generate images with FLUX
    python gurukul_fractions.py --tts      # generate narration
    python wan_animate.py --full           # animate + assemble
"""

import argparse, json, re, shutil, sys, textwrap
from pathlib import Path

MLX_PYTHON   = "/Volumes/bujji1/sravya/ai_vidgen/venv/bin/python"
GEMMA_MODEL  = "mlx-community/gemma-3-4b-it-4bit"
TEMPLATE     = Path(__file__).parent / "gurukul_island.py"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM = textwrap.dedent("""
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

PROMPT_TEMPLATE = textwrap.dedent("""
Create a 10-scene Gurukul Island episode teaching "{topic}" to kids aged 6-10.

Rules:
1. Each scene is a pure LANDSCAPE — the environment itself visualises the concept.
   No human characters, no text in images, no narrators shown.
2. Scene 1: introduce the island / world for this topic.
3. Scenes 2-9: each scene introduces one key concept about {topic}, building up.
4. Scene 10: triumphant aerial wide shot showing the whole island — everything learned.
5. Narration is warm, simple, wonder-filled. 2-4 short sentences per scene.
   Age 6-10 level. No jargon without explanation.
6. Image descriptions must be vivid and specific — mention exact colours, shapes,
   lighting, scale. The AI image model has no context — every detail must be spelled out.

Return a JSON object with this exact structure:
{{
  "topic": "{topic}",
  "island_name": "...",
  "scene_defs": [
    [1, "image description for scene 1"],
    [2, "image description for scene 2"],
    ...
    [10, "image description for scene 10"]
  ],
  "scenes": [
    {{"id": 1, "narration": "narration text for scene 1"}},
    {{"id": 2, "narration": "narration text for scene 2"}},
    ...
    {{"id": 10, "narration": "narration text for scene 10"}}
  ]
}}
""").strip()


def _run_gemma(topic: str) -> str:
    """Run Gemma via mlx-lm and return the raw output string."""
    user_prompt = PROMPT_TEMPLATE.format(topic=topic)

    # Build the chat messages as JSON for mlx-lm
    messages = json.dumps([
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_prompt},
    ])

    script = f"""
import sys
try:
    from mlx_lm import load, generate
except ImportError:
    print("INSTALL_NEEDED", flush=True)
    sys.exit(1)

print("Loading Gemma model...", file=sys.stderr, flush=True)
model, tokenizer = load({GEMMA_MODEL!r})

messages = {messages}
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("Generating scenes...", file=sys.stderr, flush=True)
response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=4096,
    verbose=False,
)
print(response, flush=True)
"""

    import subprocess
    result = subprocess.run(
        [MLX_PYTHON, "-c", script],
        capture_output=True, text=True, timeout=300,
    )

    if "INSTALL_NEEDED" in result.stdout:
        print("Installing mlx-lm...")
        subprocess.run([MLX_PYTHON, "-m", "pip", "install", "mlx-lm", "-q"], check=True)
        # Retry
        result = subprocess.run(
            [MLX_PYTHON, "-c", script],
            capture_output=True, text=True, timeout=300,
        )

    if result.returncode != 0:
        print("Gemma stderr:", result.stderr[-500:], file=sys.stderr)
        raise RuntimeError(f"Gemma failed (exit {result.returncode})")

    return result.stdout.strip()


def _extract_json(raw: str) -> dict:
    """Extract the JSON object from Gemma output (may have markdown fences)."""
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

    # Find outermost { ... }
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in output:\n{raw[:500]}")

    return json.loads(raw[start:end])


def _build_script(data: dict, topic: str) -> str:
    """Produce a complete gurukul_<topic>.py by patching the template."""
    template = TEMPLATE.read_text()

    # ── Replace SCENE_DEFS ──────────────────────────────────────────────────
    scene_defs_lines = ["SCENE_DEFS = ["]
    for scene_id, desc in data["scene_defs"]:
        # Wrap long descriptions for readability
        escaped = desc.replace('"', '\\"')
        scene_defs_lines.append(f"    ({scene_id},")
        scene_defs_lines.append(f'     "{escaped}"),')
        scene_defs_lines.append("")
    scene_defs_lines.append("]")
    new_scene_defs = "\n".join(scene_defs_lines)

    # ── Replace SCENES ───────────────────────────────────────────────────────
    scenes_lines = ["SCENES = ["]
    for s in data["scenes"]:
        narration = s["narration"].replace('"', '\\"')
        scenes_lines.append(f'    {{')
        scenes_lines.append(f'        "id": {s["id"]},')
        scenes_lines.append(f'        "narration": (')
        scenes_lines.append(f'            "{narration}"')
        scenes_lines.append(f'        )')
        scenes_lines.append(f'    }},')
    scenes_lines.append("]")
    new_scenes = "\n".join(scenes_lines)

    # Find and replace the SCENE_DEFS block
    scene_defs_pat = re.compile(
        r"^SCENE_DEFS\s*=\s*\[.*?^\]",
        re.MULTILINE | re.DOTALL,
    )
    scenes_pat = re.compile(
        r"^SCENES\s*=\s*\[.*?^\]",
        re.MULTILINE | re.DOTALL,
    )

    result = scene_defs_pat.sub(new_scene_defs, template, count=1)
    result = scenes_pat.sub(new_scenes, result, count=1)

    # Update the docstring / header comment with the new topic
    island_name = data.get("island_name", f"{topic.title()} Island")
    result = result.replace("Gurukul Probability", f"Gurukul {topic.title()}", 1)
    result = result.replace("Probability Island", island_name, 1)
    result = result.replace("probability_island.mp4", f"{topic.lower().replace(' ', '_')}_island.mp4", 1)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", help='e.g. "fractions", "geometry", "multiplication"')
    ap.add_argument("--out", default=None, help="Output .py file (default: gurukul_<topic>.py)")
    ap.add_argument("--preview", action="store_true", help="Print scenes without writing file")
    args = ap.parse_args()

    topic = args.topic.strip()
    out_path = Path(args.out) if args.out else Path(__file__).parent / f"gurukul_{topic.lower().replace(' ', '_')}.py"

    print(f"Generating '{topic}' episode with Gemma ({GEMMA_MODEL})...")
    print("First run downloads ~2.5 GB model — subsequent runs are instant.\n")

    raw = _run_gemma(topic)

    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Failed to parse Gemma output: {e}", file=sys.stderr)
        print("Raw output:", raw[:1000], file=sys.stderr)
        sys.exit(1)

    # ── Preview ──────────────────────────────────────────────────────────────
    island_name = data.get("island_name", f"{topic.title()} Island")
    print(f"Island: {island_name}\n")
    for (sid, desc), scene in zip(data["scene_defs"], data["scenes"]):
        print(f"Scene {sid}: {desc[:80]}...")
        print(f"         \"{scene['narration'][:80]}...\"\n")

    if args.preview:
        print("(--preview: file not written)")
        return

    # ── Write script ─────────────────────────────────────────────────────────
    script_text = _build_script(data, topic)
    out_path.write_text(script_text)
    print(f"\nWritten: {out_path}")
    print(f"\nNext steps:")
    print(f"  python {out_path.name} --scenes   # generate images with FLUX (~8 min)")
    print(f"  python {out_path.name} --tts       # generate narration audio")
    print(f"  python wan_animate.py --full        # animate + assemble video")


if __name__ == "__main__":
    main()

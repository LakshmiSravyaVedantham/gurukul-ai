"""
Gurukul Probability — Probability Island style
- 10 cinematic landscape scenes, each scene IS the concept
- No human characters — zero consistency problems
- Single warm narrator voice (Kokoro am_adam)
- Style: Pixar animated movie landscape, vibrant, magical

Run:
    python gurukul_island.py --scenes      # generate 10 scenes (~8 min with schnell)
    python gurukul_island.py --tts         # generate narration audio
    python gurukul_island.py --showcase    # assemble MP4
    python gurukul_island.py --all         # everything end-to-end
"""

import subprocess, sys
import numpy as np
from pathlib import Path

AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
SCENES_DIR   = AI_EDU_DIR / "output" / "island_scenes"
AUDIO_DIR    = AI_EDU_DIR / "output" / "island_audio"
SHOWCASE_OUT = AI_EDU_DIR / "output" / "probability_island.mp4"
MFLUX        = Path("/Volumes/bujji1/sravya/ai_vidgen/venv/bin/mflux-generate")

for d in [SCENES_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Style anchor — every prompt ends with this ────────────────────────────────
STYLE = (
    "Pixar animated movie landscape, vibrant saturated colors, "
    "cinematic wide shot, golden warm light, magical atmosphere, "
    "stunning environmental storytelling, ultra detailed, "
    "beautiful Pixar 3D render, no text, no people, no characters"
)

# ── 10 scenes ─────────────────────────────────────────────────────────────────
SCENE_DEFS = [
    (1,
     "aerial view of a magical fantasy island at golden hour, "
     "one corner has towering golden coin cliffs by the ocean, "
     "another area has giant red dice scattered like boulders across plains, "
     "a lush forest has glowing red and blue fruit trees, "
     "a sparkling river winds through the whole island, "
     "warm sunset sky with soft glowing clouds"),

    (2,
     "towering golden cliffs by a sparkling ocean, "
     "giant golden coins embedded in the cliff face, "
     "one massive coin clearly shows H on its face, "
     "another massive coin beside it clearly shows T on its face, "
     "ocean waves crashing below, warm golden light, "
     "lush green trees on top of the cliffs"),

    (3,
     "one giant plain gold coin spinning and tumbling through the air "
     "above a sparkling ocean between two cliffs, "
     "coin has letter H on one face and letter T on the other face, "
     "NO bitcoin logo, NO cryptocurrency symbol, NO B symbol, plain smooth coin surface, "
     "coin catching the sunlight dramatically mid-spin, "
     "golden rays radiating from the spinning coin, "
     "sea spray below, dramatic sky, sense of suspense and wonder"),

    (4,
     "vast open plains with six enormous red dice scattered like ancient boulders, "
     "each dice clearly shows different numbers of dots on their faces, "
     "numbered one through six, warm desert sunset light, "
     "long dramatic shadows, a winding path between the dice, "
     "sense of scale and wonder"),

    (5,
     "close up of one giant red dice boulder in the plains, "
     "the face showing four dots is glowing bright gold, "
     "other five faces are in shadow, "
     "one spotlight of light hitting the four-dot face, "
     "dramatic contrast, the other five dice visible in background, "
     "sense of one being chosen out of many"),

    (6,
     "enchanted forest with magical glowing trees, "
     "three trees have bright red glowing fruit like lanterns, "
     "two trees have bright blue glowing fruit like lanterns, "
     "golden light filtering through the canopy, "
     "mystical forest floor with soft grass, "
     "fireflies in the air, warm magical atmosphere"),

    (7,
     "one magical glowing red fruit falling from a tree in the enchanted forest, "
     "the fruit glowing brightly mid-fall catching the light, "
     "other red and blue glowing fruits still on the surrounding trees, "
     "rays of golden light streaming down, "
     "magical sparkles in the air, sense of selection and chance"),

    (8,
     "breathtaking golden sunrise over the magical island, "
     "massive blazing sun rising above the horizon, "
     "golden rays flooding the entire landscape, "
     "the coin cliffs and dice plains and forest all lit up in warm gold, "
     "the most certain and guaranteed moment of the day, "
     "overwhelming warmth and brightness"),

    (9,
     "dramatic rocky shore at the edge of the island, "
     "one giant red dice boulder with a carved stone seven on its face, "
     "thick red vines growing over the seven in an X shape, "
     "mist and dark stormy clouds on this side, "
     "the impossible shore, "
     "contrast with the warm sunny island behind it"),

    (10,
     "wide view of the entire magical probability island from above at dusk, "
     "coin cliffs glowing gold, dice plains with warm light, "
     "enchanted fruit forest with red and blue glowing lights, "
     "a sparkling river winding through labeled zero at one end "
     "and one at the other in glowing stones, "
     "magical golden sunset, triumphant and beautiful"),
]

# ── Narration script ──────────────────────────────────────────────────────────
SCENES = [
    {
        "id": 1,
        "narration": (
            "Welcome to Probability Island. "
            "Every corner of this island holds a secret about chance. "
            "The golden cliffs by the ocean. The dice plains. The enchanted forest. "
            "Today, this island will teach you everything about probability. "
            "Let us explore."
        )
    },
    {
        "id": 2,
        "narration": (
            "These are the Coin Cliffs. "
            "Every rock here is a coin — with two faces. "
            "One face shows H, for Heads. The other shows T, for Tails. "
            "Just two possibilities. Nothing else can happen. "
            "When you flip a coin, you enter a world of exactly two outcomes."
        )
    },
    {
        "id": 3,
        "narration": (
            "Watch this coin spin through the air. "
            "While it is in the air — nobody knows which side will land. "
            "Heads? Or Tails? "
            "That moment of not knowing — that is exactly what probability is. "
            "But here is the key. Each side has an equal chance. One out of two. "
            "We write that as one half. Or fifty percent."
        )
    },
    {
        "id": 4,
        "narration": (
            "Welcome to the Dice Plains. "
            "These six ancient boulders are dice — each with six faces. "
            "One, two, three, four, five, six. "
            "Roll a dice and six things can happen. One for each face. "
            "Each face has an equal chance of landing face up. "
            "One out of six."
        )
    },
    {
        "id": 5,
        "narration": (
            "See how this face glows gold — the number four. "
            "Out of six possible faces, only one shows four. "
            "So the probability of rolling a four is one out of six. "
            "One favorable outcome. Six total outcomes. "
            "That is all probability ever is — "
            "count what you want, count everything possible, divide."
        )
    },
    {
        "id": 6,
        "narration": (
            "This is the Enchanted Forest. "
            "Three trees glow red. Two trees glow blue. Five trees total. "
            "If you closed your eyes and walked to any tree — "
            "which color would you reach? "
            "Red is more likely. Three out of five. "
            "Blue is less likely. Two out of five. "
            "But both are possible."
        )
    },
    {
        "id": 7,
        "narration": (
            "A red fruit falls. "
            "Was that lucky? Not really — it was probable. "
            "Three out of five trees are red. "
            "The math told us red was more likely before it even happened. "
            "That is the power of probability. "
            "It does not tell you exactly what will happen. "
            "It tells you what to expect."
        )
    },
    {
        "id": 8,
        "narration": (
            "Every morning, the sun rises over Probability Island. "
            "Will it rise tomorrow? Absolutely. Without any doubt. "
            "When something is completely certain to happen, "
            "its probability is one. "
            "One is the highest probability there is. "
            "It means — this will definitely happen."
        )
    },
    {
        "id": 9,
        "narration": (
            "But here, on the Impossible Shore — things are different. "
            "Can a dice ever show the number seven? "
            "No. It only has faces one through six. "
            "Seven is impossible. Its probability is zero. "
            "Zero means it can never happen. "
            "So all probability lives between zero and one. "
            "Zero — impossible. One — certain. Everything else — somewhere in between."
        )
    },
    {
        "id": 10,
        "narration": (
            "From above, Probability Island tells the whole story. "
            "Coin Cliffs — two outcomes, equal chance. "
            "Dice Plains — six outcomes, one in six. "
            "The Enchanted Forest — more red than blue, so red is more likely. "
            "And the river flowing from zero to one — "
            "from impossible, all the way to certain. "
            "You now understand probability. "
            "The math of chance. See you on the island."
        )
    },
]


# ── Image generation ──────────────────────────────────────────────────────────

def generate_scene(scene_id: int, description: str):
    out = SCENES_DIR / f"scene_{scene_id:02d}.png"
    if out.exists():
        print(f"  Scene {scene_id:02d}: cached")
        return out
    prompt = f"{description}, {STYLE}"
    cmd = [
        str(MFLUX), "--model", "dev",
        "--prompt", prompt,
        "--width", "1360", "--height", "768",
        "--steps", "20",
        "--seed", str(3000 + scene_id),
        "--output", str(out),
    ]
    print(f"  Generating scene {scene_id:02d}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out.exists():
        print(f"  Saved: {out.name}")
    else:
        print(f"  ERROR scene {scene_id}:\n{r.stderr[-300:]}")
    return out


def generate_all_scenes():
    print(f"Generating {len(SCENE_DEFS)} island scenes (dev model, 20 steps)...")
    print("One at a time — never in parallel on Apple Silicon.\n")
    for scene_id, description in SCENE_DEFS:
        generate_scene(scene_id, description)
    print("\nAll scenes done!")


# ── TTS narration: ElevenLabs (primary) → Kokoro (fallback) ──────────────────

# Set ELEVENLABS_API_KEY in your environment to use ElevenLabs.
# If not set or if generation fails, falls back to local Kokoro TTS automatically.
ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # "Adam" — deep warm male narrator

def _generate_elevenlabs(text: str, out_path: Path) -> bool:
    """Try ElevenLabs. Returns True on success, False on failure."""
    import os
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        return False
    try:
        from elevenlabs.client import ElevenLabs
        import soundfile as sf

        client = ElevenLabs(api_key=api_key)
        audio_gen = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",
        )
        pcm = b"".join(audio_gen)
        audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(str(out_path), audio_np, 24000)
        print(f"    [ElevenLabs] saved {out_path.name} ({len(audio_np)/24000:.1f}s)")
        return True
    except Exception as e:
        print(f"    [ElevenLabs] failed: {e} — falling back to Kokoro")
        return False


def _generate_kokoro(text: str, out_path: Path):
    """Kokoro TTS fallback — fully local, no API key needed."""
    from kokoro import KPipeline
    import soundfile as sf

    pipeline = KPipeline(lang_code='a')
    segments = []
    silence  = np.zeros(int(0.3 * 24000), dtype=np.float32)
    for _, _, audio in pipeline(text, voice='am_adam', speed=0.88):
        segments.append(audio)
    segments.append(silence)
    combined = np.concatenate(segments)
    sf.write(str(out_path), combined, 24000)
    print(f"    [Kokoro] saved {out_path.name} ({len(combined)/24000:.1f}s)")


def generate_audio(scene: dict) -> Path:
    out_path = AUDIO_DIR / f"scene_{scene['id']:02d}.wav"
    if out_path.exists():
        print(f"  Audio cached: {out_path.name}")
        return out_path

    print(f"  Scene {scene['id']:02d} TTS...")
    text = scene["narration"]
    if not _generate_elevenlabs(text, out_path):
        _generate_kokoro(text, out_path)
    return out_path


def generate_all_audio():
    import os
    backend = "ElevenLabs" if os.environ.get("ELEVENLABS_API_KEY") else "Kokoro (local)"
    print(f"Generating narration audio — primary: {backend}, fallback: Kokoro...")
    for scene in SCENES:
        generate_audio(scene)
    print("All audio done!")


# ── Showcase assembly ─────────────────────────────────────────────────────────

def build_showcase():
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    clips = []
    for scene in SCENES:
        img_path   = SCENES_DIR / f"scene_{scene['id']:02d}.png"
        audio_path = AUDIO_DIR  / f"scene_{scene['id']:02d}.wav"

        if not img_path.exists() or not audio_path.exists():
            print(f"  SKIP scene {scene['id']}: missing files")
            continue

        audio    = AudioFileClip(str(audio_path))
        duration = audio.duration + 1.0
        clip     = ImageClip(str(img_path)).set_duration(duration).set_audio(audio)
        clips.append(clip)
        print(f"  Scene {scene['id']:02d}: {duration:.1f}s")

    if not clips:
        print("No clips to assemble.")
        return None

    print(f"\nAssembling {len(clips)} clips...")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        str(SHOWCASE_OUT), fps=24,
        codec="libx264", audio_codec="aac",
        logger="bar"
    )
    print(f"\nSaved: {SHOWCASE_OUT}")
    return SHOWCASE_OUT


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenes",   action="store_true", help="Generate 10 island scenes")
    p.add_argument("--tts",      action="store_true", help="Generate narration audio")
    p.add_argument("--showcase", action="store_true", help="Assemble showcase MP4")
    p.add_argument("--all",      action="store_true", help="Run everything")
    a = p.parse_args()

    if a.all or a.scenes:   generate_all_scenes()
    if a.all or a.tts:      generate_all_audio()
    if a.all or a.showcase:
        out = build_showcase()
        if out:
            subprocess.run(["open", str(out)])
    if not any(vars(a).values()):
        print(__doc__)

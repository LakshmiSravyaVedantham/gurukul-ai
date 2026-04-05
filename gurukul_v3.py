"""
Gurukul Probability v3
- img2img at strength 0.35 from approved base_scene.png
- NO overlays — props appear naturally IN the scene via prompt
- Natural Guru-kid conversation (not narration)
- Kokoro TTS: Guru slow+warm (am_adam 0.85), Kid energetic (af_sky 1.0)
- 0.5s silence between speaker turns

Run:
    python gurukul_v3.py --scenes      # generate 12 scenes (~18 min)
    python gurukul_v3.py --tts         # generate voices
    python gurukul_v3.py --showcase    # assemble MP4
    python gurukul_v3.py --all         # everything end-to-end
"""

import subprocess, sys
import numpy as np
from pathlib import Path

AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
BASE_IMAGE   = AI_EDU_DIR / "output" / "base_scene.png"
SCENES_DIR   = AI_EDU_DIR / "output" / "v3_scenes"
AUDIO_DIR    = AI_EDU_DIR / "output" / "v3_audio"
SHOWCASE_OUT = AI_EDU_DIR / "output" / "gurukul_v3_showcase.mp4"
MFLUX        = Path("/Volumes/bujji1/sravya/ai_vidgen/venv/bin/mflux-generate")

for d in [SCENES_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Character base — identical every prompt so faces stay consistent ──────────
CHARS = (
    "Disney Pixar 3D CGI animated movie, Pixar Studios quality render, "
    "wise old bald Indian guru NO GLASSES with long white beard wearing saffron orange robes, "
    "two young Indian children in simple orange and red clothes sitting cross-legged facing him, "
    "under giant ancient banyan tree, warm golden Indian ashram, "
    "dappled afternoon sunlight, polished 3D render, cinematic, "
    "full body visible including hands"
)

# ── 12 scenes: prop described naturally IN the scene, no overlay ──────────────
# Each prompt = CHARS + scene-specific action/prop
SCENE_DEFS = [
    (1,
     "guru opening both arms wide in a big warm welcoming gesture, "
     "kids smiling brightly and clapping hands with delight, "
     "golden sunlight and sparkles in the air, joyful and inviting"),

    (2,
     "guru extending his right arm toward the viewer holding one large shiny gold coin "
     "pinched clearly between his thumb and index finger in the foreground, "
     "coin is big and clearly visible with golden glint, "
     "kids leaning forward staring at the coin with wide excited eyes"),

    (3,
     "guru holding a large gold coin on his thumb ready to flick it into the air, "
     "thumb cocked under the coin about to toss it, "
     "both kids leaning forward with open mouths in anticipation, "
     "coin gleaming in the sunlight prominently visible in guru's hand"),

    (4,
     "guru holding a large bright red six-sided dice prominently in both cupped hands "
     "extended toward the kids, dice clearly shows dots on all visible faces, "
     "one kid counting dots on their fingers, other kid pointing at the dice excitedly"),

    (5,
     "guru holding the large red dice up high with one hand "
     "and pointing with one finger of other hand at the face showing four dots, "
     "four dot face glowing bright yellow, one kid raising hand with one finger up, "
     "both kids smiling and excited"),

    (6,
     "guru holding a large round cloth bag open toward the kids "
     "with three red balls and two blue balls clearly visible inside the open bag, "
     "bag is prominent in the foreground, kids counting balls on fingers, "
     "all looking at the colorful balls inside"),

    (7,
     "one kid reaching hand into a large cloth bag with eyes tightly closed, "
     "other kid watching with both hands over their mouth in excited suspense, "
     "guru leaning forward encouraging with big smile, "
     "bag prominently in front of the kid reaching in"),

    (8,
     "guru holding both hands out open like weighing scales, "
     "one hand slightly higher than other in a balancing gesture, "
     "looking at both kids with wise warm expression, "
     "kids leaning forward listening very attentively"),

    (9,
     "guru raising his arm high and pointing one finger straight up at the bright golden sun "
     "streaming through the banyan tree leaves above them, "
     "both kids tilting heads back shielding eyes looking up at the sun, "
     "dramatic golden god rays streaming down beautifully"),

    (10,
     "guru holding the large red dice in one hand and dramatically waving his other "
     "index finger back and forth in an exaggerated NO gesture, "
     "both kids laughing out loud with open mouths, "
     "everyone shaking heads with big smiles, playful funny energy"),

    (11,
     "one kid flicking a gold coin upward into the air with thumb, "
     "coin clearly visible spinning in the air above them, "
     "other kid clapping hands excitedly watching the coin, "
     "guru laughing with hands on knees watching joyfully"),

    (12,
     "guru placing both hands warmly on the two kids shoulders as all three "
     "smile broadly at each other in a proud happy moment, "
     "golden sparkles and warm light all around, "
     "heartwarming celebratory group moment, triumph and joy"),
]

# ── Dialogue — natural conversation, not narration ────────────────────────────
# guru speaks slow and warm; kid speaks quick and curious
SCENES = [
    {
        "id": 1, "title": "Welcome to Gurukul!",
        "dialogue": [
            ("guru", "Arey! My students are here! Come, come, sit down!"),
            ("kid",  "Guru Ji! What are we learning today?"),
            ("guru", "Today... I have something very special for you both."),
            ("kid",  "What is it? What is it?"),
            ("guru", "The secret of CHANCE. Welcome... to Probability!"),
        ]
    },
    {
        "id": 2, "title": "Look at This Coin!",
        "dialogue": [
            ("guru", "Look! What do I have here?"),
            ("kid",  "A coin Guru Ji! A shiny coin!"),
            ("guru", "Tell me — what do you see on it?"),
            ("kid",  "Two sides! One side has H... and the other has T!"),
            ("guru", "Heads and Tails. Just two sides. Simple yes?"),
        ]
    },
    {
        "id": 3, "title": "What Will Happen?",
        "dialogue": [
            ("guru", "Now I am going to toss this coin. What will happen?"),
            ("kid",  "Umm... maybe Heads?"),
            ("guru", "Maybe! Or maybe Tails. We do NOT know for sure."),
            ("kid",  "So we cannot predict it Guru Ji?"),
            ("guru", "We cannot be CERTAIN. But we can be SMART about it!"),
        ]
    },
    {
        "id": 4, "title": "The Dice Has Six Faces!",
        "dialogue": [
            ("guru", "Now look at THIS! What is this?"),
            ("kid",  "A dice! A big red dice!"),
            ("guru", "Count the faces for me!"),
            ("kid",  "One... two... three... four... five... SIX faces!"),
            ("guru", "Six faces! So how many things can happen when we roll it?"),
            ("kid",  "Six things! One number for each face!"),
        ]
    },
    {
        "id": 5, "title": "What Are the Chances of Getting 4?",
        "dialogue": [
            ("kid",  "Guru Ji! I want to get the number four when I roll!"),
            ("guru", "Hmm. How many faces show the number four?"),
            ("kid",  "Just ONE face!"),
            ("guru", "And how many faces are there total?"),
            ("kid",  "Six total. So... one out of six!"),
            ("guru", "That is the probability. ONE out of SIX. You got it!"),
        ]
    },
    {
        "id": 6, "title": "Red Balls and Blue Balls",
        "dialogue": [
            ("guru", "Look at what I have in this bag!"),
            ("kid",  "Balls! Red ones and blue ones!"),
            ("guru", "Count them with me. How many red?"),
            ("kid",  "One... two... three RED balls!"),
            ("guru", "And blue?"),
            ("kid",  "Two BLUE balls. Five balls total!"),
            ("guru", "If you close your eyes and pick one — which color do you think will come?"),
        ]
    },
    {
        "id": 7, "title": "Reach In and Pick!",
        "dialogue": [
            ("kid",  "Red! Because there are MORE red balls!"),
            ("guru", "Clever! Three red out of five — red has the bigger chance."),
            ("kid",  "Can I try Guru Ji? Can I pick one?"),
            ("guru", "Go on! Close your eyes and reach in!"),
            ("kid",  "I got a RED one! I knew it!"),
            ("guru", "Math helped you predict that before you even tried!"),
        ]
    },
    {
        "id": 8, "title": "The Simple Secret",
        "dialogue": [
            ("guru", "Listen carefully. Here is the whole secret of probability."),
            ("kid",  "Tell us Guru Ji!"),
            ("guru", "Count how many ways you CAN win."),
            ("kid",  "Okay..."),
            ("guru", "Then count ALL the possible things that can happen."),
            ("kid",  "And then?"),
            ("guru", "Divide the first by the second. That IS probability."),
        ]
    },
    {
        "id": 9, "title": "When We Are 100% Sure",
        "dialogue": [
            ("guru", "Look up there! Will the sun rise tomorrow morning?"),
            ("kid",  "Yes! Of course Guru Ji! It always does!"),
            ("guru", "When we are completely certain something will happen..."),
            ("kid",  "What is the probability?"),
            ("guru", "The probability is ONE. Full. Complete. Certain."),
            ("kid",  "So one is the highest probability possible!"),
        ]
    },
    {
        "id": 10, "title": "When It Is Impossible!",
        "dialogue": [
            ("guru", "Now tell me — can this dice ever give us the number NINE?"),
            ("kid",  "No no no! It only goes up to six!"),
            ("guru", "So what is the chance of rolling a nine?"),
            ("kid",  "ZERO! It can never happen!"),
            ("guru", "Exactly. Zero means impossible. One means certain."),
            ("kid",  "And everything else is somewhere in between!"),
        ]
    },
    {
        "id": 11, "title": "Your Turn!",
        "dialogue": [
            ("kid",  "Guru Ji can I toss the coin? Please?"),
            ("guru", "Yes! But first — predict! Heads or Tails?"),
            ("kid",  "I say HEADS!"),
            ("guru", "Good! What is your probability of being right?"),
            ("kid",  "One out of two! Fifty percent!"),
            ("guru", "Toss it!"),
            ("kid",  "HEADS! I was right Guru Ji!"),
        ]
    },
    {
        "id": 12, "title": "See You Next Time!",
        "dialogue": [
            ("guru", "You have both done wonderfully today!"),
            ("kid",  "We learned about coins and dice and bags of balls!"),
            ("guru", "Remember — probability is just counting carefully."),
            ("kid",  "Favorable outcomes divided by total outcomes!"),
            ("guru", "Zero means impossible. One means certain. In between — possible!"),
            ("kid",  "Thank you Guru Ji! Same time tomorrow?"),
            ("guru", "Gurukul is always open for curious minds. See you next time!"),
        ]
    },
]


# ── img2img scene generation ──────────────────────────────────────────────────

def generate_scene(scene_id: int, prop_desc: str):
    out = SCENES_DIR / f"scene_{scene_id:02d}.png"
    if out.exists():
        print(f"  Scene {scene_id:02d}: cached")
        return out
    prompt = f"{CHARS}, {prop_desc}"
    cmd = [
        str(MFLUX), "--model", "schnell",
        "--prompt", prompt,
        "--image-path", str(BASE_IMAGE),
        "--image-strength", "0.55",
        "--width", "1360", "--height", "768",
        "--steps", "4",
        "--seed", str(2000 + scene_id),
        "--output", str(out),
    ]
    print(f"  Generating scene {scene_id:02d}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out.exists():
        print(f"  Saved: {out.name}")
    else:
        print(f"  ERROR scene {scene_id}:\n{r.stderr[-400:]}")
    return out


def generate_all_scenes():
    print(f"Generating {len(SCENE_DEFS)} scenes via img2img (strength=0.35)...")
    for scene_id, prop_desc in SCENE_DEFS:
        generate_scene(scene_id, prop_desc)
    print("All scenes done!")


# ── Kokoro TTS ────────────────────────────────────────────────────────────────

def generate_audio(scene: dict) -> Path:
    from kokoro import KPipeline
    import soundfile as sf

    out_path = AUDIO_DIR / f"scene_{scene['id']:02d}.wav"
    if out_path.exists():
        print(f"  Audio cached: {out_path.name}")
        return out_path

    pipeline = KPipeline(lang_code='a')
    silence_between = np.zeros(int(0.5 * 24000), dtype=np.float32)  # 0.5s gap between turns
    silence_end     = np.zeros(int(0.3 * 24000), dtype=np.float32)  # 0.3s after last line

    segments = []
    for speaker, text in scene["dialogue"]:
        if speaker == 'guru':
            voice = 'am_adam'
            speed = 0.85   # slow, warm, storytelling
        else:
            voice = 'af_sky'
            speed = 1.0    # energetic, curious

        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            segments.append(audio)
        segments.append(silence_between)

    # Replace last silence_between with shorter end silence
    if segments:
        segments[-1] = silence_end
        combined = np.concatenate(segments)
        sf.write(str(out_path), combined, 24000)
        print(f"  Audio: {out_path.name} ({len(combined)/24000:.1f}s)")
    return out_path


def generate_all_audio():
    print("Generating Kokoro TTS audio...")
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

        if not img_path.exists():
            print(f"  SKIP scene {scene['id']}: no image")
            continue
        if not audio_path.exists():
            print(f"  SKIP scene {scene['id']}: no audio")
            continue

        audio    = AudioFileClip(str(audio_path))
        duration = audio.duration + 0.8   # tiny tail after audio ends
        clip     = ImageClip(str(img_path)).set_duration(duration).set_audio(audio)
        clips.append(clip)
        print(f"  Scene {scene['id']:02d}: {duration:.1f}s")

    if not clips:
        print("No clips — run --scenes and --tts first")
        return

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
    p.add_argument("--scenes",   action="store_true", help="Generate 12 img2img scenes")
    p.add_argument("--tts",      action="store_true", help="Generate Kokoro TTS audio")
    p.add_argument("--showcase", action="store_true", help="Assemble showcase MP4")
    p.add_argument("--all",      action="store_true", help="Run everything end-to-end")
    a = p.parse_args()

    if a.all or a.scenes:   generate_all_scenes()
    if a.all or a.tts:      generate_all_audio()
    if a.all or a.showcase:
        out = build_showcase()
        if out:
            subprocess.run(["open", str(out)])
    if not any(vars(a).values()):
        print(__doc__)

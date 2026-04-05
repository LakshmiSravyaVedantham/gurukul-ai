"""
Gurukul Probability v2 — Proper pipeline
- img2img at strength 0.25 from base scene → consistent characters, changing props
- Large centered prop visuals drawn with Pillow
- Kokoro TTS for high-quality Guru + kid voices

Run:
    python gurukul_v2.py --scenes      # generate 12 img2img scenes (~18 min)
    python gurukul_v2.py --overlay     # add large prop overlays
    python gurukul_v2.py --tts         # generate voices (Kokoro)
    python gurukul_v2.py --showcase    # assemble MP4
    python gurukul_v2.py --all         # everything end-to-end
"""

import subprocess, sys, shutil, math, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

AI_EDU_DIR   = Path("/Volumes/bujji1/sravya/ai_edu")
BASE_IMAGE   = AI_EDU_DIR / "output" / "base_scene.png"
SCENES_DIR   = AI_EDU_DIR / "output" / "v2_scenes"
FRAMES_DIR   = AI_EDU_DIR / "output" / "v2_frames"
AUDIO_DIR    = AI_EDU_DIR / "output" / "v2_audio"
SHOWCASE_OUT = AI_EDU_DIR / "output" / "gurukul_v2_showcase.mp4"
MFLUX        = Path("/Volumes/bujji1/sravya/ai_vidgen/venv/bin/mflux-generate")

for d in [SCENES_DIR, FRAMES_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Character base (same in every prompt for consistency) ────────────────────
CHARS = (
    "Disney Pixar 3D CGI animated movie, Pixar Studios quality, "
    "wise old bald Indian guru with long white beard in saffron orange robes "
    "sitting cross-legged, two young Indian children sitting facing him, "
    "under giant ancient banyan tree, warm golden Indian ashram, "
    "dappled afternoon sunlight, polished 3D render, cinematic"
)

# ── 12 scenes: each adds specific prop/action to base ────────────────────────
SCENE_DEFS = [
    (1,  "guru and two kids sitting together warmly, guru gesturing welcome with open hands, kids smiling excitedly, magical golden sparkles in the air, welcoming atmosphere"),
    (2,  "guru holding up one large shiny golden coin in his hand showing it to the two kids, coin clearly visible with H on one side, kids leaning forward with wide curious eyes looking at the coin"),
    (3,  "guru and kids looking at a large golden coin floating in the air between them, one side showing H for heads glowing, other side showing T for tails, magical sparkle effect"),
    (4,  "guru holding a large colorful red dice in both hands showing it to the kids, dice clearly showing dots on all visible faces, kids counting the dots on their fingers excitedly"),
    (5,  "guru tossing the red dice in the air, dice spinning with number 4 face highlighted in yellow glow, kids watching with mouths open in amazement, dynamic action shot"),
    (6,  "guru holding a large transparent bag filled with colorful balls, red balls and blue balls clearly visible inside the bag, kids peering into the bag counting the balls"),
    (7,  "one kid reaching into the bag with eyes closed to pick a ball, other kid watching anxiously, guru smiling encouragingly, suspense and excitement on all faces"),
    (8,  "guru holding up a big wooden board showing a simple diagram, kids looking at it attentively, guru pointing at the board with finger, teaching moment"),
    (9,  "guru pointing up at the bright golden sun shining through the banyan tree leaves, kids looking up at the sun with hand over eyes, sun rays streaming down beautifully"),
    (10, "guru shaking head and holding up the dice with a big X gesture, kids laughing and shaking heads too, playful funny moment showing impossibility"),
    (11, "guru and two kids playing a coin toss game together, laughing joyfully, coin in the air, happy celebratory moment, golden light all around"),
    (12, "guru and two kids in a joyful group pose with hands raised in celebration, big smiles all around, golden sparkles and light rays, triumphant happy ending scene"),
]

# ── Dialogue script ───────────────────────────────────────────────────────────
SCENES = [
    {
        "id": 1, "title": "Welcome to Gurukul!",
        "visual": "welcome",
        "dialogue": [
            ("guru", "Welcome to Gurukul, dear students!"),
            ("kid",  "Guru Ji, what are we learning today?"),
            ("guru", "Today — we learn PROBABILITY. The math of CHANCE!"),
            ("kid",  "What is chance, Guru Ji?"),
            ("guru", "Chance is how LIKELY something is to happen. Let me show you!"),
        ]
    },
    {
        "id": 2, "title": "The Coin — 2 Outcomes",
        "visual": "coin_show",
        "dialogue": [
            ("guru", "Look at this coin I am holding."),
            ("kid",  "It has two sides, Guru Ji!"),
            ("guru", "Exactly! If I toss it — what can happen?"),
            ("kid",  "It can land on Heads... or Tails!"),
            ("guru", "Two possible outcomes. That is the start of probability!"),
        ]
    },
    {
        "id": 3, "title": "Heads or Tails?",
        "visual": "coin_both",
        "dialogue": [
            ("kid",  "Guru Ji, which side will come up?"),
            ("guru", "We do not know for sure. But each side has EQUAL chance!"),
            ("kid",  "So the chance of Heads is 1 out of 2?"),
            ("guru", "Brilliant! We write it as one divided by two. Or fifty percent!"),
            ("kid",  "So half the time Heads, half the time Tails!"),
        ]
    },
    {
        "id": 4, "title": "The Dice — 6 Outcomes",
        "visual": "dice_show",
        "dialogue": [
            ("guru", "Now look at this dice I am holding. Count the faces!"),
            ("kid",  "One, two, three, four, five, SIX faces!"),
            ("guru", "So how many possible outcomes when we roll it?"),
            ("kid",  "SIX outcomes — one for each face!"),
            ("guru", "Perfect! And each number has an equal chance of coming up."),
        ]
    },
    {
        "id": 5, "title": "Chance of Rolling 4",
        "visual": "dice_4",
        "dialogue": [
            ("kid",  "Guru Ji! What is the chance of getting number 4?"),
            ("guru", "How many faces show number 4?"),
            ("kid",  "Just ONE face out of SIX!"),
            ("guru", "So the probability is one divided by six!"),
            ("kid",  "That is a small chance — only about 17 percent!"),
        ]
    },
    {
        "id": 6, "title": "The Bag of Balls",
        "visual": "bag_show",
        "dialogue": [
            ("guru", "Students — look at this bag I am holding!"),
            ("kid",  "It has red balls and blue balls inside!"),
            ("guru", "Three red and two blue. Total of five balls."),
            ("kid",  "If we pick one without looking — what is the chance of red?"),
            ("guru", "Three red out of five total. So three divided by five!"),
        ]
    },
    {
        "id": 7, "title": "Picking Without Looking",
        "visual": "bag_pick",
        "dialogue": [
            ("kid",  "Can I try picking a ball, Guru Ji?"),
            ("guru", "Yes! Close your eyes and pick one!"),
            ("kid",  "I got a RED one! Was I lucky?"),
            ("guru", "The probability was three out of five — so quite likely!"),
            ("kid",  "Math helped us predict it before we even tried!"),
        ]
    },
    {
        "id": 8, "title": "The Magic Formula",
        "visual": "formula_board",
        "dialogue": [
            ("guru", "Students, here is our golden formula!"),
            ("kid",  "Tell us Guru Ji!"),
            ("guru", "FAVORABLE outcomes divided by TOTAL outcomes."),
            ("kid",  "Like 3 red divided by 5 total equals three fifths!"),
            ("guru", "Exactly! Simple, powerful, and it always works!"),
        ]
    },
    {
        "id": 9, "title": "When Chance is Certain",
        "visual": "sun",
        "dialogue": [
            ("guru", "Look up at that beautiful sun! Will it rise tomorrow?"),
            ("kid",  "Of course Guru Ji! It ALWAYS rises!"),
            ("guru", "When something is absolutely certain — probability is ONE!"),
            ("kid",  "So P equals 1 means it will definitely happen?"),
            ("guru", "Yes! Between 0 and 1 — 1 is the highest possible probability!"),
        ]
    },
    {
        "id": 10, "title": "When It Is Impossible",
        "visual": "impossible",
        "dialogue": [
            ("guru", "Can we roll a number NINE on this dice?"),
            ("kid",  "No! It only has numbers one to six!"),
            ("guru", "Correct! That is IMPOSSIBLE. Probability is ZERO!"),
            ("kid",  "P equals zero means it can never happen!"),
            ("guru", "So probability is always between zero and one. Never outside!"),
        ]
    },
    {
        "id": 11, "title": "Let us Play!",
        "visual": "play",
        "dialogue": [
            ("kid",  "Guru Ji can we practice with a game?"),
            ("guru", "Of course! Toss the coin — predict Heads or Tails!"),
            ("kid",  "I predict HEADS!"),
            ("guru", "Good! You have a fifty percent chance of being right!"),
            ("kid",  "Probability makes guessing smarter, not just lucky!"),
        ]
    },
    {
        "id": 12, "title": "See You Next Time!",
        "visual": "celebrate",
        "dialogue": [
            ("guru", "Today we mastered PROBABILITY — the math of chance!"),
            ("kid",  "Favorable divided by total — I will remember!"),
            ("guru", "Zero means impossible. One means certain. In between — possible!"),
            ("kid",  "Thank you Guru Ji! This was so fun to learn!"),
            ("guru", "Well done students! See you next time at Gurukul!"),
            ("kid",  "Jai Gurukul!"),
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
        "--image-strength", "0.35",
        "--width", "1360", "--height", "768",
        "--steps", "4",
        "--seed", str(1000 + scene_id),
        "--output", str(out),
    ]
    print(f"  Generating scene {scene_id:02d}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out.exists():
        print(f"  Saved: {out.name}")
    else:
        print(f"  ERROR scene {scene_id}: {r.stderr[-300:]}")
    return out


def generate_all_scenes():
    print(f"Generating {len(SCENE_DEFS)} scenes via img2img (strength=0.35)...")
    for scene_id, prop_desc in SCENE_DEFS:
        generate_scene(scene_id, prop_desc)
    print("All scenes done!")


# ── Large prop overlays ───────────────────────────────────────────────────────

def get_fonts():
    try:
        big  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
        med  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
        sml  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        bold = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 48)
    except Exception:
        big = med = sml = bold = ImageFont.load_default()
    return big, med, sml, bold


def draw_coin(draw, cx, cy, r, label, gold=True):
    bg = (255, 200, 0) if gold else (180, 180, 180)
    rim = (200, 150, 0) if gold else (120, 120, 120)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=bg, outline=rim, width=6)
    draw.ellipse([cx-r+10, cy-r+10, cx+r-10, cy+r-10], outline=rim, width=3)
    _, _, _, bold = get_fonts()
    bb = draw.textbbox((0,0), label, font=bold)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((cx - tw//2, cy - th//2 - 4), label, fill=(80,40,0), font=bold)


def draw_dice(draw, x, y, size, n, highlight=False):
    col = (255, 80, 80) if not highlight else (255, 200, 0)
    rim = (180, 30, 30) if not highlight else (200, 140, 0)
    draw.rounded_rectangle([x, y, x+size, y+size], radius=14, fill=col, outline=rim, width=5)
    dots = {1:[(0.5,0.5)], 2:[(0.28,0.28),(0.72,0.72)], 3:[(0.28,0.28),(0.5,0.5),(0.72,0.72)],
            4:[(0.28,0.28),(0.72,0.28),(0.28,0.72),(0.72,0.72)],
            5:[(0.28,0.28),(0.72,0.28),(0.5,0.5),(0.28,0.72),(0.72,0.72)],
            6:[(0.28,0.2),(0.72,0.2),(0.28,0.5),(0.72,0.5),(0.28,0.8),(0.72,0.8)]}
    for dx, dy in dots.get(n, []):
        dr = max(9, size//9)
        dcx, dcy = int(x+dx*size), int(y+dy*size)
        draw.ellipse([dcx-dr, dcy-dr, dcx+dr, dcy+dr], fill=(255,255,255))


def draw_ball(draw, cx, cy, r, color):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline=(0,0,0,80), width=2)
    draw.ellipse([cx-r//3, cy-r//2, cx+r//5, cy-r//6], fill=(255,255,255,160))


def make_prop_overlay(W, H, visual):
    ov = Image.new("RGBA", (W, H), (0,0,0,0))
    d  = ImageDraw.Draw(ov)
    big, med, sml, bold = get_fonts()
    cx, cy = W//2, H//2

    # ── title bar ──────────────────────────────────────────────────────
    scene = next((s for s in SCENES if s["visual"] == visual), None)
    title = scene["title"] if scene else ""
    d.rectangle([0, 0, W, 72], fill=(0,0,0,190))
    d.text((cx, 36), title, fill=(255,220,60), font=med, anchor="mm")

    if visual == "welcome":
        d.text((cx, 160), "🏫 GURUKUL", fill=(255,220,60), font=big, anchor="mm")
        d.text((cx, 250), "PROBABILITY", fill=(255,255,255), font=bold, anchor="mm")
        d.text((cx, 320), "The Math of Chance!", fill=(200,240,255), font=med, anchor="mm")

    elif visual in ("coin_show", "coin_both"):
        # Two large coins side by side in center
        draw_coin(d, cx-140, cy+20, 110, "H", gold=True)
        d.text((cx, cy+20), "OR", fill=(255,255,255), font=bold, anchor="mm")
        draw_coin(d, cx+140, cy+20, 110, "T", gold=False)
        if visual == "coin_both":
            d.text((cx, cy+160), "P(Heads) = 1/2 = 50%", fill=(100,255,150), font=med, anchor="mm")

    elif visual in ("dice_show", "dice_4"):
        ds = 105
        gap = 18
        row_w = 3*(ds+gap)
        sx = cx - row_w//2
        sy = cy - ds - gap//2
        for i, n in enumerate([1,2,3,4,5,6]):
            col, row = i%3, i//3
            fx = sx + col*(ds+gap)
            fy = sy + row*(ds+gap)
            draw_dice(d, fx, fy, ds, n, highlight=(n==4 and visual=="dice_4"))
        if visual == "dice_4":
            d.text((cx, cy+ds+gap+30), "P(4) = 1/6 ≈ 17%", fill=(255,220,60), font=bold, anchor="mm")

    elif visual in ("bag_show", "bag_pick"):
        # Draw a big bag with balls
        bx, by, bw, bh = cx-160, cy-120, 320, 280
        d.ellipse([bx+bw//3, by-30, bx+2*bw//3, by+30], fill=(140,90,40))
        d.rounded_rectangle([bx, by, bx+bw, by+bh], radius=40, fill=(100,65,25,180), outline=(140,90,40), width=5)
        balls = [(255,60,60)]*3 + [(60,120,255)]*2
        positions = [(bx+70,by+80),(bx+160,by+65),(bx+250,by+80),(bx+100,by+170),(bx+210,by+175)]
        for (bcy, bcx_), col in zip(positions, balls):
            draw_ball(d, bcy, bcx_, 38, col)
        d.text((cx, by+bh+40), "3 Red + 2 Blue = 5 balls", fill=(255,220,100), font=med, anchor="mm")
        if visual == "bag_show":
            d.text((cx, by+bh+90), "P(Red) = 3/5 = 60%", fill=(100,255,150), font=bold, anchor="mm")
        else:
            d.text((cx, by+bh+90), "P(Red) = 3/5 → Quite Likely!", fill=(100,255,150), font=bold, anchor="mm")

    elif visual == "formula_board":
        # Chalkboard style formula
        bx, by, bw, bh = cx-280, cy-130, 560, 260
        d.rounded_rectangle([bx, by, bx+bw, by+bh], radius=15, fill=(30,80,40,220), outline=(80,140,80), width=4)
        d.text((cx, by+50), "P(event) =", fill=(255,255,200), font=bold, anchor="mm")
        d.text((cx, by+120), "Favorable outcomes", fill=(100,255,150), font=med, anchor="mm")
        d.line([cx-200, by+155, cx+200, by+155], fill=(200,200,200), width=3)
        d.text((cx, by+190), "Total outcomes", fill=(255,150,100), font=med, anchor="mm")

    elif visual == "sun":
        sun_r = 90
        sun_cx, sun_cy = cx, cy+10
        for a in range(0, 360, 30):
            import math as m
            rad = m.radians(a)
            x1, y1 = sun_cx+int((sun_r+15)*m.cos(rad)), sun_cy+int((sun_r+15)*m.sin(rad))
            x2, y2 = sun_cx+int((sun_r+50)*m.cos(rad)), sun_cy+int((sun_r+50)*m.sin(rad))
            d.line([x1,y1,x2,y2], fill=(255,220,0,200), width=7)
        d.ellipse([sun_cx-sun_r, sun_cy-sun_r, sun_cx+sun_r, sun_cy+sun_r], fill=(255,200,0))
        d.text((sun_cx, sun_cy+sun_r+50), "P = 1   →   CERTAIN! ✓", fill=(255,220,60), font=bold, anchor="mm")

    elif visual == "impossible":
        # Dice with no 7
        draw_dice(d, cx-60, cy-80, 120, 6)
        d.text((cx+100, cy-30), "7 ?", fill=(255,80,80), font=big, anchor="mm")
        d.line([cx+55, cy-80, cx+175, cy+20], fill=(255,50,50), width=10)
        d.line([cx+175, cy-80, cx+55, cy+20], fill=(255,50,50), width=10)
        d.text((cx, cy+80), "P = 0   →   IMPOSSIBLE! ✗", fill=(255,80,80), font=bold, anchor="mm")

    elif visual in ("play", "celebrate"):
        d.text((cx, cy-30), "🎉" if visual=="celebrate" else "🎲", font=big, anchor="mm", fill=(255,220,60))
        msg = "You learned PROBABILITY!" if visual == "celebrate" else "Your turn! Predict the outcome!"
        d.text((cx, cy+80), msg, fill=(255,220,60), font=bold, anchor="mm")
        if visual == "celebrate":
            for line in ["P = Favorable ÷ Total", "0 = Impossible | 1 = Certain"]:
                d.text((cx, cy+150 + (["P = Favorable ÷ Total","0 = Impossible | 1 = Certain"].index(line))*55),
                       line, fill=(200,240,255), font=med, anchor="mm")

    # ── dialogue subtitles (bottom) ────────────────────────────────────
    if scene:
        lines = scene["dialogue"][-2:]
        dh = 110
        d.rectangle([0, H-dh, W, H], fill=(0,0,0,200))
        for i, (spk, txt) in enumerate(lines):
            color = (255, 200, 60) if spk == "guru" else (130, 210, 255)
            label = "Guru Ji:" if spk == "guru" else "Student :"
            d.text((24, H-dh+12+i*46), label, fill=color, font=sml)
            d.text((160, H-dh+12+i*46), txt, fill=(240,240,240), font=sml)

    return ov


def make_frame(scene_id: int, visual: str):
    src = SCENES_DIR / f"scene_{scene_id:02d}.png"
    if not src.exists():
        src = BASE_IMAGE
    dst = FRAMES_DIR / f"scene_{scene_id:02d}.png"
    img = Image.open(src)
    W, H = img.size
    ov = make_prop_overlay(W, H, visual)
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    out.save(dst)
    print(f"  Frame {scene_id:02d}: {dst.name}")
    return dst


def generate_all_frames():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(SCENES)} frames...")
    for s in SCENES:
        make_frame(s["id"], s["visual"])
    print("All frames done!")


# ── Kokoro TTS ────────────────────────────────────────────────────────────────

def generate_kokoro_audio(scene: dict) -> Path:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np

    out_path = AUDIO_DIR / f"scene_{scene['id']:02d}.wav"
    if out_path.exists():
        print(f"  Audio cached: {out_path.name}")
        return out_path

    pipeline_guru = KPipeline(lang_code='a')  # American English
    pipeline_kid  = KPipeline(lang_code='a')

    segments = []
    silence = np.zeros(int(0.4 * 24000), dtype=np.float32)

    for speaker, text in scene["dialogue"]:
        voice = 'am_adam' if speaker == 'guru' else 'af_sky'
        pipeline = pipeline_guru if speaker == 'guru' else pipeline_kid
        for _, _, audio in pipeline(text, voice=voice, speed=0.95):
            segments.append(audio)
            segments.append(silence)

    if segments:
        combined = np.concatenate(segments)
        sf.write(str(out_path), combined, 24000)
        print(f"  Audio: {out_path.name} ({len(combined)/24000:.1f}s)")
    return out_path


def generate_all_audio():
    print("Generating Kokoro TTS audio...")
    for scene in SCENES:
        generate_kokoro_audio(scene)
    print("All audio done!")


# ── Showcase assembly ─────────────────────────────────────────────────────────

def build_showcase():
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    clips = []
    for scene in SCENES:
        img_path   = FRAMES_DIR / f"scene_{scene['id']:02d}.png"
        audio_path = AUDIO_DIR  / f"scene_{scene['id']:02d}.wav"
        if not img_path.exists() or not audio_path.exists():
            print(f"  SKIP {scene['id']}: missing files")
            continue
        audio    = AudioFileClip(str(audio_path))
        duration = audio.duration + 1.5
        clip     = ImageClip(str(img_path)).set_duration(duration).set_audio(audio)
        clips.append(clip)
        print(f"  Scene {scene['id']}: {duration:.1f}s")

    print(f"\nAssembling {len(clips)} clips...")
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(str(SHOWCASE_OUT), fps=24, codec="libx264", audio_codec="aac", logger="bar")
    print(f"\nSaved: {SHOWCASE_OUT}")
    return SHOWCASE_OUT


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenes",   action="store_true")
    p.add_argument("--overlay",  action="store_true")
    p.add_argument("--tts",      action="store_true")
    p.add_argument("--showcase", action="store_true")
    p.add_argument("--all",      action="store_true")
    a = p.parse_args()

    if a.all or a.scenes:   generate_all_scenes()
    if a.all or a.overlay:  generate_all_frames()
    if a.all or a.tts:      generate_all_audio()
    if a.all or a.showcase:
        out = build_showcase()
        subprocess.run(["open", str(out)])
    if not any(vars(a).values()):
        print(__doc__)

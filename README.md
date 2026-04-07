# Gurukul AI — Kids Educational Video Pipeline

> Fully local, free, Apple Silicon. No cloud APIs needed.
> Generates Pixar-style animated educational videos end-to-end.

**🌐 [gurukul-ai.vercel.app](https://web-five-lime-42.vercel.app)** &nbsp;|&nbsp; 

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Stage0["Stage 0 — Research"]
        R[🌐 web_research.py\nDuckDuckGo + Wikipedia\nno API key needed]
    end

    subgraph Stage1["Stage 1 — Script"]
        G[🧠 Gemma 3 4B\nmlx-lm · Apple Silicon\n10-scene island script]
    end

    subgraph Stage2["Stage 2 — Images"]
        F[🖼️ FLUX Dev\nmflux · Apple Silicon\n1360×768 per scene]
    end

    subgraph Stage3["Stage 3 — Audio"]
        TTS{API key set?}
        TTS -- Yes --> E[☁️ ElevenLabs\nDaniel voice]
        TTS -- No  --> K[🔊 Kokoro TTS\nfully local · free]
    end

    subgraph Stage4["Stage 4 — Animation"]
        C{ComfyUI\nrunning?}
        C -- LTX 2B    --> V1[⚡ LTX Video 2B\n~40s/scene]
        C -- LTX 13B   --> V2[🎬 LTX Video 13B\n~11min/scene]
        C -- LTX 2.3   --> V3[✨ LTX-2.3 22B GGUF\n~4-6min/scene]
        C -- Wan Fun5B --> V4[🌊 Wan2.2 Fun-5B GGUF\n~8-10min/scene]
        C -- Wan I2V   --> V5[🏆 Wan2.2 I2V-A14B GGUF\n~15-20min/scene]
        C -- No        --> KB[🎞️ Ken Burns\nffmpeg · instant]
    end

    subgraph Stage5["Stage 5 — Self-Improve ⚡"]
        D[🎭 Director\nGemma 4 26B\ncinematic prompt expansion]
        CR[🔍 Critic\nQwen2.5-VL 7B\nvideo scoring 1-10]
        RF[🔄 Refiner\nauto-escalate model\nif score below threshold]
        P[✨ Polisher\nTopaz Video AI\n4K upscale]
        D --> CR --> RF --> P
    end

    subgraph Stage6["Stage 6 — Post-Process"]
        SUB[🔤 subtitles.py\nmlx-whisper\nword-level captions]
        XF[🎞️ xfade transitions\nffmpeg dissolve/wipe\nbetween scenes]
    end

    R --> G
    G --> F
    G --> TTS
    F --> C
    K --> C
    E --> C
    V1 & V2 & V3 & V4 & V5 & KB --> XF
    XF --> SUB
    SUB --> OUT[📹 Final Video\n1280×720 MP4]

    V1 & V2 & V3 & V4 & V5 --> Stage5
    Stage5 --> LB[📊 Model Leaderboard\n+ Training Dataset]
```

## Features

| Feature | Tool | Cost |
|---|---|---|
| Script generation | Gemma 3 4B (mlx-lm) | Free |
| Topic research | DuckDuckGo + Wikipedia | Free |
| Image generation | FLUX Dev (mflux) | Free |
| Narration TTS | Kokoro am_adam | Free |
| Animation | LTX 2B / LTX 13B / LTX-2.3 / Wan2.2 Fun / Wan2.2 I2V | Free |
| Prompt expansion | Gemma 4 26B (mlx-lm) | Free |
| Video scoring | Qwen2.5-VL 7B (mlx-vlm) | Free |
| Word subtitles | mlx-whisper (Whisper Small) | Free |
| Scene transitions | ffmpeg xfade | Free |
| 4K upscale | Topaz Video AI (optional) | Paid app |

## ⚡ /selfimprove — Agentic Pipeline

5-stage self-improving loop that auto-escalates to better models until quality passes:

```
Director (Gemma 4) → Creator → Critic (Qwen2.5-VL scores 1-10) → Refiner → Polisher (Topaz 4K)
```

Escalation order: `LTX-2B → LTX-2.3 GGUF → Wan2.2 Fun-5B GGUF → LTX-13B → Wan2.2 I2V-A14B GGUF`

```bash
# Single model
python agentic_pipeline.py "coin slowly flipping" --scene 3 --model ltx-2b

# Benchmark all models + build leaderboard
python agentic_pipeline.py "coin slowly flipping" --scene 3 --models all

# View leaderboard
python agentic_pipeline.py --leaderboard
```

## Setup

```bash
# Clone and install
pip install -r requirements.txt
brew install ffmpeg

# Start the Gradio app
python app.py
# → Open http://localhost:7860
```

Start ComfyUI on port 8288 for video generation:
```bash
cd /path/to/ComfyUI && python main.py --port 8288 --preview-method none
```

## Models

| Stage | Model | Size | Format |
|---|---|---|---|
| Script | Gemma 3 4B | ~2.5 GB | MLX 4-bit |
| Director | Gemma 4 26B | ~14 GB | MLX 4-bit MoE |
| Critic | Qwen2.5-VL 7B | ~4 GB | MLX 4-bit |
| Animation (fast) | LTX Video 2B | ~4 GB | ComfyUI |
| Animation (quality) | LTX-2.3 22B | ~16 GB | GGUF Q4_0 |
| Animation (objects) | Wan2.2 Fun-5B | ~5 GB | GGUF Q8_0 |
| Animation (best) | Wan2.2 I2V-A14B | ~20 GB | GGUF dual |
| Subtitles | Whisper Small | ~150 MB | MLX |
| Images | FLUX Dev | ~16 GB | MLX BF16 |

## Quick commands

```bash
# Generate a new topic
python generate_topic.py "fractions"           # researches topic + writes script

# Add subtitles to any video
python subtitles.py output/animated.mp4

# Run web research only
python web_research.py "black holes"

# Assemble with xfade transitions
python assemble_video.py --all
```

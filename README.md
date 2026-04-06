# Gurukul AI — Kids Educational Video Pipeline

```mermaid
flowchart TD
    A[📝 Scene Definitions\nnarration script · 10 scenes] --> B
    A --> C

    B[🖼️ mflux FLUX Dev\nText-to-image · 1360×768\nApple Silicon · sequential only]

    C{ELEVENLABS_API_KEY set?}
    C -- Yes --> D[☁️ ElevenLabs TTS\nDaniel voice · multilingual v2]
    C -- No  --> E[🖥️ Kokoro TTS\nFully local · am_adam · free]

    B --> F
    D --> F
    E --> F

    F{ComfyUI running?}
    F -- Yes --> G[🎬 LTX Video 2B Distilled\nImage-to-Video · 8s clips\n4 steps · Apple Silicon MPS]
    F -- No  --> H[🎞️ Ken Burns Fallback\nffmpeg zoom + pan · instant\nno GPU needed]

    G -- success --> I
    G -- error/OOM --> H
    H --> I

    I[🎬 MoviePy Assembly\nping-pong loop · audio sync\nper-scene duration match]

    I --> J[📹 probability_island_animated.mp4]
```

## Setup

```bash
pip install -r requirements.txt
brew install ffmpeg
python download_models.py        # downloads LTX Video + T5 encoder (~9 GB)
```

Start ComfyUI on port 8288:
```bash
cd /path/to/ComfyUI && python main.py --port 8288 --preview-method none
```

## Run

```bash
python gurukul_island.py --scenes      # generate scene images
python gurukul_island.py --tts         # generate narration audio
python wan_animate.py --full           # animate + assemble final video
```

Single scene test:
```bash
python wan_animate.py --test
```

Static version (no ComfyUI needed):
```bash
python gurukul_island.py --showcase
```

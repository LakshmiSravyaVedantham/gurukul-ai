"""
Download LTX Video models into ComfyUI.
Run once before using wan_animate.py.

Usage:
    python download_models.py --comfyui /path/to/ComfyUI
"""

import argparse, shutil, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", default=None, help="Path to ComfyUI directory")
    args = p.parse_args()

    # Try to find ComfyUI directory
    if args.comfyui:
        comfy = Path(args.comfyui)
    else:
        # Common locations
        candidates = [
            Path.home() / "ComfyUI",
            Path("/Volumes/bujji1/sravya/ComfyUI"),
            Path.cwd().parent / "ComfyUI",
        ]
        comfy = next((c for c in candidates if c.exists()), None)
        if not comfy:
            print("Could not find ComfyUI. Pass --comfyui /path/to/ComfyUI")
            sys.exit(1)

    print(f"ComfyUI: {comfy}")
    checkpoints  = comfy / "models" / "checkpoints"
    text_encoders = comfy / "models" / "text_encoders"
    checkpoints.mkdir(parents=True, exist_ok=True)
    text_encoders.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    # ── LTX Video 2B distilled FP8 ────────────────────────────────────────────
    ltxv_dest = checkpoints / "ltxv-2b-0.9.8-distilled-fp8.safetensors"
    if ltxv_dest.exists():
        print(f"Already exists: {ltxv_dest.name}")
    else:
        print("Downloading LTX Video 2B distilled FP8 (~4.2 GB)...")
        hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename="ltxv-2b-0.9.8-distilled-fp8.safetensors",
            local_dir=str(checkpoints),
        )
        print(f"  Saved: {ltxv_dest}")

    # ── T5-XXL FP8 text encoder ────────────────────────────────────────────────
    t5_dest = text_encoders / "t5xxl_fp8_e4m3fn.safetensors"
    if t5_dest.exists():
        print(f"Already exists: {t5_dest.name}")
    else:
        # Check HuggingFace cache first (may already exist from FLUX)
        import glob, os
        cache_hits = glob.glob(
            str(Path.home() / ".cache/huggingface/hub/**/t5xxl_fp8_e4m3fn.safetensors"),
            recursive=True
        )
        if cache_hits:
            print(f"Found in cache — copying: {Path(cache_hits[0]).name}")
            shutil.copy2(cache_hits[0], str(t5_dest))
        else:
            print("Downloading T5-XXL FP8 (~4.6 GB)...")
            hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                filename="t5xxl_fp8_e4m3fn.safetensors",
                local_dir=str(text_encoders),
            )
        print(f"  Saved: {t5_dest}")

    print("\nAll models ready.")
    print("Start ComfyUI: cd", comfy, "&& python main.py --port 8288 --preview-method none")


if __name__ == "__main__":
    main()

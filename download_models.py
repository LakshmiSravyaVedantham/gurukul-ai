"""
Download all models needed for the Gurukul AI pipeline into ComfyUI.
Run once before using wan_animate.py.

Models downloaded:
  - LTX Video 2B distilled FP8       (~4.2 GB)  — fast fallback
  - LTX Video 13B 0.9.7 dev FP8      (~13 GB)   — high quality primary
  - Wan 2.1 Fun InP 1.3B BF16        (~2.6 GB)  — start+end frame I2V
  - T5-XXL FP8 text encoder          (~4.6 GB)  — shared by LTX models
  - UMT5-XXL FP8 text encoder        (~4.6 GB)  — used by Wan 2.1
  - Wan 2.1 VAE                      (~0.4 GB)  — used by Wan 2.1
  - CLIP Vision H                    (~0.6 GB)  — used by Wan 2.1 Fun InP

Usage:
    python3 download_models.py
    python3 download_models.py --comfyui /path/to/ComfyUI
"""

import argparse, shutil, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", default=None, help="Path to ComfyUI directory")
    p.add_argument("--skip-ltx13b", action="store_true", help="Skip LTX 13B download (large)")
    args = p.parse_args()

    if args.comfyui:
        comfy = Path(args.comfyui)
    else:
        candidates = [
            Path("/Volumes/bujji1/sravya/ComfyUI"),
            Path.home() / "ComfyUI",
            Path.cwd().parent / "ComfyUI",
        ]
        comfy = next((c for c in candidates if c.exists()), None)
        if not comfy:
            print("Could not find ComfyUI. Pass --comfyui /path/to/ComfyUI")
            sys.exit(1)

    print(f"ComfyUI: {comfy}")

    checkpoints   = comfy / "models" / "checkpoints"
    diff_models   = comfy / "models" / "diffusion_models"
    text_encoders = comfy / "models" / "text_encoders"
    vae_dir       = comfy / "models" / "vae"
    clip_vision   = comfy / "models" / "clip_vision"
    for d in [checkpoints, diff_models, text_encoders, vae_dir, clip_vision]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    def dl(repo, filename, dest_dir, label, size_hint=""):
        dest = dest_dir / filename
        if dest.exists():
            print(f"  Already exists: {filename}")
            return dest
        print(f"  Downloading {label} {size_hint}...")
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest_dir))
        print(f"  Saved: {dest}")
        return dest

    # ── LTX Video 2B distilled FP8 (fast fallback) ───────────────────────────
    print("\n[1/7] LTX Video 2B distilled FP8")
    dl("Lightricks/LTX-Video",
       "ltxv-2b-0.9.8-distilled-fp8.safetensors",
       checkpoints, "LTX 2B", "(~4.2 GB)")

    # ── LTX Video 13B 0.9.7 dev FP8 (high quality primary) ──────────────────
    print("\n[2/7] LTX Video 13B 0.9.7 dev FP8 (primary animation model)")
    if args.skip_ltx13b:
        print("  Skipped (--skip-ltx13b)")
    else:
        dl("Lightricks/LTX-Video",
           "ltxv-13b-0.9.7-dev-fp8.safetensors",
           checkpoints, "LTX 13B", "(~13 GB)")

    # ── T5-XXL FP8 (shared LTX text encoder) ─────────────────────────────────
    print("\n[3/7] T5-XXL FP8 text encoder")
    t5_dest = text_encoders / "t5xxl_fp8_e4m3fn.safetensors"
    if t5_dest.exists():
        print(f"  Already exists: {t5_dest.name}")
    else:
        import glob
        cache_hits = glob.glob(
            str(Path.home() / ".cache/huggingface/hub/**/t5xxl_fp8_e4m3fn.safetensors"),
            recursive=True
        )
        if cache_hits:
            print(f"  Found in HF cache — copying")
            shutil.copy2(cache_hits[0], str(t5_dest))
        else:
            dl("comfyanonymous/flux_text_encoders",
               "t5xxl_fp8_e4m3fn.safetensors",
               text_encoders, "T5-XXL FP8", "(~4.6 GB)")

    # ── UMT5-XXL FP8 (Wan 2.1 text encoder) ──────────────────────────────────
    print("\n[4/7] UMT5-XXL FP8 text encoder (Wan 2.1)")
    dl("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
       "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
       text_encoders, "UMT5-XXL FP8", "(~4.6 GB)")

    # ── Wan 2.1 VAE ───────────────────────────────────────────────────────────
    print("\n[5/7] Wan 2.1 VAE")
    dl("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
       "split_files/vae/wan_2.1_vae.safetensors",
       vae_dir, "Wan 2.1 VAE", "(~0.4 GB)")

    # ── CLIP Vision H (Wan 2.1 Fun InP image encoder) ────────────────────────
    print("\n[6/7] CLIP Vision H (Wan 2.1 Fun InP)")
    dl("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
       "split_files/clip_vision/clip_vision_h.safetensors",
       clip_vision, "CLIP Vision H", "(~0.6 GB)")

    # ── Wan 2.1 Fun InP 1.3B (start+end frame I2V) ───────────────────────────
    print("\n[7/7] Wan 2.1 Fun InP 1.3B BF16")
    dl("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
       "split_files/diffusion_models/wan2.1_fun_inp_1.3B_bf16.safetensors",
       diff_models, "Wan Fun InP 1.3B", "(~2.6 GB)")

    print("\nAll models ready!")
    print(f"Start ComfyUI: cd {comfy} && python3 main.py --port 8288 --preview-method none")


if __name__ == "__main__":
    main()

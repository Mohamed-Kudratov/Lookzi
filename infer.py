#!/usr/bin/env python
"""Headless inference for Layering VTON.

Gradio needs a browser; over SSH there isn't one. This runs the pipeline
directly so a pod can be verified, timed and iterated on from a terminal.

    # single run
    python infer.py --person assets/person_1.png --garment assets/pants.png \
        --mode swap --description "swap the deep blue jeans for dark wash jeans"

    # smoke test: the three bundled examples, into ./outputs
    python infer.py --examples

    # time the model without writing anything useful
    python infer.py --examples --steps 8
"""
import argparse
import os
import sys
import time

import torch
from PIL import Image

from pipeline import LayeringVTONPipeline, detect_device, detect_dtype, total_vram_gb
from utils import process_inputs

# The three examples from the upstream Gradio demo.
EXAMPLES = [
    ("assets/person_1.png", "assets/pants.png",   "swap", "swap the deep blue jeans for dark wash jeans"),
    ("assets/person_2.png", "assets/sweater.png", "add",  "add a light gray turtleneck sweater"),
    ("assets/person_3.png", "assets/coat.png",    "add",  "add a black leather jacket"),
]


def build_parser():
    p = argparse.ArgumentParser(description="Layering VTON, headless")
    p.add_argument("--person", help="path to the person image")
    p.add_argument("--garment", help="path to the garment image")
    p.add_argument("--pose", default=None, help="optional custom pose image; extracted with DWPose if omitted")
    p.add_argument("--mode", default="swap", choices=["swap", "add"])
    p.add_argument("--description", default="", help='e.g. "swap the beige leggings for dark wash jeans"')
    p.add_argument("--examples", action="store_true", help="run the three bundled examples instead")

    p.add_argument("--steps", type=int, default=40, help="inference steps (paper default: 40)")
    p.add_argument("--cfg", type=float, default=4.0, help="true CFG scale; 1.0 skips the negative pass (~2x faster)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model", default=os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509"),
                   help="HF repo id; use ovedrive/Qwen-Image-Edit-2509-4bit under 40 GB of VRAM")
    p.add_argument("--lora", default="./weights")
    p.add_argument("--out", default=None, help="output path (single run) or directory (--examples)")
    p.add_argument("--low-vram", dest="low_vram", action="store_true", default=None,
                   help="force sequential component loading")
    p.add_argument("--no-low-vram", dest="low_vram", action="store_false")
    return p


def main():
    args = build_parser().parse_args()

    if not args.examples and not (args.person and args.garment):
        build_parser().error("give --person and --garment, or use --examples")

    device = detect_device()
    if device == "cpu":
        print("No CUDA GPU. This is a 20B model; it will not run on CPU.", file=sys.stderr)
        return 1

    vram = total_vram_gb()
    print(f"GPU:   {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    print(f"dtype: {detect_dtype(device)}")
    print(f"model: {args.model}")

    # 57.7 GB of bf16 weights against a 4-bit build at ~17 GB -- warn rather than
    # let it fail deep inside the transformer load.
    if vram < 40 and "4bit" not in args.model.lower():
        print(f"\nWARNING: {vram:.0f} GB of VRAM with the full bf16 model. Expect an OOM.\n"
              f"         Use --model ovedrive/Qwen-Image-Edit-2509-4bit\n", file=sys.stderr)

    t0 = time.time()
    pipe = LayeringVTONPipeline(args.model, args.lora, low_vram=args.low_vram)
    print(f"\nPipeline ready in {time.time() - t0:.0f}s")

    if args.examples:
        jobs = EXAMPLES
        outdir = args.out or "outputs"
        os.makedirs(outdir, exist_ok=True)
    else:
        jobs = [(args.person, args.garment, args.mode, args.description)]
        outdir = None

    failures = 0
    for i, (person_path, garment_path, mode, description) in enumerate(jobs, 1):
        label = f"[{i}/{len(jobs)}] {os.path.basename(person_path)} + {os.path.basename(garment_path)}"
        print(f"\n{label}\n  mode={mode!r} steps={args.steps} cfg={args.cfg} seed={args.seed}")
        print(f"  {description!r}")

        try:
            person = Image.open(person_path)
            garment = Image.open(garment_path)
            pose = Image.open(args.pose) if args.pose else None

            t = time.time()
            pp, pg, ppose = process_inputs(person, garment, pose)
            print(f"  preprocessing: {time.time() - t:.1f}s")

            t = time.time()
            result = pipe(
                person_img=pp,
                garment_img=pg,
                pose_img=ppose,
                description=description,
                mode=mode,
                num_inference_steps=args.steps,
                true_cfg_scale=args.cfg,
                seed=args.seed,
            )
            elapsed = time.time() - t

            if outdir:
                out_path = os.path.join(outdir, f"{i:02d}_{mode}_{os.path.basename(person_path)}")
            else:
                out_path = args.out or "result.png"
            result.save(out_path)

            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  generated in {elapsed:.0f}s ({elapsed / args.steps:.1f}s/step), peak VRAM {peak:.1f} GB")
            print(f"  -> {out_path}")

        except torch.cuda.OutOfMemoryError:
            failures += 1
            print("  OOM. Try --low-vram, fewer --steps, --cfg 1.0, or the 4-bit model.", file=sys.stderr)
            torch.cuda.empty_cache()
        except Exception as exc:
            failures += 1
            print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print(f"\n{len(jobs) - failures}/{len(jobs)} succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

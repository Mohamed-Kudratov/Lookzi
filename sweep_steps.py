#!/usr/bin/env python
"""Find the cheapest sampling settings that still match the reference output.

40 steps with CFG 4.0 costs two transformer passes per step on a 20B model.
Whether that is *necessary* is measurable, not a matter of taste: generate the
same input at several settings with the same seed, then compare each against the
40-step reference.

The comparison is the point. You do not need to look at the images to learn that
28 steps is indistinguishable from 40 -- SSIM says so, and you keep the 30% you
would otherwise burn on every render.

    python sweep_steps.py --person assets/person_1.png --garment assets/pants.png \
        --mode swap --description "swap the deep blue jeans for dark wash jeans"

Rules of thumb for the SSIM column, against the 40-step reference:
    > 0.99   indistinguishable -- take the speedup
    > 0.97   very close; check one by eye before committing
    < 0.95   visibly different; the extra steps are doing something
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from pipeline import LayeringVTONPipeline
from utils import process_inputs

# (steps, true_cfg_scale, lightning). The reference must be first.
# lightning=None is the undistilled path; 4 or 8 stacks the distillation LoRA.
DEFAULT_GRID = [
    (40, 4.0, None),   # reference: the paper's default, 80 transformer passes
    (24, 4.0, None),   # fewer steps alone
    (24, 1.0, None),   # CFG 1.0 skips the negative pass -- half the work per step
    (8,  1.0, 8),      # Lightning 8-step: 8 passes
    (4,  1.0, 4),      # Lightning 4-step: 4 passes
]


def ssim(a: Image.Image, b: Image.Image) -> float:
    """Global SSIM on greyscale. No scikit-image dependency."""
    x = np.asarray(a.convert("L"), dtype=np.float64)
    y = np.asarray(b.convert("L"), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(
        ((2 * mx * my + C1) * (2 * cov + C2)) / ((mx**2 + my**2 + C1) * (vx + vy + C2))
    )


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.int16)
    y = np.asarray(b.convert("RGB"), dtype=np.int16)
    return float(np.abs(x - y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="assets/person_1.png")
    ap.add_argument("--garment", default="assets/pants.png")
    ap.add_argument("--mode", default="swap", choices=["swap", "add"])
    ap.add_argument("--description", default="swap the deep blue jeans for dark wash jeans")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509"))
    ap.add_argument("--lora", default="./weights")
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--grid", default=None,
                    help='override, e.g. "40:4.0,24:4.0,24:1.0"')
    args = ap.parse_args()

    grid = DEFAULT_GRID
    if args.grid:
        grid = []
        for item in args.grid.split(","):
            parts = item.split(":")
            s, c = int(parts[0]), float(parts[1])
            light = int(parts[2]) if len(parts) > 2 and parts[2] not in ("", "none") else None
            grid.append((s, c, light))

    os.makedirs(args.out, exist_ok=True)

    # Lightning needs a different scheduler and an extra adapter, so each
    # distillation setting is its own pipeline -- and at 57.7 GB each, two do not
    # fit on an 80 GB card. Group the grid by setting, load once per group, and
    # free before the next. First-appearance order is preserved so the reference
    # still runs first.
    groups = []
    for entry in grid:
        for lightning, entries in groups:
            if lightning == entry[2]:
                entries.append(entry)
                break
        else:
            groups.append((entry[2], [entry]))

    pp, pg, ppose = process_inputs(
        Image.open(args.person), Image.open(args.garment), None
    )

    rows, reference = [], None
    for lightning, entries in groups:
        print(f"\n--- loading pipeline (lightning={lightning}) ---")
        pipe = LayeringVTONPipeline(args.model, args.lora, lightning=lightning)

        for steps, cfg, _ in entries:
            tag = f"s{steps}_cfg{cfg}" + (f"_L{lightning}" if lightning else "")
            t = time.time()
            img = pipe(
                person_img=pp, garment_img=pg, pose_img=ppose,
                description=args.description, mode=args.mode,
                num_inference_steps=steps, true_cfg_scale=cfg, seed=args.seed,
            )
            elapsed = time.time() - t
            path = os.path.join(args.out, f"{tag}.png")
            img.save(path)

            if reference is None:
                reference = img
                s, d = 1.0, 0.0
            else:
                s, d = ssim(reference, img), mean_abs_diff(reference, img)

            # Passes through the transformer: CFG > 1 runs the negative branch too.
            passes = steps * (2 if cfg > 1 else 1)
            rows.append({
                "steps": steps, "cfg": cfg, "lightning": lightning,
                "seconds": round(elapsed, 1),
                "transformer_passes": passes, "ssim_vs_reference": round(s, 4),
                "mean_abs_diff": round(d, 2), "path": path,
            })
            print(f"  {tag:20} {elapsed:6.1f}s  {passes:3d} passes  ssim={s:.4f}  diff={d:.2f}")

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    base = rows[0]["seconds"]
    print(f"\n{'setting':22}{'time':>8}{'speedup':>9}{'passes':>8}{'SSIM':>9}{'verdict':>26}")
    for r in rows:
        speed = base / r["seconds"]
        if r is rows[0]:
            verdict = "reference"
        elif r["ssim_vs_reference"] > 0.99:
            verdict = "indistinguishable"
        elif r["ssim_vs_reference"] > 0.97:
            verdict = "very close, eyeball it"
        elif r["ssim_vs_reference"] > 0.95:
            verdict = "slightly different"
        else:
            verdict = "visibly different"
        name = f"s{r['steps']}/cfg{r['cfg']}" + (f"/L{r['lightning']}" if r["lightning"] else "")
        print(f"  {name:<20}{r['seconds']:>7.1f}s{speed:>8.2f}x{r['transformer_passes']:>8}"
              f"{r['ssim_vs_reference']:>9.4f}{verdict:>26}")

    out = os.path.join(args.out, "sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": rows}, f, indent=2)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

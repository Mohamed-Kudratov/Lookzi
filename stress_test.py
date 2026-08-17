#!/usr/bin/env python
"""Run every sampling setting across several inputs, modes and seeds.

The single-image sweep said 24 steps at CFG 1.0 was indistinguishable from the
40-step reference (SSIM 0.9996) and that Lightning was far away (0.71). Looking
at them reversed that: Lightning was the better image. SSIM measures distance
from the reference, and the reference is not necessarily the best output -- so
the decision needs more than one image, more than one seed, and eyes.

This produces the evidence: N inputs x M settings x K seeds, timed, checked for
outright failure, and assembled into one contact sheet per input so the
comparison is a glance rather than a file hunt.

    python stress_test.py --seeds 3 --out /workspace/stress
    python view_sweep.py --dir /workspace/stress
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from pipeline import LayeringVTONPipeline
from utils import process_inputs

# (label, steps, cfg, lightning)
SETTINGS = [
    ("ref",   40, 4.0, None),
    ("s24",   24, 1.0, None),
    ("L8",     8, 1.0, 8),
    ("L4",     4, 1.0, 4),
]

INPUTS = [
    ("person_1+pants",   "assets/person_1.png", "assets/pants.png",   "swap",
     "swap the deep blue jeans for dark wash jeans"),
    ("person_2+sweater", "assets/person_2.png", "assets/sweater.png", "add",
     "add a light gray turtleneck sweater"),
    ("person_3+coat",    "assets/person_3.png", "assets/coat.png",    "add",
     "add a black leather jacket"),
]


def looks_broken(img):
    """Cheap detector for the failures that matter at scale."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    if not np.isfinite(a).all():
        return "non-finite pixels"
    if a.std() < 3.0:
        return "flat image (near-constant)"
    if a.mean() < 8.0:
        return "almost black"
    if a.mean() > 247.0:
        return "almost white"
    return None


def border_change_pct(before, after, frac=0.10, thresh=8):
    """Share of border-band pixels that moved -- a proxy for background damage.

    Mirrors the A1 criterion in the VTON project's eval/metrics.py: the outer
    band is nearly always background, so a model repainting it shows up here.
    """
    a = np.asarray(before.convert("RGB"), dtype=np.int16)
    b = np.asarray(after.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return None
    h, w = a.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    by, bx = max(1, int(h * frac)), max(1, int(w * frac))
    m[:by, :] = m[-by:, :] = True
    m[:, :bx] = m[:, -bx:] = True
    d = np.abs(a - b).max(axis=2)[m]
    return round(float((d > thresh).mean() * 100), 2)


def contact_sheet(rows, labels, out_path, pad=8):
    """One strip per input: settings across, seeds down."""
    if not rows or not rows[0]:
        return
    w, h = rows[0][0].size
    cols = max(len(r) for r in rows)
    sheet = Image.new("RGB", (cols * (w + pad) + pad, len(rows) * (h + pad) + pad + 28),
                      (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(labels[:cols]):
        draw.text((pad + c * (w + pad) + 4, 6), label, fill=(20, 20, 20))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            sheet.paste(img, (pad + c * (w + pad), 28 + pad + r * (h + pad)))
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="stress")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509"))
    ap.add_argument("--lora", default="./weights")
    ap.add_argument("--only", default=None,
                    help="comma-separated setting labels, e.g. 'L8,s24'")
    args = ap.parse_args()

    settings = SETTINGS
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        settings = [s for s in SETTINGS if s[0] in keep]

    os.makedirs(args.out, exist_ok=True)
    seeds = [42 + i * 1000 for i in range(args.seeds)]

    prepared = {}
    for name, person, garment, mode, desc in INPUTS:
        pp, pg, ppose = process_inputs(Image.open(person), Image.open(garment), None)
        prepared[name] = (pp, pg, ppose, mode, desc)
        print(f"prepared {name}")

    results, images = [], {}

    # One pipeline load per distillation setting, never two resident at once.
    by_lightning = {}
    for label, steps, cfg, lightning in settings:
        by_lightning.setdefault(lightning, []).append((label, steps, cfg))

    for lightning, entries in by_lightning.items():
        print(f"\n=== pipeline lightning={lightning} ===")
        pipe = LayeringVTONPipeline(args.model, args.lora, lightning=lightning)

        for label, steps, cfg in entries:
            for name, (pp, pg, ppose, mode, desc) in prepared.items():
                for seed in seeds:
                    tag = f"{name}__{label}__seed{seed}"
                    t = time.time()
                    try:
                        img = pipe(person_img=pp, garment_img=pg, pose_img=ppose,
                                   description=desc, mode=mode,
                                   num_inference_steps=steps, true_cfg_scale=cfg,
                                   seed=seed)
                        elapsed = time.time() - t
                        broken = looks_broken(img)
                        bg = border_change_pct(pp, img)
                        img.save(os.path.join(args.out, tag + ".png"))
                        images[(name, label, seed)] = img
                    except Exception as exc:
                        elapsed = time.time() - t
                        broken, bg, img = f"{type(exc).__name__}: {exc}", None, None
                        print(f"  {tag}  FAILED  {broken}")

                    results.append({
                        "input": name, "setting": label, "seed": seed,
                        "steps": steps, "cfg": cfg, "lightning": lightning,
                        "seconds": round(elapsed, 1),
                        "passes": steps * (2 if cfg > 1 else 1),
                        "broken": broken, "bg_changed_pct": bg,
                    })
                    if img is not None:
                        print(f"  {tag:44} {elapsed:6.1f}s  bg{bg:>6}%"
                              + ("  BROKEN: " + broken if broken else ""))

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    labels = [s[0] for s in settings]
    for name in prepared:
        rows = []
        for seed in seeds:
            row = [images[(name, lab, seed)] for lab in labels if (name, lab, seed) in images]
            if row:
                rows.append(row)
        contact_sheet(rows, labels, os.path.join(args.out, f"sheet__{name}.png"))
        print(f"sheet__{name}.png")

    with open(os.path.join(args.out, "stress.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "results": results}, f, indent=2)

    print(f"\n{'setting':10}{'runs':>6}{'failed':>8}{'median s':>10}{'mean bg%':>10}")
    for label in labels:
        rows = [r for r in results if r["setting"] == label]
        ok = [r for r in rows if not r["broken"]]
        times = sorted(r["seconds"] for r in ok)
        bgs = [r["bg_changed_pct"] for r in ok if r["bg_changed_pct"] is not None]
        med = times[len(times) // 2] if times else 0
        print(f"  {label:<8}{len(rows):>6}{len(rows) - len(ok):>8}{med:>10.1f}"
              f"{(sum(bgs) / len(bgs) if bgs else 0):>10.2f}")

    print(f"\n  -> {args.out}/stress.json and {len(prepared)} contact sheets")
    return 0


if __name__ == "__main__":
    sys.exit(main())

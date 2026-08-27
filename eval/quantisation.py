#!/usr/bin/env python3
"""Does the 4-bit checkpoint cost us anything visible?

The whole infrastructure argument rests on this. bf16 is 57.7 GB and needs an
80 GB card at $1.39/hour; the published 4-bit checkpoint is 15.8 GB, fits a
48 GB A6000 at $0.33, fits the container disk so it loads off local NVMe
instead of a network volume, and is small enough that a new worker can be
useful in seconds rather than minutes -- which is what makes autoscaling
possible at all.

So the question is not "is 4-bit good enough in general". It is whether, on our
inputs, with our LoRA and our step count, the difference is one a seller would
notice. That is measurable, and guessing at it would decide weeks of work.

    python eval/quantisation.py --pairs 20

Generates the same job twice, once per checkpoint, from identical inputs and
seeds, and reports three things per pair: whether the face survived, whether
the garment survived, and how long each took.

Loading two 20B models one after another needs the memory freed in between,
which is why this runs them in phases rather than side by side.
"""
import argparse
import gc
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "elements"))

BF16 = "Qwen/Qwen-Image-Edit-2509"
NF4 = "ovedrive/Qwen-Image-Edit-2509-4bit"


def _free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load(model_path, lora_dir, lightning):
    from pipeline import LayeringVTONPipeline
    t = time.time()
    pipe = LayeringVTONPipeline(model_path, lora_dir, lightning=lightning)
    return pipe, round(time.time() - t, 1)


def run_set(pipe, jobs, out_dir, tag):
    """Generate every job with one checkpoint, recording the time for each."""
    from PIL import Image
    from utils import process_inputs

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, job in enumerate(jobs):
        path = os.path.join(out_dir, f"{tag}_{i:02d}.png")
        if os.path.exists(path):
            rows.append({"i": i, "path": path, "seconds": None, "cached": True})
            continue
        person = Image.open(job["person"]).convert("RGB")
        garment = Image.open(job["garment"]).convert("RGB")
        pp, pg, ppose = process_inputs(person, garment, None)
        t = time.time()
        img = pipe(person_img=pp, garment_img=pg, pose_img=ppose,
                   description=job.get("description", "the garment"),
                   mode=job.get("mode", "upper"), seed=job["seed"])
        secs = round(time.time() - t, 2)
        img.save(path)
        rows.append({"i": i, "path": path, "seconds": secs, "cached": False})
        print(f"  {tag} {i:02d}  {secs:5.2f}s", flush=True)
    return rows


def identity_scores(pairs, person_paths):
    """ArcFace similarity: each output against the person it was meant to be.

    The same measurement the roster was checked with, so the numbers are
    comparable to the 0.684 already recorded for the two-stage method.
    """
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from identity import embed, load_app
    import numpy as np

    app = load_app()
    out = []
    for p in pairs:
        ref = embed(app, person_paths[p["i"]])
        a = embed(app, p["bf16"])
        b = embed(app, p["nf4"])
        row = {"i": p["i"]}
        row["bf16_identity"] = float(a @ ref) if (a is not None and ref is not None) else None
        row["nf4_identity"] = float(b @ ref) if (b is not None and ref is not None) else None
        # The two outputs against each other: how far 4-bit moved the picture,
        # regardless of whether either matches the reference well.
        row["between"] = float(a @ b) if (a is not None and b is not None) else None
        out.append(row)
    return out


def garment_scores(pairs, garment_paths):
    """How much of the garment survived, by colour and structure.

    A face metric says nothing about whether the jacket is still the same
    jacket, and that is what a seller is paying for. Histogram correlation over
    the clothed region is crude but it moves when the garment changes, which is
    what a comparison between two checkpoints needs.
    """
    import cv2
    import numpy as np

    def hist(path):
        img = cv2.imread(path)
        if img is None:
            return None
        # The middle band: torso rather than face or background.
        h, w = img.shape[:2]
        crop = img[int(h * 0.25):int(h * 0.70), int(w * 0.15):int(w * 0.85)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hh = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv2.normalize(hh, hh).flatten()

    out = []
    for p in pairs:
        g = hist(garment_paths[p["i"]])
        a, b = hist(p["bf16"]), hist(p["nf4"])
        row = {"i": p["i"]}
        row["bf16_garment"] = float(cv2.compareHist(g, a, cv2.HISTCMP_CORREL)) if g is not None and a is not None else None
        row["nf4_garment"] = float(cv2.compareHist(g, b, cv2.HISTCMP_CORREL)) if g is not None and b is not None else None
        out.append(row)
    return out


def build_jobs(heroes_dir, garments, limit):
    """Pairs drawn from the roster we already have, not from invented inputs."""
    import glob
    people = []
    for d in sorted(glob.glob(os.path.join(heroes_dir, "*"))):
        for f in sorted(glob.glob(os.path.join(d, "*.png")))[:1]:
            people.append(f)
    if not people:
        raise SystemExit(f"no hero images under {heroes_dir}")
    jobs = []
    for i in range(limit):
        jobs.append({"person": people[i % len(people)],
                     "garment": garments[i % len(garments)],
                     "mode": "upper", "seed": 1000 + i})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--heroes", default="/workspace/elements_out/heroes")
    ap.add_argument("--garments", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--out", default="/workspace/quant_eval")
    ap.add_argument("--lora", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--lightning", type=int, default=8)
    ap.add_argument("--only", choices=["bf16", "nf4"],
                    help="run one checkpoint now and the other later, "
                         "for a card that cannot hold bf16")
    args = ap.parse_args()

    import glob
    garments = sorted(g for g in glob.glob(os.path.join(args.garments, "*.png"))
                      if "person" not in os.path.basename(g).lower())
    if not garments:
        raise SystemExit(f"no garment images in {args.garments}")

    jobs = build_jobs(args.heroes, garments, args.pairs)
    print(f"{len(jobs)} pairs, {len(garments)} garments, "
          f"{len(set(j['person'] for j in jobs))} people\n")

    os.makedirs(args.out, exist_ok=True)
    timings = {}

    for tag, path in (("bf16", BF16), ("nf4", NF4)):
        if args.only and args.only != tag:
            continue
        print(f"--- {tag} ---", flush=True)
        pipe, load_secs = load(path, args.lora, args.lightning)
        timings[tag] = {"load": load_secs}
        rows = run_set(pipe, jobs, args.out, tag)
        done = [r["seconds"] for r in rows if r["seconds"] is not None]
        timings[tag]["per_image"] = round(sum(done) / len(done), 2) if done else None
        print(f"  load {load_secs}s, per image {timings[tag]['per_image']}s\n", flush=True)
        del pipe
        _free()

    have_both = all(os.path.exists(os.path.join(args.out, f"{t}_{i:02d}.png"))
                    for t in ("bf16", "nf4") for i in range(len(jobs)))
    if not have_both:
        print("one checkpoint still to run; re-run with --only for the other")
        json.dump(timings, open(os.path.join(args.out, "timings.json"), "w"), indent=2)
        return 0

    pairs = [{"i": i,
              "bf16": os.path.join(args.out, f"bf16_{i:02d}.png"),
              "nf4": os.path.join(args.out, f"nf4_{i:02d}.png")}
             for i in range(len(jobs))]

    print("--- measuring ---", flush=True)
    ident = identity_scores(pairs, [j["person"] for j in jobs])
    garm = garment_scores(pairs, [j["garment"] for j in jobs])

    def mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    report = {
        "pairs": len(pairs),
        "timings": timings,
        "identity": {"bf16": mean(ident, "bf16_identity"),
                     "nf4": mean(ident, "nf4_identity"),
                     "between": mean(ident, "between")},
        "garment": {"bf16": mean(garm, "bf16_garment"),
                    "nf4": mean(garm, "nf4_garment")},
        "rows": [dict(**a, **{k: v for k, v in b.items() if k != "i"})
                 for a, b in zip(ident, garm)],
    }
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=2)

    print(f"\n{'':14}{'bf16':>10}{'4-bit':>10}{'difference':>13}")
    for name, block in (("identity", report["identity"]), ("garment", report["garment"])):
        a, b = block["bf16"], block["nf4"]
        d = round(b - a, 4) if (a is not None and b is not None) else None
        print(f"  {name:12}{a:>10}{b:>10}{str(d):>13}")
    print(f"  {'load, s':12}{timings.get('bf16',{}).get('load','-'):>10}"
          f"{timings.get('nf4',{}).get('load','-'):>10}")
    print(f"  {'image, s':12}{timings.get('bf16',{}).get('per_image','-'):>10}"
          f"{timings.get('nf4',{}).get('per_image','-'):>10}")
    print(f"\n  the two outputs against each other: {report['identity']['between']}")
    print(f"  full detail -> {os.path.join(args.out, 'report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Does the 4-bit checkpoint cost us anything visible?

The infrastructure argument rests on this. bf16 is 57.7 GB and needs an 80 GB
card at $1.39/hour; the published 4-bit checkpoint is 15.8 GB, fits a 48 GB
A6000 at $0.33, fits a container disk so it loads off local NVMe instead of a
network volume, and is small enough that a new worker becomes useful in seconds
rather than minutes -- which is what makes autoscaling possible at all.

So the question is not whether 4-bit is good in general. It is whether, on our
inputs, with our LoRA at our step count, the difference is one a seller would
notice. Guessing decides weeks of work either way.

    python eval/quantisation.py --pairs 20

Generates every job twice, once per checkpoint, from identical inputs and
seeds, then reports three things: whether the face survived, whether the
garment survived, and what each cost in time.

The checkpoints are loaded one after another with the memory freed in between.
They never need to be resident together: what gets compared is the images, and
those are on disk.
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "elements"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

BF16 = "Qwen/Qwen-Image-Edit-2509"
NF4 = "ovedrive/Qwen-Image-Edit-2509-4bit"

# What each garment is, and where it goes. Running trousers through the "upper"
# prompt gives something incoherent from both checkpoints, which would make the
# pair agree for a reason having nothing to do with quantisation.
GARMENTS = {
    "coat.png": ("upper", "the coat"),
    "sweater.png": ("upper", "the sweater"),
    "pants.png": ("lower", "the trousers"),
}


def _free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_jobs(heroes_dir, assets_dir, limit):
    """Pairs drawn from the roster we already have, not from invented inputs.

    One frame per person, so the comparison spans faces instead of testing one
    face repeatedly, with garments cycling underneath.
    """
    people = []
    for d in sorted(glob.glob(os.path.join(heroes_dir, "*"))):
        frames = sorted(glob.glob(os.path.join(d, "*.png")))
        if frames:
            people.append(frames[0])
    if not people:
        raise SystemExit(f"no hero images under {heroes_dir}")

    garments = []
    for name, (mode, desc) in GARMENTS.items():
        path = os.path.join(assets_dir, name)
        if os.path.exists(path):
            garments.append((path, mode, desc))
    if not garments:
        raise SystemExit(f"none of {list(GARMENTS)} found in {assets_dir}")

    jobs = []
    for n in range(limit):
        path, mode, desc = garments[n % len(garments)]
        jobs.append({"person": people[n % len(people)], "garment": path,
                     "mode": mode, "description": desc, "seed": 1000 + n})
    return jobs


def load(model_path, lora_dir, lightning):
    from pipeline import LayeringVTONPipeline
    t = time.time()
    pipe = LayeringVTONPipeline(model_path, lora_dir, lightning=lightning)
    return pipe, round(time.time() - t, 1)


def run_set(pipe, jobs, out_dir, tag):
    """Generate every job with one checkpoint, recording the time for each.

    Images already on disk are kept, so an interrupted run resumes rather than
    paying for the same picture twice.
    """
    from PIL import Image
    from utils import process_inputs

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, job in enumerate(jobs):
        path = os.path.join(out_dir, f"{tag}_{i:02d}.png")
        if os.path.exists(path):
            rows.append({"i": i, "seconds": None})
            continue
        person = Image.open(job["person"]).convert("RGB")
        garment = Image.open(job["garment"]).convert("RGB")
        pp, pg, ppose = process_inputs(person, garment, None)
        t = time.time()
        img = pipe(person_img=pp, garment_img=pg, pose_img=ppose,
                   description=job["description"], mode=job["mode"],
                   seed=job["seed"])
        secs = round(time.time() - t, 2)
        img.save(path)
        rows.append({"i": i, "seconds": secs})
        print(f"  {tag} {i:02d}  {secs:6.2f}s  {job['mode']:6} "
              f"{os.path.basename(os.path.dirname(job['person']))}", flush=True)
    return rows


def identity_scores(pairs, person_paths):
    """ArcFace similarity: each output against the person it was meant to be.

    The measurement the roster was checked with, so these numbers sit on the
    same scale as the 0.684 already recorded for the two-stage method.
    """
    from identity import embed, load_app

    app = load_app()
    refs = {}
    out = []
    for p in pairs:
        i = p["i"]
        if i not in refs:
            refs[i] = embed(app, person_paths[i])
        ref, a, b = refs[i], embed(app, p["bf16"]), embed(app, p["nf4"])
        out.append({
            "i": i,
            "bf16_identity": float(a @ ref) if a is not None and ref is not None else None,
            "nf4_identity": float(b @ ref) if b is not None and ref is not None else None,
            # The two outputs against each other: how far 4-bit moved the
            # picture, regardless of whether either matches the reference.
            "between": float(a @ b) if a is not None and b is not None else None,
        })
    return out


def garment_scores(pairs, garment_paths):
    """How much of the garment survived, by colour and structure.

    A face metric says nothing about whether the jacket is still the same
    jacket, and that is what a seller is paying for. Histogram correlation over
    the clothed band is crude, but it moves when the garment changes, which is
    all a comparison between two checkpoints needs it to do.
    """
    import cv2

    def hist(path, band):
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        top, bottom = band
        crop = img[int(h * top):int(h * bottom), int(w * 0.15):int(w * 0.85)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hh = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv2.normalize(hh, hh).flatten()

    out = []
    for p in pairs:
        # Look where the garment actually is. Comparing a torso crop against a
        # photograph of trousers measures the background.
        band = (0.55, 0.95) if p["mode"] == "lower" else (0.25, 0.70)
        # The reference is a flat product shot, so read all of it.
        g = hist(garment_paths[p["i"]], (0.0, 1.0))
        a, b = hist(p["bf16"], band), hist(p["nf4"], band)
        out.append({
            "i": p["i"],
            "bf16_garment": float(cv2.compareHist(g, a, cv2.HISTCMP_CORREL))
            if g is not None and a is not None else None,
            "nf4_garment": float(cv2.compareHist(g, b, cv2.HISTCMP_CORREL))
            if g is not None and b is not None else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--heroes", default="/workspace/elements_out/heroes")
    ap.add_argument("--garments", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--out", default="/workspace/quant_eval")
    ap.add_argument("--lora", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--lightning", type=int, default=8)
    ap.add_argument("--only", choices=["bf16", "nf4"],
                    help="run one checkpoint now and the other later, for a "
                         "card that cannot hold bf16 at all")
    args = ap.parse_args()

    jobs = build_jobs(args.heroes, args.garments, args.pairs)
    print(f"{len(jobs)} pairs, "
          f"{len(set(j['garment'] for j in jobs))} garments, "
          f"{len(set(j['person'] for j in jobs))} people")
    print()

    os.makedirs(args.out, exist_ok=True)
    timings_path = os.path.join(args.out, "timings.json")
    timings = {}
    if os.path.exists(timings_path):
        with open(timings_path) as fh:
            timings = json.load(fh)

    for tag, path in (("bf16", BF16), ("nf4", NF4)):
        if args.only and args.only != tag:
            continue
        if all(os.path.exists(os.path.join(args.out, f"{tag}_{i:02d}.png"))
               for i in range(len(jobs))):
            print(f"--- {tag}: already generated ---\n", flush=True)
            continue
        print(f"--- {tag} ---", flush=True)
        pipe, load_secs = load(path, args.lora, args.lightning)
        rows = run_set(pipe, jobs, args.out, tag)
        fresh = [r["seconds"] for r in rows if r["seconds"] is not None]
        timings[tag] = {
            "load": load_secs,
            "per_image": round(sum(fresh) / len(fresh), 2) if fresh else None,
        }
        print(f"  load {load_secs}s, per image {timings[tag]['per_image']}s\n",
              flush=True)
        del pipe
        _free()
        with open(timings_path, "w") as fh:
            json.dump(timings, fh, indent=2)

    have_both = all(os.path.exists(os.path.join(args.out, f"{t}_{i:02d}.png"))
                    for t in ("bf16", "nf4") for i in range(len(jobs)))
    if not have_both:
        print("one checkpoint still to run; re-run with --only for the other")
        return 0

    pairs = [{"i": i, "mode": jobs[i]["mode"],
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
        "rows": [{**a, **{k: v for k, v in b.items() if k != "i"},
                  "mode": jobs[a["i"]]["mode"],
                  "person": os.path.basename(os.path.dirname(jobs[a["i"]]["person"])),
                  "garment": os.path.basename(jobs[a["i"]]["garment"])}
                 for a, b in zip(ident, garm)],
    }
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    def fmt(v):
        return "-" if v is None else str(v)

    print()
    print(f"{'':16}{'bf16':>10}{'4-bit':>10}{'difference':>13}")
    for name, block in (("identity", report["identity"]),
                        ("garment", report["garment"])):
        a, b = block["bf16"], block["nf4"]
        d = f"{b - a:+.4f}" if a is not None and b is not None else "-"
        print(f"  {name:14}{fmt(a):>10}{fmt(b):>10}{d:>13}")
    for name, key in (("load, s", "load"), ("image, s", "per_image")):
        a = timings.get("bf16", {}).get(key)
        b = timings.get("nf4", {}).get(key)
        print(f"  {name:14}{fmt(a):>10}{fmt(b):>10}")

    print(f"\n  the two outputs against each other: "
          f"{fmt(report['identity']['between'])}")
    print(f"  full detail -> {os.path.join(args.out, 'report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

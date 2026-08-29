#!/usr/bin/env python3
"""Read a run and say what happened, in numbers and in pictures.

    python -m bench.score --run stress1

Two outputs, and the second matters more than the first.

**Numbers.** Coarse on purpose. There is no face model and no CLIP here, so
nothing pretends to score beauty. What these measure are the specific ways this
system has actually been seen to fail:

  a packshot that cut everything away, or cut nothing
  a packshot that did not land on white
  a try-on that changed the background -- the seller's room replacing the
    studio, which was measured and fixed once and can come back
  a try-on that did nothing at all: the torso comes out the same as it went in

Every one of these came from a real defect, not from a list of metrics someone
thought sounded thorough. FID and LPIPS would score all four as fine.

**Pictures.** Contact sheets of input beside output, because the eye is the
ground truth for everything else and no number here replaces looking.
"""
import argparse
import json
import os
import statistics

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, size=(256, 342)):
    im = Image.open(path).convert("RGB")
    return np.asarray(im.resize(size, Image.LANCZOS)).astype(np.float32)


def border(a, frac=0.06):
    """The frame around the edge, where the subject almost never is."""
    h, w = a.shape[:2]
    m = max(2, int(min(h, w) * frac))
    return np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3),
                           a[:, :m].reshape(-1, 3), a[:, -m:].reshape(-1, 3)])


def torso(a):
    """The band a garment lands on. Rough, and the same rough band every time."""
    h, w = a.shape[:2]
    return a[int(h * 0.28):int(h * 0.58), int(w * 0.28):int(w * 0.72)]


def measure_packshot(src, out):
    a, b = load(src), load(out)
    edge = border(b)
    ground = edge.mean(axis=0)
    # How far each pixel is from the paper it was placed on. Anything well
    # clear of it is garment; the fraction says whether the cut-out kept the
    # item, kept the whole photograph, or kept nothing.
    dist = np.linalg.norm(b - ground, axis=2)
    subject = float((dist > 40).mean())
    return {
        "white_mean": round(float(edge.mean()), 1),
        "white_sd": round(float(edge.std()), 1),
        "subject_frac": round(subject, 3),
        # Did the colour survive the cut? The input's middle against the
        # output's subject, in mean RGB. A big shift is the correction washing
        # a navy garment to grey, which has happened.
        "colour_shift": round(float(np.linalg.norm(
            torso(a).reshape(-1, 3).mean(axis=0)
            - b[dist > 40].mean(axis=0) if subject > 0.01 else 0.0)), 1),
    }


def measure_tryon(model_src, garment_src, out):
    m, g, b = load(model_src), load(garment_src), load(out)
    return {
        # The studio backdrop replaced by the seller's wardrobe. Measured once
        # on a real failure; cheap to keep watching.
        "bg_change": round(float(np.abs(border(b) - border(m)).mean()), 1),
        # High is good here: the torso is where the garment goes, and a result
        # that matches the model's own torso is a tool that did nothing.
        "torso_change": round(float(np.abs(torso(b) - torso(m)).mean()), 1),
        # Whether what landed there is the colour of the garment that was sent.
        "garment_colour_gap": round(float(np.linalg.norm(
            torso(b).reshape(-1, 3).mean(axis=0)
            - torso(g).reshape(-1, 3).mean(axis=0))), 1),
    }


def sheets(rows, outdir, per_sheet=20):
    """Input beside output, in batches small enough to look at."""
    made = []
    by_tool = {}
    for r in rows:
        if r.get("image"):
            by_tool.setdefault(r["tool"], []).append(r)
    for tool, rs in by_tool.items():
        for start in range(0, len(rs), per_sheet):
            batch = rs[start:start + per_sheet]
            cell, pad, lab = 150, 6, 12
            cols = 5
            wide = 3 if tool != "packshot" else 2
            rows_n = (len(batch) + cols - 1) // cols
            W = cols * (cell * wide + pad) + pad
            H = rows_n * (cell + lab + pad) + pad
            s = Image.new("RGB", (W, H), (15, 23, 42))
            d = ImageDraw.Draw(s)
            for n, r in enumerate(batch):
                parts = [r["source"]]
                if r.get("model_source"):
                    parts.insert(0, r["model_source"])
                parts.append(os.path.join(outdir, "img", r["image"]))
                x0 = pad + (n % cols) * (cell * wide + pad)
                y0 = pad + (n // cols) * (cell + lab + pad)
                d.text((x0 + 1, y0), r["id"].split("/")[-1][:26], fill=(148, 163, 184))
                for k, p in enumerate(parts[:wide]):
                    try:
                        im = Image.open(p).convert("RGB")
                    except Exception:                         # noqa: BLE001
                        continue
                    im.thumbnail((cell, cell), Image.LANCZOS)
                    s.paste(im, (x0 + k * cell + (cell - im.width) // 2, y0 + lab))
            fn = os.path.join(outdir, f"sheet_{tool}_{start // per_sheet + 1}.png")
            s.save(fn)
            made.append(fn)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="stress1")
    args = ap.parse_args()
    outdir = os.path.join(HERE, "runs", args.run)
    with open(os.path.join(outdir, "results.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data["rows"]

    for r in rows:
        if not r.get("image"):
            continue
        out = os.path.join(outdir, "img", r["image"])
        try:
            if r["tool"] == "packshot":
                r["m"] = measure_packshot(r["source"], out)
            else:
                r["m"] = measure_tryon(r["model_source"], r["source"], out)
        except Exception as exc:                              # noqa: BLE001
            r["m"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Reported apart, always. The men's photographs are 3024x4032 and the
    # women's are 400x533; a difference between them may be the compression.
    print(f"\n  run {args.run}  commit {data.get('commit')}\n")
    keys = {}
    for r in rows:
        keys.setdefault((r["tool"], r["gender"], r["category"]), []).append(r)
    print(f"  {'tool':16} {'who':6} {'category':11} {'n':>3} {'done':>5} "
          f"{'med s':>7}  measurements")
    for k in sorted(keys):
        rs = keys[k]
        ok = [r for r in rs if r["status"] == "done"]
        secs = [r["seconds"] for r in ok if r.get("seconds")]
        ms = [r["m"] for r in ok if isinstance(r.get("m"), dict) and "error" not in r["m"]]
        agg = ""
        if ms:
            names = [n for n in ms[0] if n != "error"]
            agg = "  ".join(
                f"{n}={statistics.median([m[n] for m in ms]):.2f}" for n in names)
        print(f"  {k[0]:16} {k[1]:6} {k[2]:11} {len(rs):3} {len(ok):5} "
              f"{statistics.median(secs) if secs else 0:7.2f}  {agg}")

    bad = [r for r in rows if r["status"] != "done"]
    if bad:
        print(f"\n  {len(bad)} did not finish:")
        for r in bad[:12]:
            print(f"    {r['id'][:52]:54} {r['status']} {str(r.get('error'))[:60]}")

    with open(os.path.join(outdir, "scored.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    made = sheets(rows, outdir)
    print(f"\n  {len(made)} contact sheets in bench/runs/{args.run}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

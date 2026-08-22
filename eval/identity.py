#!/usr/bin/env python3
"""Does the face survive the variation grid?

The roster is built in two stages: one hero image fixes the identity, then the
try-on model re-poses and re-lights that person thirty times. The whole question
for Phase 1 is whether the person in image 27 is still the person in the hero,
because that is what an identity LoRA would be trained to guarantee. If the
two-stage method already holds, the LoRA is twenty minutes of training per face
and an entire pipeline that buys nothing.

Looking at the images cannot settle it. Faces that a person calls "the same"
routinely fail a face matcher, and vice versa -- and the answer decides weeks of
work, so it should not rest on an impression.

What is measured. ArcFace embeddings (insightface buffalo_l), cosine similarity.
Two numbers per roster face:

  within   each variation against its own hero -- how well identity transferred
  between  each variation against *other* faces' heroes -- the control

`within` alone means nothing. A model that returned the same generic face for
every roster entry would score a perfect `within` and be useless; only the gap
between the two says identity is both preserved and distinct.

Thresholds. ArcFace's conventional same-person cut is ~0.4 at 1e-4 FAR on
frontal photos. Above 0.5 is a confident match. These are synthetic images at
angles a verification benchmark never contains, so treat the margin as the
signal, not the absolute.

    python eval/identity.py --dir /workspace/elements_out
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def load_app():
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("needs insightface:\n  pip install insightface onnxruntime-gpu",
              file=sys.stderr)
        raise SystemExit(3)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def embed(app, path):
    """The largest face in the image, or None.

    Largest, not first: a variation shot at a distance can catch a reflection or
    a bystander the generator invented, and the subject is the biggest face in
    every frame the grid produces.
    """
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    v = face.normed_embedding
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/elements_out")
    ap.add_argument("--picks", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "elements", "picks.txt"))
    ap.add_argument("--out", default="/workspace/identity.csv")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "elements"))
    from hero import read_picks
    picks = read_picks(args.picks)

    app = load_app()

    # Heroes first: they are the reference every variation is measured against.
    heroes = {}
    for fid, idx in picks.items():
        p = os.path.join(args.dir, "heroes", fid, f"{idx}.png")
        v = embed(app, p)
        if v is None:
            print(f"  WARNING no face found in hero {fid}/{idx}.png")
            continue
        heroes[fid] = v
    print(f"heroes embedded: {len(heroes)}/{len(picks)}\n")
    if len(heroes) < 2:
        raise SystemExit("need at least two heroes for the between-identity control")

    rows = []
    models = os.path.join(args.dir, "models")
    for fid in sorted(heroes):
        paths = sorted(p for p in os.listdir(models)
                       if p.startswith(fid + "__") and p.endswith(".png"))
        for name in paths:
            v = embed(app, os.path.join(models, name))
            if v is None:
                rows.append(dict(face=fid, image=name, within="", between="",
                                 note="no face detected"))
                continue
            within = float(v @ heroes[fid])
            # Highest similarity to any *other* hero: the strictest control,
            # since a roster fails if a variation matches any other member
            # better than a stranger would.
            between = max(float(v @ heroes[o]) for o in heroes if o != fid)
            rows.append(dict(face=fid, image=name, within=round(within, 4),
                             between=round(between, 4), note=""))

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["face", "image", "within", "between", "note"])
        w.writeheader()
        w.writerows(rows)

    by_face = defaultdict(list)
    undetected = defaultdict(int)
    for r in rows:
        if r["note"]:
            undetected[r["face"]] += 1
        else:
            by_face[r["face"]].append((r["within"], r["between"]))

    print(f"{'face':22} {'n':>3} {'within':>8} {'min':>7} {'between':>8} {'margin':>7} {'<0.4':>5}")
    all_w, all_b = [], []
    for fid in sorted(by_face):
        w_ = [x for x, _ in by_face[fid]]
        b_ = [y for _, y in by_face[fid]]
        all_w += w_
        all_b += b_
        weak = sum(1 for x in w_ if x < 0.4)
        print(f"{fid:22} {len(w_):3d} {np.mean(w_):8.3f} {min(w_):7.3f} "
              f"{np.mean(b_):8.3f} {np.mean(w_) - np.mean(b_):7.3f} {weak:5d}"
              + (f"   ({undetected[fid]} no face)" if undetected[fid] else ""))
    print("-" * 68)
    print(f"{'ALL':22} {len(all_w):3d} {np.mean(all_w):8.3f} {min(all_w):7.3f} "
          f"{np.mean(all_b):8.3f} {np.mean(all_w) - np.mean(all_b):7.3f} "
          f"{sum(1 for x in all_w if x < 0.4):5d}")
    print(f"\nper-image detail -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Upload each roster model's photograph and record where it went.

A model without a photograph cannot be chosen: product-to-model uses that image
as the person the garment goes onto. The roster rows come from catalog.py via
seed_models; the pictures come from wherever they were generated.

    python -m service.seed_heroes --dir /workspace/elements_out/heroes

Expects one directory per model id, each holding the chosen candidate:

    heroes/f_cauz_20s_avg/002.png

which is the layout elements/hero.py already writes. Which candidate was
chosen is read from elements/picks.txt, so the choice lives in one place.

For local development, when the real photographs are on a pod that is switched
off:

    python -m service.seed_heroes --placeholder assets

Every model then points at one of a handful of stand-in images. Enough to
exercise the stack; obviously not the roster.
"""
import argparse
import os
import sys

from . import queue as q
from . import storage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def chosen_index(face_id):
    """Which candidate was curated for this face, if any."""
    sys.path.insert(0, os.path.join(ROOT, "elements"))
    try:
        from hero import read_picks
        return read_picks(os.path.join(ROOT, "elements", "picks.txt")).get(face_id)
    except (SystemExit, ImportError, OSError):
        return None


def upload(conn, model_id, path, placeholder=False):
    with open(path, "rb") as fh:
        data = fh.read()
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    key = storage.key_for("heroes", 0, ext=ext)
    storage.put_bytes(key, data, content_type=f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}")
    conn.execute(
        "UPDATE models SET hero_key = %s, hero_is_placeholder = %s WHERE id = %s",
        (key, placeholder, model_id))
    return key, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="heroes/<model_id>/<index>.png")
    ap.add_argument("--placeholder", metavar="DIR",
                    help="assign stand-in images from this directory, for development")
    ap.add_argument("--force", action="store_true",
                    help="replace photographs that are already uploaded")
    args = ap.parse_args()
    if not args.dir and not args.placeholder:
        ap.error("give --dir or --placeholder")

    storage.ensure_bucket()
    conn = q.connect()
    models = conn.execute(
        "SELECT id, display_name, hero_key FROM models ORDER BY id").fetchall()

    stand_ins = []
    if args.placeholder:
        stand_ins = sorted(
            os.path.join(args.placeholder, f)
            for f in os.listdir(args.placeholder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and "person" in f.lower())
        if not stand_ins:
            raise SystemExit(f"no person images in {args.placeholder}")
        print(f"{len(stand_ins)} stand-in image(s) — NOT the roster")

    done = skipped = missing = 0
    for i, m in enumerate(models):
        if m["hero_key"] and not args.force:
            skipped += 1
            continue

        if args.dir:
            idx = chosen_index(m["id"])
            path = os.path.join(args.dir, m["id"], f"{idx}.png") if idx else None
            if not path or not os.path.exists(path):
                # A face generated but never curated has no chosen candidate,
                # which is a normal state during curation rather than an error.
                missing += 1
                continue
        else:
            path = stand_ins[i % len(stand_ins)]

        key, size = upload(conn, m["id"], path, placeholder=bool(args.placeholder))
        done += 1
        print(f"  {m['id']:20} {m['display_name']:10} {size/1024:6.0f} KB  {key}")

    print(f"\n{done} uploaded, {skipped} already had one, {missing} not curated yet")
    conn.close()


if __name__ == "__main__":
    main()

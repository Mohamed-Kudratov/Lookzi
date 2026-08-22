#!/usr/bin/env python3
"""Are the roster's members actually different people?

The variation measurement answers whether identity *transferred*. It cannot
answer whether the roster is a roster: a library where two entries are the same
face fails in a catalogue, because a brand shooting forty products across two
models gets forty pictures that look like one person in different clothes.

This compares every chosen hero to every other one directly. No variations
involved -- if two heroes collide, everything generated from them collides too,
and no amount of downstream work separates them.

    python eval/roster_distinctness.py

ArcFace convention: ~0.4 is the same-person threshold at 1e-4 FAR, ~0.5 is a
confident match. Two roster members scoring above 0.4 against each other are a
defect however different they look to a person.
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/elements_out")
    ap.add_argument("--picks", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "elements", "picks.txt"))
    ap.add_argument("--warn", type=float, default=0.40)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "elements"))
    sys.path.insert(0, os.path.join(here, "eval"))
    from hero import read_picks
    from identity import embed, load_app

    picks = read_picks(args.picks)
    app = load_app()

    ids, vecs = [], []
    for fid, idx in sorted(picks.items()):
        v = embed(app, os.path.join(args.dir, "heroes", fid, f"{idx}.png"))
        if v is None:
            print(f"  no face in hero {fid}/{idx}.png")
            continue
        ids.append(fid)
        vecs.append(v)
    M = np.array(vecs) @ np.array(vecs).T
    np.fill_diagonal(M, -1.0)

    w = max(len(i) for i in ids) + 1
    print("\nclosest other roster member\n")
    print(f"{'face':{w}} {'nearest':{w}} {'cosine':>7}")
    flagged = []
    for i, fid in enumerate(ids):
        j = int(np.argmax(M[i]))
        mark = "  <-- too close" if M[i][j] >= args.warn else ""
        print(f"{fid:{w}} {ids[j]:{w}} {M[i][j]:7.3f}{mark}")
        if M[i][j] >= args.warn:
            flagged.append(tuple(sorted((fid, ids[j]))))

    print("\nfull matrix\n")
    print(" " * w + "".join(f"{i[:9]:>10}" for i in ids))
    for i, fid in enumerate(ids):
        cells = "".join(f"{(M[i][j] if i != j else 1.0):10.3f}" for j in range(len(ids)))
        print(f"{fid:{w}}{cells}")

    pairs = sorted(set(flagged))
    print(f"\n{len(pairs)} colliding pair(s) at >= {args.warn}")
    for a, b in pairs:
        print(f"  {a}  ==  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read a cutter comparison and say which one to ship.

    python -m bench.cutscore --run cutters

One number decides it: how often the hanger came with the garment. The others
are there to catch a model that wins by cutting too much away -- a cut-out
with no hanger and no sleeves is not an improvement.
"""
import argparse
import json
import os
import statistics

from service.fidelity import cut_quality

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cutters")
    args = ap.parse_args()
    outdir = os.path.join(HERE, "runs", args.run)
    with open(os.path.join(outdir, "results.json"), encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]

    by = {}
    for r in rows:
        if r.get("image"):
            by.setdefault(r["model"], []).append(r)
    print(f"  {'model':24} {'n':>4} {'sec':>6} {'hanger':>8} {'>0.05':>7} "
          f"{'kept':>7}")
    for model, rs in by.items():
        scores, kept, secs = [], [], []
        for r in rs:
            q = cut_quality(os.path.join(outdir, r["image"]))
            if q.get("hanger") is None:
                continue
            scores.append(q["hanger"])
            kept.append(q["kept"])
            secs.append(r.get("seconds") or 0)
        if not scores:
            continue
        over = sum(1 for s in scores if s > 0.05)
        print(f"  {model:24} {len(scores):4} {statistics.median(secs):6.2f} "
              f"{statistics.median(scores):8.3f} {over:4}/{len(scores):<3}"
              f" {statistics.median(kept):7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

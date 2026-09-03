#!/usr/bin/env python3
"""Can it work out the garment type on its own, or do we ask the seller?

    python tools/garment_type_eval.py            # on the pod, where torch is

The hundred photographs in `test products/` are filed by category, so the
folder names are the answer sheet. `man/overall` is left out: it holds street
photographs of men wearing outfits rather than garments laid out, and it was
excluded from the benchmark for the same reason.

What this has to be good enough for: a sentence in front of the retouch
instruction saying what the garment is. A confident wrong answer is worse than
none -- it would tell the model to turn a dress into a skirt -- so the report
also shows what happens if the uncertain ones are left alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from service.garment_type import classify

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "test products")
FOLDERS = [("man/upper", "tops"), ("man/lower", "bottoms"),
           ("Woman/Upper", "tops"), ("Woman/lower", "bottoms"),
           ("Woman/overall", "one-pieces")]
EXT = (".jpg", ".jpeg", ".png", ".webp")


def main():
    rows = []
    for folder, truth in FOLDERS:
        d = os.path.join(ROOT, folder)
        if not os.path.isdir(d):
            print(f"  missing: {folder}")
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(EXT):
                continue
            got = classify(Image.open(os.path.join(d, name)))
            rows.append({"folder": folder, "truth": truth, "file": name, **got})

    right = [r for r in rows if r["kind"] == r["truth"]]
    print(f"\n  {len(right)} of {len(rows)} correct "
          f"({100 * len(right) / max(len(rows), 1):.0f}%)\n")

    print(f"  {'folder':16} {'n':>3} {'right':>6}   commonest mistake")
    for folder, truth in FOLDERS:
        rs = [r for r in rows if r["folder"] == folder]
        if not rs:
            continue
        ok = [r for r in rs if r["kind"] == truth]
        wrong = {}
        for r in rs:
            if r["kind"] != truth:
                wrong[r["kind"]] = wrong.get(r["kind"], 0) + 1
        worst = max(wrong.items(), key=lambda kv: kv[1]) if wrong else ("", 0)
        print(f"  {folder:16} {len(rs):>3} {len(ok):>6}   "
              f"{worst[0]} x{worst[1]}" if worst[1] else
              f"  {folder:16} {len(rs):>3} {len(ok):>6}")

    # The margin is the whole question: if the wrong answers are the unconfident
    # ones, they can be dropped and the rest trusted.
    print(f"\n  {'if we only speak when the margin is at least':44} "
          f"{'kept':>6} {'right':>6}")
    for cut in (0.0, 0.01, 0.02, 0.03, 0.05):
        kept = [r for r in rows if r["margin"] >= cut]
        ok = [r for r in kept if r["kind"] == r["truth"]]
        share = 100 * len(ok) / max(len(kept), 1)
        print(f"  {cut:>44.2f} {len(kept):>6} {share:>5.0f}%")

    bad = [r for r in rows if r["kind"] != r["truth"]]
    print(f"\n  worst mistakes (most confident and wrong):")
    for r in sorted(bad, key=lambda r: -r["margin"])[:6]:
        print(f"    {r['truth']:11} -> {r['kind']:11} margin {r['margin']:.3f}  "
              f"{r['folder']}/{r['file'][:26]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

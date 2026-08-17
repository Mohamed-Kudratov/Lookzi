#!/usr/bin/env python
"""Serve sweep outputs side by side so the quality call can be made by eye.

SSIM answers "how close is this to the reference", which is the wrong question
for a distilled model -- Lightning produces a *different* image, and different is
not the same as worse. Only looking at them settles it.

    python view_sweep.py --dir /workspace/sweep

Prints a public URL. Images are shown at full size, in grid order, labelled with
the timing and SSIM from sweep.json.
"""
import argparse
import glob
import json
import os

import gradio as gr
from PIL import Image


def load(dir_path):
    meta = {}
    jpath = os.path.join(dir_path, "sweep.json")
    if os.path.exists(jpath):
        with open(jpath, encoding="utf-8") as f:
            for row in json.load(f).get("results", []):
                meta[os.path.basename(row["path"])] = row

    items = []
    for path in sorted(glob.glob(os.path.join(dir_path, "*.png"))):
        name = os.path.basename(path)
        r = meta.get(name)
        if r:
            label = (f"{name}  ·  {r['seconds']}s  ·  {r['transformer_passes']} passes"
                     f"  ·  SSIM {r['ssim_vs_reference']}")
        else:
            label = name
        items.append((Image.open(path), label))
    # Reference first, then cheapest last -- reading order matches the decision.
    items.sort(key=lambda it: -(meta.get(os.path.basename(it[1].split("  ·  ")[0]), {})
                                .get("transformer_passes", 0)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/sweep")
    # 0 means "let Gradio find a free one". A fixed port collides with whatever
    # app.py or an earlier viewer left behind, and the failure is a hard exit.
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    items = load(args.dir)
    if not items:
        raise SystemExit(f"no PNGs in {args.dir}")

    with gr.Blocks(title="Sweep comparison") as demo:
        gr.Markdown("# Sweep comparison")
        gr.Markdown(
            "Same inputs, same seed, different sampling settings. The question is "
            "not which is closest to the reference — it is which you would put on "
            "a product page."
        )
        gr.Gallery(value=items, columns=len(items), height=760,
                   object_fit="contain", show_label=True)
        for img, label in items:
            with gr.Accordion(label, open=False):
                gr.Image(value=img, height=900, show_label=False)

    demo.launch(server_name="0.0.0.0", server_port=args.port or None, share=True)


if __name__ == "__main__":
    main()

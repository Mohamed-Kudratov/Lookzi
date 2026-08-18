#!/usr/bin/env python
"""Keep/reject candidates fast, and see what coverage is still missing.

    python elements/curate.py --group f_cauz_20s_hijab
    python elements/curate.py --dir some/folder        # any folder of images

One image at a time, large, with Keep and Reject. That is deliberately not a
grid: judging identity means comparing ears, jaw and eye spacing, and those
decisions are wrong at thumbnail size. A grid feels faster and produces a worse
dataset.

The coverage panel is the point of the tool. It reads the manifest attributes of
the images you have KEPT and shows which angle, distance and lighting buckets
are still empty -- because a gap there is what makes an identity LoRA work only
at the angle it saw (CURATION.md), and it is invisible while you are clicking.

Decisions are written to decisions.csv on every click, so nothing is lost if
this dies halfway.
"""
import argparse
import csv
import os
import sys
from collections import Counter

import gradio as gr
from PIL import Image

AXES = ("angle", "distance", "lighting", "background", "expression")


def load_rows(manifest, out_dir, group=None, category=None, folder=None):
    """Pair every generated file with its manifest attributes."""
    if folder:
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
        return [{"id": os.path.splitext(os.path.basename(p))[0], "path": p,
                 "group": os.path.basename(folder), "category": "",
                 **{a: "" for a in AXES}} for p in files]

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if group:
        rows = [r for r in rows if r["group"] == group]
    if category:
        rows = [r for r in rows if r["category"] == category]

    out = []
    for r in rows:
        p = os.path.join(out_dir, r["category"], r["id"] + ".png")
        if os.path.exists(p):
            out.append({**r, "path": p})
    return out


def read_decisions(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["status"] for r in csv.DictReader(f) if r.get("id")}


def write_decisions(path, rows, decisions):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "status"] + list(AXES))
        for r in rows:
            w.writerow([r["id"], decisions.get(r["id"], "")] + [r.get(a, "") for a in AXES])


def coverage_table(rows, decisions, targets):
    kept = [r for r in rows if decisions.get(r["id"]) == "keep"]
    if not kept:
        return "**Nothing kept yet.**"

    lines = [f"**{len(kept)} kept**, {sum(1 for r in rows if decisions.get(r['id']) == 'reject')} rejected",
             "", "| axis | value | kept |", "|---|---|---|"]
    gaps = []
    for axis in AXES:
        counts = Counter(r.get(axis, "") for r in kept if r.get(axis))
        seen = set(counts)
        for value, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {axis} | {value} | {n} |")
        for value in targets.get(axis, set()) - seen:
            lines.append(f"| {axis} | {value} | **0** |")
            gaps.append(f"{axis}: {value}")

    if gaps:
        lines += ["", "### Still missing", ""] + [f"- {g}" for g in gaps]
        lines += ["", "*Generate more for these before training — an empty bucket "
                  "is what makes the LoRA fail outside the angles it saw.*"]
    else:
        lines += ["", "### Every bucket covered."]
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(here, "manifest.csv"))
    ap.add_argument("--out-dir", default=os.path.join(here, "out"))
    ap.add_argument("--group", default=None, help="one face id, e.g. f_cauz_20s_hijab")
    ap.add_argument("--category", default=None)
    ap.add_argument("--dir", default=None, help="curate a plain folder instead")
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.manifest, args.out_dir, args.group, args.category, args.dir)
    if not rows:
        print("no generated images found for that selection — run generate.py first",
              file=sys.stderr)
        return 2

    name = args.group or args.category or (os.path.basename(args.dir) if args.dir else "all")
    dec_path = args.decisions or os.path.join(args.out_dir, f"decisions__{name}.csv")
    decisions = read_decisions(dec_path)

    # What "full coverage" means is taken from the manifest itself, so the tool
    # never invents a target the catalogue does not actually produce.
    targets = {a: {r.get(a, "") for r in rows if r.get(a)} for a in AXES}

    def render(i):
        i = max(0, min(i, len(rows) - 1))
        r = rows[i]
        status = decisions.get(r["id"], "")
        mark = {"keep": "KEPT", "reject": "REJECTED", "": "undecided"}[status]
        meta = "  ·  ".join(f"{a}: {r[a]}" for a in AXES if r.get(a))
        header = (f"### {i + 1} / {len(rows)}  —  {mark}\n\n"
                  f"`{r['id']}`\n\n{meta}")
        return Image.open(r["path"]), header, coverage_table(rows, decisions, targets), i

    def decide(i, status):
        rows_i = max(0, min(i, len(rows) - 1))
        decisions[rows[rows_i]["id"]] = status
        write_decisions(dec_path, rows, decisions)
        return render(min(rows_i + 1, len(rows) - 1))

    def export():
        kept = [r for r in rows if decisions.get(r["id"]) == "keep"]
        if not kept:
            return "Nothing kept."
        dest = os.path.join(args.out_dir, "datasets", name)
        os.makedirs(dest, exist_ok=True)
        cap_lines = []
        for r in kept:
            target = os.path.join(dest, os.path.basename(r["path"]))
            Image.open(r["path"]).save(target)
            # Caption what varies, never what stays -- otherwise the trigger
            # token carries nothing and the description carries the face.
            varying = ", ".join(r[a] for a in ("angle", "distance", "lighting",
                                               "background", "expression") if r.get(a))
            cap_lines.append(f"{os.path.basename(target)}\t{name}, {varying}")
        with open(os.path.join(dest, "captions.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(cap_lines))
        return f"Exported **{len(kept)}** images and captions to `{dest}`"

    with gr.Blocks(title=f"Curate — {name}") as demo:
        gr.Markdown(f"# Curating `{name}`")
        gr.Markdown(
            "Reject anything that reads as the person's *sibling* rather than the "
            "person — check ears, jaw line and eye spacing. A near-miss kept out "
            "of politeness is 1/25th of a wrong face in the result."
        )
        idx = gr.State(0)
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(height=760, show_label=False)
                info = gr.Markdown()
                with gr.Row():
                    b_prev = gr.Button("← Back")
                    b_rej = gr.Button("Reject", variant="stop")
                    b_keep = gr.Button("Keep", variant="primary")
                    b_next = gr.Button("Skip →")
            with gr.Column(scale=2):
                cov = gr.Markdown()
                b_exp = gr.Button("Export kept + captions")
                exp_out = gr.Markdown()

        outs = [img, info, cov, idx]
        demo.load(lambda: render(0), outputs=outs)
        b_keep.click(lambda i: decide(i, "keep"), idx, outs)
        b_rej.click(lambda i: decide(i, "reject"), idx, outs)
        b_next.click(lambda i: render(i + 1), idx, outs)
        b_prev.click(lambda i: render(i - 1), idx, outs)
        b_exp.click(export, outputs=exp_out)

    demo.launch(server_name="0.0.0.0", server_port=args.port or None, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())

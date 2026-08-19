#!/usr/bin/env python
"""Two-stage identity generation: pick one face, then vary *that* face.

The problem this solves. Z-Image-Turbo is text-to-image, so every call invents a
new person. Thirty prompts for "Central Asian woman, mid 20s" return thirty
different women who merely fit the description -- and an identity LoRA needs
thirty pictures of *one* woman. Generating the dataset directly from text cannot
work, no matter how detailed the prompt.

So it is done in two stages, with two models we already run:

  stage 1   Z-Image-Turbo, text only     -> candidate faces, pick one "hero"
  stage 2   Qwen-Image-Edit-2509, with   -> the same face at every angle,
            the hero as a reference         distance and light in the grid
  stage 3   train the LoRA on stage 2    -> identity finally locked

Stage 2 is the try-on model doing what it already does: multi-reference,
instruction-driven editing. "Show this person, full body, three-quarter view,
hard light" is the same operation as putting a garment on someone.

    # stage 1 -- 12 candidates for one roster entry
    python elements/hero.py candidates --face f_cauz_20s_hijab --n 12

    # look, choose, then stage 2 -- the full coverage grid from that hero
    python elements/hero.py variations --face f_cauz_20s_hijab --hero 003

Stage 1 needs the Z-Image venv; stage 2 needs the system interpreter and the
pinned diffusers fork. They cannot share one, which is why this is two
subcommands rather than one script that does both.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catalog import (ANGLES, BACKGROUNDS_TRAIN, CLOTHING_MODEST, CLOTHING_NEUTRAL,
                     DISTANCE_MIX, EXPRESSION_MIX, LIGHTING, REALISM_PERSON,
                     ROSTER)  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def face_by_id(face_id):
    for f in ROSTER:
        if f["id"] == face_id:
            return f
    raise SystemExit(f"unknown face {face_id!r}; known: {', '.join(f['id'] for f in ROSTER)}")


def hero_prompt(face):
    """A neutral, well-lit, unambiguous reference shot.

    Deliberately plain: front on, soft light, plain backdrop, half body. This
    image is the source of truth for the identity, so nothing in it should be
    hard to read -- dramatic light or an extreme angle hides the very features
    stage 2 has to carry over.
    """
    pronoun = "her" if face["gender"] == "woman" else "his"
    clothes = (CLOTHING_MODEST if face["modest"] else CLOTHING_NEUTRAL)[0]
    return (f"photorealistic photograph of a {face['appearance']} {face['gender']} "
            f"in {pronoun} {face['age']}, {face['build']} build, {face['skin']}, "
            f"{face['hair']}, {face['detail']}, wearing {clothes}, "
            f"half body from the waist up, front view facing camera, "
            f"soft diffused daylight, plain light grey studio backdrop, "
            f"neutral expression, {REALISM_PERSON}")


def cmd_candidates(args):
    face = face_by_id(args.face)
    out = os.path.join(args.out, "heroes", face["id"])
    os.makedirs(out, exist_ok=True)

    prompt = hero_prompt(face)
    print(f"face   : {face['id']}")
    print(f"prompt : {prompt[:160]}...")
    print(f"out    : {out}\n")
    if args.dry_run:
        return 0

    import torch
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        print("needs the Z-Image venv:\n"
              "  bash elements/setup_zimage.sh\n"
              "  /opt/zimage-venv/bin/python elements/hero.py candidates ...",
              file=sys.stderr)
        return 3

    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo",
                                          torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=False).to("cuda")
    for i in range(args.n):
        path = os.path.join(out, "%03d.png" % i)
        if os.path.exists(path) and not args.redo:
            continue
        t = time.time()
        img = pipe(prompt=prompt, height=1152, width=768, num_inference_steps=9,
                   guidance_scale=0.0,
                   generator=torch.Generator("cuda").manual_seed(args.seed + i)).images[0]
        img.save(path)
        print(f"  {i:03d}  {time.time() - t:.1f}s")

    print(f"\nLook at them, pick one, then:\n"
          f"  python elements/hero.py variations --face {face['id']} --hero 000")
    return 0


def variation_specs(per_face=30):
    """The same coverage grid catalog.py uses, as instructions rather than prompts."""
    specs = []
    for i in range(per_face):
        specs.append({
            "index": i,
            "angle": ANGLES[i % len(ANGLES)],
            "distance": DISTANCE_MIX[i % len(DISTANCE_MIX)],
            "lighting": LIGHTING[i % len(LIGHTING)],
            "background": BACKGROUNDS_TRAIN[i % len(BACKGROUNDS_TRAIN)],
            "expression": EXPRESSION_MIX[i % len(EXPRESSION_MIX)],
        })
    return specs


def cmd_variations(args):
    face = face_by_id(args.face)
    hero_path = os.path.join(args.out, "heroes", face["id"], f"{args.hero}.png")
    specs = variation_specs(args.per_face)
    if args.indices:
        wanted = {int(x) for x in args.indices.replace(" ", "").split(",") if x != ""}
        specs = [s for s in specs if s["index"] in wanted]
        if not specs:
            raise SystemExit(f"no spec indices matched {sorted(wanted)}")

    print(f"face  : {face['id']}")
    print(f"hero  : {hero_path}")
    print(f"making: {len(specs)} variations\n")

    # Checked after the summary so --dry-run can be used to review the grid
    # before any hero exists.
    if args.dry_run:
        for s in specs[:4]:
            print(f"  {s['index']:03d}  {s['distance']}, {s['angle']}, {s['lighting']}")
        print("  ...")
        return 0
    if not os.path.exists(hero_path):
        raise SystemExit(f"no hero at {hero_path} -- run `candidates` first")

    out = os.path.join(args.out, "models")
    os.makedirs(out, exist_ok=True)

    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    clothes = (CLOTHING_MODEST if face["modest"] else CLOTHING_NEUTRAL)
    hero = Image.open(hero_path).convert("RGB")

    # The stock edit pipeline, not LayeringVTONPipeline. Ours hardcodes the
    # try-on instruction -- "...and change the pose of the person in the first
    # image to the pose in the third image" -- which fights an instruction that
    # is trying to set the pose in words. The base model is a general
    # multi-reference editor and that is what this stage needs.
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509"),
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    steps, cfg = 40, 4.0
    if args.lightning:
        # Same distillation LoRA the try-on path uses: 8 passes instead of 80.
        from pipeline import LIGHTNING_REPO, LIGHTNING_WEIGHTS
        pipe.load_lora_weights(LIGHTNING_REPO,
                               weight_name=LIGHTNING_WEIGHTS[args.lightning])
        steps, cfg = args.lightning, 1.0
    print(f"  sampling: {steps} steps, cfg {cfg}\n")

    # A partial run must not wipe the log of the full one. Read what is there,
    # replace only the rows this run touches, write it all back at the end.
    log_path = os.path.join(out, f"variations__{face['id']}.csv")
    fields = ["id", "angle", "distance", "lighting", "background",
              "expression", "seconds", "error"]
    existing = {}
    if os.path.exists(log_path):
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    existing[row["id"]] = row

    ok = failed = 0
    for s in specs:
        path = os.path.join(out, "%s__%03d.png" % (face["id"], s["index"]))
        if os.path.exists(path) and not args.redo:
            continue
        outfit = clothes[s["index"] % len(clothes)]
        # The identity comes from the reference image; the instruction carries
        # only what should change. Describing the face here would fight the
        # reference rather than reinforce it.
        instruction = (f"Keep the same person with exactly the same face, "
                       f"same skin tone and same hair. Show them {s['distance']}, "
                       f"{s['angle']}, {s['lighting']}, {s['background']}, "
                       f"{s['expression']} expression, wearing {outfit}. "
                       f"Photorealistic, natural skin texture.")
        t = time.time()
        try:
            img = pipe(
                image=[hero],
                prompt=instruction,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                generator=torch.Generator("cuda").manual_seed(args.seed + s["index"]),
            ).images[0]
            img.save(path)
            elapsed, err, ok = time.time() - t, "", ok + 1
            print(f"  {s['index']:03d}  {elapsed:5.1f}s  {s['distance']}, {s['angle'][:28]}")
        except Exception as exc:
            elapsed, err, failed = time.time() - t, f"{type(exc).__name__}: {exc}", failed + 1
            print(f"  {s['index']:03d}  FAILED  {err}", file=sys.stderr)
        existing[os.path.basename(path)] = dict(
            id=os.path.basename(path), angle=s["angle"], distance=s["distance"],
            lighting=s["lighting"], background=s["background"],
            expression=s["expression"], seconds=round(elapsed, 1), error=err)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for k in sorted(existing):
                wr.writerow({c: existing[k].get(c, "") for c in fields})

    print(f"\n{ok} made, {failed} failed -> {out}")
    print(f"Now curate:  python elements/curate.py --group {face['id']}")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("candidates", help="stage 1: text -> candidate faces")
    c.add_argument("--face", required=True)
    c.add_argument("--n", type=int, default=12)
    c.add_argument("--out", default=os.path.join(HERE, "out"))
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--redo", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(fn=cmd_candidates)

    v = sub.add_parser("variations", help="stage 2: hero -> the coverage grid")
    v.add_argument("--face", required=True)
    v.add_argument("--hero", default="000", help="which candidate, e.g. 003")
    v.add_argument("--per-face", type=int, default=30)
    v.add_argument("--out", default=os.path.join(HERE, "out"))
    v.add_argument("--lora", default="./weights")
    v.add_argument("--lightning", type=int, choices=[4, 8], default=8)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--indices", default=None,
                   help="regenerate only these grid positions, e.g. 3,11,23,27 -- "
                        "used with --redo when one axis value turns out bad")
    v.add_argument("--redo", action="store_true")
    v.add_argument("--dry-run", action="store_true")
    v.set_defaults(fn=cmd_variations)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

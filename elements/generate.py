#!/usr/bin/env python
"""Run the Elements manifest through Z-Image-Turbo.

    python elements/generate.py --category poses
    python elements/generate.py --group f_cauz_20s_hijab
    python elements/generate.py --limit 5          # smoke test first

**Needs its own environment.** Z-Image-Turbo requires diffusers from source,
while the try-on pipeline is pinned to the bundled 0.36.0.dev0 fork, and
installing one replaces the other. Run `bash elements/setup_zimage.sh` to build
a separate venv, then call this with that interpreter. In production these are
two different workers anyway -- see ARCHITECTURE.md.

Resumes by default: anything already on disk is skipped, so an interrupted run
costs nothing. 641 prompts at roughly a second each is minutes, but the pod is
billed either way.
"""
import argparse
import csv
import os
import sys
import time

MODEL = "Tongyi-MAI/Z-Image-Turbo"


def load_manifest(path, category=None, group=None, limit=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if category:
        wanted = {c.strip() for c in category.split(",")}
        rows = [r for r in rows if r["category"] in wanted]
    if group:
        wanted = {g.strip() for g in group.split(",")}
        rows = [r for r in rows if r["group"] in wanted]
    return rows[:limit] if limit else rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(here, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--category", default=None, help="comma separated")
    ap.add_argument("--group", default=None, help="comma separated, e.g. one face id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=9,
                    help="Turbo does 8 forwards at 9; more is not better here")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1152,
                    help="3:2 portrait suits full-body figures; products use --square")
    ap.add_argument("--square", action="store_true", help="1024x1024, for products")
    ap.add_argument("--seed", type=int, default=0, help="base seed; each row adds its index")
    ap.add_argument("--redo", action="store_true", help="regenerate even if the file exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.category, args.group, args.limit)
    if not rows:
        print("nothing matched", file=sys.stderr)
        return 2

    # Products and backdrops want square framing; people want portrait.
    def size_for(cat):
        if args.square or cat in ("clothing", "accessories", "shoes", "backgrounds"):
            return 1024, 1024
        return args.width, args.height

    todo = []
    for i, r in enumerate(rows):
        path = os.path.join(args.out, r["category"], r["id"] + ".png")
        if os.path.exists(path) and not args.redo:
            continue
        todo.append((i, r, path))

    print(f"{len(rows)} in manifest, {len(rows) - len(todo)} already done, {len(todo)} to generate")
    for cat in sorted({r["category"] for _, r, _ in todo}):
        n = sum(1 for _, r, _ in todo if r["category"] == cat)
        w, h = size_for(cat)
        print(f"  {cat:<20}{n:>5}  at {w}x{h}")

    if args.dry_run or not todo:
        return 0

    import torch
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        print("\nZImagePipeline not found. Z-Image-Turbo needs diffusers from source,\n"
              "which conflicts with the pinned fork used by the try-on pipeline.\n"
              "Build the separate environment first:\n\n"
              "    bash elements/setup_zimage.sh\n"
              "    /workspace/zimage-venv/bin/python elements/generate.py ...\n",
              file=sys.stderr)
        return 3

    print(f"\nloading {MODEL} ...")
    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=False).to("cuda")
    print(f"ready in {time.time() - t0:.0f}s")

    log_path = os.path.join(args.out, "generate_log.csv")
    os.makedirs(args.out, exist_ok=True)
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["id", "category", "group", "seconds", "error"])

    ok = failed = 0
    started = time.time()
    for n, (i, r, path) in enumerate(todo, 1):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        w, h = size_for(r["category"])
        t = time.time()
        try:
            # guidance_scale must be 0.0: the Turbo variant is distilled and is
            # not built for classifier-free guidance.
            img = pipe(
                prompt=r["prompt"],
                height=h, width=w,
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(args.seed + i),
            ).images[0]
            img.save(path)
            elapsed, err, ok = time.time() - t, "", ok + 1
        except Exception as exc:  # one bad prompt must not end the run
            elapsed, err, failed = time.time() - t, f"{type(exc).__name__}: {exc}", failed + 1
            print(f"  FAILED {r['id']}: {err}", file=sys.stderr)

        writer.writerow([r["id"], r["category"], r["group"], round(elapsed, 2), err])
        log.flush()

        if n % 10 == 0 or n == len(todo):
            rate = (time.time() - started) / n
            left = (len(todo) - n) * rate
            print(f"  {n}/{len(todo)}  {rate:.1f}s each  ~{left / 60:.0f} min left")

    log.close()
    print(f"\n{ok} generated, {failed} failed  ->  {args.out}")
    print(f"  log: {log_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

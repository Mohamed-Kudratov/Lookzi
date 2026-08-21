#!/usr/bin/env python3
"""Write a bf16 copy of Z-Image-Turbo, once.

The published repo is fp32: 22.9 GB of transformer plus 7.5 GB of text encoder,
30.6 GB in total. Every load reads all of it off the network volume and casts it
down to bf16 in memory -- so half the bytes crossing the wire are thrown away on
arrival, and the wire is the slow part.

That matters more here than it would locally. RunPod's volume is fine at bulk
sequential reads (measured 655 MB/s) but safetensors loads through mmap, which
turns into a long tail of small page faults over a network filesystem. One load
took 9m44s; an earlier attempt wedged the process in uninterruptible sleep
entirely, moving 570 MB in nine minutes before it had to be killed.

Halving the bytes halves the faults. The cast happens once here instead of on
every load, and the result is written where `hero.py` will find it.

    /opt/zimage-venv/bin/python elements/save_bf16.py

Costs one slow load plus a ~15 GB write. Skips itself if the copy exists.
"""
import argparse
import os
import shutil
import time

DEFAULT_OUT = "/workspace/models/Z-Image-Turbo-bf16"


def resolved_model_path(out=DEFAULT_OUT):
    """The bf16 copy if it is there, otherwise the hub id.

    Callers do not choose -- if the copy exists it is strictly better, and
    making it a flag means someone eventually forgets to pass it.
    """
    if os.path.isfile(os.path.join(out, "model_index.json")):
        return out
    return "Tongyi-MAI/Z-Image-Turbo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    done = os.path.join(args.out, "model_index.json")
    if os.path.isfile(done) and not args.redo:
        print(f"already there: {args.out}")
        return 0
    # A half-written copy is worse than none -- it satisfies the existence check
    # above and then fails at load time, on a pod, minutes into a run.
    if os.path.isdir(args.out):
        print(f"removing incomplete copy at {args.out}")
        shutil.rmtree(args.out)

    import torch
    from diffusers import ZImagePipeline

    t = time.time()
    print("loading fp32 from the hub cache (this is the slow part)", flush=True)
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo",
                                          torch_dtype=torch.bfloat16)
    print(f"  loaded in {time.time() - t:.0f}s", flush=True)

    t = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pipe.save_pretrained(args.out)
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(args.out) for f in fs)
    print(f"  wrote {size / 1024**3:.1f} GB in {time.time() - t:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

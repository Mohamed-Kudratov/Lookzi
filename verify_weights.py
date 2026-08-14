#!/usr/bin/env python
"""Delete weight files that are corrupt, so the next download refetches them.

A stalled or interrupted HuggingFace download can leave a blob that is the right
*length* but not valid content. `snapshot_download` compares sizes, so it treats
such a file as complete forever, and the damage only surfaces much later as:

    OSError: Unable to load weights from checkpoint file for
    '.../transformer/diffusion_pytorch_model-00001-of-00005.safetensors'

This opens every shard and reads its header, which is cheap (safetensors is
mmap'd -- no full read) and catches exactly that case. Corrupt files are removed
along with the blob they point at, since the snapshot entries are symlinks and
deleting only the link leaves the bad blob in the cache.

    python verify_weights.py --repo Qwen/Qwen-Image-Edit-2509
    python verify_weights.py --path /workspace/.cache/huggingface
"""
import argparse
import json
import os
import sys


def _resolve_targets(path):
    """A snapshot entry and the blob behind it."""
    out = [path]
    real = os.path.realpath(path)
    if real != os.path.abspath(path):
        out.append(real)
    return out


def check_safetensors(path):
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
        if not keys:
            return False, "no tensors"
        return True, f"{len(keys)} tensors"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen-Image-Edit-2509",
                    help="HF repo id, used to locate the cache directory")
    ap.add_argument("--path", default=None,
                    help="HF cache root (defaults to HF_HOME or ~/.cache/huggingface)")
    ap.add_argument("--dry-run", action="store_true", help="report without deleting")
    args = ap.parse_args()

    root = args.path or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = os.path.join(root, "hub") if os.path.isdir(os.path.join(root, "hub")) else root
    model_dir = os.path.join(hub, "models--" + args.repo.replace("/", "--"))

    if not os.path.isdir(model_dir):
        print(f"No cache for {args.repo} under {hub}", file=sys.stderr)
        return 2

    files = []
    for dirpath, _, names in os.walk(model_dir):
        if os.path.basename(dirpath) == "blobs":
            continue          # checked through the snapshot symlinks instead
        for n in names:
            if n.endswith(".safetensors") or n.endswith(".json"):
                files.append(os.path.join(dirpath, n))

    if not files:
        print(f"No weight files found under {model_dir}", file=sys.stderr)
        return 2

    bad = []
    for path in sorted(files):
        if path.endswith(".safetensors"):
            ok, detail = check_safetensors(path)
        else:
            ok, detail = check_json(path)
        name = os.path.relpath(path, model_dir)
        if ok:
            print(f"  ok    {name}  ({detail})")
        else:
            print(f"  BAD   {name}  ({detail})")
            bad.append(path)

    incomplete = []
    for dirpath, _, names in os.walk(model_dir):
        for n in names:
            if n.endswith(".incomplete"):
                incomplete.append(os.path.join(dirpath, n))

    print(f"\n{len(files) - len(bad)}/{len(files)} files valid")
    if incomplete:
        print(f"{len(incomplete)} interrupted downloads")

    if args.dry_run:
        print("(dry run -- nothing deleted)")
        return 1 if bad else 0

    removed = 0
    for path in bad:
        for target in _resolve_targets(path):
            try:
                os.remove(target)
                removed += 1
            except FileNotFoundError:
                pass
    for path in incomplete:
        try:
            os.remove(path)
            removed += 1
        except FileNotFoundError:
            pass

    if removed:
        print(f"removed {removed} files -- re-run the download to refetch them")
        return 1
    print("nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())

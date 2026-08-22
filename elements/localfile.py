#!/usr/bin/env python3
"""Keep a file on local disk when reading it from the volume is the bottleneck.

safetensors opens weights with mmap, and mmap over RunPod's network volume
degenerates into a long tail of page faults -- work that shows up as neither
CPU time nor read() bytes, so the process looks idle and hung when it is
neither. The volume itself is not consistently slow: the same blob read
sequentially measured 655 MB/s on one pod and 43 MB/s on another, and mmap
multiplies whichever it is.

Copying first turns the read into one sequential pass, which is the access
pattern a network filesystem is good at, and every load afterwards comes off
local NVMe. For the 810 MB Lightning LoRA that is the difference between
seconds and not finishing.

Only worth it for files that are read repeatedly and fit -- the container disk
is 30 GB with the Python environments already on it. The 57.7 GB try-on model
does not qualify and is loaded shard by shard straight to the GPU instead.
"""
import os
import shutil
import time

CACHE = os.environ.get("LOCAL_CACHE", "/root/.localcache")


def cached_local(path, min_free_gb=4.0):
    """Return a local copy of `path`, making one if needed.

    Falls back to the original on any problem. A slow read is better than a
    failed run, and this is an optimisation -- it must never be the reason
    something breaks.
    """
    try:
        size = os.path.getsize(path)
        dest = os.path.join(CACHE, f"{os.path.basename(path)}.{size}")
        # Size is part of the name, so a changed file cannot be served stale.
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            return dest

        os.makedirs(CACHE, exist_ok=True)
        free = shutil.disk_usage(CACHE).free
        if free - size < min_free_gb * 1024**3:
            print(f"  [localfile] only {free / 1024**3:.1f} GB free, "
                  f"reading {os.path.basename(path)} in place", flush=True)
            return path

        t = time.time()
        # Copy to a temporary name first: an interrupted copy that kept the
        # final name would pass the size check on the next run only if it
        # happened to be complete, and silently corrupt the load if not.
        tmp = dest + ".part"
        with open(path, "rb", buffering=0) as src, open(tmp, "wb", buffering=0) as out:
            shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
        os.replace(tmp, dest)
        secs = time.time() - t
        print(f"  [localfile] cached {size / 1024**2:.0f} MB in {secs:.1f}s "
              f"({size / 1024**2 / max(secs, 0.001):.0f} MB/s) -> {dest}", flush=True)
        return dest
    except OSError as exc:
        print(f"  [localfile] {type(exc).__name__}: {exc}; using the original",
              flush=True)
        return path

# Measured timings

Numbers, not estimates. Everything here was timed on the pod, on an A100-SXM4
80GB with the 120 GB network volume attached. `elements/timing.py` appends new
measurements to `/workspace/timings.csv`, which survives the container.

    python elements/timing.py          # summarise the most recent run

## Pod setup

| | |
|---|---|
| Stock RunPod image, full setup | **601 s** |
| Same, after the fork became a wheel | **116 s** |
| Same, from the Lookzi Docker image | not yet measured |
| Recovering the fork: `sparse-checkout disable` | minutes, and it wedged |
| Recovering the fork: `git archive` to local disk | **0.223 s** |

The drop from 601 s to 116 s is two changes. The diffusers fork is no longer
recovered by undoing sparse-checkout, which materialised 2247 files onto the
network volume, rewrote the index, and on one pod wedged completely when a
killed git left index.lock behind; `git archive` streams the same tree out of
the object store without touching the index, onto the container's local NVMe.
And the built wheel is kept on the volume, so a later pod installs one file
instead of unpacking a source tree at all.

The 601 s was entirely package installation — torch, the diffusers fork, the
isolated Z-Image venv. No weights were downloaded: they were already on the
volume and reported `present`. This is the cost the Docker image removes, and
it is paid on every restart and every forced migration, because stopping a pod
wipes the container disk.

## Model loading

| | |
|---|---|
| Z-Image-Turbo, fp32 from the hub cache | **584 s** |
| Z-Image-Turbo, bf16 copy | **1 s** |
| One-time bf16 conversion | **467 s** (202 s load, 257 s write, 19.1 GB) |
| Qwen-Image-Edit-2509 transformer, device_map | ~170 s |
| Qwen text encoder, device_map | ~120 s |
| Lightning 8-step adapter, before the thread fix | **> 1200 s**, never finished |
| Lightning 8-step adapter, after | seconds |

The published Z-Image repo is fp32 and 30.6 GB. Loading it read all of that off
the volume and cast it down to bf16 in memory, so half the bytes crossing the
wire were discarded on arrival. The bf16 copy pays that once.

## Generation

| | |
|---|---|
| Z-Image candidate (768x1152, 9 steps) | **5.9 s** |
| Stage 1, 13 faces x 6 candidates = 78 images | **462 s** |
| Try-on, 40 steps CFG 4.0 | 94.8 s |
| Try-on, 24 steps CFG 1.0 | 30.0 s |
| Try-on, Lightning 8-step | **14.3 s** |
| Try-on, Lightning 4-step | 8.3 s |

Cost per image fell from $0.066 to $0.0061 across that row. The market price
for virtual try-on is about $0.04.

## Volume behaviour

| | |
|---|---|
| Sequential read, `dd` with `iflag=direct` | **655 MB/s** |
| Write | 1.1 GB/s |
| `git reset --hard` restoring the diffusers tree | minutes |
| Same, with the tree excluded by sparse-checkout | **1 s** |

The volume is fast in bulk and slow at everything else. safetensors loads
through mmap, which becomes a long tail of small page faults over a network
filesystem, and thousands of tiny files -- a git checkout, a pip install -- is
the worst case. Both are avoidable rather than inherent.

## The thread quota

The pod's host reports 128 cores. The container's cgroup allows 13.6. torch
sized its OpenMP pool from the host figure and spent nearly all of it in futex
contention: the Lightning conversion ran over twenty minutes at roughly 0.6
cores of useful throughput, presenting exactly as a deadlock.

It was misdiagnosed twice -- first as accelerate's dispatch hooks, then as the
network volume -- before sampling `utime` twice thirty seconds apart showed the
process was running the whole time, just wasting it.

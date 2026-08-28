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
| **Whole pod to working, via the admin panel** | **258 s** |

The 258 s is a pod taken from a stock template to a state where it has loaded
the model and produced a real image, unattended, with every step measured:
connect 2.5, inspect 2.4, volume 5.5, clone 6.6, packages 6.8, fork 5.6 (from
the cached wheel), verify 11.9, weights 2.5, download skipped, load and
generate 168 + 18. Everything before the model load costs under a minute.
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
| Try-on, Lightning 8-step, **4-bit** | **17.8 s** |

## A40 against A100

The card was changed on purpose: A100 PCIe reads "out of capacity" and A100 SXM
offers one at a time, and a pod on the scarcest card in the fleet stopped by
itself three times in a day. A40 shows six free.

| | A100 SXM, $1.59/hr | A40, $0.44/hr |
|---|---|---|
| Model load, cold | 167.7 s | **171.1 s** |
| Model load, warm page cache | 15.9 s | 16.4 s |
| One image, 4-bit, 8 steps | 16.4 s | **31.1 s** |
| **Cost per image** | $0.0072 | **$0.0038** |

Loading is the same because it is bound by the volume, not the card. Generating
is 1.9x slower, which is what a third of the memory bandwidth buys. The card is
3.6x cheaper by the hour, so an image costs about half as much even though the
customer waits twice as long -- and the market price for a try-on is $0.04, so
both are far inside it.

The trade is deliberate: twenty seconds of a customer's patience against a card
that does not get taken away mid-job.
| Model load, bf16 (transformer + text encoder) | ~290 s |
| Model load, **4-bit** | **167.7 s** |

The first real numbers on the quantisation question, though not yet the answer:
4-bit is about 24% slower per image and loads in a little over half the time.
Speed was never the argument for it -- fitting a 48 GB card at a quarter the
price, and a container disk rather than a network volume, is. Whether the
pictures hold up is what eval/quantisation.py is for.

Cost per image fell from $0.066 to $0.0061 across that row. The market price
for virtual try-on is about $0.04.

## Volume behaviour

| | |
|---|---|
| Sequential read, `dd` with `iflag=direct` | **655 MB/s** |
| Same volume, a bad morning | **16 MB/s** |
| Same volume, that afternoon | **930 MB/s** |
| HF download, plain `huggingface_hub` | 1.7 MB/s |
| HF download, same file with `curl -L` | 47 MB/s |
| HF download, `hf_transfer` on | **43 MB/s** |
| Write | 1.1 GB/s |
| `git reset --hard` restoring the diffusers tree | minutes |
| Same, with the tree excluded by sparse-checkout | **1 s** |

**These numbers are weather, not physics.** On 2026-08-28 the same volume read
at **16 MB/s** -- forty times slower -- for a whole session. Small writes stayed
normal, so it was reads specifically. A model load that takes four minutes at
655 MB/s takes fifty-six at 16, and one Lightning adapter load stalled
completely: 45 seconds with `read_bytes` unmoved, one thread parked in
`folio_wait_bit_common`, which is the kernel waiting on an mmap page that never
arrived.

Nothing about that is fixable from inside the pod, and it makes the network
volume a single point of failure for load time. The container's local disk
measured 3.4 GB/s the same minute. That is the argument for keeping the
checkpoint small enough to live on local NVMe, and it is the same architecture
serverless needs anyway.

Downloads throttle too, and separately. A fresh pull of the 4-bit checkpoint
ran at 36 MB/s for about 75 seconds and then settled to 2 MB/s -- to the local
disk, so not the volume's fault. Unauthenticated Hugging Face traffic is rate
limited; a token is the lever, not more patience.

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

# Running on RunPod

The point of paying for a pod is that you stop needing the 4-bit build. With
enough VRAM this runs the **full bf16 `Qwen/Qwen-Image-Edit-2509`**, and the
quantisation quality loss that Colab forces on you disappears.

## Which GPU

| GPU | VRAM | Model | Both components resident | ~Time / image (40 steps) |
|---|---|---|---|---|
| **A100 80GB** | 80 GB | full bf16 | yes | **~2–3 min** |
| **H100 80GB** | 80 GB | full bf16 | yes | ~1–2 min |
| A100 40GB | 40 GB | full bf16 | no — sequential | ~5–8 min |
| L40S / A40 | 48 GB | full bf16 | no — sequential | ~6–10 min |
| RTX 4090 | 24 GB | 4-bit only | no | ~8–12 min |

The bf16 weights are 57.7 GB — transformer 40.9 GB plus text encoder 16.6 GB.
Above ~70 GB of VRAM both stay loaded. Between 40 and 70 GB the pipeline loads
them one at a time, which costs a model load per generation. Below 40 GB only
the 4-bit build fits.

**A100 80GB is the sweet spot.** H100 is roughly twice the price for a gain you
will not notice on single images.

## Pod configuration

- **Template:** any RunPod PyTorch image (CUDA 12.1+), e.g.
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container disk:** 30 GB
- **Volume:** **100 GB**, mounted at `/workspace` — the model alone is 57.7 GB,
  and a volume is what survives a pod stop
- **Expose HTTP port:** `7860`

> **The volume size is the one setting people get wrong.** A 50 GB volume cannot
> hold the bf16 model, and you will not find out until the download dies at 46 GB
> with `Errno 122 Disk quota exceeded`. `df -h /workspace` is no help — on RunPod
> it reports the whole MFS cluster as free, not your quota. Set 100 GB up front;
> the volume costs cents per hour next to the GPU.

## Setup

In the pod's web terminal:

```bash
cd /workspace
git clone https://github.com/Mohamed-Kudratov/Lookzi.git Layering-Virtual-Try-On
bash Layering-Virtual-Try-On/runpod_setup.sh
```

The script detects VRAM, picks the model accordingly, installs everything, works
around the `diffusers` shadowing problem (see
[`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md)), and downloads the weights into
`/workspace/hf_cache` so they survive a restart.

Then either the web UI:

```bash
bash /workspace/Layering-Virtual-Try-On/run.sh
```

Open `https://<YOUR_POD_ID>-7860.proxy.runpod.net`. RunPod's proxy handles TLS,
so Gradio's `share=True` tunnel is neither needed nor enabled.

Or headless, which is what you want over SSH — there is no browser there:

```bash
cd /workspace/Layering-Virtual-Try-On

# the three bundled examples, into ./outputs -- use this as a smoke test
python infer.py --examples

# a single run
python infer.py \
    --person assets/person_1.png \
    --garment assets/pants.png \
    --mode swap \
    --description "swap the deep blue jeans for dark wash jeans" \
    --steps 40 --cfg 4.0 --seed 42 \
    --out result.png
```

`infer.py` prints per-step timing and peak VRAM, so it doubles as the benchmark
for deciding whether a given pod size is worth its hourly rate.

## Cost

Billing is per second while the pod runs, and **a stopped pod still bills for
its volume**. Two habits worth keeping:

- Stop the pod when you are done. The 100 GB volume costs a few cents an hour on
  its own; an idle A100 costs a hundred times that.
- The first run downloads 57.7 GB. On a stopped-and-restarted pod with the same
  volume, that download does not repeat.

## Defaults

Set for a proper GPU, not a free-tier one:

- **Model:** `Qwen/Qwen-Image-Edit-2509`, full bf16 — `runpod_setup.sh` drops to
  the 4-bit build only under 40 GB of VRAM
- **Steps:** 40, the paper's default
- **True CFG:** 4.0

`pipeline.py` adapts the rest on its own: bf16 is selected over fp16 by GPU
capability, the fp16 numerical guards become no-ops, and `low_vram` switches off
above 20 GB so nothing is loaded twice.

## RunPod plugin for Claude Code

Optional, and unrelated to the pipeline — it lets an agent manage pods for you:

```bash
claude plugin marketplace add runpod/runpod-plugins-official
```

```bash
claude plugin install runpod@runpod
```

Then, in an interactive `claude` session, run `/reload-plugins`, followed by
`/mcp` → **runpod** → **Sign in with Runpod**. Both steps need a real terminal;
neither can be done from a non-interactive session.

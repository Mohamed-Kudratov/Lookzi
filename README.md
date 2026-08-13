# Layering Virtual Try-On — RunPod

Virtual try-on with **add** (layering) and **swap** modes, packaged to run on a
RunPod GPU pod.

Upstream research code: [ChuenFung/Layering-Virtual-Try-On](https://github.com/ChuenFung/Layering-Virtual-Try-On)
· [paper](https://arxiv.org/pdf/2607.22924)
· [project page](https://chunfeng-projects.github.io/layering-virtual-tryon/)

**Authors:** [Chun Feng](https://github.com/ChuenFung), [Bowei Chen](https://armastuschen.github.io/), [Mengyi Shan](https://shanmy.github.io/), and [Ira Kemelmacher-Shlizerman](https://www.irakemelmacher.com/)

![Teaser Image](./assets/teaser.png)

---

## What this needs

The base model is **`Qwen-Image-Edit-2509`: 20.4 B parameters, 57.7 GB** in bf16
— transformer 40.9 GB plus text encoder 16.6 GB. That number drives every
decision here.

| GPU | VRAM | Model | Both resident | ~Time / image (40 steps) |
|---|---|---|---|---|
| **A100 80GB** | 80 GB | full bf16 | yes | **~2–3 min** |
| H100 80GB | 80 GB | full bf16 | yes | ~1–2 min |
| A100 40GB | 40 GB | full bf16 | no — sequential | ~5–8 min |
| L40S / A40 | 48 GB | full bf16 | no — sequential | ~6–10 min |
| RTX 4090 | 24 GB | 4-bit only | no | ~8–12 min |

**A100 80GB is the sweet spot.** H100 costs roughly double for a difference you
will not notice on single images.

## Quick start

Pod: any RunPod PyTorch template (CUDA 12.1+), container disk 30 GB,
**volume 100 GB at `/workspace`**, HTTP port **7860** exposed.

```bash
cd /workspace
git clone https://github.com/Mohamed-Kudratov/Lookzi.git Layering-Virtual-Try-On
bash Layering-Virtual-Try-On/runpod_setup.sh
```

The script detects VRAM, picks the model to match, installs everything, and
caches the weights on the volume so a pod restart does not re-download 57.7 GB.

Then either the web UI:

```bash
bash /workspace/Layering-Virtual-Try-On/run.sh
```

at `https://<POD_ID>-7860.proxy.runpod.net`, or headless over SSH:

```bash
python infer.py --examples
```

See [`RUNPOD_README.md`](./RUNPOD_README.md) for the full guide and
[`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) when something breaks.

## Usage

1. **Person image** — full-body.
2. **Garment image** — the garment to try on.
3. **Mode** — `swap` (traditional try-on) or `add` (layering).
4. **Description** — e.g. `"swap the beige leggings for dark wash jeans"` or
   `"add a light gray turtleneck sweater"`.
5. **Pose** *(optional)* — extracted from the person image with DWPose if left
   blank.

Everything is padded to 512×896 before inference.

## Changes from upstream

The demo assumes a bf16 datacentre GPU and breaks in several places on anything
else. Fixed here:

- **Pose extraction crashed on GPU machines.** `utils.py` chose the DWPose
  device with `torch.cuda.is_available()`, so `easy-dwpose` demanded
  `CUDAExecutionProvider` while `environment.yml` installs the CPU-only
  `onnxruntime`. Provider availability is now queried.
- **The swap/add radio did nothing** — `mode` was collected by the UI and never
  passed to the pipeline.
- **`pip install -e ./diffusers` cannot work**; the source tree shadows the
  install. See [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md).
- `DWposeDetector` was rebuilt (two ONNX sessions) on every call.
- `torch.cuda.empty_cache()` ran before any GPU check.
- `if not person_img` — PIL images define neither `__bool__` nor `__len__`, so
  this was always false and worked by accident.
- Device and dtype were hardcoded to `cuda` + `bfloat16`; both are detected.
- `.to(device)` was called on loaded models, which a bitsandbytes 4-bit module
  cannot do — quantized checkpoints are placed via `device_map`.

Added: `infer.py` (headless CLI), `runpod_setup.sh`, text-embedding caching,
sampling controls in the UI, and fp32 guards for the VAE and CFG
renormalisation that keep fp16 hardware from producing NaN.

## Files

| | |
|---|---|
| `runpod_setup.sh` | one-shot pod setup |
| `infer.py` | headless CLI inference |
| `app.py` | Gradio web UI |
| `pipeline.py` | sampler |
| `utils.py` | pose extraction, padding |
| `weights/` | the try-on LoRA (rank 32) |
| `diffusers/` | required diffusers fork, v0.36.0.dev0 |

The upstream conda instructions live in
[`README_UPSTREAM.md`](./README_UPSTREAM.md).

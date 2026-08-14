# Model choice against the product plan

Written 2026-08-14, from measurements on a RunPod A100 SXM 80GB.

## The short answer

**This model is wrong for Service 1 and right for Service 2.** It is not a question of
quality — it is that the two services have opposite constraints, and one model cannot
satisfy both.

## What was measured, not estimated

`Qwen/Qwen-Image-Edit-2509` + the Layering VTON LoRA, bf16, A100 80GB at $1.59/hr:

| | |
|---|---|
| Sampling, 40 steps | **91 s** (2.3 s/step) |
| Wall clock incl. DWPose + text encoding | **~150 s** |
| Peak VRAM | 55.7 GB |
| Weights on disk | 57.7 GB |
| Cold start (load from volume) | ~3 min |
| Cost per image at full utilisation | **$0.066** |

`SUCCESS_CRITERIA.md` sets **A3: latency p95 < 15 s**. This is **10× over that bar.**

Optimisation does not close the gap. CFG 1.0 plus 24 steps gets to roughly 45 s on an
A100 and roughly 22 s on an H100 — still over, and at the cost of quality that has not
been measured yet (`sweep_steps.py` exists to measure it).

## Why it is heavy

Not incidental — it is architectural, and the reasons multiply:

1. **20.4B transformer + 8.3B Qwen2.5-VL text encoder.** An SD1.5-based try-on model is
   ~0.86B UNet + 123M CLIP.
2. **Flat DiT, 60 layers, no downsampling.** A UNet does most of its attention at 32×32
   and below; this one does full-resolution attention in every layer.
3. **Four latent streams in one sequence.** `pipeline.py` concatenates noise, person,
   garment and pose — 4 × 1792 = **7168 tokens** at 512×896. Attention is O(n²), so that
   is 16× the attention cost of a single image.
4. **CFG 4.0 runs the transformer twice per step.** 40 steps → 80 forward passes.

Roughly 290 TFLOP per forward pass, against ~0.7 for an SD1.5 UNet.

## The two services want different things

| | Service 1 — customer tries on | Service 2 — photo studio |
|---|---|---|
| Runs | live, in a product page | offline, batch |
| Latency | **decisive** (< 15 s) | irrelevant (minutes are fine) |
| Quality bar | "good enough to judge fit" | **decisive** — must drive a purchase |
| Volume | high, thin margin per call | low, thick margin per job |
| Layering needed | rarely | **yes** — styled outfits, outerwear |
| Right model | **FASHN VTON 1.5** | **this one** |

Service 2 is where 150 s costs nothing and $0.066 per image is a rounding error against
what a seller pays for a photoshoot. It is also where layering — the one thing this model
does that nothing else does — actually matters.

## What else exists

Surveyed on HuggingFace, 2026-08-14:

| Model | Params | VRAM | Speed | Layering | License |
|---|---|---|---|---|---|
| **This repo** (Qwen-Image-Edit-2509) | 20.4B + 8.3B | ~58 GB | 150 s (A100) | **yes** | **none** ⚠ |
| **`fashn-ai/fashn-vton-1.5`** | 972M | ~8 GB | **5 s** (H100) | no | Apache 2.0 |
| `fal/flux-klein-9b-virtual-tryon-lora` | 9B | ~24 GB | 28 steps | no (swap) | Apache 2.0 |
| `yisol/IDM-VTON` | SD1.5 | ~16 GB | ~20 s | no | research only |

**FASHN VTON 1.5 is the strongest option for Service 1**, by a wide margin:

- 21× smaller, ~30× faster, 7× less VRAM
- **maskless** — no segmentation step to build, run or get wrong
- uses DWPose for keypoints, exactly like this repo, so the preprocessing already written
  here carries over
- Apache 2.0, no commercial restriction
- capped at 576×864, and explicitly cannot layer

## Survey of the 2025–2026 field

Checked 2026-08-14, against the shortlist from YouTube reviews plus what the
[Awesome-Try-On-Models](https://github.com/Zheng-Chong/Awesome-Try-On-Models)
index turned up. **Licence is the sharpest filter** — several of the strongest
models cannot be used commercially at all.

| Model | What it is | Licence | Verdict for us |
|---|---|---|---|
| **FLUX.2 Klein 9B** | top-ranked open image-editing model; `fal/flux-klein-9b-virtual-tryon-lora` adds try-on | **non-commercial** | **blocked.** The 4B sibling is Apache 2.0; the 9B is not, and the try-on LoRA targets the 9B |
| **Z-Image-Turbo** | 6B, **8 NFEs**, sub-second on H800, 16 GB VRAM | Apache 2.0 | **not a try-on model** — text-to-image. Interesting as a *base* to train on, and for generating studio backdrops |
| **OmniTry** | mask-free try-on of **anything** — jewellery, watches, glasses | Apache 2.0 | **useful for Service 2.** Accessories are a real gap in garment-only models |
| **PG-VTON** (CVPR 2026) | training-free, mask-free, **single pass** | released | worth testing. "CGPR-VTON" appears to be this; no model by that exact name exists |
| Mobile-VTON (CVPR 2026) | on-device | released | wrong target — we have GPUs |
| **FitVTON** (2026-06) | **fit-aware sizing control** | released | closest thing to the sizing requirement in the plan |
| **Voost** | bidirectional try-on **and try-off**, DiT | released | try-off is interesting for catalogue prep |
| Garments2Look (CVPR 2026) | outfit-level, clothing **+ accessories** | dataset | matches the photo-studio brief |
| MagicTryOn / DreamVVT / CatV2TON / OmniTryOn | **video** try-on, DiT-based | released | this is the video path, when we get there |

"AI Virtual Try-On | My Project Showcased at IBM Pre-AI Summit" is a conference
talk, not a model — nothing to evaluate.

**Nothing in this list beats what we already run on quality.** FASHN 1.5 was
tested and fell short; the rest are either licence-blocked, not try-on models,
or in the same weight class as what we have. So the answer is not a different
model — it is fewer forward passes through this one.

## Unit economics

**Service 1 on this model.** 57.7 GB of weights means a ~3 minute cold start, so
serverless is impossible — the GPU has to stay warm. That is **$1,161/month** for one
A100 before a single request arrives, and one A100 serves 24 images/hour.

**Service 1 on FASHN 1.5.** 2 GB of weights load in seconds, so RunPod serverless works
and you pay per second of actual use. On an RTX A6000 at $0.53/hr it lands near
**$0.002/image** — roughly **35× cheaper**, with no monthly floor.

The floor is the real difference, not the per-image number. A model you must keep warm
turns every idle hour into cost; a model that cold-starts in seconds does not.

**Service 2 on this model.** Batch jobs on a spot/on-demand A100 spun up per batch.
100 images ≈ 4 GPU-hours ≈ $6.40, plus ~5 min of load time amortised across the batch.
Against the price of a real photoshoot, that is nothing.

## The licensing problem

**The upstream repo ships no LICENSE file.** The only licence in the tree belongs to the
bundled `diffusers` fork (Apache 2.0, HuggingFace's, not the authors'). No licence granted
means all rights reserved — **commercial use is not permitted by default.**

This must be resolved before Service 2 ships. Options, in order of preference:

1. Ask the authors for written permission or an explicit licence.
2. Use the base `Qwen/Qwen-Image-Edit-2509` (Apache 2.0) and **train your own layering
   LoRA**. The LoRA here is rank 32 on `to_k/to_q/to_v/to_out.0` — a modest training job,
   and it removes the dependency entirely.
3. Drop layering for now and ship Service 2 on FASHN 1.5 as well.

Option 2 is also the strategically better answer: a layering LoRA trained on **your own**
model roster and product catalogue is a real asset, and nobody else has it.

## What the plan should not use a generative model for

**Size matching** is body-measurement estimation from a photo (SMPL/SMPL-X fitting) plus
garment size charts. It runs in milliseconds on a CPU.

**Style recommendation** is embedding retrieval and ranking — CLIP or SigLIP vectors, a
vector index, and a feedback loop from what actually sold.

Both are orders of magnitude cheaper than image generation, and both get better with usage
data. That is where a defensible product lives; the generative model is a commodity that
anyone can rent.

**Video** is a separate stack — Qwen-Image-Edit does not generate video at all. That is a
later decision, not an extension of this one.

## Recommendation

1. **Service 1 on FASHN VTON 1.5**, serverless. It meets the latency bar, it is Apache
   2.0, and the DWPose preprocessing in this repo already fits it.
2. **Keep this model for Service 2**, as an offline batch worker — but resolve the licence
   first, and plan to train your own layering LoRA on the Apache-2.0 base.
3. **Run `sweep_steps.py`** before tuning anything else, so the steps/CFG decision is made
   on measured quality rather than guesswork.
4. **Build sizing and style recommendation as their own services.** They are cheap, they
   compound with data, and they do not belong in the image pipeline.

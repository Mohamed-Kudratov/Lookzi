# Build plan — the seven tools

Target: the toolset FASHN ships (Virtual Try-On, Product to Model, Packshot,
Model Swap, Model Creation, Consistent Models, Short Videos), built for the CIS
market where none of the incumbents operate.

## The architectural insight

**FASHN ships seven tools. Six of them are one model with different inputs.**

`Qwen-Image-Edit-2509` is an instruction-driven editor that takes multiple
reference images. "Put this garment on this person", "put this flat-lay on a
model", "swap the model but keep the garment" — these are the same operation
with different references and a different instruction. That means:

- **one** base model resident in VRAM
- task-specific **LoRAs swapped per request** — they are ~90 MB and load in about
  a second, against ~3 minutes for a base model
- **one GPU serves six products**

Running six separate models would mean six warm GPUs or six cold starts. This is
the difference between a service that can be afforded and one that cannot. The
existing `pipeline.py` already loads adapters this way, so the mechanism is
built — it needs a registry and a swap call, not a rewrite.

Only Short Videos needs a genuinely different model.

## Model per tool

| Tool | Model | Licence | Notes |
|---|---|---|---|
| **Virtual Try-On** | Qwen-Image-Edit-2509 + the layering LoRA we run + Lightning | negotiate | already working; quality proven against FASHN 1.5 |
| **Product to Model** | same base, own LoRA | Apache 2.0 base | prior art: `Any2AnyTryon`, `DreamFit`, `Voost` |
| **Packshot** (try-off) | `TryOffDiff` or `TryOffAnyone`; `Voost` does both directions | open | small dedicated models; see [awesome-virtual-try-off](https://github.com/rizavelioglu/awesome-virtual-try-off) |
| **Model Swap** | same base, multi-image reference | Apache 2.0 base | prior art: `RefTon` |
| **Model Creation** | **Z-Image-Turbo** (6B, 8 NFE, sub-second on H800, 16 GB) | **Apache 2.0** | or FLUX.2 Klein **4B** (the 9B is non-commercial) |
| **Consistent Models** | **per-model LoRA** on our own roster + reference conditioning | ours | the keystone — see below |
| **Short Videos** | **Wan 2.2** (MoE; TI2V-5B runs on a 4090) | **Apache 2.0** | alternatives: LTX-2.3, HunyuanVideo 1.5, both Apache 2.0 |

Beyond FASHN's list, the product plan also needs:

- **Sizing** — SMPL/SMPL-X body fitting from a photo plus garment size charts.
  Runs on CPU in milliseconds. Closest published work: **FitVTON** (2026-06),
  which has explicit fit-aware sizing control.
- **Style recommendation** — CLIP or SigLIP embeddings, a vector index, and a
  ranking model trained on what actually sold.

Neither is a generative problem, and neither should be built with a diffusion
model.

## Consistent Models is the keystone

Everything commercial in the photo studio depends on it. A brand shooting forty
products needs the *same* model across all forty, or the catalogue looks wrong.
FASHN sells this as a separate tool because it is the hard part.

Two approaches, and the right one is not the fashionable one:

- **Zero-shot identity adapters** (IP-Adapter, InstantID, PuLID) — no training,
  but identity drifts across poses and lighting. Fine for a demo, visible in a
  catalogue.
- **A LoRA per model** — train once per face on 15–30 images, ~20 minutes on an
  A100. Identity is locked because it is baked into weights, not inferred from a
  reference at inference time.

**Train a LoRA per model.** The roster then becomes a real asset: a library of
consistent, exclusive, owned models that no competitor can copy and that
improves every time you add one. That is the moat — not the base model, which
anyone can rent.

## Sequence

**Phase 0 — get the cost under a cent.** Lightning 8-step, measured with
`sweep_steps.py`. At $0.066/image the business does not work; the market price
is $0.04. Nothing else matters until this lands. *Days.*

**Phase 1 — Model Creation + Consistent Models.** Build the roster: 10–20 models
covering the demographics your sellers actually sell to, one LoRA each. This is
the asset every other tool draws on, and it is worth doing carefully. Z-Image-Turbo
generates candidates, then a LoRA per chosen face. *2–3 weeks.*

**Phase 2 — Product to Model.** The core studio product and the one sellers pay
for repeatedly. A seller uploads a flat-lay; it comes back on a roster model.
Offline batch, so latency is free and the heavy model is fine here. *2–4 weeks.*

**Phase 3 — Packshot.** Cheap to add, and the inverse of Phase 2 — a small
dedicated try-off model, not the big one. Useful on its own for cleaning up
seller photos into catalogue images. *1 week.*

**Phase 4 — Virtual Try-On, productised.** Already working. Ship it inside the
local product with sizing, not as a bare API — that market is priced at $0.04
by a competitor who open-sourced their model. *2 weeks.*

**Phase 5 — Model Swap.** Same base, another LoRA. *1 week.*

**Phase 6 — Short Videos.** Wan 2.2 image-to-video over the stills from Phase 2.
Bursty and expensive, so price it separately. *2–3 weeks.*

Sizing and style recommendation run in parallel from Phase 2 onward — they are a
different stack and a different person.

## Infrastructure

| | GPU | Mode | Serves |
|---|---|---|---|
| Image worker | A100 80GB or H100 | **warm**, LoRAs hot-swapped | tools 1, 2, 4 |
| Model creation | L4 / 4090 | on demand | tool 5 |
| Try-off | L4 / 4090 | on demand | tool 3 |
| LoRA training | A100 | per job, ~20 min | tool 6 |
| Video | A100 / 4090 | burst | tool 7 |

One warm image worker is the only standing cost. Everything else is bursty and
should be serverless.

## On the team question

The model work here is integration and LoRA training, not research. Nothing in
this plan requires inventing an architecture — every component exists with
released weights. The genuinely hard parts are elsewhere:

- **keeping cost per image under a cent** — an ops problem
- **making images that actually sell** — a product and taste problem
- **the roster** — a curation problem

FASHN reached this toolset on roughly $2M and a small team. This is not a
fifty-engineer problem. One strong ML engineer, one backend engineer, and
someone with taste in fashion imagery will get further than a large team without
the third.

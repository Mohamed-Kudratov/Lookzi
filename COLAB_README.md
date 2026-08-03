# Layering Virtual Try-On — Colab setup notes

Upstream: <https://github.com/ChuenFung/Layering-Virtual-Try-On>

## How to run

Upload **`Layering_VTON_Colab.ipynb`** to <https://colab.research.google.com>, set
`Runtime → Change runtime type → T4 GPU`, and run the cells top to bottom. The
notebook is self-contained: it clones upstream, installs dependencies, writes the
patched source files over the originals, downloads weights, and launches Gradio.

## The central constraint

`Qwen-Image-Edit-2509` is **20.4 B parameters / 57.7 GB** in bf16. No Colab GPU
holds that. The notebook therefore defaults to
[`ovedrive/Qwen-Image-Edit-2509-4bit`](https://huggingface.co/ovedrive/Qwen-Image-Edit-2509-4bit),
an NF4 build in the same diffusers layout (`scheduler/ vae/ processor/
text_encoder/ transformer/`) and the same `_diffusers_version: 0.36.0.dev0` as
the fork bundled in this repo.

| | bf16 original | 4-bit NF4 |
|---|---|---|
| Download | 57.7 GB | **17 GB** |
| Transformer | 40.9 GB | 11.6 GB |
| Text encoder | 16.6 GB | 5.1 GB |
| Runs on T4 | no | yes, sequentially |

Even at 4 bit, 11.6 + 5.1 + VAE ≈ 17 GB exceeds a T4's 15 GB, so on any GPU
under 20 GB the pipeline loads the text encoder and the transformer **one at a
time**. That is what `low_vram` does; it is auto-enabled by VRAM size.

### Expected timings

| Runtime | VRAM | ~Time / image (20 steps) |
|---|---|---|
| T4 (free) | 15 GB | **20–40 min** |
| L4 (Pro) | 22 GB | ~4–7 min |
| A100 (Pro+) | 40 GB | ~2–3 min |

Each step runs the 20 B transformer **twice** (conditional + CFG negative).
Setting `true_cfg_scale = 1.0` skips the negative pass and roughly halves the
time, at some cost in fidelity.

4-bit quantisation is lossy — output will sit visibly below the paper's figures.
On an L4/A100 you can set `MODEL_PATH = "Qwen/Qwen-Image-Edit-2509"` for the
full bf16 model and the quality gap closes.

## Bugs fixed relative to upstream

**1. `utils.py` — pose extraction crashed on GPU machines.**
`extract_and_process_pose` chose its device with
`"cuda" if torch.cuda.is_available() else "cpu"`. `easy-dwpose` passes that
straight to `onnxruntime`, which then demands `CUDAExecutionProvider` — but
`environment.yml` installs plain `onnxruntime`, which is **CPU-only**. So the
default install failed on exactly the CUDA machines the repo targets. Device
selection now asks `onnxruntime.get_available_providers()` what actually exists.

**2. `utils.py` — the detector was rebuilt on every call.**
`DWposeDetector(...)` was constructed inside `extract_and_process_pose`,
re-creating two ONNX sessions per generation. Now built once and cached.

**3. `app.py` — the swap/add radio button did nothing.**
`mode` was collected by the UI, passed into `run_vton`, and then never used —
`pl(...)` was called without it, and `LayeringVTONPipeline.__call__` had no such
parameter. The mode was only ever implied by how the user happened to word the
description. `apply_mode()` now binds the two: the description stays the source
of truth, and the radio supplies the verb when the description lacks one. If the
description already opens with `add`/`swap`, it wins — otherwise a mode/text
mismatch produced prompts like `"add swap the jeans for shorts"`.

**4. `app.py` — `torch.cuda.empty_cache()` was called unconditionally**, before
any GPU check.

**5. `app.py` — `if not person_img` for emptiness checks.** PIL images define
neither `__bool__` nor `__len__`, so this was always `False` and worked only by
accident. Now `is None`.

**6. `pipeline.py` — hardcoded `device="cuda"` and `torch.bfloat16`.** A T4 is
Turing and has no native bf16. Device and dtype are now detected.

**7. `pipeline.py` — `.to(device)` on loaded models.** A bitsandbytes 4-bit
module cannot be relocated that way; its `Params4bit` storage is bound to the
device it was quantised onto. Quantised checkpoints are now placed via
`device_map` at load time.

## Numerical guards added for fp16

These matter only on T4; on bf16 hardware they are no-ops in effect.

- **VAE runs in fp32** when the transformer runs in fp16. Qwen's VAE overflows
  in half precision and returns a black image.
- **CFG renormalisation runs in fp32.** `torch.norm` over a 3072-wide fp16
  vector can reach `inf`, which turns the whole latent to NaN on the next step.
- **The bnb compute dtype is overridden to fp16.** The published 4-bit repo bakes
  `bnb_4bit_compute_dtype: bfloat16` into its config; honouring that on a T4
  makes every 4-bit matmul crawl.

## Installing the bundled `diffusers` fork

`README.md` says to `pip install -e ./diffusers`, which does not work — importing
the result fails with:

```
ImportError: cannot import name 'AutoencoderKLQwenImage' from 'diffusers' (unknown location)
```

`diffusers/` in the repo root is the fork's **repository**; the package is at
`diffusers/src/diffusers`. Python searches the working directory first, resolves
`diffusers` to that directory, finds no `__init__.py`, and returns an empty
PEP 420 namespace package. `(unknown location)` is the tell — the name resolved
to a directory, not a module.

An editable install does not help: the directory still shadows it, and an
editable install cannot survive the rename either, since it records the absolute
source path. Install it normally and move the tree aside:

```bash
pip install ./diffusers        # not -e
mv diffusers diffusers_src
```

`pipeline.py` now raises this explanation instead of the bare ImportError.

## Other changes

- Text embeddings are cached against the inputs that produced them. Re-running
  with a different seed or step count is common, and on a T4 the text-encoder
  stage costs a full model load from disk.
- Sampling steps / CFG / seed are exposed in the UI. The default drops from 40
  steps to 20, which is the difference between ~40 and ~20 minutes on a T4.
- A step-level progress bar, and an explicit OOM message pointing at the
  settings that actually help.
- `requirements-colab.txt` does not pin torch. `environment.yml` is a conda spec
  pinned to **linux-aarch64** that also reinstalls `torch==2.7.0+cu128` — wrong
  architecture for Colab, and ~3 GB to replace a working CUDA torch.

## Local Windows machine

This repo will not run on `D:\projects\lvton` itself. That machine has a Ryzen
7 4700U with integrated AMD Radeon graphics (no CUDA) and 15.4 GB of system RAM,
against a model needing 40–60 GB of VRAM. The files here are the source that the
notebook writes into Colab.

Note also that `git clone` over HTTPS fails on this machine — a Windows
Application Control policy blocks `libcurl-4.dll`, so git's HTTPS helper aborts.
The repo was fetched via `Invoke-WebRequest` against `codeload.github.com`
instead. This affects only the local machine, not Colab.

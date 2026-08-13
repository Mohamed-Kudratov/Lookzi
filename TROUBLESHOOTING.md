# Troubleshooting

## `ImportError: cannot import name 'AutoencoderKLQwenImage' from 'diffusers' (unknown location)`

`diffusers/` in the repo root is the fork's **repository**, not its package — the
package is at `diffusers/src/diffusers`. Python searches the working directory
first, resolves `diffusers` to that directory, finds no `__init__.py`, and hands
back an empty [PEP 420](https://peps.python.org/pep-0420/) namespace package.
The `(unknown location)` is the tell: the name resolved to a directory, not a
module.

The repo's own `README.md` says `pip install -e ./diffusers`, which cannot work
— the directory still shadows the editable install, and an editable install
cannot survive moving the tree either, because it records the absolute source
path. Install normally, then move the tree aside:

```bash
pip install ./diffusers        # not -e
mv diffusers diffusers_src
```

`runpod_setup.sh` does this. `pipeline.py` also catches the ImportError and
prints this explanation rather than the bare message.

## `AssertionError` on a `site-packages` check

Debian-based images (Colab, some RunPod templates) install into
`/usr/local/lib/pythonX.Y/**dist-packages**`. Any check for `site-packages` is
wrong there. What actually distinguishes the failure is that a shadowed
namespace package has `__file__ = None`.

## `CUDA out of memory`

The bf16 weights are 57.7 GB — transformer 40.9 GB plus text encoder 16.6 GB.

- **Above ~70 GB VRAM:** both stay resident.
- **40–70 GB:** `low_vram` loads them one at a time. Auto-enabled under 20 GB;
  force it with `--low-vram` (CLI) or `low_vram=True`.
- **Under 40 GB:** use `ovedrive/Qwen-Image-Edit-2509-4bit` (~17 GB).

Other levers, in order of how much they buy you: `--cfg 1.0` skips the negative
pass and roughly halves both time and peak activations; fewer `--steps` cuts
time but not peak memory. Fragmentation is real — if a rerun OOMs where the
first run succeeded, restart the process.

## `CUDAExecutionProvider is not in available providers`

`easy-dwpose` asks onnxruntime for CUDA whenever its device is not `"cpu"`, but
`environment.yml` installs plain `onnxruntime`, which is CPU-only. Upstream's
`utils.py` chose the device with `torch.cuda.is_available()`, so pose extraction
failed on exactly the GPU machines this repo targets.

`utils.py` now queries `onnxruntime.get_available_providers()`. DWPose on CPU
takes about two seconds, so plain `onnxruntime` is the right choice — do not
"fix" this by installing `onnxruntime-gpu`.

## Black output, or NaN latents

Only on fp16 hardware (T4 and other Turing cards). Two guards are already in
`pipeline.py`:

- The VAE runs in **fp32** when the transformer runs in fp16. Qwen's VAE
  overflows in half precision.
- The CFG renormalisation runs in **fp32**. `torch.norm` over a 3072-wide fp16
  vector can reach `inf`, which turns the whole latent to NaN on the next step.

If it still happens, the fp16 path is unstable for those inputs. Any Ampere or
newer GPU uses bf16 and does not have the problem.

## The swap/add radio seems to do nothing

It did nothing upstream — `mode` was collected by the UI and never passed to the
pipeline. `apply_mode()` now binds them: the description stays the source of
truth, and the radio supplies the verb when the description lacks one. If the
description already opens with `add` or `swap`, it wins, so a mode/text mismatch
cannot produce `"add swap the jeans for shorts"`.

## Unexpected LoRA keys

The LoRA targets `to_k/to_q/to_v/to_out.0` at rank 32. A handful of unexpected
keys is tolerable. If *every* key is unexpected, the base model and the LoRA
disagree and output will be garbage — check `MODEL_PATH`.

## Gradio is unreachable on RunPod

`app.py` binds `0.0.0.0:7860`. The pod template must expose **HTTP port 7860**,
and you reach it at `https://<POD_ID>-7860.proxy.runpod.net` — not at the pod's
IP. `share=True` is off by default because RunPod's proxy already does the job;
set `GRADIO_SHARE=1` only if you need the `gradio.live` tunnel.

## The model re-downloads after a pod restart

`HF_HOME` must point at the **volume**, not the container disk. `runpod_setup.sh`
sets `/workspace/hf_cache`. Anything outside the volume mount is wiped when the
pod stops.

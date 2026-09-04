#!/usr/bin/env python3
"""LTX-2.5 behind HTTP, so the weights are read once instead of once per clip.

This exists because of a measurement. Run from the command line, a five-second
clip took 651 seconds and then 640 -- and inside that, the denoising loop took
eight. Everything else was loading: 42 GB of transformer and 26 GB of text
encoder off a network volume, cast to fp8, used once, and thrown away when the
process exited. The two-stage pipeline builds the transformer more than once
per run, so the same weights were read several times in a single clip.

`DistilledPipeline` builds every component in `__init__` -- prompt encoder,
image conditioner, diffusion stage, upsampler, decoder -- and `__call__` only
samples. So holding one instance turns a 645-second job into an 8-second one
plus whatever the offloaded parts cost to move back. That is the whole design
of this file.

Third interpreter, third port. LTX pins its own torch and builds two local
packages; it cannot share the try-on stack's pins any more than Z-Image could.

    /opt/ltx/.venv/bin/python -m service.ltx_server

Port 8021, and /health names the model. That is not decoration: the pod image
runs nginx on 8001, which answered a health check that only asked whether
something was listening, so Z-Image never started and two tools were quietly
dead for a day. Every server here says who it is.
"""
import io
import os
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

MODELS = os.environ.get("LTX_MODELS", "/workspace/models/ltx-2.5")
PORT = int(os.environ.get("LTX_PORT", "8021"))

# bf16 weights with fp8-cast, not the `comfy-int8-convrot` files in the same
# repository. Those are ComfyUI's quantisation and this loader cannot read them
# -- it dies on `mlp.down_proj.comfy_quant`. fp8-cast stores the weights in fp8
# and upcasts to bf16 to compute, so it needs no fp8 tensor cores and runs on
# an A100. Thirty-seven gigabytes were downloaded before that was understood.
TRANSFORMER = os.environ.get(
    "LTX_TRANSFORMER",
    f"{MODELS}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
TEXT_ENCODER = os.environ.get(
    "LTX_TEXT_ENCODER",
    f"{MODELS}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
VIDEO_VAE = os.environ.get(
    "LTX_VIDEO_VAE", f"{MODELS}/vae/ltx-2.5-video-vae-bf16.safetensors")
AUDIO_VAE = os.environ.get(
    "LTX_AUDIO_VAE", f"{MODELS}/vae/ltx-2.5-audio-vae-bf16.safetensors")
UPSAMPLER = os.environ.get(
    "LTX_UPSAMPLER",
    f"{MODELS}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")

# The card already holds the try-on model and Z-Image -- about 38 GB of 80 --
# so the text encoder is offloaded rather than resident. It runs once per clip,
# to turn a sentence into embeddings, and the pod has 1889 GB of system memory
# and 252 cores to do it with. Keeping 26 GB on the card for that would cost
# more than it saves.
OFFLOAD = os.environ.get("LTX_OFFLOAD", "cpu")
QUANT = os.environ.get("LTX_QUANT", "fp8-cast")

# Portrait, because these are clothes. Stage one samples at half of this and
# stage two upsamples, which is why the numbers must stay divisible by 32.
WIDTH = int(os.environ.get("LTX_WIDTH", "704"))
HEIGHT = int(os.environ.get("LTX_HEIGHT", "1280"))
FPS = int(os.environ.get("LTX_FPS", "24"))
MAX_SECONDS = float(os.environ.get("LTX_MAX_SECONDS", "10"))

# What the model is told when the seller says nothing. Deliberately small
# motion: this is a garment on a white background, and a clip that invents a
# spin or a runway is a clip the seller cannot use. The camera line is
# restrained on purpose -- "the camera pushes in very slowly" produced a clip
# that had cropped the hem away by the last frame.
DEFAULT_PROMPT = (
    "A garment photographed on a plain white studio background. The fabric "
    "moves and settles very gently. The camera holds almost still. Soft even "
    "studio light, sharp focus, no people, no hands.")

app = FastAPI(title="lookzi-ltx")

_pipe = None
_error = None
_load_seconds = None
_stats = {"served": 0, "failed": 0, "seconds": 0.0}
# One card, one clip at a time. Two sampling at once would contend for the same
# VRAM on a card that is already holding two other models, and the failure mode
# of losing that race is an allocation error rather than a slow clip.
_gpu = threading.Lock()


def load():
    global _pipe, _load_seconds, _error
    t = time.time()
    print(f"[ltx] loading {os.path.basename(TRANSFORMER)}", flush=True)
    try:
        import torch
        from ltx_core.model.video_vae.transformer import DiffVAEMode
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode

        paths = ModelPaths.from_split(
            transformer_path=TRANSFORMER,
            text_encoder_path=TEXT_ENCODER,
            video_vae_path=VIDEO_VAE,
            audio_vae_path=AUDIO_VAE,
            duration_head_path=None,
        )
        _pipe = DistilledPipeline(
            model_paths=paths,
            spatial_upsampler_path=UPSAMPLER,
            loras=(),
            quantization=QuantizationPolicy(QUANT),
            offload_mode=OffloadMode(OFFLOAD),
            diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
        )
        _load_seconds = round(time.time() - t, 1)
        print(f"[ltx] ready in {_load_seconds}s", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        # Held rather than raised, so /health can say what went wrong. A server
        # that dies at startup leaves whoever is waiting to guess between a
        # crash, a slow load and a wrong address.
        _error = f"{type(exc).__name__}: {exc}"
        print(f"[ltx] failed to load: {_error}", flush=True)


threading.Thread(target=load, daemon=True).start()


@app.get("/health")
def health():
    return {"ready": _pipe is not None, "error": _error,
            # Named, so a health check cannot be satisfied by whatever else
            # happens to be listening on this port.
            "model": "LTX-2.5-distilled",
            "load_seconds": _load_seconds, "busy": _gpu.locked(),
            "served": _stats["served"], "failed": _stats["failed"],
            "mean_seconds": (round(_stats["seconds"] / _stats["served"], 2)
                             if _stats["served"] else None)}


@app.post("/video")
def video(image: UploadFile = File(...),
          prompt: str = Form(""),
          seconds: float = Form(5.0),
          seed: int = Form(42),
          width: int = Form(0), height: int = Form(0)):
    """One still in, one mp4 out.

    `seconds` rather than a frame count, because that is what a seller has an
    opinion about. The model wants 8n+1 frames, so the number is rounded to the
    nearest one that fits and the returned header says what was actually made.
    """
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    if _pipe is None:
        raise HTTPException(503, "still loading")

    import tempfile

    from PIL import Image

    raw = image.file.read()
    if not raw:
        raise HTTPException(400, "empty image")
    try:
        still = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(400, f"not an image: {exc}") from exc

    w = int(width) or WIDTH
    h = int(height) or HEIGHT
    # The conditioning frame has to be the frame the model is generating, or
    # the first frame of the clip is a resize of the seller's photograph
    # stitched onto footage of a different shape.
    still = still.resize((w, h), Image.LANCZOS)

    secs = max(1.0, min(float(seconds), MAX_SECONDS))
    # 8n+1: the temporal compression works in eights and the pipeline rejects
    # anything else, so this is rounded here rather than left to fail deep
    # inside the sampler with a shape error.
    frames = int(round((secs * FPS - 1) / 8)) * 8 + 1

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "still.png")
        out_path = os.path.join(tmp, "clip.mp4")
        still.save(src)

        with _gpu:
            try:
                import torch
                from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
                from ltx_pipelines.utils.args import ImageConditioningInput
                from ltx_pipelines.utils.media_io import encode_video

                result = _pipe(
                    prompt=(prompt or "").strip() or DEFAULT_PROMPT,
                    seed=int(seed),
                    height=h, width=w,
                    num_frames=frames,
                    frame_rate=FPS,
                    # Frame 0 at full strength: this is the picture the seller
                    # chose, and the clip should begin on it rather than near it.
                    # crf is the fourth field and is not optional in the
                    # tuple; None means "use the value that matches the
                    # checkpoint's version", which is what the CLI does when
                    # the flag is left off.
                    images=[ImageConditioningInput(src, 0, 1.0, None)],
                    vae_dtype=torch.bfloat16,
                    tiling_config=AUTO_TILING,
                )
                encode_video(
                    video=result.video, fps=FPS,
                    # No audio. LTX generates it, and a product listing does not
                    # want a soundtrack the seller did not ask for and cannot
                    # hear before they publish.
                    audio=None,
                    output_path=out_path,
                    video_chunks_number=get_video_chunks_number(
                        result.num_frames, result.tiling_config),
                )
            except Exception as exc:                          # noqa: BLE001
                _stats["failed"] += 1
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

        data = open(out_path, "rb").read()

    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed
    return Response(content=data, media_type="video/mp4",
                    headers={"X-Seconds": str(elapsed),
                             "X-Frames": str(frames),
                             "X-Fps": str(FPS),
                             "X-Width": str(w), "X-Height": str(h)})


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

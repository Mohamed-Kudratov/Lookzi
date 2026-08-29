#!/usr/bin/env python3
"""FASHN VTON 1.5 behind HTTP on the pod, beside the other two.

    /opt/fashn-venv/bin/python -m service.fashn_server

A third process for the same reason there is a second one: this model brings
its own dependency tree -- onnxruntime, its own huggingface_hub -- and the
try-on stack caps versions it wants. Three interpreters, one card.

It earns its place on measurements taken on this pod, against our own pipeline,
on the same four pairs: 17.5s against 32s, 3.6 GiB of VRAM against 15.8, and a
garment category that is actually read. Ours turns a dress into a tunic over
the trousers the model already wore, and takes a headscarf off a woman who was
wearing one. This one does neither.

Apache-2.0, which is what makes it usable at all. It is a competitor's model,
published deliberately; using it is allowed and worth being clear-eyed about.
"""
import io
import os
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

WEIGHTS = os.environ.get("FASHN_WEIGHTS", "/workspace/models/fashn-vton-1.5")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

# tops / bottoms / one-pieces. The one control the customer picks that this
# model genuinely answers -- ours ignores the equivalent, measured five ways in
# docs/CONTROLS.md.
CATEGORIES = ("tops", "bottoms", "one-pieces")

app = FastAPI(title="Lookzi fashn-vton")

_pipe = None
_load_seconds = None
_error = None
_gpu = threading.Lock()
_stats = {"served": 0, "failed": 0, "seconds": 0.0}


def load():
    global _pipe, _load_seconds, _error
    from fashn_vton import TryOnPipeline
    t = time.time()
    print(f"[fashn] loading {WEIGHTS}", flush=True)
    try:
        _pipe = TryOnPipeline(weights_dir=WEIGHTS)
        _load_seconds = round(time.time() - t, 1)
        print(f"[fashn] ready in {_load_seconds}s", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        _error = f"{type(exc).__name__}: {exc}"
        print(f"[fashn] failed to load: {_error}", flush=True)


@app.on_event("startup")
def _startup():
    threading.Thread(target=load, daemon=True).start()


@app.get("/health")
def health():
    return {"ready": _pipe is not None, "error": _error, "model": WEIGHTS,
            "load_seconds": _load_seconds, "busy": _gpu.locked(),
            "served": _stats["served"], "failed": _stats["failed"],
            "mean_seconds": round(_stats["seconds"] / _stats["served"], 2)
            if _stats["served"] else None}


def _read(upload: UploadFile, name: str) -> Image.Image:
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{name} is larger than {MAX_BYTES} bytes")
    if not data:
        raise HTTPException(400, f"{name} is empty")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:                                         # noqa: BLE001
        raise HTTPException(400, f"{name} is not an image this can read")


@app.post("/generate")
def generate(person: UploadFile = File(...), garment: UploadFile = File(...),
             category: str = Form("tops")):
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    if _pipe is None:
        raise HTTPException(503, "the model is still loading")
    cat = (category or "tops").strip().lower()
    if cat not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")

    p = _read(person, "person")
    g = _read(garment, "garment")

    started = time.time()
    with _gpu:
        try:
            out = _pipe(person_image=p, garment_image=g, category=cat).images[0]
        except Exception as exc:                              # noqa: BLE001
            _stats["failed"] += 1
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed

    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Seconds": str(elapsed), "X-Width": str(out.width),
                             "X-Height": str(out.height), "X-Category": cat})


def main():
    import uvicorn
    port = int(os.environ.get("FASHN_PORT", "8002"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

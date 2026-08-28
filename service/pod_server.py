#!/usr/bin/env python3
"""The model, behind HTTP, on the pod.

    python -m service.pod_server        # on the pod

One process holding one loaded model, answering one request at a time. It knows
nothing about jobs, credits, users or the queue -- those live where they can be
backed up, and this lives where the GPU is.

Why HTTP rather than the queue-polling worker in gpu_worker.py. That one needs
to reach Postgres, which means Postgres has to be reachable from a rented
machine on someone else's network, which means either exposing the database to
the internet or paying for a managed one before the product has a single user.
This inverts it: the pod exposes one endpoint on loopback, we reach it through
the ssh connection we already have, and nothing about our side becomes public.

    ssh -N -L 18000:127.0.0.1:8000 root@<pod-ip> -p <pod-port>

Bound to 127.0.0.1 deliberately, and that is the whole security model. The pod
is a rented machine with a public address; a service on 0.0.0.0 there is a
service on the internet. On loopback it is reachable only by something already
inside the pod, and an ssh tunnel is exactly that.

Images travel as bytes in the request. That is the right trade at this size and
the wrong one later: at scale the bytes should go straight between the customer
and object storage with only signed links crossing the wire, which is what
runpod_handler.py does. Doing it that way now would mean a public bucket and an
account, before knowing whether anyone wants the product.
"""
import io
import os
import sys
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH = os.environ.get("MODEL_PATH", "ovedrive/Qwen-Image-Edit-2509-4bit")
LORA_DIR = os.environ.get("LORA_DIR", os.path.join(ROOT, "weights"))
LIGHTNING = int(os.environ.get("LIGHTNING", "8"))
MAX_BYTES = int(os.environ.get("MAX_INPUT_BYTES", str(32 * 1024 * 1024)))

app = FastAPI(title="Lookzi pod server")

_pipe = None
_load_seconds = None
_error = None
# One card, one model, one image at a time. Two requests sampling at once would
# not go twice as fast; they would contend for the same VRAM and make both slow
# and occasionally kill one with an allocation failure. Queueing here keeps the
# failure mode boring.
_gpu = threading.Lock()
_stats = {"served": 0, "failed": 0, "seconds": 0.0}


def load():
    global _pipe, _load_seconds, _error
    from pipeline import LayeringVTONPipeline
    t = time.time()
    print(f"[pod] loading {MODEL_PATH}", flush=True)
    try:
        _pipe = LayeringVTONPipeline(MODEL_PATH, LORA_DIR, lightning=LIGHTNING)
        _load_seconds = round(time.time() - t, 1)
        print(f"[pod] ready in {_load_seconds}s", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        # Held rather than raised, so /health can say what went wrong. A worker
        # that dies at startup leaves whoever is waiting to guess between a
        # crash, a slow load and a wrong address.
        _error = f"{type(exc).__name__}: {exc}"
        print(f"[pod] failed to load: {_error}", flush=True)


@app.on_event("startup")
def _startup():
    # In a thread, so the port is listening while the weights load. Otherwise
    # every health check during the three minutes of loading is a connection
    # refused, which reads as "the server is not running".
    threading.Thread(target=load, daemon=True).start()


@app.get("/health")
def health():
    return {"ready": _pipe is not None, "error": _error,
            "model": MODEL_PATH, "lightning": LIGHTNING,
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
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(400, f"{name} is not an image: {exc}")


@app.post("/generate")
def generate(person: UploadFile = File(...),
             garment: UploadFile = File(...),
             mode: str = Form("upper"),
             description: str = Form(""),
             seed: int = Form(42)):
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    if _pipe is None:
        raise HTTPException(503, "the model is still loading")

    from utils import process_inputs

    person_img = _read(person, "person")
    garment_img = _read(garment, "garment")
    p, g, pose = process_inputs(person_img, garment_img, None)

    started = time.time()
    with _gpu:
        try:
            out = _pipe(person_img=p, garment_img=g, pose_img=pose,
                        description=description or "the garment",
                        mode=mode, seed=int(seed))
        except Exception as exc:                              # noqa: BLE001
            _stats["failed"] += 1
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed

    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(
        content=buf.getvalue(), media_type="image/png",
        # In headers rather than a JSON envelope, so the body stays a plain
        # image that anything can open, including a browser pointed at it.
        headers={"X-Seconds": str(elapsed),
                 "X-Width": str(out.width), "X-Height": str(out.height)})


def main():
    import uvicorn
    port = int(os.environ.get("POD_SERVER_PORT", "8000"))
    host = os.environ.get("POD_SERVER_HOST", "127.0.0.1")
    if host != "127.0.0.1":
        print(f"[pod] WARNING: binding {host}, not loopback. This machine has "
              "a public address and this endpoint has no authentication.",
              flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The real worker: the same contract, backed by the model.

Everything structural was settled in worker.py and proved against
fake_worker.py, so this file is only the part that needs a card: load the
pipeline once, then turn a job into an image.

    python -m service.gpu_worker

    MODEL_PATH=ovedrive/Qwen-Image-Edit-2509-4bit python -m service.gpu_worker

The model loads at start-up and stays resident. That is the whole reason a
worker is a long-lived process rather than a function: the weights cost
minutes to load and seconds to use, so they are loaded once and used for
every job the process ever sees.
"""
import io
import os
import time

from PIL import Image

from . import queue as q
from . import storage
from .worker import Worker

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509")
LORA_DIR = os.environ.get("LORA_DIR", "weights")
LIGHTNING = int(os.environ.get("LIGHTNING", "8"))

_pipe = None


def load():
    """Load once, at start-up, and report how long it took.

    Timed on purpose. Load time is the number that decides whether autoscaling
    is possible at all: a worker that takes seven minutes to become useful
    cannot answer a traffic spike, and one that takes thirty seconds can.
    """
    global _pipe
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline import LayeringVTONPipeline

    t = time.time()
    print(f"[gpu] loading {MODEL_PATH}", flush=True)
    _pipe = LayeringVTONPipeline(MODEL_PATH, LORA_DIR, lightning=LIGHTNING)
    print(f"[gpu] ready in {time.time() - t:.0f}s", flush=True)


def handle(job):
    from utils import process_inputs

    p = job["params"] or {}
    person = Image.open(io.BytesIO(storage.get_bytes(p["person_key"]))).convert("RGB")
    garment = Image.open(io.BytesIO(storage.get_bytes(p["garment_key"]))).convert("RGB")

    pp, pg, ppose = process_inputs(person, garment, None)
    t = time.time()
    result = _pipe(person_img=pp, garment_img=pg, pose_img=ppose,
                   description=p.get("description") or "the garment",
                   mode=p.get("mode", "upper"),
                   seed=int(p.get("seed", 42)))
    elapsed = round(time.time() - t, 2)

    buf = io.BytesIO()
    result.save(buf, "PNG")
    key = storage.key_for("results", job["user_id"])
    storage.put_bytes(key, buf.getvalue())
    return {"object_key": key, "kind": "image",
            "width": result.width, "height": result.height, "seconds": elapsed}


def main():
    storage.ensure_bucket()
    load()
    name = os.environ.get("WORKER_NAME", f"gpu:{q.WORKER_ID}")
    Worker(handle, name=name).run()


if __name__ == "__main__":
    main()

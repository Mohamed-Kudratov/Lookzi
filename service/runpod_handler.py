#!/usr/bin/env python3
"""The GPU side of a RunPod Serverless endpoint.

gpu_worker.py and this file do the same work in opposite directions. That one
is a long-lived process that reaches into the queue and takes what it wants;
this one sits still and is handed a job by RunPod's own dispatcher. The shape
follows from who does the scaling: a process we run cannot be started by a
queue that is filling up, and a function RunPod runs can.

    runpod.serverless.start({"handler": handler})

Nothing here touches Postgres, and nothing here holds our storage credentials.
Everything the job needs arrives as three signed links -- fetch the person,
fetch the garment, put the result -- each one good for a few minutes and for
one object. A worker rented by the minute on somebody else's hardware should
not be given a key to the whole bucket, and with this design it never is.

The images do not travel through RunPod's job payload either. Their request
limit is 10 MB and a garment photograph can approach it, but the better reason
is that a payload is copied through their queue, their storage and their API on
the way to a card that is going to fetch from object storage anyway.

The model loads at import, once per worker, and stays resident for every
invocation that worker ever serves. That is the entire economic argument for
serverless here: the weights cost minutes to load and seconds to use.
"""
import io
import os
import time
import urllib.error
import urllib.request

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.environ.get("MODEL_PATH", "ovedrive/Qwen-Image-Edit-2509-4bit")
# Anchored to the repository, not to the working directory. A relative "weights"
# resolves under the image's WORKDIR and nowhere else, and the failure is silent:
# no adapter loads, and the endpoint returns plausible pictures of the wrong
# thing instead of an error anybody would notice.
LORA_DIR = os.environ.get("LORA_DIR", os.path.join(ROOT, "weights"))
LIGHTNING = int(os.environ.get("LIGHTNING", "8"))
FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "60"))
MAX_BYTES = int(os.environ.get("MAX_INPUT_BYTES", str(32 * 1024 * 1024)))

_pipe = None
_load_seconds = None


def load():
    """Load the model once, on the first invocation this worker serves.

    Not at import: RunPod imports the module to discover the handler, and a
    failure there is reported as a broken worker rather than as a failed job.
    Loading on first use means a bad checkpoint fails one job with a message,
    and the endpoint stays diagnosable.
    """
    global _pipe, _load_seconds
    if _pipe is not None:
        return _pipe
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline import LayeringVTONPipeline

    t = time.time()
    print(f"[handler] loading {MODEL_PATH}", flush=True)
    _pipe = LayeringVTONPipeline(MODEL_PATH, LORA_DIR, lightning=LIGHTNING)
    _load_seconds = round(time.time() - t, 1)
    print(f"[handler] ready in {_load_seconds}s", flush=True)
    return _pipe


def _fetch(url):
    """Read a signed link into memory, refusing anything implausible.

    The size cap is not about malice -- the links are ours -- but about a
    truncated or wrong object turning into an out-of-memory kill that takes the
    worker down instead of failing the one job.
    """
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        declared = resp.headers.get("Content-Length")
        if declared and int(declared) > MAX_BYTES:
            raise ValueError(f"input is {int(declared)} bytes, over the "
                             f"{MAX_BYTES} limit")
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"input exceeds the {MAX_BYTES} byte limit")
    return Image.open(io.BytesIO(data)).convert("RGB")


def _put(url, payload, content_type):
    """Upload the result to the signed link.

    Content-Type has to match what the link was signed with, byte for byte, or
    the signature will not verify and the upload comes back as 403 -- which
    reads like a permissions problem and is not one.
    """
    req = urllib.request.Request(url, data=payload, method="PUT",
                                 headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        if resp.status not in (200, 201, 204):
            raise RuntimeError(f"upload returned {resp.status}")


def handler(event):
    """One image. Everything needed arrives in the payload."""
    from utils import process_inputs

    inp = event.get("input") or {}
    for required in ("person_url", "garment_url", "result_put_url"):
        if not inp.get(required):
            return {"error": f"missing {required}"}

    started = time.time()
    pipe = load()
    cold = _pipe is not None and time.time() - started > 5

    try:
        person = _fetch(inp["person_url"])
        garment = _fetch(inp["garment_url"])
    except (urllib.error.URLError, ValueError, OSError) as exc:
        # A signed link expires. Saying so beats a stack trace, because the
        # fix is on the calling side and it is a different fix from a bad image.
        return {"error": f"could not read the inputs: {type(exc).__name__}: {exc}"}

    pp, pg, ppose = process_inputs(person, garment, None)

    t = time.time()
    result = pipe(person_img=pp, garment_img=pg, pose_img=ppose,
                  description=inp.get("description") or "the garment",
                  mode=inp.get("mode", "upper"),
                  seed=int(inp.get("seed", 42)))
    generate_seconds = round(time.time() - t, 2)

    buf = io.BytesIO()
    result.save(buf, "PNG")
    _put(inp["result_put_url"], buf.getvalue(),
         inp.get("result_content_type", "image/png"))

    return {"width": result.width, "height": result.height,
            "seconds": generate_seconds,
            "total_seconds": round(time.time() - started, 2),
            # Reported so the bridge can tell a slow queue from a cold worker
            # without guessing, and so the cost of scaling to zero is visible
            # in the numbers rather than argued about.
            "cold_start": bool(cold), "load_seconds": _load_seconds}


if __name__ == "__main__":
    import runpod

    # Warm the model before accepting anything, so the first customer is not
    # the one who pays for the load. RunPod counts this against the worker's
    # start-up, which is exactly where it belongs.
    if os.environ.get("PRELOAD", "1") == "1":
        load()
    runpod.serverless.start({"handler": handler})

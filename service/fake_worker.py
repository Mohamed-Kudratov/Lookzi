#!/usr/bin/env python3
"""A worker that satisfies the contract without a GPU.

The point is not to fake a demo. It is that every part of the product other
than the model -- sign-in, credits, the queue, retries, refunds, history,
Telegram, the web app -- can be built and corrected on a laptop, and only
plugged into a real card once it already works.

It takes the time a real job takes, so the interface is designed against the
real latency rather than against an instant reply that will never happen.

    python -m service.fake_worker

    FAKE_SECONDS=0.2 python -m service.fake_worker    # fast, for tests
    FAKE_FAIL_RATE=0.2 python -m service.fake_worker  # exercise retries
"""
import io
import os
import random
import time

from PIL import Image, ImageDraw

from . import queue as q
from . import storage
from .worker import Worker

# The measured figure for one image at 8 steps on an A100. Defaulting to the
# real number keeps the wait honest while the interface is being designed.
FAKE_SECONDS = float(os.environ.get("FAKE_SECONDS", "14.3"))
FAKE_FAIL_RATE = float(os.environ.get("FAKE_FAIL_RATE", "0"))
SIZE = (768, 1024)


def _placeholder(job):
    """An image that says what it is, so nobody mistakes it for output.

    A stub that returns something plausible is worse than one that returns
    something obviously fake: plausible output ends up in a screenshot, a
    pitch, or a customer's catalogue.
    """
    img = Image.new("RGB", SIZE, "#16181C")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, SIZE[0] - 40, SIZE[1] - 40], outline="#3B4BA0", width=3)
    lines = [
        "PLACEHOLDER",
        "",
        f"tool   {job['tool']}",
        f"model  {job.get('model_id') or '-'}",
        f"job    {str(job['id'])[:8]}",
        "",
        "no GPU was involved",
    ]
    y = SIZE[1] // 2 - len(lines) * 14
    for line in lines:
        d.text((80, y), line, fill="#F2F1ED")
        y += 28
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def handle(job):
    if FAKE_FAIL_RATE and random.random() < FAKE_FAIL_RATE:
        raise RuntimeError("simulated failure")

    time.sleep(FAKE_SECONDS)
    key = storage.key_for("results", job["user_id"])
    storage.put_bytes(key, _placeholder(job))
    return {"object_key": key, "kind": "image",
            "width": SIZE[0], "height": SIZE[1]}


def main():
    storage.ensure_bucket()
    name = os.environ.get("WORKER_NAME", f"fake:{q.WORKER_ID}")
    Worker(handle, name=name).run()


if __name__ == "__main__":
    main()

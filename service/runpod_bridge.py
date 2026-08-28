#!/usr/bin/env python3
"""The worker that owns no GPU.

RunPod Serverless scales cards up and down on its own, but only for work it is
handed through its own API -- it cannot reach into our queue and take a job.
So something has to stand between them, and this is it: it claims from our
queue exactly like every other worker, then pays RunPod to do the sampling and
writes the result back.

    RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... python -m service.runpod_bridge

What that buys is the whole point of the exercise. This process holds the
queue, the credit ledger and the history -- the parts that must not be lost --
and needs no GPU, so it runs on the same ten-dollar box as the web tier and
stays up whether or not a card exists anywhere. RunPod holds the cards and the
scaling, which is the part we are bad at and they are good at.

    web + queue + ledger        RunPod Serverless
    ---------------------       ------------------
    this process claims a job
    signs three links       ->  a worker fetches, samples, uploads
    records the result      <-  and reports how long it took

The images never pass through here. This process signs a link to fetch the
person, a link to fetch the garment and a link to store the result, and sends
those three strings. A rented worker gets no credentials and no bucket, only
three doors that close by themselves.
"""
import json
import os
import time
import urllib.error
import urllib.request

from . import queue as q
from . import storage
from .worker import Worker

API = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# How long a signed link stays good. Long enough to cover a cold start and a
# queue behind it; short enough that a link found in a log later is useless.
LINK_SECONDS = int(os.environ.get("RUNPOD_LINK_SECONDS", "1800"))

# The wait is capped below JOB_STALE_AFTER on purpose. A job held past that is
# requeued by the stale sweep while this process is still waiting on it, and
# then two workers generate one image for one charge. Failing first keeps the
# accounting honest, and a job that has taken this long is not coming back.
POLL_SECONDS = float(os.environ.get("RUNPOD_POLL_SECONDS", "1.0"))
POLL_MAX_SECONDS = float(os.environ.get("RUNPOD_POLL_MAX_SECONDS", "3.0"))
WAIT_LIMIT = float(os.environ.get("RUNPOD_WAIT_SECONDS", "600"))

TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodError(RuntimeError):
    pass


def _call(path, payload=None, method=None, timeout=30):
    if not ENDPOINT_ID or not API_KEY:
        raise RunPodError(
            "RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY must both be set. The key "
            "belongs in the environment, never in the repository.")
    url = f"{API}/{ENDPOINT_ID}/{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        # 401 here is almost always a key for the wrong account rather than a
        # malformed one, and saying which endpoint was asked saves the usual
        # twenty minutes of checking the key that was fine all along.
        raise RunPodError(f"{method or 'POST'} {path} -> {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RunPodError(f"cannot reach {API}: {exc.reason}") from exc


def submit(payload):
    out = _call("run", {"input": payload})
    rid = out.get("id")
    if not rid:
        raise RunPodError(f"no job id in the response: {out}")
    return rid


def cancel(rid):
    try:
        _call(f"cancel/{rid}", payload={})
    except RunPodError:
        # Best effort. A job that cannot be cancelled will finish and be
        # ignored, which costs a few cents; failing the local job because the
        # cancellation failed would cost the customer their image.
        pass


def wait(rid, limit=None):
    """Poll until the remote job finishes, or give up and say so."""
    limit = WAIT_LIMIT if limit is None else limit
    started = time.time()
    interval = POLL_SECONDS
    while True:
        state = _call(f"status/{rid}", method="GET")
        status = state.get("status")
        if status in TERMINAL_OK:
            return state.get("output") or {}
        if status in TERMINAL_BAD:
            raise RunPodError(f"remote job {status}: {state.get('error') or state}")
        if time.time() - started > limit:
            cancel(rid)
            raise RunPodError(
                f"remote job still {status} after {limit:.0f}s; cancelled it")
        time.sleep(interval)
        interval = min(interval * 1.5, POLL_MAX_SECONDS)


def handle(job):
    p = job["params"] or {}
    key = storage.key_for("results", job["user_id"])
    payload = {
        "person_url": storage.presigned_get(p["person_key"], seconds=LINK_SECONDS),
        "garment_url": storage.presigned_get(p["garment_key"], seconds=LINK_SECONDS),
        "result_put_url": storage.presigned_put(key, seconds=LINK_SECONDS,
                                                content_type="image/png"),
        "result_content_type": "image/png",
        "mode": p.get("mode", "upper"),
        "description": p.get("description") or "the garment",
        "seed": int(p.get("seed", 42)),
    }

    rid = submit(payload)
    print(f"[bridge] {job['id']} -> runpod {rid}", flush=True)
    try:
        out = wait(rid)
    except BaseException:
        # Includes the SIGTERM path: a pod being reclaimed should not leave a
        # remote job running and billing with nobody waiting for it.
        cancel(rid)
        raise

    if out.get("error"):
        raise RunPodError(out["error"])
    if out.get("cold_start"):
        print(f"[bridge] {job['id']} paid a cold start "
              f"({out.get('load_seconds')}s of loading)", flush=True)

    return {"object_key": key, "kind": "image",
            "width": out.get("width"), "height": out.get("height"),
            "seconds": out.get("seconds")}


def preflight():
    """Fail at start-up on the two mistakes that otherwise fail per job.

    Both are worth catching here because both look like something else later:
    a missing key reads as a RunPod outage, and a storage endpoint only this
    machine can resolve reads as a broken model, since what actually happens is
    that a worker on the other side of the internet gets a link to localhost.
    """
    if not ENDPOINT_ID or not API_KEY:
        raise SystemExit(
            "RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY must both be set.\n"
            "The key goes in the environment or .env, never in the repository.")

    probe = storage.presigned_get("preflight-probe.png", seconds=60)
    if any(h in probe for h in ("localhost", "127.0.0.1", "://storage",
                                "://minio")):
        raise SystemExit(
            f"S3_PUBLIC_ENDPOINT signs links as {probe.split('/')[2]}, which a "
            "RunPod worker cannot resolve.\nPoint it at a public address (R2, "
            "or a tunnel) before running the bridge.")

    health = _call("health", method="GET")
    workers = health.get("workers", {})
    print(f"[bridge] endpoint {ENDPOINT_ID}: "
          f"{workers.get('ready', 0)} ready, {workers.get('running', 0)} running, "
          f"{workers.get('idle', 0)} idle", flush=True)
    if not any(workers.get(k) for k in ("ready", "running", "idle", "initializing")):
        print("[bridge] warning: the endpoint reports no workers at all. "
              "Jobs will queue until one starts.", flush=True)


def _check_wait_below_lease():
    """The wait cap must stay under the queue's lease, and be checked, not hoped.

    A job held past JOB_STALE_AFTER is requeued by the stale sweep while this
    process is still waiting on it, and then two workers generate one image
    for one charge. That invariant was written in a comment and enforced by
    nothing, so lowering the lease would have broken it silently.
    """
    import re
    m = re.match(r"^(\d+)\s*(second|minute|hour)s?$", q.STALE_AFTER.strip().lower())
    if not m:
        return  # an interval we cannot parse is not one to guess at
    unit = {"second": 1, "minute": 60, "hour": 3600}[m.group(2)]
    lease = int(m.group(1)) * unit
    if WAIT_LIMIT >= lease:
        raise SystemExit(
            f"RUNPOD_WAIT_SECONDS is {WAIT_LIMIT:.0f}s but JOB_STALE_AFTER is "
            f"{lease}s. The wait must end first, or the stale sweep requeues a "
            "job this process is still waiting on: two generations, one charge.")


def main():
    preflight()
    _check_wait_below_lease()
    name = os.environ.get("WORKER_NAME", f"runpod:{q.WORKER_ID}")
    Worker(handle, name=name).run()


if __name__ == "__main__":
    main()

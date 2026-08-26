#!/usr/bin/env python3
"""One job, all the way through, against the running stack.

The other two suites test the queue directly. This one goes through the door a
customer goes through: the HTTP API, a presigned upload to object storage, a
worker in another container, and a signed link back.

    docker compose up -d
    python tests/smoke_api.py

It uses the stub worker, so the image that comes back says PLACEHOLDER. What
is being proved is the path, not the picture.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
WHO = os.environ.get("SMOKE_IDENTITY", "smoke-test-1")

# A 1x1 PNG. The stub never opens it; it is here to prove the presigned upload
# accepts a body and that the key round-trips.
PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

failures = []


def ok(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name} {detail}")


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"X-Telegram-Id": WHO, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def main():
    try:
        health = call("GET", "/health")
    except (urllib.error.URLError, OSError) as exc:
        print(f"no API at {BASE}: {exc}")
        print("start it with:  docker compose up -d")
        return 0

    print("\nservice")
    ok("health responds", health.get("ok") is True, health)
    me = call("GET", "/me")
    ok("an unknown identity is given an account", me["id"] > 0, me)
    ok("and trial credits", me["credits"] > 0, me)
    start_credits = me["credits"]

    print("\ncatalogue")
    models = call("GET", "/models")
    ok("models are listed", len(models) > 0, f"{len(models)} returned")
    ok("duplicates are hidden", len(models) == 13, f"{len(models)} returned, expected 13")
    ok("models have human names",
       all(m["display_name"] and not m["display_name"].startswith(("f_", "m_"))
           for m in models),
       [m["display_name"] for m in models[:3]])

    print("\nupload")
    up = call("POST", "/uploads", {"kind": "garment", "content_type": "image/png"})
    req = urllib.request.Request(up["url"], data=PIXEL, method="PUT",
                                 headers={"Content-Type": "image/png"})
    with urllib.request.urlopen(req, timeout=30) as r:
        ok("presigned upload accepted", r.status in (200, 204), r.status)

    print("\njob")
    job = call("POST", "/jobs", {"tool": "product-to-model",
                                 "garment_key": up["key"],
                                 "model_id": models[0]["id"],
                                 "mode": "upper"})
    ok("job accepted", job["status"] == "queued", job)
    ok("a credit was taken", call("GET", "/me")["credits"] == start_credits - 1)

    print("\nworker")
    deadline = time.time() + 90
    state = None
    while time.time() < deadline:
        state = call("GET", f"/jobs/{job['job_id']}")
        if state["status"] in ("done", "failed"):
            break
        time.sleep(2)
    ok("a worker picked it up and finished", state and state["status"] == "done",
       state)
    ok("timing was recorded", bool(state and state.get("seconds")), state)

    if state and state.get("result_url"):
        with urllib.request.urlopen(state["result_url"], timeout=30) as r:
            body = r.read()
        ok("the result downloads", len(body) > 1000, f"{len(body)} bytes")
        ok("and it is a PNG", body[:8] == b"\x89PNG\r\n\x1a\n")
    else:
        ok("the result downloads", False, "no result_url")

    print("\nhistory")
    hist = call("GET", "/jobs")
    ok("the job appears in history",
       any(h["job_id"] == job["job_id"] for h in hist), f"{len(hist)} rows")

    print()
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

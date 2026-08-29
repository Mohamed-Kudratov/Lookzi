#!/usr/bin/env python3
"""The worker that owns no GPU and needs nothing to be public.

    POD_SSH="root@154.54.102.46 -p 13504" python -m service.tunnel_worker

It claims from our queue exactly like every other worker, then sends the two
images to a model running on a rented pod and writes the result back. The queue,
the ledger and the history stay here, on a machine we control; only the pixels
travel, and they travel inside the ssh connection.

What this avoids is the reason it exists. The obvious arrangement -- run
gpu_worker.py on the pod -- needs the pod to reach Postgres, which means either
exposing the database to the internet or renting a managed one. The other
obvious arrangement -- signed links to object storage, as runpod_bridge.py does
-- needs a public bucket. Both are the right answer later and both are an
account, a bill and a new attack surface before the product has a user. An ssh
tunnel is none of those and it already works.

The tunnel is this process's own responsibility. A separate `ssh -L` in another
terminal is a thing that dies quietly at three in the morning and leaves a
worker failing every job with a connection error; here it is opened at start,
checked before every job, and reopened when it drops.

    web + queue + ledger                    pod
    --------------------                    ---------------
    claim a job
    read both images from storage
    POST them through the tunnel   ------>  one loaded model
    store the PNG that comes back  <------  answers with a PNG
"""
import io
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from . import queue as q
from . import storage
from .worker import Worker

POD_SSH = os.environ.get("POD_SSH", "")
SSH_KEY = os.environ.get("POD_SSH_KEY",
                         os.path.expanduser("~/.ssh/id_ed25519_github"))
LOCAL_PORT = int(os.environ.get("POD_LOCAL_PORT", "18000"))
REMOTE_PORT = int(os.environ.get("POD_SERVER_PORT", "8000"))
# The model maker is a second process on the pod, because Z-Image needs
# diffusers from source and the try-on stack cannot have it. One ssh connection
# carries both.
ZIMAGE_LOCAL = int(os.environ.get("ZIMAGE_LOCAL_PORT", "18001"))
# The third model on the card. 3.6 GiB, which is the only reason three fit.
FASHN_LOCAL = int(os.environ.get("FASHN_LOCAL_PORT", "18002"))
FASHN_REMOTE = int(os.environ.get("FASHN_PORT", "8002"))
ZIMAGE_REMOTE = int(os.environ.get("ZIMAGE_PORT", "8001"))
# Generous, because it covers the model still loading on a pod that has just
# started. The queue's own lease is fifteen minutes and this must end first.
REQUEST_TIMEOUT = int(os.environ.get("POD_REQUEST_TIMEOUT", "600"))
# The link to the pod drops occasionally and heals in seconds. Worth
# waiting out rather than throwing away work somebody is waiting for.
TUNNEL_ATTEMPTS = int(os.environ.get("POD_TUNNEL_ATTEMPTS", "4"))
TUNNEL_BACKOFF = float(os.environ.get("POD_TUNNEL_BACKOFF", "4"))

BASE = f"http://127.0.0.1:{LOCAL_PORT}"
ZBASE = f"http://127.0.0.1:{ZIMAGE_LOCAL}"
FBASE = f"http://127.0.0.1:{FASHN_LOCAL}"


class PodDown(RuntimeError):
    """The pod cannot be reached or is not ready."""


class RunPodInput(ValueError):
    """The job is missing something the tool needs. Not the pod's fault."""


class Tunnel:
    """An ssh port forward, kept alive.

    RunPod's ssh.runpod.io proxy refuses port forwarding, so this uses the pod's
    direct address -- the one RUNPOD_PUBLIC_IP and RUNPOD_TCP_PORT_22 name from
    inside the pod. That route also allows plain command execution and scp,
    neither of which the proxy does.
    """

    def __init__(self, target, key=None, local=LOCAL_PORT, remote=REMOTE_PORT):
        if not target:
            raise SystemExit(
                "POD_SSH is not set. It is the pod's direct address, which the\n"
                "pod itself reports:\n"
                "    echo root@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22\n"
                "The ssh.runpod.io proxy will not do; it refuses forwarding.")
        self.target = shlex.split(target)
        self.key = key or SSH_KEY
        self.local, self.remote = local, remote
        self.proc = None

    def _spawn(self):
        cmd = ["ssh", "-N",
               "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
               "-o", "ExitOnForwardFailure=yes",
               # Without these a dropped link leaves a process that looks alive
               # and forwards nothing, which is the worst of both.
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
               "-o", "IdentitiesOnly=yes", "-i", self.key,
               "-L", f"{self.local}:127.0.0.1:{self.remote}",
               "-L", f"{ZIMAGE_LOCAL}:127.0.0.1:{ZIMAGE_REMOTE}",
               "-L", f"{FASHN_LOCAL}:127.0.0.1:{FASHN_REMOTE}"] + self.target
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)

    def up(self, attempts=None):
        """True once the tunnel carries traffic, opening it if it does not.

        Retried, because the link to the pod drops. It happened twice in one
        afternoon -- 'Network is unreachable' for a few seconds, over a path
        with 220 ms of latency -- and each time a customer's job failed. The
        credit came back, which is right, but the image did not, and a job
        thrown away over a blip that heals in ten seconds is a bad trade.
        """
        attempts = attempts or TUNNEL_ATTEMPTS
        if self.proc is not None and self.proc.poll() is None and self._reaches():
            return True

        last = None
        for attempt in range(1, attempts + 1):
            self.close()
            try:
                self._open_once()
                if attempt > 1:
                    print(f"[tunnel] reopened on attempt {attempt}", flush=True)
                return True
            except PodDown as exc:
                last = exc
                if attempt < attempts:
                    pause = TUNNEL_BACKOFF * attempt
                    print(f"[tunnel] {exc}; retrying in {pause:.0f}s "
                          f"({attempt}/{attempts})", flush=True)
                    time.sleep(pause)
        raise last

    def _open_once(self):
        self._spawn()
        for _ in range(20):
            time.sleep(0.5)
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode(errors="replace")
                raise PodDown(f"ssh exited: {err.strip()[:200]}")
            if self._reaches():
                print(f"[tunnel] {self.local} -> pod {self.remote}", flush=True)
                return
        raise PodDown(f"the tunnel opened but nothing answers on {BASE}")

    def _reaches(self):
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=5).read()
            return True
        except Exception:                                     # noqa: BLE001
            return False

    def close(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


_tunnel = None


def health():
    _tunnel.up()
    import json
    with urllib.request.urlopen(f"{BASE}/health", timeout=15) as r:
        return json.loads(r.read())


def _multipart(fields, files):
    """Build a multipart body without pulling in requests.

    The web tier is deliberately free of heavy dependencies -- there is a test
    that fails if torch ever appears in it -- and one function is cheaper than
    another package in the image.
    """
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buf.write(f"{value}\r\n".encode())
    for name, (filename, data) in files.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write((f'Content-Disposition: form-data; name="{name}"; '
                   f'filename="{filename}"\r\n').encode())
        buf.write(b"Content-Type: image/png\r\n\r\n")
        buf.write(data)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


# Which endpoint each tool wants, and what it needs sent. Kept here rather than
# branched inside handle(), so a new tool is an entry rather than another if.
ROUTES = {
    "packshot": ("/packshot", ("garment",)),
    "model-creation": (ZBASE + "/create", ()),
    # Same two photographs as product-to-model, a different model behind them.
    "try-on-v2": (FBASE + "/generate", ("person", "garment")),
}
DEFAULT_ROUTE = ("/generate", ("person", "garment"))


def _post(url, fields, files, timeout=None):
    """One request to one of the pod's servers, returning bytes and headers."""
    body, content_type = _multipart(fields, files)
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout or REQUEST_TIMEOUT) as resp:
            return resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise PodDown(f"pod returned {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise PodDown(f"the tunnel dropped mid-request: {exc.reason}")


def handle_scene(job, p):
    """Describe the shot, then dress the person in it.

    Two models and two stages, because neither can do this alone. The try-on
    model reads the garment image and ignores text entirely, so a prompt in
    front of it would change nothing; Z-Image answers text and knows nothing
    about the garment. So the prompt makes the person and the scene, and the
    try-on stage puts the seller's garment on them -- which is exactly how the
    roster itself was built.
    """
    prompt = (p.get("prompt") or "").strip()
    if not prompt:
        raise RunPodInput("a scene needs something written about it")

    person, headers = _post(f"{ZBASE}/prompt",
                            {"prompt": prompt, "seed": int(p.get("seed", 0))}, {})
    scene_seconds = float(headers.get("X-Seconds") or 0)
    print(f"[bridge] {job['id']} scene in {scene_seconds}s", flush=True)

    png, headers = _post(f"{BASE}/generate",
                         {"mode": "", "description": "the garment",
                          "seed": int(p.get("seed", 42))},
                         {"person": ("person.png", person),
                          "garment": ("garment.png",
                                      storage.get_bytes(p["garment_key"]))})
    # Both stages, because a customer waiting nineteen seconds is owed the
    # truth about where they went.
    total = scene_seconds + float(headers.get("X-Seconds") or 0)
    return png, headers, round(total, 2)


def handle(job):
    p = job["params"] or {}
    _tunnel.up()

    if job["tool"] == "product-in-scene":
        png, headers, seconds = handle_scene(job, p)
        key = storage.key_for("results", job["user_id"])
        storage.put_bytes(key, png)
        return {"object_key": key, "kind": "image",
                "width": int(headers.get("X-Width") or 0) or None,
                "height": int(headers.get("X-Height") or 0) or None,
                "seconds": seconds}

    path, wants = ROUTES.get(job["tool"], DEFAULT_ROUTE)
    files = {}
    for name in wants:
        key = p.get(f"{name}_key")
        if not key:
            raise RunPodInput(f"{job['tool']} needs a {name} and none was sent")
        files[name] = (f"{name}.png", storage.get_bytes(key))

    # A written description used to send this to /prompt, the scene endpoint.
    # That endpoint frames for a place -- natural light, wherever the customer
    # says -- and a model is a studio photograph: plain backdrop, even light,
    # neutral expression. Worse, it briefly carried a "plain fitted sleeveless
    # top" instruction, so "white skin, uzbek girl, very slim" came back as a
    # woman in a black mini dress. /create takes the description now and frames
    # it as what it is.

    fields = {}
    if path.endswith("/prompt"):
        fields = {"prompt": p["prompt"].strip(), "seed": int(p.get("seed", 0))}
    elif path.endswith("/create"):
        # The choices a seller has an opinion about. Everything else varies by
        # seed, so asking twice gives two people.
        fields = {k: p.get(k) for k in ("gender", "age", "build", "look", "modest")
                  if p.get(k) is not None}
        fields["seed"] = int(p.get("seed", 0))
        # The written description, which the studio offers and this used to
        # drop. A customer who typed "white skin, uzbek girl, very slim" got
        # whatever the five dropdowns happened to say instead.
        if p.get("prompt"):
            fields["prompt"] = p["prompt"]
    elif path.startswith(FBASE):
        # The one control this model actually answers. Ours takes a mode too
        # and does nothing with it.
        fields = {"category": p.get("category") or "tops"}
    elif path == "/generate":
        fields = {
            "mode": p.get("mode") or "",
            "description": p.get("description") or "the garment",
            "seed": int(p.get("seed", 42))}
    body, content_type = _multipart(fields, files)

    url = path if path.startswith("http") else f"{BASE}{path}"
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            png = resp.read()
            seconds = float(resp.headers.get("X-Seconds") or 0) or None
            width = int(resp.headers.get("X-Width") or 0) or None
            height = int(resp.headers.get("X-Height") or 0) or None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        # 503 is the pod still loading its weights, which is a wait rather than
        # a fault; the queue retries and the customer keeps their credit.
        raise PodDown(f"pod returned {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise PodDown(f"the tunnel dropped mid-request: {exc.reason}")

    key = storage.key_for("results", job["user_id"])
    storage.put_bytes(key, png)
    return {"object_key": key, "kind": "image",
            "width": width, "height": height, "seconds": seconds}


def _refuse_a_second_bridge():
    """One bridge per local port, checked before anything else starts.

    Two of these ran at once for a while, because the first survived a restart
    that was meant to replace it. They both claimed jobs and both wanted the
    same forwarded port, so one had a working tunnel and the other failed every
    job it won -- intermittently, depending on which claimed first. The health
    endpoint reported two workers ready, which was true and useless.
    """
    import socket
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", LOCAL_PORT))
    except OSError:
        raise SystemExit(
            f"port {LOCAL_PORT} is already taken, which almost certainly"
            " means another bridge is running. Two of them claim the same"
            " jobs and fight over the same tunnel.\n"
            "Stop the other one, or set POD_LOCAL_PORT for a second pod.")
    finally:
        probe.close()


def main():
    global _tunnel
    _refuse_a_second_bridge()
    storage.ensure_bucket()
    _tunnel = Tunnel(POD_SSH)

    state = health()
    if state.get("error"):
        raise SystemExit(f"the pod's model did not load: {state['error']}")
    if not state.get("ready"):
        print("[tunnel] the pod is still loading its weights; "
              "jobs will wait for it", flush=True)
    else:
        print(f"[tunnel] pod ready: {state['model']} "
              f"(loaded in {state.get('load_seconds')}s)", flush=True)

    name = os.environ.get("WORKER_NAME", f"tunnel:{q.WORKER_ID}")
    try:
        Worker(handle, name=name).run()
    finally:
        _tunnel.close()


if __name__ == "__main__":
    main()

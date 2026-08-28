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
# Generous, because it covers the model still loading on a pod that has just
# started. The queue's own lease is fifteen minutes and this must end first.
REQUEST_TIMEOUT = int(os.environ.get("POD_REQUEST_TIMEOUT", "600"))

BASE = f"http://127.0.0.1:{LOCAL_PORT}"


class PodDown(RuntimeError):
    """The pod cannot be reached or is not ready."""


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
               "-L", f"{self.local}:127.0.0.1:{self.remote}"] + self.target
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)

    def up(self):
        """True once the tunnel carries traffic, opening it if it does not."""
        if self.proc is not None and self.proc.poll() is None and self._reaches():
            return True
        self.close()
        self._spawn()
        for _ in range(20):
            time.sleep(0.5)
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode(errors="replace")
                raise PodDown(f"ssh exited: {err.strip()[:300]}")
            if self._reaches():
                print(f"[tunnel] {self.local} -> pod {self.remote}", flush=True)
                return True
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


def handle(job):
    p = job["params"] or {}
    _tunnel.up()

    person = storage.get_bytes(p["person_key"])
    garment = storage.get_bytes(p["garment_key"])
    body, content_type = _multipart(
        {"mode": p.get("mode", "upper"),
         "description": p.get("description") or "the garment",
         "seed": int(p.get("seed", 42))},
        {"person": ("person.png", person), "garment": ("garment.png", garment)})

    req = urllib.request.Request(f"{BASE}/generate", data=body, method="POST",
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


def main():
    global _tunnel
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

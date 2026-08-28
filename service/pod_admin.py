#!/usr/bin/env python3
"""Bring a fresh pod to a working state, and show the work happening.

    python -m service.pod_admin          # then open http://localhost:8090

Paste the ssh line RunPod gives you, press Start, and watch. Nothing else is
asked for.

Why this exists. Every pod so far has been set up by hand, and every one of
them found a new way to go wrong: a container disk wiped by a migration, a
diffusers fork hidden by sparse-checkout, an OpenMP pool sized from the host's
128 cores against a 13.6-core quota, three git processes fighting over one
working tree, a stale index.lock, CRLF in a shell script, a network volume that
quietly dropped from 655 MB/s to 16. Each one cost an hour and was diagnosed
once. Keeping that knowledge in a person's head means paying for it again on
the next pod; keeping it here means the pod either comes up or says precisely
what is wrong with it.

The important design change is where the code lives. It used to sit on the
network volume, which is fast in bulk and terrible at thousands of small files
-- and a source tree is nothing but small files. Now the repository is cloned
to the container's local disk, which measured 3.4 GB/s against the volume's 16
MB/s on a bad day, and is re-cloned on every pod because that takes seconds.
The volume keeps only what it is good at: the 90 GB of weights, and the output.

On progress. Every step carries the number of seconds it actually took when
measured, and the bar is the weighted sum of those -- so it moves at roughly
real speed rather than jumping between round numbers. Downloads report the
bytes on disk against the bytes expected. A step that cannot report honestly
says so instead of animating.
"""
import base64
import json
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REPO_URL = os.environ.get("LOOKZI_REPO",
                          "https://github.com/Mohamed-Kudratov/Lookzi.git")
REPO_REF = os.environ.get("LOOKZI_REF", "main")
# The code goes on the container's local disk. See the module docstring.
POD_REPO = "/opt/lookzi"
SSH_KEY = os.environ.get("POD_SSH_KEY",
                         os.path.expanduser("~/.ssh/id_ed25519_github"))

# What a working pod must be able to load. Sizes are what the repositories
# actually occupy on disk, used to turn a download into a percentage.
MODELS = {
    "4bit": ("ovedrive/Qwen-Image-Edit-2509-4bit",
             "models--ovedrive--Qwen-Image-Edit-2509-4bit", 15_800),
    "bf16": ("Qwen/Qwen-Image-Edit-2509",
             "models--Qwen--Qwen-Image-Edit-2509", 57_700),
    "lightning": ("lightx2v/Qwen-Image-Lightning",
                  "models--lightx2v--Qwen-Image-Lightning", 1_600),
}

# Below this, the volume is having a bad day and every later step will take
# forty times longer than the numbers in TIMINGS.md. Saying so in the first
# minute is the whole point; on 2026-08-28 it was found after an hour.
VOLUME_SLOW_MBS = 100.0

# Where things live on the volume.
CACHE = "/workspace/.cache/huggingface/hub"
# Z-Image is published in fp32 and loads at 584 s that way. The bf16 copy is
# made once, from the fp32, and is what every load actually reads: same
# weights, half the bytes off a network filesystem, one second instead.
ZIMAGE_BF16 = "/workspace/models/Z-Image-Turbo-bf16"


# ---------------------------------------------------------------------------
# talking to the pod

class Ssh:
    """Run a script on the pod and get its output back.

    RunPod's ssh proxy forces a PTY and ignores a command argument -- it always
    opens a login shell -- so the script goes in over standard input. Three
    things follow from that and all three are handled here:

      the script is base64'd onto one line, so a quote or a newline inside it
      cannot be re-interpreted by the shell receiving it

      the terminal's own echo is turned off, and the sentinels are split so the
      shell's echo of the line that prints them does not match first

      escape sequences are stripped, because a login shell colours everything
    """

    def __init__(self, host, key=None):
        self.host = host
        self.key = key or SSH_KEY

    def run(self, script, timeout=600):
        blob = base64.b64encode(script.encode()).decode()
        stdin = (
            "stty -echo 2>/dev/null; export PS1= PS2=; unset PROMPT_COMMAND\n"
            'echo "__""B__"\n'
            f"echo {blob} | base64 -d > /tmp/_admin.sh && bash /tmp/_admin.sh; __rc=$?\n"
            'echo "__""E__"$__rc\n'
            "exit\n")
        cmd = ["ssh", "-tt", "-q",
               "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=25",
               "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=30",
               "-o", "IdentitiesOnly=yes", "-i", self.key, self.host]
        try:
            proc = subprocess.run(cmd, input=stdin, capture_output=True,
                                  text=True, timeout=timeout,
                                  encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            raise PodError(f"the pod did not answer within {timeout}s")

        raw = (proc.stdout or "") + (proc.stderr or "")
        raw = raw.replace("\r", "")
        raw = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", raw)
        raw = re.sub(r"\x1b\][^\x07]*\x07", "", raw)

        if "container not found" in raw:
            raise PodError("RunPod says the container does not exist. The pod "
                           "was stopped or terminated; start it and paste the "
                           "new address.")
        if "Permission denied (publickey)" in raw:
            raise PodError(f"the key at {self.key} was refused. Add its public "
                           "half to RunPod's SSH settings.")

        begin, end = raw.find("__B__"), raw.rfind("__E__")
        if begin < 0 or end < 0:
            raise PodError("could not read the pod's reply.\n" + raw[-600:])
        body = raw[begin + len("__B__"):end].strip("\n")
        m = re.match(r"__E__(-?\d+)", raw[end:])
        return int(m.group(1)) if m else 1, body


class PodError(RuntimeError):
    """Something about the pod, said plainly enough to act on."""


# ---------------------------------------------------------------------------
# the steps
#
# Each one takes (ssh, state) and either returns or raises PodError. `weight`
# is the seconds it took when measured, which is what makes the bar move at
# roughly real speed instead of in equal jumps.

@dataclass
class Step:
    key: str
    title: str
    weight: float
    run: object
    detail: str = ""


def _env_prefix():
    """The environment every command on the pod needs.

    The thread caps are a correctness fix, not tuning. RunPod grants the
    container 13.6 CPUs but leaves /proc showing the host's 128, so anything
    sizing a pool from nproc opens 128 threads against 13.6 cores and spends
    its life in the scheduler. That presented as a hung model load and cost a
    day to find.
    """
    return (
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "
        "NUMEXPR_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false\n"
        "export HF_HOME=/workspace/.cache/huggingface\n"
        # xet stalls past ~10 GB so it stays off. hf_transfer is decided
        # per download rather than here -- see step_download.
        "export HF_HUB_DISABLE_XET=1\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n")


def step_connect(ssh, st):
    rc, out = ssh.run("hostname; echo POD=${RUNPOD_POD_ID:-unknown}", timeout=90)
    if rc != 0:
        raise PodError(out or "the pod refused the connection")
    for line in out.splitlines():
        if line.startswith("POD="):
            st.facts["pod_id"] = line[4:].strip()
    st.log(out)


def step_inspect(ssh, st):
    rc, out = ssh.run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n"
        "echo DISK=$(df -BG --output=avail / | tail -1 | tr -d ' G')\n"
        "echo VOLUME=$(df -BG --output=avail /workspace | tail -1 | tr -d ' G')\n"
        "echo CORES=$(awk '{printf \"%.1f\", $1/$2}' /sys/fs/cgroup/cpu.max)\n"
        "echo PROC=$(nproc)\n", timeout=120)
    st.log(out)
    for line in out.splitlines():
        if "," in line and "MiB" in line:
            name, mem = [p.strip() for p in line.split(",")[:2]]
            st.facts["gpu"] = name
            st.facts["vram_gb"] = round(int(mem.split()[0]) / 1024)
        for k in ("DISK", "VOLUME", "CORES", "PROC"):
            if line.startswith(k + "="):
                st.facts[k.lower()] = line.split("=", 1)[1]
    if not st.facts.get("gpu"):
        raise PodError("no GPU visible on this pod")


def step_volume(ssh, st):
    """Measure the volume before trusting it.

    This step exists because of one day. The volume normally reads at 655 MB/s;
    it spent a whole session at 16, every later step took forty times longer
    than planned, and one model load stopped entirely with a thread parked in
    the kernel waiting for an mmap page that never arrived. None of that is
    fixable from inside the pod, and all of it is visible in thirty seconds.
    """
    rc, out = ssh.run(
        "mkdir -p /workspace/.admin\n"
        "dd if=/dev/urandom of=/workspace/.admin/probe bs=1M count=400 "
        "  oflag=direct 2>&1 | tail -1\n"
        "sync\n"
        "echo READ=$(dd if=/workspace/.admin/probe of=/dev/null bs=4M "
        "  iflag=direct 2>&1 | tail -1)\n"
        "rm -f /workspace/.admin/probe\n", timeout=300)
    st.log(out)
    m = re.search(r"READ=.*?([\d.]+)\s*(GB|MB)/s", out)
    if not m:
        st.warn("could not measure the volume; continuing anyway")
        return
    speed = float(m.group(1)) * (1024 if m.group(2) == "GB" else 1)
    st.facts["volume_mbs"] = round(speed, 1)
    if speed < VOLUME_SLOW_MBS:
        st.warn(
            f"the volume is reading at {speed:.0f} MB/s. It normally does 655. "
            "Everything after this will be roughly "
            f"{655 / max(speed, 1):.0f}x slower than the recorded timings, and "
            "a model load may stall outright. This is RunPod's side, not "
            "something the pod can fix.")


def step_code(ssh, st):
    """Clone to the container's local disk, fresh, every time.

    Not to /workspace. The volume is a network filesystem that is fast in bulk
    and terrible at thousands of small files, which is exactly what a source
    tree is; keeping the repository there is what made sparse-checkout
    necessary, and undoing that exclusion is what wedged a pod for good. Local
    disk makes the whole problem disappear, and a shallow clone over HTTPS
    takes seconds.
    """
    rc, out = ssh.run(
        f"rm -rf {POD_REPO}\n"
        f"git clone --depth 1 --branch {shlex.quote(REPO_REF)} "
        f"  {shlex.quote(REPO_URL)} {POD_REPO} 2>&1 | tail -3\n"
        f"cd {POD_REPO} && echo COMMIT=$(git log --oneline -1)\n"
        f"echo FILES=$(git ls-files | wc -l)\n", timeout=420)
    st.log(out)
    if rc != 0 or "FILES=" not in out:
        raise PodError("could not clone the repository:\n" + out[-500:])
    # Tagged rather than positional: git writes "Cloning into ..." to stderr,
    # which arrives first and was being recorded as the commit.
    for line in out.splitlines():
        if line.startswith("COMMIT="):
            st.facts["commit"] = line[7:].strip()[:60]


def step_packages(ssh, st):
    rc, out = ssh.run(
        _env_prefix() +
        f"cd {POD_REPO}\n"
        "if python -c 'import peft, cv2, fastapi' 2>/dev/null; then\n"
        "  echo 'already installed'\n"
        "else\n"
        "  pip install --no-cache-dir -r requirements.txt 2>&1 | tail -4\n"
        "  pip install --no-cache-dir easy-dwpose==1.0.2 --no-deps 2>&1 | tail -2\n"
        "fi\n", timeout=1500)
    st.log(out)
    if rc != 0:
        raise PodError("pip failed:\n" + out[-700:])


def step_fork(ssh, st):
    """The bundled diffusers fork: from a cached wheel if there is one.

    The fork's repository is not its package -- the package is at src/diffusers
    -- so a directory named `diffusers` in the working directory shadows the
    install and yields an empty namespace package. Building it once and keeping
    the wheel on the volume means later pods install a single file.
    """
    rc, out = ssh.run(
        _env_prefix() +
        f"cd {POD_REPO}\n"
        "mkdir -p /workspace/wheels\n"
        # `|| true` is load-bearing: under pipefail an ls matching nothing
        # fails the pipeline and takes the script with it.
        "wheel=$(ls -1t /workspace/wheels/diffusers-*.whl 2>/dev/null | head -1 || true)\n"
        "pip uninstall -q -y diffusers 2>/dev/null || true\n"
        "if [ -n \"$wheel\" ]; then\n"
        "  echo \"cached wheel: $(basename $wheel)\"\n"
        "  pip install -q --no-cache-dir --no-deps \"$wheel\"\n"
        "elif [ -f diffusers/setup.py ]; then\n"
        "  echo 'building the wheel once, for every pod after this one'\n"
        "  pip wheel -q --no-cache-dir --no-deps -w /workspace/wheels ./diffusers \\\n"
        "    && pip install -q --no-cache-dir --no-deps \\\n"
        "         \"$(ls -1t /workspace/wheels/diffusers-*.whl | head -1)\" \\\n"
        "    || pip install -q --no-cache-dir ./diffusers\n"
        "else\n"
        "  echo 'NOFORK'; exit 1\n"
        "fi\n"
        # It must not be importable from the working directory afterwards.
        "rm -rf diffusers\n", timeout=1500)
    st.log(out)
    if rc != 0:
        raise PodError("could not install the diffusers fork:\n" + out[-700:])


def step_verify(ssh, st):
    rc, out = ssh.run(
        _env_prefix() +
        f"cd {POD_REPO}\n"
        "python - <<'PY'\n"
        "import torch, peft, diffusers\n"
        "assert 'packages' in diffusers.__file__, diffusers.__file__\n"
        "from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel\n"
        "print(f'TORCH={torch.__version__}')\n"
        "print(f'DIFFUSERS={diffusers.__version__}')\n"
        "print(f'PEFT={peft.__version__}')\n"
        "print(f'CUDA={torch.cuda.is_available()}')\n"
        "PY\n", timeout=600)
    st.log(out)
    if rc != 0 or "CUDA=True" not in out:
        raise PodError("the stack does not import cleanly:\n" + out[-700:])
    for line in out.splitlines():
        if "=" in line and line.split("=")[0] in ("TORCH", "DIFFUSERS", "PEFT"):
            k, v = line.split("=", 1)
            st.facts[k.lower()] = v


def step_weights(ssh, st):
    """What the volume already holds, read before anything depends on it.

    Moved to the front. It used to run eighth, so a fresh volume announced
    itself only after seven steps of setup -- and the thing a person wants to
    know when they press start is whether this is a five minute run or a forty
    minute one.
    """
    checks = "\n".join(
        f"echo {k}=$(du -sm {H} 2>/dev/null | cut -f1 || echo 0)"
        for k, H in (("4bit", CACHE + "/models--ovedrive--Qwen-Image-Edit-2509-4bit"),
                     ("bf16", CACHE + "/models--Qwen--Qwen-Image-Edit-2509"),
                     ("lightning", CACHE + "/models--lightx2v--Qwen-Image-Lightning"),
                     ("zimage_fp32", CACHE + "/models--Tongyi-MAI--Z-Image-Turbo"),
                     ("zimage_bf16", ZIMAGE_BF16)))
    rc, out = ssh.run(_env_prefix() + checks + "\n"
                      "echo FREE=$(df -BG --output=avail /workspace | tail -1 | tr -d ' G')\n",
                      timeout=420)
    st.log(out)
    have = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower()
            try:
                have[k] = int(v.strip() or 0)
            except ValueError:
                have[k] = 0
    st.facts["weights"] = have
    st.facts["volume_free_gb"] = have.pop("free", None)

    want = st.options.get("model", "4bit")
    need = []
    if have.get(want, 0) < MODELS[want][2] * 0.97:
        need.append(f"{MODELS[want][0]} ({MODELS[want][2] // 1000} GB)")
    if have.get("lightning", 0) < 1500:
        need.append("the Lightning adapter (1.6 GB)")
    if have.get("zimage_bf16", 0) < 18000:
        need.append("Z-Image (31 GB down, 20 GB kept)")

    if need:
        st.log("to fetch: " + ", ".join(need))
        st.note("weights", "needs " + str(len(need)) + " download(s)")
    else:
        st.log("every model this pod needs is already on the volume")
        st.note("weights", "all present")


def _ensure_dwpose(ssh, st):
    """The pose model, which the repository does not carry.

    easy-dwpose fetches these to ./checkpoints relative to the working
    directory the first time it is used, which means the first customer pays
    for a 220 MB download in the middle of their job. Worse, moving the code to
    local disk left the copy an earlier pod had put on the volume behind, so a
    pod could look completely set up and then fail at the one step that needs
    a pose -- which is every step.
    """
    have = st.facts.get("weights", {}).get("dwpose")
    if have == "yes":
        return
    st.log("fetching the DWPose checkpoints")
    rc, out = ssh.run(
        _env_prefix() +
        f"cd {POD_REPO}\n"
        "mkdir -p checkpoints\n"
        # An earlier pod may have left them on the volume. Copying 220 MB from
        # there beats downloading it again.
        "for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do\n"
        "  if [ ! -f checkpoints/$f ]; then\n"
        "    src=$(find /workspace -name $f -not -path '*/.admin/*' 2>/dev/null | head -1)\n"
        "    if [ -n \"$src\" ]; then cp \"$src\" checkpoints/$f && echo \"copied $f\"; fi\n"
        "  fi\n"
        "done\n"
        "python - <<'PY'\n"
        "import os\n"
        "from huggingface_hub import hf_hub_download\n"
        "for f in ('yolox_l.onnx', 'dw-ll_ucoco_384.onnx'):\n"
        "    if not os.path.exists(os.path.join('checkpoints', f)):\n"
        "        hf_hub_download('RedHash/DWPose', f, local_dir='./checkpoints')\n"
        "        print('downloaded', f, flush=True)\n"
        "PY\n"
        # Keep a copy on the volume so the next pod takes the fast path.
        "mkdir -p /workspace/dwpose && cp -n checkpoints/*.onnx /workspace/dwpose/ 2>/dev/null\n"
        "ls -la checkpoints/*.onnx | awk '{print $9, $5}'\n", timeout=900)
    st.log(out)
    if rc != 0 or "dw-ll_ucoco_384.onnx" not in out:
        raise PodError("could not obtain the DWPose checkpoints:\n" + out[-500:])
    st.facts.setdefault("weights", {})["dwpose"] = "yes"
    st.resolve("DWPose")


def step_zimage_weights(ssh, st):
    """Get Z-Image onto a volume that has never seen it.

    The bf16 copy cannot be downloaded: it does not exist anywhere but here.
    The published repository is fp32, and the copy is written from it once.
    So on a fresh volume this is download 31 GB, write 20, delete the 31 --
    and the deletion happens immediately, before anything else is fetched,
    because 51 GB is the peak and a 60 GB volume has no room for the peak and
    the try-on checkpoint at the same time.
    """
    have = st.facts.get("weights", {})
    if have.get("zimage_bf16", 0) >= 18000:
        st.log("the Z-Image bf16 copy is already here")
        st.note("zimage_weights", "already here")
        return

    rc, out = ssh.run(
        "test -x /opt/zimage-venv/bin/python && echo HAVE || echo NONE", timeout=120)
    if "HAVE" not in out:
        # The conversion runs in that interpreter, so the venv has to exist
        # first. Ordering the steps the other way round was tried and the
        # error it produced blamed the weights.
        st.log("building the venv first; the conversion runs inside it")
        step_zimage(ssh, st)

    st.log("fetching Z-Image (31 GB) and writing the bf16 copy (20 GB)")
    ssh.run(
        _env_prefix() +
        f"cd {POD_REPO}\n"
        "setsid nohup /opt/zimage-venv/bin/python elements/save_bf16.py "
        "  > /workspace/.zimage_convert.log 2>&1 < /dev/null &\n"
        "echo started\n", timeout=180)

    started = time.time()
    while not st.cancelled:
        time.sleep(12)
        rc, out = ssh.run(
            f"du -sm {ZIMAGE_BF16} 2>/dev/null | cut -f1 || echo 0\n"
            f"du -sm {CACHE}/models--Tongyi-MAI--Z-Image-Turbo 2>/dev/null | cut -f1 || echo 0\n"
            "pgrep -f save_bf16.py >/dev/null && echo ALIVE || echo GONE\n",
            timeout=300)
        nums = [int(x) for x in out.split() if x.isdigit()]
        made = nums[0] if nums else 0
        pulled = nums[1] if len(nums) > 1 else 0
        st.facts["zimage"] = f"{pulled} MB fetched, {made} MB written"
        # Two thirds of the wait is the download, a third the write.
        st.step_progress("zimage_weights",
                         min(0.97, (pulled / 31000) * 0.66 + (made / 20000) * 0.34))
        if made >= 18000:
            break
        if "GONE" in out:
            rc, tail = ssh.run("tail -15 /workspace/.zimage_convert.log", timeout=180)
            raise PodError("the Z-Image conversion stopped:\n" + tail[-700:])

    # The fp32 goes now, not later. It is 31 GB of source for a copy that has
    # been written, and leaving it is what makes a 60 GB volume too small.
    rc, out = ssh.run(
        f"rm -rf {CACHE}/models--Tongyi-MAI--Z-Image-Turbo\n"
        "echo FREE=$(df -BG --output=avail /workspace | tail -1 | tr -d ' G')\n",
        timeout=420)
    st.log("removed the fp32 source. " + out.strip())


def step_download(ssh, st):
    """Fetch whatever is missing, reporting bytes rather than a spinner."""
    _ensure_dwpose(ssh, st)
    want = st.options.get("model", "4bit")
    repo, cache_dir, size_mb = MODELS[want]
    need = [(repo, cache_dir, size_mb)]
    if MODELS["lightning"][1] != cache_dir:
        need.append(MODELS["lightning"])

    for repo, cache_dir, size_mb in need:
        path = f"/workspace/.cache/huggingface/hub/{cache_dir}"
        rc, out = ssh.run(f"du -sm {path} 2>/dev/null | cut -f1 || echo 0",
                          timeout=180)
        got = int((out.strip() or "0").splitlines()[-1] or 0)
        if got >= size_mb * 0.97:
            st.log(f"{repo}: already here ({got} MB)")
            continue

        st.log(f"{repo}: {got} of {size_mb} MB")
        # Adopt a download already in flight rather than starting a second one.
        # The panel can be restarted mid-run -- that is the normal way to pick
        # up a fix -- and two snapshot_downloads writing the same blobs is a
        # good way to corrupt both.
        rc, out = ssh.run(
            _env_prefix() +
            f"cd {POD_REPO}\n"
            "if pgrep -f '/tmp/dl.py' >/dev/null; then echo ADOPTED; exit 0; fi\n"
            # hf_transfer fetches one file in parallel chunks. It was
            # switched off across this project after raising part-way
            # through a 57.7 GB pull, and that decision quietly cost hours:
            # measured on the pod, the plain path managed 1.7 MB/s against
            # curl's 47 on the same file, while hf_transfer finished the
            # remaining 4.4 GB in 35 seconds at 43 MB/s. Try it first and
            # fall back if it breaks -- partial blobs are kept either way,
            # so a failure costs seconds rather than the download.
            "cat > /tmp/dl.py <<'PY'\n"
            "import os, sys\n"
            "repo = sys.argv[1]\n"
            "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n"
            "from huggingface_hub import snapshot_download\n"
            "try:\n"
            "    print(snapshot_download(repo, max_workers=8), flush=True)\n"
            "except Exception as exc:\n"
            "    print(f'hf_transfer failed: {exc}', flush=True)\n"
            "    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'\n"
            "    import importlib, huggingface_hub.constants as c\n"
            "    importlib.reload(c)\n"
            "    import huggingface_hub.file_download as fd\n"
            "    importlib.reload(fd)\n"
            "    from huggingface_hub import snapshot_download as sd\n"
            "    print('falling back to the plain path', flush=True)\n"
            "    print(sd(repo), flush=True)\n"
            "PY\n"
            f"setsid nohup python /tmp/dl.py {shlex.quote(repo)} "
            f"  > /workspace/.admin_dl.log 2>&1 < /dev/null &\n"
            "echo started\n", timeout=180)
        if "ADOPTED" in out:
            st.log(f"{repo}: a download is already running; watching it")

        # Poll the bytes on disk. snapshot_download has no usable callback, and
        # the size on disk is the honest number anyway -- it includes partial
        # blobs, so a resumed download does not restart the bar at zero.
        stalled_for = 0.0
        last = got
        while not st.cancelled:
            time.sleep(6)
            rc, out = ssh.run(
                f"du -sm {path} 2>/dev/null | cut -f1 || echo 0\n"
                "pgrep -f /tmp/dl.py >/dev/null && echo ALIVE || echo GONE\n",
                timeout=180)
            lines = [l for l in out.splitlines() if l.strip()]
            now = int(lines[0]) if lines and lines[0].isdigit() else last
            alive = "ALIVE" in out
            st.step_progress("download", min(0.99, now / size_mb))
            st.facts["download"] = f"{now} / {size_mb} MB"

            if now == last:
                stalled_for += 6
                if stalled_for >= 180:
                    st.warn(f"{repo}: no bytes for three minutes. Hugging Face "
                            "throttles unauthenticated downloads, and the "
                            "volume was slow today; both look like this.")
                    stalled_for = 0
            else:
                stalled_for = 0
            last = now

            if not alive:
                if now >= size_mb * 0.95:
                    st.log(f"{repo}: done ({now} MB)")
                    break
                rc, tail = ssh.run("tail -5 /workspace/.admin_dl.log", timeout=120)
                raise PodError(f"the download stopped at {now} of {size_mb} MB:\n"
                               + tail[-500:])


def step_warm(ssh, st):
    """Load the model once and generate one image, so 'ready' means ready.

    Every earlier failure -- the shadowed package, the thread quota, the LoRA
    path that resolved under one working directory and nowhere else -- was
    invisible until something actually ran. A pod that has produced a picture
    has proved all of it at once.
    """
    want = st.options.get("model", "4bit")
    repo = MODELS[want][0]
    ssh.run(
        _env_prefix() +
        f"cd {POD_REPO}\n"
        "cat > /tmp/warm.py <<'PY'\n"
        "import os, sys, time, glob\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from PIL import Image\n"
        "from pipeline import LayeringVTONPipeline\n"
        "from utils import process_inputs\n"
        "t = time.time()\n"
        "pipe = LayeringVTONPipeline(sys.argv[1], 'weights', lightning=8)\n"
        "load = time.time() - t\n"
        "print(f'LOAD={load:.1f}', flush=True)\n"
        "person = sorted(glob.glob('assets/person_*.png'))[0]\n"
        "p, g, pose = process_inputs(Image.open(person).convert('RGB'),\n"
        "                            Image.open('assets/coat.png').convert('RGB'), None)\n"
        "t = time.time()\n"
        "img = pipe(person_img=p, garment_img=g, pose_img=pose,\n"
        "           description='the coat', mode='upper', seed=7)\n"
        "img.save('/workspace/.admin/warm.png')\n"
        "print(f'IMAGE={time.time() - t:.1f}', flush=True)\n"
        "print('READY', flush=True)\n"
        "PY\n"
        "mkdir -p /workspace/.admin\n"
        f"setsid nohup python /tmp/warm.py {shlex.quote(repo)} "
        f"  > /workspace/.admin_warm.log 2>&1 < /dev/null &\n"
        "echo started\n", timeout=300)

    started = time.time()
    # The expected load time, so the bar can move while a model that reports
    # nothing for four minutes is loading. It is an estimate and is labelled as
    # one; the step only completes on the process actually saying READY.
    expected = 300.0 if want == "4bit" else 600.0
    while not st.cancelled:
        time.sleep(8)
        rc, out = ssh.run(
            "tail -3 /workspace/.admin_warm.log\n"
            "pgrep -f /tmp/warm.py >/dev/null && echo ALIVE || echo GONE\n",
            timeout=180)
        for line in out.splitlines():
            if line.startswith("LOAD="):
                st.facts["load_seconds"] = float(line.split("=")[1])
            if line.startswith("IMAGE="):
                st.facts["image_seconds"] = float(line.split("=")[1])
        if "READY" in out:
            st.step_progress("warm", 1.0)
            return
        if "GONE" in out:
            rc, tail = ssh.run("tail -20 /workspace/.admin_warm.log", timeout=120)
            raise PodError("the model did not come up:\n" + tail[-800:])
        st.step_progress("warm", min(0.95, (time.time() - started) / expected))


def step_zimage(ssh, st):
    """The second interpreter, for the half of the product that answers text.

    Z-Image needs diffusers from source, which wants huggingface_hub 1.x, while
    the try-on stack's transformers caps it below 1.0. No single set of versions
    satisfies both, so it lives in its own venv -- and that venv sits on the
    container disk, which means it is gone after every migration.

    Built whether or not the weights are here yet: the conversion that produces
    them runs in this interpreter.
    """
    rc, out = ssh.run(
        "test -x /opt/zimage-venv/bin/python && echo HAVE || echo BUILD",
        timeout=120)
    if "HAVE" in out:
        st.log("the z-image venv is already here")
        st.note("zimage", "already built")
    else:
        st.log("building the z-image venv (about four minutes, once per pod)")
        ssh.run(
            f"cd {POD_REPO}\n"
            "setsid nohup bash tools/setup_pod.sh --zimage "
            "  > /workspace/zimage_setup.log 2>&1 < /dev/null &\n"
            "echo started\n", timeout=180)
        started = time.time()
        while not st.cancelled:
            time.sleep(10)
            rc, out = ssh.run(
                "test -x /opt/zimage-venv/bin/python && echo DONE || echo BUILDING\n"
                "pgrep -f setup_pod.sh >/dev/null && echo ALIVE || echo GONE\n",
                timeout=180)
            if "DONE" in out:
                break
            if "GONE" in out:
                rc, tail = ssh.run("tail -12 /workspace/zimage_setup.log",
                                   timeout=120)
                raise PodError("the z-image venv did not build:\n" + tail[-600:])
            st.step_progress("zimage", min(0.95, (time.time() - started) / 260))

    rc, out = ssh.run(
        "/opt/zimage-venv/bin/python -c \"from diffusers import ZImagePipeline; "
        "import fastapi; print(\'ZIMAGE_OK\')\" 2>&1 | tail -2",
        timeout=300)
    st.log(out)
    if "ZIMAGE_OK" not in out:
        raise PodError("the z-image venv is incomplete:\n" + out[-400:])


def step_serve(ssh, st):
    """Leave the pod actually serving, not merely proven.

    The panel used to stop after generating one image, so it reported ready and
    left nothing running. Everything after that was four commands typed by hand,
    and forgetting the second one meant two tools silently did not work.
    """
    rc, out = ssh.run(
        f"cd {POD_REPO}\n"
        "bash tools/pod_serve.sh start 2>&1 | tail -2\n", timeout=600)
    st.log(out)
    if "ready" not in out and "starting" not in out:
        raise PodError("the try-on server did not start:\n" + out[-400:])

    rc, out = ssh.run(
        "test -x /opt/zimage-venv/bin/python && echo HAVE || echo NONE",
        timeout=120)
    if "HAVE" in out:
        rc, out = ssh.run(
            f"cd {POD_REPO}\n"
            "bash tools/pod_serve.sh start zimage 2>&1 | tail -2\n",
            timeout=600)
        st.log(out)
    else:
        st.warn("no z-image venv, so only the try-on tools will work")

    # What the bridge needs, read once, so nobody has to go and find it.
    rc, out = ssh.run(
        "echo BRIDGE=root@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22", timeout=120)
    for line in out.splitlines():
        if line.startswith("BRIDGE="):
            st.facts["bridge_ssh"] = line[7:].strip()
            st.log("bridge address: " + line[7:].strip())


STEPS = [
    Step("connect", "Connect", 5, step_connect, "ssh, and which pod this is"),
    Step("inspect", "Inspect", 8, step_inspect, "GPU, disk, CPU quota"),
    Step("volume", "Measure the volume", 40, step_volume,
         "read speed, before anything depends on it"),
    Step("weights", "What the volume holds", 20, step_weights,
         "read first, so you know if this is five minutes or forty"),
    Step("code", "Clone the code", 25, step_code, "to local disk, not the volume"),
    Step("packages", "Install packages", 90, step_packages, "pip"),
    Step("fork", "Install the fork", 60, step_fork, "from the cached wheel"),
    Step("verify", "Verify the stack", 20, step_verify, "imports and CUDA"),
    Step("zimage", "Build the model maker", 260, step_zimage,
         "a second interpreter; the two stacks cannot share one"),
    Step("zimage_weights", "Get Z-Image", 900, step_zimage_weights,
         "31 GB down, 20 GB kept, the rest deleted at once"),
    Step("download", "Download what is missing", 420, step_download,
         "measured in bytes"),
    Step("warm", "Load and generate", 360, step_warm,
         "one real image, so ready means ready"),
    Step("serve", "Start serving", 45, step_serve,
         "both servers, and the address the bridge needs"),
]
TOTAL_WEIGHT = sum(s.weight for s in STEPS)


# ---------------------------------------------------------------------------
# the run

@dataclass
class State:
    host: str = ""
    options: dict = field(default_factory=dict)
    status: str = "idle"          # idle | running | ready | failed | cancelled
    current: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    cancelled: bool = False
    facts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    steps: dict = field(default_factory=dict)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def reset(self, host, options):
        with self._lock:
            self.host, self.options = host, options
            self.status, self.current, self.error = "running", "", ""
            self.cancelled = False
            self.started_at, self.finished_at = time.time(), 0.0
            self.facts, self.warnings, self.lines = {}, [], []
            self.steps = {s.key: {"state": "waiting", "progress": 0.0,
                                  "seconds": 0.0, "note": ""} for s in STEPS}

    def log(self, text):
        with self._lock:
            for line in str(text).splitlines():
                if line.strip():
                    self.lines.append(f"{time.strftime('%H:%M:%S')}  {line}")
            del self.lines[:-400]

    def warn(self, text):
        with self._lock:
            self.warnings.append(text)
        self.log("! " + text)

    def resolve(self, fragment):
        """Drop a warning that a later step has dealt with.

        A panel that keeps showing a problem after fixing it teaches people to
        ignore its warnings, which costs more than the warning was worth.
        """
        with self._lock:
            self.warnings = [w for w in self.warnings if fragment not in w]

    def note(self, key, text):
        """Why a step took no time, said where the step is.

        A run where seven of twelve steps finish instantly looks broken unless
        each one says it found its work already done.
        """
        with self._lock:
            if key in self.steps:
                self.steps[key]["note"] = text

    def step_progress(self, key, value):
        with self._lock:
            if key in self.steps:
                self.steps[key]["progress"] = max(0.0, min(1.0, value))

    def percent(self):
        done = sum(s.weight * self.steps.get(s.key, {}).get("progress", 0.0)
                   for s in STEPS)
        return round(100 * done / TOTAL_WEIGHT, 1)

    def snapshot(self):
        with self._lock:
            return {
                "host": self.host, "status": self.status,
                "current": self.current, "percent": self.percent(),
                "elapsed": round((self.finished_at or time.time())
                                 - self.started_at) if self.started_at else 0,
                "error": self.error, "facts": dict(self.facts),
                "warnings": list(self.warnings), "log": self.lines[-160:],
                "steps": [{"key": s.key, "title": s.title, "detail": s.detail,
                           "weight": s.weight, **self.steps.get(s.key, {})}
                          for s in STEPS],
            }


STATE = State()
_thread = None


def _run_all(host, key, options):
    ssh = Ssh(host, key)
    for step in STEPS:
        if STATE.cancelled:
            STATE.status, STATE.current = "cancelled", ""
            return
        with STATE._lock:
            STATE.current = step.key
            STATE.steps[step.key]["state"] = "running"
        STATE.log(f"- {step.title}")
        began = time.time()
        try:
            step.run(ssh, STATE)
        except PodError as exc:
            with STATE._lock:
                STATE.steps[step.key]["state"] = "failed"
                STATE.status, STATE.error = "failed", str(exc)
                STATE.finished_at = time.time()
            STATE.log(f"failed: {exc}")
            return
        except Exception as exc:                      # noqa: BLE001
            with STATE._lock:
                STATE.steps[step.key]["state"] = "failed"
                STATE.status = "failed"
                STATE.error = f"{type(exc).__name__}: {exc}"
                STATE.finished_at = time.time()
            STATE.log(f"failed: {type(exc).__name__}: {exc}")
            return
        with STATE._lock:
            STATE.steps[step.key]["state"] = "done"
            STATE.steps[step.key]["progress"] = 1.0
            STATE.steps[step.key]["seconds"] = round(time.time() - began, 1)
    with STATE._lock:
        STATE.status, STATE.current = "ready", ""
        STATE.finished_at = time.time()
    STATE.log("the pod is ready")


# ---------------------------------------------------------------------------
# the API

app = FastAPI(title="Lookzi pod admin")


class StartRequest(BaseModel):
    ssh: str
    model: str = "4bit"
    key: str | None = None


def parse_host(text):
    """Accept the whole line RunPod hands you, not a cleaned-up version of it.

    It gives `ssh abc-123@ssh.runpod.io -i ~/.ssh/id_ed25519`, and asking
    somebody to edit that down to the middle part is asking them to make a
    mistake at four in the morning.
    """
    text = (text or "").strip()
    m = re.search(r"([A-Za-z0-9_.\-]+@[A-Za-z0-9_.\-]+)", text)
    if not m:
        raise HTTPException(400, "no user@host found in that line")
    return m.group(1)


@app.post("/api/start")
def start(req: StartRequest):
    global _thread
    if STATE.status == "running":
        raise HTTPException(409, "a run is already in progress")
    if req.model not in MODELS:
        raise HTTPException(400, f"unknown model {req.model}")
    host = parse_host(req.ssh)
    key = req.key or SSH_KEY
    if not os.path.exists(key):
        raise HTTPException(400, f"no ssh key at {key}")
    STATE.reset(host, {"model": req.model})
    STATE.log(f"pod {host}, model {req.model}")
    _thread = threading.Thread(target=_run_all, args=(host, key, STATE.options),
                               daemon=True)
    _thread.start()
    return {"host": host}


@app.post("/api/cancel")
def cancel():
    STATE.cancelled = True
    return {"cancelled": True}


@app.get("/api/status")
def status():
    return STATE.snapshot()


@app.get("/api/steps")
def steps():
    return [{"key": s.key, "title": s.title, "detail": s.detail,
             "weight": s.weight} for s in STEPS]


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(HERE, "static", "admin.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>admin.html is missing</h1>", status_code=500)
    with open(path, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


def main():
    import uvicorn
    port = int(os.environ.get("ADMIN_PORT", "8090"))
    print(f"pod admin on http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

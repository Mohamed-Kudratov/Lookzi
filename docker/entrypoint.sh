#!/usr/bin/env bash
# Runs on every pod start. Everything heavy is already in the image, so this
# only reconciles what genuinely differs between pods: the code revision, the
# volume layout, and whether the big model is present.
set -uo pipefail

REPO=/opt/lookzi
WORKSPACE="${WORKSPACE:-/workspace}"
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null

# A pod's GPU is not always as empty as its spec sheet. Memory held by a dead
# process, or by another tenant outside this container, is invisible here and
# cannot be reclaimed from inside -- but it will OOM a load later. Say so now.
python - <<'PY' 2>/dev/null
import torch
free, total = (x / 1024**3 for x in torch.cuda.mem_get_info())
print(f"VRAM: {free:.1f} GB free of {total:.1f} GB")
if free < total * 0.9:
    print(f"WARNING: {total - free:.1f} GB held by something else.")
    print("         If nothing of yours is running, redeploy this pod.")
PY
echo "=============================================="

# Latest code, without losing the image's already-installed state.
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch -q origin 2>/dev/null \
      && git -C "$REPO" reset -q --hard "origin/${REPO_REF:-main}" 2>/dev/null \
      && echo "code: $(git -C "$REPO" log --oneline -1)"

    # reset --hard restores ./diffusers, which then shadows the installed
    # package. The install is already in the image; only the directory needs
    # moving back out of the way.
    if [ -d "$REPO/diffusers" ]; then
        rm -rf "$REPO/diffusers_src"
        mv "$REPO/diffusers" "$REPO/diffusers_src"
    fi
fi

# The volume is the only thing that survives a pod, so the 57.7 GB model lives
# there. Anything smaller belongs on local disk, where mmap page faults do not
# cross a network.
mkdir -p "$HF_HOME" "$WORKSPACE/.pip-cache"

if [ -n "${MODEL_PATH:-}" ]; then
    echo "MODEL_PATH=$MODEL_PATH"
fi

python - <<PY
import os
p = os.path.join("$HF_HOME", "hub", "models--Qwen--Qwen-Image-Edit-2509")
if os.path.isdir(p):
    n = sum(len(f) for _, _, f in os.walk(p))
    print(f"try-on weights: present ({n} files)")
else:
    print("try-on weights: NOT on this volume -- first run will fetch 57.7 GB")
PY

echo
echo "Ready. Nothing to install."
echo "  try-on:   python infer.py --examples --lightning 8"
echo "  elements: /opt/zimage-venv/bin/python elements/hero.py candidates --face all"
echo

# Keep the container alive for SSH, or run whatever the pod was given.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exec sleep infinity

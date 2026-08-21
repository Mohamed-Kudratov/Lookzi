#!/usr/bin/env bash
# One command per pod. Idempotent, unattended, resumable.
#
#   bash go.sh                    # bring the pod up, report, wait
#   bash go.sh ui                 # ... and launch the try-on web UI
#   bash go.sh heroes <face,...>  # stage 1: candidate faces
#   bash go.sh vary <face,...>    # stage 2: the coverage grid from hero 000
#   bash go.sh view <dir>         # serve any folder of images for review
#
# RunPod hands out a new pod often -- a restart, a reclaimed GPU, a migration --
# and each one arrives with the container disk wiped. Doing that recovery by
# hand turns every session into a setup session. This does it in one line.
#
# Everything here is skippable: if the image already carries the packages, or a
# previous run already produced the images, nothing is redone.

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
ZVENV="${ZIMAGE_VENV:-/opt/zimage-venv}"

export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/.pip-cache}"
# Both of RunPod's preset accelerators break large HF downloads; plain HTTP
# sustains ~190 MB/s. See TROUBLESHOOTING.md.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

cd "$REPO"

say() { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------------------
say "pod"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python - <<'PY' 2>/dev/null
import torch
free, total = (x / 1024**3 for x in torch.cuda.mem_get_info())
print(f"VRAM {free:.1f} GB free of {total:.1f} GB")
# A pod's GPU is not always as empty as its spec sheet: memory held by a dead
# process or another tenant is invisible here and cannot be reclaimed from
# inside, but it will OOM a load later.
if free < total * 0.9:
    print(f"WARNING: {total-free:.1f} GB held by something else -- redeploy this pod")
PY

# ---------------------------------------------------------------------------
say "code"
git fetch -q origin 2>/dev/null && git reset -q --hard origin/main 2>/dev/null
git log --oneline -1
# A reset restores ./diffusers, which then shadows the installed package --
# the fork's repo root has no __init__.py, so the import yields an empty
# namespace package. Move it aside; the install itself is unaffected.
if [ -d "$REPO/diffusers" ]; then
    rm -rf "$REPO/diffusers_src"
    mv "$REPO/diffusers" "$REPO/diffusers_src"
fi

# ---------------------------------------------------------------------------
say "try-on stack"
if python -c "from diffusers import QwenImageEditPlusPipeline" 2>/dev/null; then
    echo "already installed"
else
    echo "installing (a few minutes; skipped entirely on the pod image)"
    bash start.sh --no-ui 2>&1 | grep -aE '^---|weights ok|ERROR' || true
fi
python -c "import diffusers; print('diffusers', diffusers.__version__, '->', diffusers.__file__)" 2>&1 | tail -1

# ---------------------------------------------------------------------------
say "elements stack"
if [ -x "$ZVENV/bin/python" ] && "$ZVENV/bin/python" -c "from diffusers import ZImagePipeline" 2>/dev/null; then
    echo "already installed"
else
    echo "building isolated venv (one time)"
    bash elements/setup_zimage.sh 2>&1 | grep -aE '^---|import OK|ERROR' || true
fi

# ---------------------------------------------------------------------------
say "weights"
python - <<PY
import os
hub = os.path.join("$HF_HOME", "hub")
for name, label in (("models--Qwen--Qwen-Image-Edit-2509", "try-on 57.7 GB"),
                    ("models--Tongyi-MAI--Z-Image-Turbo", "z-image 12 GB")):
    p = os.path.join(hub, name)
    print(f"  {label:16} {'present' if os.path.isdir(p) else 'MISSING -- will download on first use'}")
PY

# ---------------------------------------------------------------------------
CMD="${1:-}"
shift || true

case "$CMD" in
  ui)
    say "web ui"
    pkill -f 'python app.py' 2>/dev/null; sleep 2
    nohup setsid python app.py > "$WORKSPACE/gradio.log" 2>&1 </dev/null &
    for _ in $(seq 1 40); do
        URL=$(grep -ao 'https://[a-z0-9]*\.gradio\.live' "$WORKSPACE/gradio.log" 2>/dev/null | tail -1)
        [ -n "${URL:-}" ] && break
        sleep 3
    done
    echo "${URL:-http://localhost:7860  (expose port 7860 for a stable URL)}"
    ;;
  heroes)
    say "stage 1: candidate faces"
    nohup setsid "$ZVENV/bin/python" elements/hero.py candidates \
        --face "${1:-all}" --n "${N:-6}" --out "$WORKSPACE/elements_out" \
        > "$WORKSPACE/heroes.log" 2>&1 </dev/null &
    echo "running -> $WORKSPACE/heroes.log"
    ;;
  vary)
    say "stage 2: coverage grid"
    nohup setsid python elements/hero.py variations \
        --face "${1:-all}" --hero "${HERO:-000}" --per-face "${PER_FACE:-30}" \
        --out "$WORKSPACE/elements_out" \
        > "$WORKSPACE/vary.log" 2>&1 </dev/null &
    echo "running -> $WORKSPACE/vary.log"
    ;;
  view)
    say "viewer"
    DIR="${1:-$WORKSPACE/elements_out/models}"
    pkill -f view_sweep 2>/dev/null; sleep 2
    nohup setsid python view_sweep.py --dir "$DIR" \
        > "$WORKSPACE/viewer.log" 2>&1 </dev/null &
    for _ in $(seq 1 30); do
        URL=$(grep -ao 'https://[a-z0-9]*\.gradio\.live' "$WORKSPACE/viewer.log" 2>/dev/null | tail -1)
        [ -n "${URL:-}" ] && break
        sleep 3
    done
    echo "${URL:-check $WORKSPACE/viewer.log}"
    ;;
  "")
    echo
    echo "Ready. Next:"
    echo "  bash go.sh ui                     web UI"
    echo "  bash go.sh heroes f_cauz_30s_avg  candidate faces"
    echo "  bash go.sh vary   f_cauz_30s_avg  coverage grid"
    echo "  bash go.sh view   /workspace/elements_out/models"
    ;;
  *)
    echo "unknown command: $CMD" >&2
    exit 2
    ;;
esac

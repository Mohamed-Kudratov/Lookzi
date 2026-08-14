#!/usr/bin/env bash
# Bring a pod up and launch the web UI. Run this after every pod start.
#
#   bash /workspace/Layering-Virtual-Try-On/start.sh
#
# Stopping a RunPod pod wipes the container disk, and pip installs live there --
# so a restarted pod has its 57.7 GB of weights but no packages, and nothing
# launches the app on its own. This script fixes both, and keeps the virtualenv
# on /workspace so the reinstall only ever happens once.
#
#   --no-ui   set everything up but do not launch Gradio
#   --force   reinstall packages even if the venv looks current

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="$WORKSPACE/Layering-Virtual-Try-On"
VENV="$WORKSPACE/venv"
REPO_URL="${REPO_URL:-https://github.com/Mohamed-Kudratov/Lookzi.git}"

LAUNCH_UI=1
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --no-ui) LAUNCH_UI=0 ;;
        --force) FORCE=1 ;;
        *) echo "unknown option: $arg"; exit 2 ;;
    esac
done

export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
# Both of RunPod's preset download accelerators stall or error here; plain HTTP
# sustains ~190 MB/s. See TROUBLESHOOTING.md.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# A pod's GPU is not always as empty as its spec sheet: memory held by a dead
# process or another tenant is invisible here and cannot be reclaimed from
# inside the container. Catch it now rather than 20 minutes into a load.
python3 - <<'PY'
import sys
try:
    import torch
    free, total = torch.cuda.mem_get_info()
    free, total = free / 1024**3, total / 1024**3
    print(f"VRAM: {free:.1f} GB free of {total:.1f} GB")
    if free < total * 0.9:
        print(f"WARNING: {total - free:.1f} GB is held by something else.")
        print("         If nothing of yours is running, this pod is unusable -- redeploy it.")
except ImportError:
    print("(torch not importable yet -- checked again after install)")
PY
echo "=============================================="

# ---- repository -----------------------------------------------------------
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull -q --ff-only || echo "  (pull skipped)"
else
    command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
    git clone -q "$REPO_URL" "$REPO"
fi
cd "$REPO"

# ---- virtualenv on the volume ---------------------------------------------
# --system-site-packages keeps torch from the pod image (several GB, already
# built against the right CUDA) instead of duplicating it onto the volume.
if [ ! -x "$VENV/bin/python" ]; then
    echo "--- creating venv on the volume (one time) ---"
    python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

REQ_HASH=$(md5sum requirements.txt | cut -d' ' -f1)
STAMP="$VENV/.requirements.md5"
if [ "$FORCE" = "1" ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$REQ_HASH" ]; then
    echo "--- installing packages (one time, ~3 min) ---"
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    pip install -q easy-dwpose==1.0.2 --no-deps
    echo "$REQ_HASH" > "$STAMP"
else
    echo "--- packages already installed ---"
fi

# ---- the bundled diffusers fork -------------------------------------------
# ./diffusers is the fork's repository, not its package; the package is at
# diffusers/src/diffusers. Python searches the working directory first, so that
# directory shadows any install. Hence: install normally, then move it aside.
if ! python -c "import diffusers, os; assert getattr(diffusers,'__file__',None)" 2>/dev/null; then
    echo "--- installing the diffusers fork ---"
    if [ -d "$REPO/diffusers" ]; then
        pip uninstall -q -y diffusers 2>/dev/null || true
        pip install -q "$REPO/diffusers"
        mv "$REPO/diffusers" "$REPO/diffusers_src"
    elif [ -d "$REPO/diffusers_src" ]; then
        pip install -q "$REPO/diffusers_src"
    else
        echo "ERROR: neither ./diffusers nor ./diffusers_src exists."; exit 1
    fi
fi
python -c "
import diffusers
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
print(f'    diffusers {diffusers.__version__}')
"

# ---- weights ---------------------------------------------------------------
for f in yolox_l.onnx dw-ll_ucoco_384.onnx; do
    [ -f "checkpoints/$f" ] || python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('RedHash/DWPose', '$f', local_dir='./checkpoints')"
done

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen-Image-Edit-2509}"
echo "--- verifying weights ---"
if ! python verify_weights.py --repo "$MODEL_PATH" --path "$HF_HOME" >/dev/null 2>&1; then
    echo "    refetching missing or corrupt files"
    for i in 1 2 3 4 5; do
        python - <<PY && break
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_PATH",
                  allow_patterns=["*.json","*.safetensors","*.txt","*.jinja"],
                  max_workers=2)
PY
        echo "    attempt $i incomplete, retrying"
    done
fi
echo "    weights ok"

# ---- launch ---------------------------------------------------------------
if [ "$LAUNCH_UI" = "0" ]; then
    echo
    echo "Setup complete. Activate with: source $VENV/bin/activate"
    exit 0
fi

pkill -f 'python app.py' 2>/dev/null || true
sleep 2
rm -f "$WORKSPACE/gradio.log"

# share=True unless port 7860 is exposed in the pod template. RunPod's proxy
# only routes ports declared there -- otherwise the proxy URL 404s.
export GRADIO_SHARE="${GRADIO_SHARE:-1}"
export MODEL_PATH
nohup setsid python app.py > "$WORKSPACE/gradio.log" 2>&1 </dev/null &

echo "--- starting the web UI ---"
for i in $(seq 1 40); do
    URL=$(grep -ao 'https://[a-z0-9]*\.gradio\.live' "$WORKSPACE/gradio.log" | tail -1)
    [ -n "$URL" ] && break
    grep -aq 'Running on local URL' "$WORKSPACE/gradio.log" && [ "$GRADIO_SHARE" = "0" ] && break
    sleep 3
done

echo
echo "=============================================="
if [ -n "${URL:-}" ]; then
    echo "  $URL"
    echo "  (public to anyone with the link; expires in 72h)"
else
    echo "  http://localhost:7860"
    echo "  Expose HTTP port 7860 on the pod for a stable URL:"
    echo "  https://<POD_ID>-7860.proxy.runpod.net"
fi
echo
echo "  log: $WORKSPACE/gradio.log"
echo "=============================================="

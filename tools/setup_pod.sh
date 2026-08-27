#!/usr/bin/env bash
# Rebuild the try-on environment on a pod whose container disk was wiped.
#
# Stopping a pod, or having a GPU reclaimed and migrated, destroys everything
# outside /workspace. The volume keeps the weights, the LoRA, the bundled
# diffusers fork and the DWPose checkpoints -- all the large, slow things --
# so what is left to do is package installs, which is what this does.
#
# docker/Dockerfile bakes the same steps into an image, and starting a pod from
# that image is faster and more reliable than running this. Use this when the
# pod is already running and cannot be restarted -- a scarce A100 given up on a
# restart may not come back, which is a worse loss than ten minutes of pip.
#
#     bash tools/setup_pod.sh          # try-on stack only
#     bash tools/setup_pod.sh --zimage # also the isolated Z-Image venv
#
# Safe to run twice: every step checks before doing anything.
set -euo pipefail

REPO=/workspace/lvton
cd "$REPO"

echo "=== environment ==="
cat > "$REPO/.podenv" <<'ENVFILE'
# Source this before running anything on a pod.
#
# The thread limit is not a tuning knob, it is a correctness fix. RunPod grants
# this container 13.6 CPUs but leaves /proc showing the host's 128, so every
# library that sizes its pool from nproc opens 128 threads against 13.6 cores
# and spends its time in the scheduler. That looked exactly like a hung model
# load and cost a day to find. Measure with two reads of utime 30s apart, not
# by watching top.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# The cache lives on the volume so a wiped container disk does not mean
# re-downloading 86 GB.
export HF_HOME=/workspace/.cache/huggingface

# Both of RunPod's preset download accelerators break large HF downloads: xet
# stalls after ~10 GB and hf_transfer raises mid-download. Plain HTTP sustained
# ~190 MB/s, which is fast enough and finishes.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Loading a 40 GB transformer shard by shard fragments the allocator enough to
# OOM with gigabytes still free.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVFILE
# shellcheck disable=SC1091
. "$REPO/.podenv"
echo "  quota: $(awk '{printf "%.1f", $1/$2}' /sys/fs/cgroup/cpu.max) cores, "\
     "nproc says $(nproc), threads capped at $OMP_NUM_THREADS"

echo "=== packages ==="
if python -c "import diffusers, peft, cv2" 2>/dev/null; then
    echo "  already installed"
else
    pip install -q --no-cache-dir -r requirements.txt
    pip install -q --no-cache-dir easy-dwpose==1.0.2 --no-deps

    # The fork's repository is not its package -- the package is at
    # src/diffusers. Installing normally and leaving the tree
    # named anything other than "diffusers" keeps Python from finding the
    # source directory first and importing an empty namespace package.
    #
    # The tracked tree is the one to trust. A leftover diffusers_src on the
    # volume is whatever survived the last pod, and on this one that was a
    # partial copy with no setup.py -- installable-looking until pip disagrees.
    pip uninstall -q -y diffusers 2>/dev/null || true
    if [ -f "$REPO/diffusers/setup.py" ]; then
        pip install -q --no-cache-dir "$REPO/diffusers"
        rm -rf "$REPO/diffusers_src"
        mv "$REPO/diffusers" "$REPO/diffusers_src"
    elif [ -f "$REPO/diffusers_src/setup.py" ]; then
        pip install -q --no-cache-dir "$REPO/diffusers_src"
    else
        echo "  !! no installable diffusers fork here." >&2
        echo "     Expected diffusers/setup.py. Restore the tree with a checkout." >&2
        exit 1
    fi
fi

python - <<'PY'
import diffusers, torch, peft
assert "packages" in diffusers.__file__, diffusers.__file__
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel  # noqa
print(f"  torch {torch.__version__}  diffusers {diffusers.__version__}  peft {peft.__version__}")
print(f"  cuda {torch.cuda.is_available()}  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
PY

echo "=== weights on the volume ==="
for d in "$HF_HOME"/hub/models--*; do
    [ -e "$d" ] || continue
    printf '  %-46s %s\n' "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
[ -f "$REPO/weights/pytorch_lora_weights.safetensors" ] \
    && echo "  VTON LoRA present" || echo "  !! VTON LoRA missing"
[ -f "$REPO/checkpoints/dw-ll_ucoco_384.onnx" ] \
    && echo "  DWPose present" || echo "  !! DWPose missing"

if [ "${1:-}" = "--zimage" ] && [ ! -d /opt/zimage-venv ]; then
    echo "=== z-image venv ==="
    python -m venv /opt/zimage-venv
    /opt/zimage-venv/bin/pip install -q --upgrade pip
    /opt/zimage-venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
    /opt/zimage-venv/bin/pip install -q "git+https://github.com/huggingface/diffusers"
    /opt/zimage-venv/bin/pip install -q accelerate safetensors sentencepiece protobuf pillow
    /opt/zimage-venv/bin/pip install -q --upgrade transformers huggingface_hub
fi

echo "=== ready ==="
echo "source $REPO/.podenv before running anything"

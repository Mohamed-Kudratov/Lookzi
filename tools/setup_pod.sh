#!/usr/bin/env bash
# Rebuild the try-on environment on a pod whose container disk was wiped.
#
# Stopping a pod, or having a GPU reclaimed and migrated, destroys everything
# outside /workspace. The volume keeps the weights, the LoRA, the DWPose
# checkpoints and the git object store -- all the large, slow things -- so what
# is left is package installation, which is what this does.
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

# Where the repository is, discovered rather than assumed. It used to be pinned
# to /workspace/lvton, which stopped being true the moment the code moved to
# the container's local disk -- and then failed outright when that stale copy
# was deleted, taking the admin panel's Z-Image step with it.
REPO="${REPO:-}"
if [ -z "$REPO" ]; then
    # The directory this script is in, which is the checkout it belongs to.
    REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fi
[ -f "$REPO/requirements.txt" ] || {
    echo "no requirements.txt under $REPO -- run this from inside a checkout" >&2
    exit 1
}
# Everything to do with the fork happens on the container disk. It is local
# NVMe; /workspace is a network filesystem that is fast in bulk and terrible at
# thousands of small files, which is exactly what a source tree is.
FORK_DIR=/opt/fork
WHEELS=/workspace/wheels

cd "$REPO"

install_fork() {
    # The bundled diffusers fork, without ever writing it to the volume.
    #
    # Three things make this awkward, and all three are handled here.
    #
    # The fork's repository is not its package: the package is at src/diffusers,
    # so a directory named "diffusers" in the working directory shadows the
    # install and yields an empty namespace package.
    #
    # It is excluded from the checkout by sparse-checkout, because materialising
    # 2247 files onto a network volume is that filesystem's worst case. Undoing
    # the exclusion to get the source back took minutes, rewrote the index, and
    # wedged outright when a killed git left index.lock behind. git archive
    # streams the same tree straight out of the object store without touching
    # the index, onto local NVMe: 0.2 seconds against several minutes.
    #
    # And the build itself is worth keeping. The wheel goes on the volume, so
    # the next pod installs one file and never unpacks a source tree at all.
    mkdir -p "$WHEELS"
    local wheel
    # `|| true` is load-bearing: under `set -o pipefail` an ls that
    # matches nothing fails the whole pipeline, the assignment inherits
    # that status, and `set -e` kills the script. So the very first run
    # on an empty cache -- the only run that needs this branch -- was the
    # one that could never reach it.
    wheel=$(ls -1t "$WHEELS"/diffusers-*.whl 2>/dev/null | head -1 || true)

    pip uninstall -q -y diffusers 2>/dev/null || true
    if [ -n "$wheel" ]; then
        echo "  installing the cached fork wheel: $(basename "$wheel")"
        pip install -q --no-cache-dir --no-deps "$wheel"
        return
    fi

    rm -rf "$FORK_DIR"
    mkdir -p "$FORK_DIR"
    if git -C "$REPO" archive HEAD diffusers 2>/dev/null | tar -x -C "$FORK_DIR" \
       && [ -f "$FORK_DIR/diffusers/setup.py" ]; then
        echo "  fork extracted from the object store to $FORK_DIR"
    elif [ -f "$REPO/diffusers_src/setup.py" ]; then
        # A tree an older pod left behind. Not preferred: the one on this pod
        # turned out to be a partial copy with no setup.py at all.
        echo "  falling back to diffusers_src on the volume"
        cp -r "$REPO/diffusers_src" "$FORK_DIR/diffusers"
    else
        echo "  !! cannot obtain the diffusers fork: git archive produced no" >&2
        echo "     setup.py and no usable diffusers_src exists." >&2
        exit 1
    fi

    echo "  building a wheel so no later pod pays for this"
    if pip wheel -q --no-cache-dir --no-deps -w "$WHEELS" "$FORK_DIR/diffusers"; then
        pip install -q --no-cache-dir --no-deps \
            "$(ls -1t "$WHEELS"/diffusers-*.whl | head -1)"
    else
        # A wheel that will not build is not worth failing the setup over. The
        # direct install still works; the next pod simply pays again.
        echo "  wheel build failed; installing directly"
        pip install -q --no-cache-dir "$FORK_DIR/diffusers"
    fi
}

echo "=== environment ==="
# Read before sourcing: nproc obeys OMP_NUM_THREADS, so asking afterwards
# prints the cap back at itself and hides the discrepancy worth seeing.
raw_cpus=$(nproc)
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
# re-downloading 90 GB.
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
echo "  quota $(awk '{printf "%.1f", $1/$2}' /sys/fs/cgroup/cpu.max) cores," \
     "/proc advertises $raw_cpus, threads capped at $OMP_NUM_THREADS"

echo "=== packages ==="
if python -c "import diffusers, peft, cv2" 2>/dev/null; then
    echo "  already installed"
else
    pip install -q --no-cache-dir -r requirements.txt
    pip install -q --no-cache-dir easy-dwpose==1.0.2 --no-deps
    install_fork
fi

python - <<'PY'
import diffusers, torch, peft
assert "packages" in diffusers.__file__, diffusers.__file__
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel  # noqa
print(f"  torch {torch.__version__}  diffusers {diffusers.__version__}  peft {peft.__version__}")
print(f"  cuda {torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
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
    # This venv also serves HTTP: zimage_server.py is a second process on the
    # same card, because the two stacks cannot share an interpreter.
    /opt/zimage-venv/bin/pip install -q fastapi "uvicorn[standard]" python-multipart
fi

echo "=== ready ==="
echo "source $REPO/.podenv before running anything"

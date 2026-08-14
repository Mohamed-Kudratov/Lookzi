#!/usr/bin/env bash
# Layering Virtual Try-On -- one-shot setup for a RunPod GPU pod.
#
#   bash runpod_setup.sh
#
# A decently sized pod runs the *full bf16* model rather than a quantized build,
# which is the whole reason to pay for one. The script picks the model by VRAM
# and puts everything under /workspace so a pod stop/start does not re-download
# 57.7 GB.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/Layering-Virtual-Try-On"
REPO_URL="${REPO_URL:-https://github.com/Mohamed-Kudratov/Lookzi.git}"

# Persist the HF cache on the network volume, not the container's ephemeral disk.
# RunPod images already point HF_HOME at /workspace/.cache/huggingface, so defer
# to that when it is set.
export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
mkdir -p "$HF_HOME"

# Both of RunPod's preset download accelerators make things worse here, measured
# on an A100 pod against the official Qwen repo:
#   xet          stalls after ~10 GB -- it writes deduplicated chunks, and the
#                small random IO that implies is pathological on RunPod's MFS
#                network volume (it also stores every blob twice while doing it)
#   hf_transfer  raises RuntimeError mid-download and hides the real error
# Plain HTTP sustained ~190 MB/s. Turn both off.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=============================================="
echo " Layering VTON -- RunPod setup"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
VRAM_GB=$((VRAM_MB / 1024))
echo "Detected ${VRAM_GB} GB of VRAM"

# `df` on a RunPod network volume reports the whole MFS cluster (hundreds of TB),
# not your quota, so it cannot be used to check for space. Probe the quota by
# writing until it refuses.
echo "Checking the volume quota (df is useless here -- it shows the whole cluster)"
dd if=/dev/zero of="$WORKSPACE/.quota_probe" bs=1M count=1 status=none 2>/dev/null || {
    echo "ERROR: cannot write to $WORKSPACE at all -- the volume is already full."
    exit 1
}
rm -f "$WORKSPACE/.quota_probe"
USED_GB=$(du -sm "$WORKSPACE" 2>/dev/null | cut -f1)
USED_GB=$((USED_GB / 1024))
echo "  $WORKSPACE currently uses ${USED_GB} GB"

# 57.7 GB of bf16 weights: transformer 40.9 + text encoder 16.6. Both stay
# resident above ~70 GB. Between 40 and 70 the pipeline loads them one at a
# time. Below that the 4-bit build is the only thing that fits.
if [ "$VRAM_GB" -ge 70 ]; then
    MODEL_PATH="Qwen/Qwen-Image-Edit-2509"
    echo "-> full bf16 model (best quality, both components resident)"
elif [ "$VRAM_GB" -ge 40 ]; then
    MODEL_PATH="Qwen/Qwen-Image-Edit-2509"
    echo "-> full bf16 model (sequential loading; slower per run)"
else
    MODEL_PATH="ovedrive/Qwen-Image-Edit-2509-4bit"
    echo "-> 4-bit model (under 40 GB of VRAM; quantisation is lossy)"
fi

echo
echo "--- 1/5  system packages ---"
apt-get update -qq
apt-get install -y -qq git wget libgl1 libglib2.0-0 >/dev/null

echo "--- 2/5  repository ---"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only || echo "(pull skipped)"
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "--- 3/5  python packages ---"
pip install -q --upgrade pip
pip install -q \
    "transformers==4.56.0" "tokenizers==0.22.0" "peft==0.17.1" \
    "accelerate==1.10.1" "huggingface-hub==0.34.4" "safetensors==0.6.2" \
    "bitsandbytes>=0.46.0" "hf-xet==1.1.9" hf_transfer \
    "omegaconf==2.3.0" "onnxruntime==1.22.0" "opencv-python==4.11.0.86" \
    "sentencepiece==0.2.1" "protobuf==6.32.0" "timm==1.0.16" \
    "pillow==11.0.0" "matplotlib==3.10.6" einops gradio

# easy-dwpose pins dependencies that conflict with the above, so it goes in with
# --no-deps. matplotlib above is not optional: draw_handpose imports it, and
# without it pose extraction dies with ModuleNotFoundError.
pip install -q easy-dwpose==1.0.2 --no-deps

echo "--- 4/5  bundled diffusers fork ---"
# `diffusers/` here is the fork's *repository*; the package is at
# diffusers/src/diffusers. Python searches the working directory first, so that
# directory shadows any install and `import diffusers` yields an empty PEP 420
# namespace package. An editable install does not help -- the directory still
# shadows it -- and cannot survive the rename, because it records the absolute
# source path. Install normally, then move the tree aside.
if [ -d "$REPO_DIR/diffusers" ]; then
    pip uninstall -q -y diffusers || true
    pip install -q "$REPO_DIR/diffusers"
    mv "$REPO_DIR/diffusers" "$REPO_DIR/diffusers_src"
    echo "    moved ./diffusers -> ./diffusers_src"
elif [ -d "$REPO_DIR/diffusers_src" ]; then
    # A previous run already moved it. Stopping a pod wipes the container disk,
    # so the install is gone while the source survives on the volume -- without
    # this branch a restarted pod silently ends up with no diffusers at all.
    echo "    source already moved; reinstalling from ./diffusers_src"
    pip install -q "$REPO_DIR/diffusers_src"
else
    echo "ERROR: neither ./diffusers nor ./diffusers_src exists."
    exit 1
fi

python - <<'PY'
import os, diffusers
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
loc = getattr(diffusers, "__file__", None)
assert loc is not None, "diffusers is still an empty namespace package"
assert not os.path.abspath(loc).startswith(os.path.join(os.getcwd(), "diffusers") + os.sep)
print(f"    diffusers {diffusers.__version__} from {loc}")
PY

echo "--- 5/5  weights ---"
# DWPose writes to ./checkpoints relative to the working directory.
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
    print("   ", hf_hub_download("RedHash/DWPose", f, local_dir="./checkpoints"))
PY

echo "    downloading $MODEL_PATH (this is the long part)"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_PATH", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("   ", p)
PY

cat > "$REPO_DIR/run.sh" <<EOF
#!/usr/bin/env bash
cd "$REPO_DIR"
export HF_HOME="$HF_HOME"
export MODEL_PATH="$MODEL_PATH"
export GRADIO_SHARE=0     # RunPod's own HTTP proxy is used instead
exec python app.py
EOF
chmod +x "$REPO_DIR/run.sh"

echo
echo "=============================================="
echo " Done."
echo
echo "   Model:  $MODEL_PATH"
echo "   Cache:  $HF_HOME"
echo
echo "   Start:  bash $REPO_DIR/run.sh"
echo
echo "   Then open  https://<YOUR_POD_ID>-7860.proxy.runpod.net"
echo "   (expose HTTP port 7860 in the pod's template)"
echo "=============================================="

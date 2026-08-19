#!/usr/bin/env bash
# Build a separate environment for Z-Image-Turbo.
#
#   bash elements/setup_zimage.sh
#   /workspace/zimage-venv/bin/python elements/generate.py --limit 5
#
# Why separate: Z-Image-Turbo needs diffusers from source, and the try-on
# pipeline is pinned to the bundled 0.36.0.dev0 fork. Installing either one
# replaces the other, so they cannot share an interpreter. In production these
# are two different workers anyway.
#
# The venv goes on the CONTAINER disk, not /workspace. pip writes thousands of
# small files and RunPod's network volume fails on that pattern with
# `OSError: [Errno 5] Input/output error` -- see TROUBLESHOOTING.md. The cost is
# that a pod restart loses it, which is a minute to rebuild against a cached
# wheel set.

set -euo pipefail

VENV="${ZIMAGE_VENV:-/opt/zimage-venv}"
WORKSPACE="${WORKSPACE:-/workspace}"

# Wheels cached on the volume: large files, sequential reads, which is what a
# network filesystem is good at.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/.pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=============================================="
echo " Z-Image-Turbo environment"
echo "   venv:  $VENV"
echo "   cache: $HF_HOME"
echo "=============================================="

if [ ! -x "$VENV/bin/python" ]; then
    # --system-site-packages keeps torch from the pod image rather than pulling
    # several GB of CUDA wheels a second time.
    python3 -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/pip" install -q --upgrade pip

if ! "$VENV/bin/python" -c "from diffusers import ZImagePipeline" 2>/dev/null; then
    echo "--- installing diffusers from source (needed for ZImagePipeline) ---"
    "$VENV/bin/pip" install -q "git+https://github.com/huggingface/diffusers"
    "$VENV/bin/pip" install -q transformers accelerate safetensors sentencepiece protobuf pillow
fi

# --system-site-packages lets the try-on stack leak in, and its pinned
# huggingface-hub==0.34.4 is too old for diffusers-from-source:
#   cannot import name 'get_cached_repo_tree' from 'huggingface_hub'
# Installing a current hub into the venv shadows the system one here without
# touching the pinned version the try-on pipeline needs.
"$VENV/bin/pip" install -q --upgrade huggingface_hub

# curate.py and view_sweep.py belong to this workflow too, and the system
# interpreter only has gradio after start.sh has run -- which it need not have,
# since nothing here touches the try-on pipeline.
"$VENV/bin/pip" install -q gradio

"$VENV/bin/python" - <<'PY'
import torch, diffusers
from diffusers import ZImagePipeline
print(f"  torch      {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  diffusers  {diffusers.__version__}")
print("  ZImagePipeline import OK")
PY

echo
echo "Ready. Generate with:"
echo "  $VENV/bin/python elements/generate.py --limit 5"
echo
echo "The try-on pipeline is unaffected -- it still uses the system interpreter"
echo "and the pinned fork."

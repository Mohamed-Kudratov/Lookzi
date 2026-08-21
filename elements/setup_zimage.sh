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
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=============================================="
echo " Z-Image-Turbo environment"
echo "   venv:  $VENV"
echo "   cache: $HF_HOME"
echo "=============================================="

if [ ! -x "$VENV/bin/python" ]; then
    # Fully isolated -- deliberately NOT --system-site-packages.
    #
    # Sharing the system packages looks like a saving, since torch is several
    # GB, but it drags the try-on stack's pins in with it and the two stacks
    # disagree. transformers caps huggingface_hub below 1.0; diffusers-from-
    # source needs one new enough for get_cached_repo_tree; the inherited
    # 0.34.4 is below both. Every attempt to satisfy one broke the other.
    #
    # Isolation costs ~3 GB of container disk once and ends the argument.
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install -q --upgrade pip

if ! "$VENV/bin/python" -c "from diffusers import ZImagePipeline" 2>/dev/null; then
    echo "--- installing torch (isolated venv, ~3 GB, one time) ---"
    "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu128

    echo "--- installing diffusers from source (needed for ZImagePipeline) ---"
    "$VENV/bin/pip" install -q "git+https://github.com/huggingface/diffusers"
    "$VENV/bin/pip" install -q transformers accelerate safetensors sentencepiece protobuf pillow gradio
fi

# diffusers-from-source tracks huggingface_hub 1.x (it wants is_offline_mode),
# while an older transformers caps hub below 1.0. Upgrading them together lets
# pip find the pair that agrees -- upgrading either alone picks a hub that
# breaks the other, which is what the earlier attempts kept doing.
"$VENV/bin/pip" install -q --upgrade transformers huggingface_hub

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
echo "--- pre-fetching Z-Image-Turbo ---"
"$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Tongyi-MAI/Z-Image-Turbo',
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
print('  cached at', p)
"

echo
echo "Ready. Generate with:"
echo "  $VENV/bin/python elements/generate.py --limit 5"
echo
echo "The try-on pipeline is unaffected -- it still uses the system interpreter"
echo "and the pinned fork."

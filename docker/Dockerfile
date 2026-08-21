# Lookzi pod image — everything installed, nothing to set up at run time.
#
# A stock RunPod pod needs ~10 minutes of pip installs after every start,
# because stopping a pod wipes the container disk. That cost is paid on every
# restart and every forced migration when a GPU is reclaimed. Freezing it into
# an image turns it into a one-time build.
#
# Two Python environments live here, and they must stay apart:
#
#   system python      the try-on stack -- the bundled diffusers 0.36.0.dev0
#                      fork, transformers 4.56, huggingface-hub 0.34.4
#   /opt/zimage-venv   Z-Image-Turbo -- diffusers from source, which needs a
#                      newer transformers and hub than the try-on stack allows
#
# They genuinely cannot share an interpreter: transformers 4.56 caps
# huggingface_hub below 1.0 while diffusers-from-source wants 1.x. Every
# attempt to find one satisfying version failed. Isolation settles it.

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Both of RunPod's preset download accelerators break large HF downloads:
    # xet stalls after ~10 GB, hf_transfer raises mid-download. Plain HTTP
    # sustained ~190 MB/s. See TROUBLESHOOTING.md.
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # One cache for both stacks, on the volume. Putting Z-Image on the container
    # disk was tried and filled it: venv plus 12 GB of weights does not fit in
    # 30 GB alongside the system packages. Loading it from the volume takes
    # ~200s, which is fine -- the half-hour stall that prompted the idea was
    # low_cpu_mem_usage=False on the try-on model, since removed.
    HF_HOME=/workspace/.cache/huggingface \
    # Loading a 40 GB transformer shard by shard fragments the allocator enough
    # to OOM with GB still free.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ---------------------------------------------------------------------------
# The repo, and the try-on stack on the system interpreter
# ---------------------------------------------------------------------------
ARG REPO_URL=https://github.com/Mohamed-Kudratov/Lookzi.git
ARG REPO_REF=main
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /opt/lookzi

WORKDIR /opt/lookzi

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir easy-dwpose==1.0.2 --no-deps

# The bundled fork: ./diffusers is the fork's *repository*, not its package --
# that lives at diffusers/src/diffusers. Python searches the working directory
# first, so leaving the directory in place shadows the install and yields an
# empty namespace package. Install normally, then move the tree aside.
RUN pip uninstall -q -y diffusers || true \
    && pip install --no-cache-dir ./diffusers \
    && rm -rf ./diffusers_src \
    && mv ./diffusers ./diffusers_src \
    && python -c "\
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel; \
import diffusers, os; \
assert diffusers.__file__ and 'site-packages' in diffusers.__file__ or 'dist-packages' in diffusers.__file__; \
print('try-on diffusers', diffusers.__version__)"

# DWPose ONNX (~220 MB). easy-dwpose fetches these to ./checkpoints relative to
# the working directory, so baking them avoids a download on every pod.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
[hf_hub_download('RedHash/DWPose', f, local_dir='./checkpoints') \
 for f in ('yolox_l.onnx','dw-ll_ucoco_384.onnx')]"

# ---------------------------------------------------------------------------
# Z-Image-Turbo, fully isolated
# ---------------------------------------------------------------------------
RUN python -m venv /opt/zimage-venv \
    && /opt/zimage-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/zimage-venv/bin/pip install --no-cache-dir torch \
         --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/zimage-venv/bin/pip install --no-cache-dir \
         "git+https://github.com/huggingface/diffusers" \
    && /opt/zimage-venv/bin/pip install --no-cache-dir \
         accelerate safetensors sentencepiece protobuf pillow gradio \
    # Upgraded together on purpose: diffusers-from-source needs hub 1.x while an
    # older transformers caps it below 1.0, so upgrading either alone picks a
    # version that breaks the other.
    && /opt/zimage-venv/bin/pip install --no-cache-dir --upgrade \
         transformers huggingface_hub \
    && /opt/zimage-venv/bin/python -c "\
from diffusers import ZImagePipeline; import diffusers, transformers, huggingface_hub as h; \
print('z-image diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| hub', h.__version__)"

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /opt/lookzi
CMD ["/opt/entrypoint.sh"]

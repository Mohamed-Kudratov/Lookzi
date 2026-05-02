# Lookzi

Lookzi is a virtual try-on demo for testing garments on person images with a Gradio interface.

## What It Does

- Upload a person image.
- Upload or select a garment image.
- Choose the target area: upper, lower, or full outfit.
- Generate a try-on preview.
- Run locally, on Google Colab, or on a CUDA GPU machine.

## Quick Start On Google Colab

Enable GPU first:

```bash
!nvidia-smi
```

Clone the project:

```bash
%cd /content
!rm -rf /content/Lookzi
!git clone https://github.com/Mohamed-Kudratov/Lookzi.git /content/Lookzi
%cd /content/Lookzi
```

Install dependencies:

```bash
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt
```

Download model files:

```python
from huggingface_hub import snapshot_download

snapshot_download("zhengchong/CatVTON", local_dir="hf_models/lookzi-vton")
snapshot_download("booksforcharlie/stable-diffusion-inpainting", local_dir="hf_models/stable-diffusion-inpainting")
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir="hf_models/sd-vae-ft-mse")
```

Run a fast GPU test:

```bash
!python app.py \
  --device cuda \
  --mixed_precision fp16 \
  --width 384 \
  --height 512 \
  --share \
  --server_name 0.0.0.0
```

Run full quality:

```bash
!python app.py \
  --device cuda \
  --mixed_precision fp16 \
  --width 768 \
  --height 1024 \
  --share \
  --server_name 0.0.0.0
```

## Local Windows Run

Create and activate a Python 3.9 virtual environment, then install:

```bash
pip install -r requirements.txt
```

Download the model folders into:

```text
hf_models/lookzi-vton
hf_models/stable-diffusion-inpainting
hf_models/sd-vae-ft-mse
```

Run:

```bash
python app.py --device auto --mixed_precision fp16
```

If no CUDA GPU is available, the app automatically falls back to CPU. CPU mode works, but it is very slow and is only useful for basic checks.

## Recommended Settings

- Fast smoke test: `384x512`, 10 steps.
- Better quality: `768x1024`, 30-50 steps.
- CFG: `2.5`.
- Use `result only` when you only need the final output.

## Model Storage

Large model files are intentionally not committed to Git. Keep them in `hf_models/`, Google Drive, or another private storage location.

## License And Credits

Lookzi includes code and model integration based on CatVTON by Zheng Chong and contributors. The upstream materials are released under Creative Commons BY-NC-SA 4.0 for non-commercial use. Keep the original license terms when redistributing or modifying this project.

# Lookzi Colab GPU Setup

Use this notebook flow when running Lookzi on Google Colab with a GPU runtime.

## 1. Enable GPU

Runtime -> Change runtime type -> Hardware accelerator -> GPU

Verify:

```bash
!nvidia-smi
```

## 2. Clone The Project

```bash
%cd /content
!rm -rf /content/Lookzi
!git clone https://github.com/Mohamed-Kudratov/Lookzi.git /content/Lookzi
%cd /content/Lookzi
```

## 3. Install Dependencies

```bash
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt
```

## 4. Download Model Files

```python
from huggingface_hub import snapshot_download

snapshot_download("zhengchong/CatVTON", local_dir="hf_models/lookzi-vton")
snapshot_download("booksforcharlie/stable-diffusion-inpainting", local_dir="hf_models/stable-diffusion-inpainting")
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir="hf_models/sd-vae-ft-mse")
```

## 5. Fast GPU Test

```bash
!python app.py \
  --device cuda \
  --mixed_precision fp16 \
  --width 384 \
  --height 512 \
  --share \
  --server_name 0.0.0.0
```

In the UI, set inference steps to 10 for the first test.

## 6. Full Quality

```bash
!python app.py \
  --device cuda \
  --mixed_precision fp16 \
  --width 768 \
  --height 1024 \
  --share \
  --server_name 0.0.0.0
```

Recommended UI settings: 30-50 inference steps, CFG 2.5, and `Show Type = result only`.

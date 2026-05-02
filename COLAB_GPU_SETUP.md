# CatVTON Colab GPU Setup

Use this when running Lookzi/CatVTON on Google Colab with a GPU runtime.

## 1. Enable GPU

In Colab:

Runtime -> Change runtime type -> Hardware accelerator -> GPU

Then verify:

```bash
!nvidia-smi
```

## 2. Clone repo

```bash
!git clone https://github.com/Mohamed-Kudratov/Lookzi.git
%cd Lookzi
```

## 3. Install CUDA PyTorch and dependencies

```bash
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt
```

Restart the Colab runtime if it asks you to.

## 4. Download model weights

```python
from huggingface_hub import snapshot_download

snapshot_download("zhengchong/CatVTON", local_dir="hf_models/CatVTON")
snapshot_download("booksforcharlie/stable-diffusion-inpainting", local_dir="hf_models/stable-diffusion-inpainting")
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir="hf_models/sd-vae-ft-mse")
```

## 5. Run Gradio demo with GPU

```bash
!python app.py \
  --device cuda \
  --mixed_precision fp16 \
  --width 768 \
  --height 1024 \
  --share \
  --server_name 0.0.0.0
```

Open the public Gradio link printed by the command.

## Faster smoke test

For a quick GPU test:

```bash
!python app.py \
  --device cuda \
  --mixed_precision fp16 \
  --width 384 \
  --height 512 \
  --share \
  --server_name 0.0.0.0
```

In the UI, use 10 inference steps for the first test.

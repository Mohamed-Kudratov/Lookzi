# Lookzi Colab GPU Setup

Bu setup modellarni **Google Drive**'ga bir marta saqlaydi.
Keyingi sessionlarda Drive'dan yuklanadi — HuggingFace'dan qayta yuklanmaydi.

---

## 1. GPU ni yoqish

Runtime → Change runtime type → T4 GPU → Save

```python
!nvidia-smi
```

---

## 2. Google Drive mount qilish

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 3. Repo clone

```python
%cd /content
!rm -rf /content/Lookzi
!git clone https://github.com/Mohamed-Kudratov/Lookzi.git /content/Lookzi
%cd /content/Lookzi
```

---

## 4. Dependencies o'rnatish

```python
# Eski build tizimi uchun
!pip install -q "setuptools<70" wheel

# fvcore --no-deps bilan: o'zi torch tortib olmasin
!pip install -q fvcore==0.1.5.post20221221 --no-deps
!pip install -q iopath portalocker yacs termcolor tabulate tqdm

# pycocotools alohida
!pip install -q pycocotools --no-build-isolation

# Qolgan asosiy kutubxonalar
!pip install -q \
  accelerate==1.10.1 \
  diffusers==0.31.0 \
  transformers==4.46.3 \
  peft==0.17.1 \
  huggingface_hub==0.36.0 \
  gradio==4.41.0 \
  gradio-client==1.3.0 \
  opencv-python==4.10.0.84 \
  scikit-image==0.24.0 \
  omegaconf==2.3.0 \
  cloudpickle==3.0.0 \
  av==12.3.0 \
  fastapi==0.112.2 \
  starlette==0.38.2 \
  pydantic==2.8.2 \
  typer==0.12.3

# Agar biror paket torch'ni eski versiyaga tushirib qo'ygan bo'lsa — qayta tiklash
import torch
if not torch.__version__.startswith("2.5"):
    import subprocess
    subprocess.run(["pip", "install", "-q", "--upgrade",
                    "torch", "torchvision",
                    "--index-url", "https://download.pytorch.org/whl/cu121"])
    print("torch qayta o'rnatildi — runtime restart kerak!")
else:
    print(f"torch OK: {torch.__version__}")
```

---

## 5. Modellarni Drive'dan yuklab olish (bir marta)

```python
import os, shutil
from huggingface_hub import snapshot_download

DRIVE_MODELS = "/content/drive/MyDrive/Lookzi/hf_models"
LOCAL_MODELS  = "/content/Lookzi/hf_models"

models = {
    "lookzi-vton":                    "zhengchong/CatVTON",
    "stable-diffusion-inpainting":    "booksforcharlie/stable-diffusion-inpainting",
    "sd-vae-ft-mse":                  "stabilityai/sd-vae-ft-mse",
}

os.makedirs(DRIVE_MODELS, exist_ok=True)
os.makedirs(LOCAL_MODELS, exist_ok=True)

for folder, repo_id in models.items():
    drive_path = os.path.join(DRIVE_MODELS, folder)
    local_path = os.path.join(LOCAL_MODELS, folder)

    if not os.path.exists(drive_path):
        print(f"⬇ Birinchi marta yuklanmoqda: {repo_id} → Drive")
        snapshot_download(repo_id, local_dir=drive_path)
    else:
        print(f"✓ Drive'da mavjud: {folder}")

    # Drive → /content symlink (tez kirish uchun)
    if os.path.islink(local_path):
        os.unlink(local_path)
    if os.path.exists(local_path) and not os.path.islink(local_path):
        shutil.rmtree(local_path)
    os.symlink(drive_path, local_path)
    print(f"  → symlink: {local_path}")

print("\nBarcha modellar tayyor.")
```

---

## 6. Appni ishga tushirish

```python
# Absolute path ishlatamiz — symlink/relative path muammolaridan xoli
!python app.py \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --base_model_path /content/Lookzi/hf_models/stable-diffusion-inpainting \
  --vae_model_path /content/Lookzi/hf_models/sd-vae-ft-mse \
  --device cuda \
  --mixed_precision fp16 \
  --width 768 \
  --height 1024 \
  --share \
  --server_name 0.0.0.0
```

Tavsiya: Steps=50, CFG=2.5, Show Type=result only

---

## Muhim eslatmalar

| | Birinchi sessiya | Keyingi sessionlar |
|---|---|---|
| Model yuklash | 15-20 daqiqa (Drive'ga saqlaydi) | 1-2 daqiqa (Drive'dan o'qiydi) |
| pip install | 2-3 daqiqa | 2-3 daqiqa |
| Render (50 steps) | ~130-150s (PyTorch SDPA bilan) | ~130-150s |

Drive'dagi model papkasi: `MyDrive/Lookzi/hf_models/`
Bu papkani o'chirmang — keyingi barcha sessionlarda ishlatiladi.

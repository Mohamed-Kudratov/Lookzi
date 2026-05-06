# Lookzi Colab GPU Setup

Bu setup modellarni Google Drive'ga bir marta saqlaydi. Keyingi sessionlarda modellar Drive'dan ulanadi va HuggingFace'dan qayta yuklanmaydi.

---

## 1. GPU ni yoqish

Colab menu:

```text
Runtime -> Change runtime type -> T4 GPU -> Save
```

Tekshirish:

```python
!nvidia-smi
```

---

## 2. Google Drive mount qilish

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

## 3. Repo clone yoki update

Birinchi marta:

```python
%cd /content
!rm -rf /content/Lookzi
!git clone https://github.com/Mohamed-Kudratov/Lookzi.git /content/Lookzi
%cd /content/Lookzi
```

Keyingi sessionlarda repo allaqachon bo'lsa:

```python
%cd /content/Lookzi
!git pull origin main
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

# Torch versiyasini oxirida qayta tiklash
!pip install -q torch==2.5.1+cu121 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

import torch
print(f"torch: {torch.__version__}")
assert torch.__version__.startswith("2.5"), "XATO: torch versiyasi noto'g'ri, runtime restart qiling!"
print("OK - keyingi cellga o'ting")
```

---

## 5. Modellarni Drive'dan ulash

```python
import os, shutil
from huggingface_hub import snapshot_download

DRIVE_MODELS = "/content/drive/MyDrive/Lookzi/hf_models"
LOCAL_MODELS = "/content/Lookzi/hf_models"

models = {
    "lookzi-vton": "zhengchong/CatVTON",
    "stable-diffusion-inpainting": "booksforcharlie/stable-diffusion-inpainting",
    "sd-vae-ft-mse": "stabilityai/sd-vae-ft-mse",
}

os.makedirs(DRIVE_MODELS, exist_ok=True)
os.makedirs(LOCAL_MODELS, exist_ok=True)

for folder, repo_id in models.items():
    drive_path = os.path.join(DRIVE_MODELS, folder)
    local_path = os.path.join(LOCAL_MODELS, folder)

    if not os.path.exists(drive_path):
        print(f"Downloading first time: {repo_id} -> Drive")
        snapshot_download(repo_id, local_dir=drive_path)
    else:
        print(f"Drive model exists: {folder}")

    if os.path.islink(local_path):
        os.unlink(local_path)
    if os.path.exists(local_path) and not os.path.islink(local_path):
        shutil.rmtree(local_path)
    os.symlink(drive_path, local_path)
    print(f"  symlink: {local_path}")

print("\nBarcha modellar tayyor.")
```

---

## 6. Appni ishga tushirish

```python
# Eski app jarayonini o'ldirish
!fuser -k 7860/tcp 2>/dev/null; sleep 2

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

Tavsiya:

```text
Steps = 50
CFG = 2.5
Show Type = result only
```

---

## 7. Catalog Benchmark + Review Gallery

Bu hozirgi asosiy debugging workflow. U demo do'kon garment/person pairlarini avtomatik test qiladi, mask/outputlarni Drive'ga saqlaydi va HTML review gallery chiqaradi.

### Tez mask/style test

```python
!python eval_benchmark.py \
  --mode fast \
  --pairs benchmark/catalog_pairs.json \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

### To'liq inference test

```python
!python eval_benchmark.py \
  --mode full \
  --engine catvton \
  --pairs benchmark/catalog_pairs.json \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --base_model_path /content/Lookzi/hf_models/stable-diffusion-inpainting \
  --vae_model_path /content/Lookzi/hf_models/sd-vae-ft-mse \
  --device cuda \
  --mixed_precision fp16 \
  --width 768 \
  --height 1024 \
  --num_inference_steps 50 \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

### Engine baseline test

Bu test kiyim kiydirmaydi: person rasmini qaytaradi. Maqsad - benchmark gate uchun nazorat chizig'i. Har qanday yangi engine bundan aniq yaxshiroq chiqishi kerak.

```python
!python eval_benchmark.py \
  --mode full \
  --engine identity_baseline \
  --pairs benchmark/catalog_pairs.json \
  --device cuda \
  --mixed_precision fp16 \
  --width 768 \
  --height 1024 \
  --num_inference_steps 50 \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

Run tugagach terminalda shunga o'xshash path chiqadi:

```text
[Eval] Review report: /content/drive/MyDrive/Lookzi/eval_logs/outputs/20260505_123456/review_report.html
```

O'sha HTML fayl self-contained bo'ladi: rasmlar HTML ichiga joylanadi. Uni Colab ichida ko'rish uchun:

Muhim: agar report eski commit bilan yaratilgan bo'lsa, rasmlar ko'rinmaydi. Avval `git pull origin main` qiling va benchmarkni qayta run qiling.

```python
from IPython.display import HTML, display

report_path = "/content/drive/MyDrive/Lookzi/eval_logs/outputs/20260505_123456/review_report.html"
with open(report_path, "r", encoding="utf-8") as f:
    display(HTML(f.read()))
```

Yoki helper bilan:

```python
%run display_report.py /content/drive/MyDrive/Lookzi/eval_logs/outputs/20260505_123456/review_report.html
```

HTML report har test uchun quyidagilarni ko'rsatadi:

- person image
- garment image
- mask preview
- output image
- engine name
- category
- coverage
- detected style
- error/debug

Human review uchun har card ostida quyidagi statuslardan birini qo'lda belgilash mumkin:

```text
GOOD / OK / BAD / MODEL_FAIL / MASK_FAIL
```

Review buttonlari browser localStorage'ga saqlanadi. Ish tugagach `Download review JSON` tugmasini bosing va JSON faylni saqlang.

Download qilingan review JSON avtomatik Drive logga yozilmaydi. JSONni Colab'ga upload qilib, keyin merge qiling:

```python
from google.colab import files
uploaded = files.upload()
review_json = next(iter(uploaded.keys()))

!python apply_human_review.py \
  --review_json "$review_json" \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json
```

Shundan keyin `results.json` ichidagi oxirgi run `human_rating`, `failure_reason`, `review_status` maydonlari bilan yangilanadi.

---

## 8. Eski Benchmark

Oldingi curated mask benchmark hali ham ishlaydi:

```python
!python eval_benchmark.py \
  --mode full \
  --pairs benchmark/pairs.json \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --base_model_path /content/Lookzi/hf_models/stable-diffusion-inpainting \
  --vae_model_path /content/Lookzi/hf_models/sd-vae-ft-mse \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

---

## Muhim eslatmalar

| | Birinchi sessiya | Keyingi sessionlar |
|---|---:|---:|
| Model yuklash | 15-20 daqiqa | 1-2 daqiqa |
| pip install | 2-3 daqiqa | 2-3 daqiqa |
| Render 50 steps | ~130-150s | ~130-150s |

Drive'dagi model papkasi:

```text
MyDrive/Lookzi/hf_models/
```

Bu papkani o'chirmang.

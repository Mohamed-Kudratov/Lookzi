# Lookzi — Agent Progress Notes

> Bu fayl kelajakdagi agentlar uchun. Har bir muhim o'zgarishdan keyin yangilab boring.
> Tartibi: yangi o'zgarishlar TEPAGA yoziladi.

---

## [2026-05-04 v2] Lower garment mask fix

### Muammo
Lower garment test natijasi: pink pants o'rniga ko'k/jigarrang shim chiqardi.
Debug info: mask coverage = 6.40% (erkak) va 13.27% (ayol). Normal: 30-40%.

### Sabab
`hands_protect_area` ichida DensePose feet (indeks 5,6) bor.
Bu area dilate qilinib SCHP Left-leg/Right-leg bilan kesishadi →
lower leg zona protected → `lower_body_area` ning katta qismi yo'q bo'ladi →
mask 6% ga tushadi → model condition garmentni ignore qilib o'zi "shim to'qiydi".

### Fix (model/cloth_masker.py, lower branch, ~278-287-qator)
```python
# Shoes (DensePose 5,6) are OUTSIDE lower_body_area → stay protected
local_strong_protect = strong_protect_area & ~lower_body_area
# hull_mask: closes gap between legs (inner thigh zone)
lower_hull = hull_mask((lower_body_area | strong_mask_area).astype(np.uint8) * 255).astype(bool)
allowed_area = lower_hull | lower_body_area | strong_mask_area
protect_area = local_strong_protect | background_area | Left-arm/Right-arm/Face
```

### Hali test qilinmagan
Bu fix GitHub'ga push qilindi lekin Colab'da test qilinmadi (torch versiyasi muammo tufayli).
**Keyingi agentga: Colab'da lower garment qayta test qilib natijani shu yerga yozing.**

---

## [2026-05-04] Mask fixes + Colab stabilization

### Nima qilindi

#### 1. mask muammolari (model/cloth_masker.py)
Uch xil kiyim turida mask xato ishlardi:

**Fix 1 — Sleeveless (yengsiz kiyim qo'l oladi)**
- Sabab: `upper_arm_area` `allowed_area`ga qo'shilardi
- Yechim: `garment_style == 'sleeveless'` bo'lsa `upper_arm_area` → `protect_area`ga ko'chirish
- Fayl: `cloth_agnostic_mask()`, `upper/inner/outer` branch, ~272-274-qator

**Fix 2 — Sleeved forearm cut (cardigan bilak kesib tashlaydi)**
- Sabab: `forearm_area` hech qachon `allowed_area`ga kirmagan
- Yechim: `garment_style == 'sleeved'` bo'lsa `forearm_area` → `allowed_area`, `local_arm_protect` = faqat qo'llar
- Fayl: `cloth_agnostic_mask()`, ~265-270-qator

**Fix 3 — Overall/dress hallucinated socks**
- Sabab: qisqa dress uchun ham `lower_leg_area` maskalanardi
- Yechim: `not covers_lower_legs` bo'lsa `lower_leg_area` → `protect_area`
- Fayl: `cloth_agnostic_mask()`, `overall` branch, ~294-295-qator

#### 2. Garment style detection (app.py)
`infer_garment_style()` endi 3 qiymat qaytaradi: `(style_str, covers_lower_legs: bool, debug_str)`
- `covers_lower_legs`: kiyim balandligi `(h / 256.0) > 0.65` → shimga/uzun ko'ylakka True
- `lower_side_mass`: bilak zonasidagi (y 38-65%) mato zichligi — uzun yeng aniqlash uchun
- `side_mass > 0.55 AND lower_side_mass > 0.42` → `sleeved`

#### 3. xformers / SkipAttnProcessor crash (model/pipeline.py)
- **Kritik bug**: `set_attn_processor(AttnProcessor2_0())` → `SkipAttnProcessor` o'chirilardi → shape crash (49152×320 vs 768×320)
- Yechim: SDPA fallback BUTUNLAY o'chirildi
- xformers faqat mavjud bo'lsa: `try: enable_xformers_memory_efficient_attention() except Exception: pass`
- T4 GPU (CUDA 7.5) xformers'ni qo'llab-quvvatlamaydi (kerak 8.0+) → default attention ishlaydi

#### 4. Colab notebook stabilization

**Muammo 1 — Drive mount ValueError**
- `drive.mount('/content/drive')` → "Mountpoint must not already contain files"
- Fix: `force_remount=True` parametri

**Muammo 2 — torch versiyasi tushib ketadi**
- `iopath` (fvcore dependencysi) eski torchni tortib oladi
- Fix: `torch==2.5.1+cu121` pip install'ning ENG OXIRIDA pin qilingan
- Agar hali ham noto'g'ri bo'lsa: Runtime restart → faqat cell 2,3,5,6 run qilish

**Muammo 3 — Port 7860 band**
- Oldingi app processi o'lmagan
- Fix: `!fuser -k 7860/tcp 2>/dev/null; sleep 2` cell 6 boshida

**Muammo 4 — 401 Unauthorized (model yuklashda)**
- `os.path.exists("hf_models/...")` → symlink orqali False qaytaradi → `snapshot_download(repo_id="hf_models/...")` → HF'ga noto'g'ri so'rov
- Fix: `--resume_path` uchun ABSOLUTE path (`/content/Lookzi/hf_models/lookzi-vton`)

#### 5. UI tozalik
- Footer yashirildi: `css="footer{display:none !important}"`
- API button yashirildi: `show_api=False`
- Null image check: `submit_function` boshida `person_image is None or cloth_image is None` tekshiruvi

### Natijalar (test 2026-05-04)
- Render vaqti: ~130-150s (steps=50, fp16, T4) — torch 2.5.1 to'g'ri bo'lganda
- Mask fixlari: kod yozildi, lekin to'liq test hali QILINMADI (qarang: Pending Tasks)

---

## Pending Tasks (hali tugallanmagan)

1. **Mask test** — barcha 4 kiyim turini sinab ko'rish:
   - `upper sleeveless` → qo'l ko'rinib turishi kerak
   - `upper sleeved` → bilak ham maskalanishi kerak (cardigan)
   - `overall` qisqa → oyoq pastki qismi ko'rinib turishi kerak
   - `lower` → o'rta sifat (model limitatsiyasi bo'lishi mumkin)

2. **Lower garment sifati** — "daxshat" deb ta'riflangan, lekin bu torch versiyasi noto'g'ri bo'lganda test qilingan. To'g'ri torch bilan qayta sinab ko'rish kerak.

3. **Server migration** — hozir Colab T4 + ngrok. Kelajakda HuggingFace Spaces yoki VPS.

---

## Muhim arxitektura ma'lumotlari

### Modellar
| Model | Manzil | Vazifasi |
|-------|--------|----------|
| CatVTON | `hf_models/lookzi-vton` | Asosiy VTON adapter |
| SD Inpainting | `hf_models/stable-diffusion-inpainting` | Base diffusion model |
| sd-vae-ft-mse | `hf_models/sd-vae-ft-mse` | VAE (encoder/decoder) |
| DensePose | `hf_models/lookzi-vton/DensePose` | Tana qismlari segmentatsiya |
| SCHP ATR | `hf_models/lookzi-vton/SCHP/exp-schp-201908301523-atr.pth` | Kiyim segmentatsiya |
| SCHP LIP | `hf_models/lookzi-vton/SCHP/exp-schp-201908261155-lip.pth` | Kiyim segmentatsiya |

### Asosiy fayllar
| Fayl | Vazifasi |
|------|----------|
| `app.py` | Gradio UI + `infer_garment_style()` + `submit_function()` |
| `model/cloth_masker.py` | `AutoMasker` + `cloth_agnostic_mask()` — mask mantiq |
| `model/pipeline.py` | `LookziPipeline` — diffusion inference |
| `model/attn_processor.py` | `SkipAttnProcessor` — cross-attention o'tkazib yuborish (MUHIM: o'chirma) |
| `model/SCHP/` | Human parsing model wrapper |
| `model/DensePose/` | DensePose wrapper |
| `utils.py` | VAE encoding, image resize/crop utilities |
| `Lookzi.ipynb` | Colab notebook (barcha fixlar kiritilgan) |
| `COLAB_GPU_SETUP.md` | Colab setup qo'llanmasi (markdown) |

### `SkipAttnProcessor` — KRITIK
CatVTON modeli `SkipAttnProcessor` ishlatsiz ishlamaydi.
`unet.set_attn_processor(...)` yoki `enable_xformers...` dan keyin `set_attn_processor(...)` chaqirish → shape crash.
**Qoida: `set_attn_processor()` ni hech qachon pipeline init dan keyin chaqirma.**

### `infer_garment_style()` qaytarish formati
```python
garment_style, covers_lower_legs, style_debug = infer_garment_style(cloth_image, cloth_type)
# garment_style: 'sleeveless' | 'short_sleeve' | 'sleeved' | 'auto'
# covers_lower_legs: bool (h/256 > 0.65)
# style_debug: str (diagnostika uchun)
```

### `cloth_agnostic_mask()` asosiy parametrlar
```python
mask = cloth_agnostic_mask(
    densepose_mask,   # PIL Image
    schp_lip_mask,    # PIL Image
    schp_atr_mask,    # PIL Image
    part='upper',     # 'upper' | 'lower' | 'overall' | 'inner' | 'outer'
    garment_style='sleeved',    # 'sleeveless' | 'short_sleeve' | 'sleeved' | 'auto'
    covers_lower_legs=True,     # Overall uchun oyoq alt qismi
)
```

### Colab T4 cheklovlar
- T4 = CUDA 7.5 → xformers ishlamaydi (kerak 8.0+)
- fp16 ishlaydi (bf16 ishlamaydi — bf16 Ampere+ kerak)
- 15GB VRAM — modellar + inference sig'adi
- ngrok bilan Gradio share ishlaydi, Colab proxy proxy orqali rasm chiqarmaydi (broken images)

### Pip install tartibi (MUHIM)
```
1. setuptools<70 wheel
2. fvcore --no-deps
3. iopath portalocker yacs ... (fvcore deps alohida)
4. pycocotools --no-build-isolation
5. accelerate diffusers transformers peft huggingface_hub gradio ...
6. torch==2.5.1+cu121 torchvision==0.20.1 (OXIRIDA — boshqa paketlar tushirib qo'ymasin)
```

---

## Xatolar tarixi (tuzatilgan)

| Xato | Sabab | Yechim |
|------|-------|--------|
| `ValueError: Mountpoint must not already contain files` | drive.mount ikkinchi marta | `force_remount=True` |
| `OSError: Cannot find empty port 7860` | Eski app processi tirik | `fuser -k 7860/tcp` |
| `shape mismatch 49152x320 vs 768x320` | `set_attn_processor(AttnProcessor2_0())` | SDPA fallback o'chirildi |
| `RepositoryNotFoundError: hf_models/lookzi-vton` | Relative path symlink'da False | Absolute path (`/content/Lookzi/...`) |
| `NotImplementedError` (xformers) | T4 CUDA 7.5 | `except Exception` (keng ushlash) |
| torch downgrade (2.0.x) | iopath dependency | torch pin OXIRIDA |
| 360s render | Torch noto'g'ri versiya (CPU yoki eski CUDA) | Runtime restart + to'g'ri torch |
| Gradio proxy broken images | Colab proxy static file URL'lari buzuq | `--share` (ngrok) ga qaytish |

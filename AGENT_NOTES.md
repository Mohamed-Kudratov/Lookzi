# Lookzi — Agent Progress Notes

> Bu fayl kelajakdagi agentlar uchun. Har bir muhim o'zgarishdan keyin yangilab boring.
> Tartibi: yangi o'zgarishlar TEPAGA yoziladi.

---

## [2026-05-05 v13] Human review import and benchmark category correction

### Muammo
Review report buttonlari browser ichida ishlaydi, lekin Colab HTML Drive'dagi `results.json` faylni bevosita o'zgartira olmaydi. User baholagan natija doimiy dataset/logga yozilishi uchun alohida import qadam kerak.

Yana C02/C03 catalog pairlari `upper` deb belgilangan edi, lekin garment rasmlari odam ustidagi long/full reference garment bo'lgani uchun benchmark vazifasi `overall` bo'lishi kerak.

### Fix
- `apply_human_review.py` qo'shildi:
  - downloaded `lookzi_human_review_*.json` faylni o'qiydi
  - oxirgi eval run bilan pair id bo'yicha merge qiladi
  - `human_rating`, `failure_reason`, `review_status`, `reviewed_at` maydonlarini `results.json`ga yozadi
- `COLAB_GPU_SETUP.md`ga review JSON upload + merge komandasi qo'shildi.
- `benchmark/catalog_pairs.json`da C02/C03 `cloth_type=overall` qilib tuzatildi va tag/coverage range moslandi.

### Eslatma
Brauzerda button bosishning o'zi repo dataset/logga yozmaydi. Har reviewdan keyin `Download review JSON` -> `apply_human_review.py` qadamini bajarish kerak.

---

## [2026-05-05 v12] Interactive human review buttons

### Muammo
Review reportdagi `Human review: GOOD / OK / BAD / MODEL_FAIL / MASK_FAIL` faqat text edi, button emas edi. Colab'da user bosib baholay olmadi.

### Fix
- `review_report.html` cardlari endi haqiqiy buttonlar chiqaradi:
  - `GOOD`
  - `OK`
  - `BAD`
  - `MODEL_FAIL`
  - `MASK_FAIL`
- Har card uchun optional failure note input qo'shildi.
- Tanlangan review browser `localStorage`ga saqlanadi.
- `Download review JSON` tugmasi qo'shildi.
- `COLAB_GPU_SETUP.md`ga eski report rasmlari ko'rinmasligi va yangi commitdan keyin qayta run qilish kerakligi yozildi.

### Eslatma
Eski `review_report.html` fayllar avtomatik tuzalmaydi. `git pull origin main`dan keyin benchmarkni qayta run qilish kerak.

---

## [2026-05-05 v11] Self-contained review report and category status

### Muammo
Colab'da `review_report.html` ochilganda rasmlar ko'rinmadi. Sabab HTML ichida `/content/drive/...` pathlar image `src` sifatida ishlatilgan, browser/Colab esa ularni to'g'ridan-to'g'ri o'qimaydi.

### Fix
- `eval_benchmark.py` review report rasmlarni base64 data URI sifatida HTML ichiga joylaydi.
- Report endi self-contained: download qilinsa ham rasmlar ko'rinadi.
- `display_report.py` helper qo'shildi:
```bash
%run display_report.py /content/drive/MyDrive/Lookzi/eval_logs/outputs/.../review_report.html
```
- `COLAB_GPU_SETUP.md` yangi display instructions bilan yangilandi.

### Qo'shimcha
Benchmark summary endi category status chiqaradi:
- `PRODUCTION_CANDIDATE`
- `NEEDS_REVIEW`
- `NOT_PRODUCTION_READY`
- `BROKEN`

Bu hozirgi CatVTON engine qaysi categorylarda productionga tayyor emasligini aniq ko'rsatadi.

---

## [2026-05-05 v10] Eval JSON serialization fix

### Muammo
Colab full catalog benchmark oxirida `save_results()` paytida xato berdi:
```text
TypeError: Object of type bool is not JSON serializable
```

### Sabab
Run result ichida `numpy.bool_` yoki boshqa numpy scalar qiymatlar bor edi. Python `json.dump()` ularni oddiy `bool/int/float` sifatida serialize qila olmaydi.

### Fix
- `to_json_safe()` recursive converter qo'shildi:
  - `np.generic -> .item()`
  - `np.ndarray -> .tolist()`
  - dict/list/tuple ichidagi qiymatlar ham tozalanadi
- `save_results()` endi `json.dump(to_json_safe(all_results), ...)` ishlatadi.
- `review_report.html` endi log save'dan oldin yaratiladi, shuning uchun Drive logda xato bo'lsa ham review report yo'qolmaydi.

---

## [2026-05-05 v9] Colab setup updated for catalog benchmark workflow

### Nima qilindi
`COLAB_GPU_SETUP.md` yangi architecture va benchmark workflowga moslab qayta yozildi.

### Yangilanishlar
- Repo clone/update alohida ko'rsatildi (`git pull origin main`).
- App run qismi saqlandi.
- Yangi `Catalog Benchmark + Review Gallery` bo'limi qo'shildi.
- `benchmark/catalog_pairs.json` bilan fast va full test commandlari qo'shildi.
- Review report HTML qayerda chiqishi tushuntirildi.
- Eski `benchmark/pairs.json` curated benchmark alohida bo'limga ko'chirildi.
- Mojibake belgilar o'rniga ASCII matn ishlatildi.

### Asosiy command
```bash
python eval_benchmark.py \
  --mode full \
  --pairs benchmark/catalog_pairs.json \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --base_model_path /content/Lookzi/hf_models/stable-diffusion-inpainting \
  --vae_model_path /content/Lookzi/hf_models/sd-vae-ft-mse \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

---

## [2026-05-05 v8] Catalog benchmark and HTML review gallery

### Nima qilindi
Benchmark tizimi do'kondagi demo garment/person rasmlariga yaqinroq review workflowga kengaytirildi.

### Yangi fayl
- `benchmark/catalog_pairs.json` - catalog-style test pairlar:
  - upper
  - lower pants
  - lower skirt
  - overall/dress

### Eval benchmark yangilanishlari
- `eval_benchmark.py` endi har run uchun output papka yaratadi, fast mode'da ham mask rasmlari saqlanadi.
- `--review_report` argumenti qo'shildi.
- Default HTML report: `<output_dir>/<timestamp>/review_report.html`.
- Har resultga quyidagilar qo'shildi:
  - `engine_name`
  - `review_status`
  - `human_rating`
  - `failure_reason`
  - `person_path`
  - `garment_path`
- HTML gallery person, garment, mask, output va JSON diagnostikani bir joyda ko'rsatadi.

### Muhim fix
`benchmark/pairs.json` ichidagi invalid trailing comma tuzatildi. Endi JSON parser bilan valid.

### Qanday ishlatiladi
```bash
python eval_benchmark.py \
  --mode full \
  --pairs benchmark/catalog_pairs.json \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --base_model_path /content/Lookzi/hf_models/stable-diffusion-inpainting \
  --vae_model_path /content/Lookzi/hf_models/sd-vae-ft-mse \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

Run tugagach `review_report.html` ochiladi va category/engine bo'yicha human review qilinadi.

---

## [2026-05-05 v7] Platform architecture and try-on engine scaffold

### Nima qilindi
Lookzi product direction rasmiy ravishda `Lookzi AI Stylist Commerce Platform` deb nomlandi.

### Muhim qaror
Recommendation AI hozircha scope'dan chiqarildi. Hozirgi asosiy vazifa:
- do'kon kiritgan kiyim rasmini olish
- user person image bilan preview qilish
- qaysi category qaysi engine bilan yaxshi ishlashini benchmark qilish

### Yangi fayllar
- `docs/ARCHITECTURE.md` - platform architecture, current scope, future scope, engine strategy.
- `tryon_engines/base.py` - `TryOnEngine`, `TryOnRequest`, `TryOnResult`.
- `tryon_engines/catvton.py` - hozirgi CatVTON pipeline uchun adapter wrapper.
- `tryon_engines/router.py` - category bo'yicha engine tanlaydigan router.

### App integratsiya
`app.py` endi CatVTON pipeline'ni bevosita chaqirmaydi. U `TryOnRequest` yaratadi va `CatVTONEngine.run()` orqali preview oladi. Bu keyin boshqa engine'larni router orqali ulashga tayyorlaydi.

### Keyingi yo'nalish
1. benchmark output schema'ni engine nomi/category bilan kengaytirish
2. category-wise production-ready status yozish
3. creative/alternative engine uchun stub qo'shish

---

## [2026-05-05 v6] Skirt silhouette can extend outside leg/background parse

### Muammo
Skirt branch source pants maskni olib tashlagandan keyin ham mask preview oyoq/pants silhouettega o'xshab qoldi.

### Sabab
`background_area` lower branch ichida `protect_area`ga juda erta qo'shilgan edi. Skirt trapezoid silhouette oyoq tanasidan tashqariga kengayganda, SCHP/DensePose u joylarni background deb ko'radi va maskadan kesib tashlaydi. Natija: skirt shape yana lower-body/leg contourga qaytadi.

### Fix
- Lower branchda `lower_background_protect = background_area & ~target_lower_area` qilindi.
- `garment_style == "skirt"` bo'lsa `target_lower_area` skirt silhouette, shuning ichidagi background endi protect emas.
- Final protect ham `~target_lower_area` orqali ishlaydi, `~lower_body_area` emas.

### Kutilgan natija
Red skirt mask preview endi oyoq ikki bo'lak ko'rinishida emas, kengroq bitta skirt/trapezoid body sifatida chiqishi kerak.

---

## [2026-05-05 v5] Skirt mask no longer inherits source pants shape

### Muammo
`skirt` subtype aniqlangandan keyin ham red skirt testlarida output pants/shorts tomonga ketdi.

### Sabab
Lower branchda `strong_mask_area` source persondagi eski lower garmentdan olinadi. Agar source odam black pants yoki jeans kiygan bo'lsa, `strong_mask_area` pants silhouette bo'ladi. Skirt branchda ham:
```python
allowed_area = ... | strong_mask_area
mask_area = ... | strong_mask_area
```
ishlayotgani uchun skirt trapezoid mask oxirida yana pants/two-leg maskga aylanib qolgan.

### Fix
- `garment_style == "skirt"` bo'lsa source `strong_mask_area` lower maskga majburan qo'shilmaydi.
- Skirt mask knee/mini skirt kabi ishlashi uchun `covers_lower_legs=False` qilib qaytariladi.
- Short/knee skirt silhouette pastga 58% emas, 76% lower bodygacha tushadigan qilindi.

### Eslatma
Bu CatVTON ichida mumkin bo'lgan eng mantiqli mask-side fix. Agar bundan keyin ham skirt geometry to'liq chiqmasa, bu CatVTON checkpoint lower cross-category (pants -> skirt) transferni yaxshi bilmasligini anglatadi. Keyingi real yechim: lower/skirt uchun IDM-VTON yoki boshqa VTON modelni solishtirish.

---

## [2026-05-05 v4] Lower garment subtype: skirt vs pants mask

### Nima qilindi
Lower garment uchun faqat "lower body" mask yetmasligi aniqlandi. Red skirt testlarida mask ikki oyoq/pants shaklida bo'lgani uchun model skirtni red pants yoki gray shorts qilib chiqarayotgan edi.

### Fix
- `infer_garment_style()` lower garment uchun endi subtype qaytaradi:
  - `skirt`
  - `pants`
  - `shorts`
- `model/cloth_masker.py` ichida `garment_style == "skirt"` bo'lsa lower mask ikki oyoq shaklida emas, beldan pastga kengayuvchi skirt/trapezoid silhouette sifatida quriladi.
- `covers_lower_legs=False` bo'lsa lower mask faqat thigh/shorts zonaga cheklanadi.
- `benchmark/pairs.json` lower expected stylelari `pants` va `skirt` qilib yangilandi.

### Sabab
CatVTON condition garmentni ko'rsa ham, mask silhouette pants bo'lsa model skirt geometriyasini chiqara olmaydi. Skirt uchun mask ham skirtga o'xshash bo'lishi kerak.

### Hali test qilish kerak
Colab/GPU'da red skirt testlarini qayta ko'rish:
- black pants person -> red skirt
- light jeans woman -> red skirt

Kutilgan natija: output pants emas, kamida skirt/loose lower silhouette tomonga o'tishi kerak. Agar hali ham pants chiqsa, keyingi bosqich CatVTON o'rniga lower/overall uchun IDM-VTON yoki OOTDiffusion adapterini sinash.

---

## [2026-05-05 v3] Lower mask fallback and final protect fix

### Nima qilindi
Lower garment mask engine yana kuchaytirildi va GitHub'ga `9e312dc` commit bilan push qilindi.

### Muammo
Screenshotlarda lower try-on mask coverage juda past chiqdi:
- skirt test: ~9.90%
- pants test: ~6.45%
- shorts/skirt test: ~11.40%

Mask ko'rinishda lower body topilgandek edi, lekin juda tor/kalta bo'lgani uchun model condition garmentni kuchli ushlamay, eski shim/shorts rangini yoki o'zi taxmin qilgan lower kiyimni chiqarib yubordi.

### Sabab
Oldingi lower fix branch ichida `strong_protect_area`ni lower bodydan olib tashlagan edi:
```python
local_strong_protect = strong_protect_area & ~lower_body_area
```
Lekin final cleanup oxirida yana umumiy quyidagi logic ishlayotgan edi:
```python
mask_area = ... & (~strong_protect_area) & (~background_area)
```
Natijada lower branchdagi fix final bosqichda qayta bekor bo'lib, oyoq/feet atrofida mask yana kesilib qolgan.

### Fix
- `final_protect_area` part-specific qilindi.
- `part == "lower"` bo'lsa final protect lower body ichidagi maskani kesmaydi.
- Lower mask coverage juda past (`<18%`) bo'lsa DensePose lower-body hull fallback ishlaydi.
- Kichik stray bloblar uchun `keep_main_components()` qo'shildi.
- `infer_garment_style()` lower garment uchun endi `sleeveless/sleeved` demaydi, `auto -> lower` deb debug qiladi.

### Hali test qilish kerak
Colab/GPU'da quyidagilarni qayta sinash:
- red skirt -> black pants person
- blue jeans -> white shorts person
- red skirt -> light jeans woman

Kutilgan natija: lower mask coverage kamida ~20-35% oralig'iga chiqishi va output garment rang/forma condition imagega yaqinlashishi kerak.

---

## [2026-05-05] Avtomatik Baholash Tizimi (Eval Benchmark)

### Nima qilindi
Kodni har safar o'zgartirgandan so'ng, mask qoplanishi (coverage), style aniqlanishi va CLIP score (garment bilan o'xshashlik) kabi metrikalarni avtomatik o'lchaydigan tizim qo'shildi. Bu model o'zini-o'zi rivojlantirishini (continuous evaluation) ta'minlaydi. 

### Yangi fayllar
- `eval_benchmark.py` — Asosiy baholash skripti. Oldingi test natijalarini Drive'dan o'qiydi va joriy run natijalari bilan solishtirib Delta hisobot chiqaradi.
- `benchmark/pairs.json` — 10 ta maxsus tanlab olingan test juftliklari. Ularda kiyim turi, kutilayotgan style va optimal mask coverage diapazoni ko'rsatilgan.

### Qanday ishlatiladi?
Kodni o'zgartirgach, quyidagi buyruqni Colab yoki lokal muhitda ishga tushiring:
```bash
# Tezkor tekshirish (faqat mask va style, 2-3 min)
python eval_benchmark.py --mode fast \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json

# To'liq tekshirish (inference + CLIP kiritilgan, ~20 min)
python eval_benchmark.py --mode full \
  --resume_path /content/Lookzi/hf_models/lookzi-vton \
  --base_model_path /content/Lookzi/hf_models/stable-diffusion-inpainting \
  --vae_model_path /content/Lookzi/hf_models/sd-vae-ft-mse \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

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

## [2026-05-05 v2] Mask disappearing on light skin / white pants

### Muammo
Oq fonli rasmlarda kalta ishton kiygan (bare legs) yoki oq shim kiygan odamlarda (Screenshot 1 va 3) maska juda kichkina bo'lib qolayotgan edi (coverage: 8-14%). Maska faqat kiyimni qoplab, oyoqlarni qoplamagan (natijada jinsi shim kalta qilib kiydirilgan).

### Sabab
SCHP (Kiyim segmentatsiyasi) modellari ba'zan oq fon bilan ochiq teri rangi yoki oq shimni adashtirib, uni `Background` deb belgilagan. 
DensePose (Tana segmentatsiyasi) oyoqlarni to'g'ri topgan, LEKIN `cloth_masker.py` dagi quyidagi qator DensePose topgan joylarni ham o'chirib tashlagan:
```python
mask_area = ... & (~background_area)
```
Ya'ni, SCHP "bu background" desa, tizim ko'r-ko'rona o'sha joyni maskadan o'chirgan, vaholanki DensePose "bu oyoq" deb turgan bo'lsa ham!

### Fix (model/cloth_masker.py)
Agar DensePose qandaydir tana qismini (`densepose_mask > 0`) ko'rsa, biz uni aslo `Background` deb hisoblamasligimiz kerak:
```python
densepose_body = (np.array(densepose_mask) > 0)
background_area = background_area & (~densepose_body)
```
Bu orqali Ochiq rangli oyoqlar va Oq shimlar endi `Background` tomonidan o'chirib yuborilmaydi va maska to'liq tushadi.

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

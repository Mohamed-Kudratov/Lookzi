"""
eval_benchmark.py — Lookzi Mask Engine Avtomatik Baholash Tizimi

Ishlatish:
  # Faqat mask + style (tez, ~3 min):
  python eval_benchmark.py --mode fast \
      --resume_path /content/Lookzi/hf_models/lookzi-vton \
      --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json

  # To'liq inference + CLIP (~20 min):
  python eval_benchmark.py --mode full \
      --resume_path /content/Lookzi/hf_models/lookzi-vton \
      --base_model_path /content/Lookzi/hf_models/stable-diffusion-inpainting \
      --vae_model_path /content/Lookzi/hf_models/sd-vae-ft-mse \
      --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
      --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
"""

import argparse
import base64
import html
import json
import mimetypes
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Lookzi benchmark eval")
    p.add_argument("--mode", choices=["fast", "full"], default="fast",
                   help="fast=mask+style only; full=mask+style+inference+CLIP")
    p.add_argument("--engine", choices=["catvton", "identity_baseline"], default="catvton",
                   help="Try-on engine to benchmark")
    p.add_argument("--pairs", default="benchmark/pairs.json",
                   help="Benchmark juftliklari fayli")
    p.add_argument("--resume_path", default="hf_models/lookzi-vton")
    p.add_argument("--base_model_path", default="hf_models/stable-diffusion-inpainting")
    p.add_argument("--vae_model_path", default="hf_models/sd-vae-ft-mse")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drive_log",
                   default="/content/drive/MyDrive/Lookzi/eval_logs/results.json",
                   help="Drive'dagi kumulativ JSON log fayli")
    p.add_argument("--output_dir",
                   default="/content/drive/MyDrive/Lookzi/eval_logs/outputs",
                   help="Natija rasmlari saqlanadigan papka (Drive)")
    p.add_argument("--high_priority_only", action="store_true",
                   help="Faqat priority=high juftliklarni test qilish (tezroq)")
    p.add_argument("--review_report", default=None,
                   help="HTML review gallery path. Default: output_dir/<timestamp>/review_report.html")
    return p.parse_args()


# ─────────────────────────────────────────────
# Git commit hash
# ─────────────────────────────────────────────

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
# Drive log: load / save
# ─────────────────────────────────────────────

def load_prev_results(log_path: str) -> list:
    """Drive'dagi JSON log faylini o'qiydi. Yo'q bo'lsa bo'sh list qaytaradi."""
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[WARN] Log o'qishda xato: {e} — bo'sh boshlanadi")
        return []


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_results(log_path: str, all_results: list):
    """Yangilangan natijalarni Drive JSON fayliga yozadi."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(all_results), f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# Mask coverage hisoblash
# ─────────────────────────────────────────────

def compute_coverage(mask_img: Image.Image) -> float:
    """Mask qoplanish foizini hisoblaydi (0–100)."""
    arr = np.array(mask_img.convert("L"))
    return float((arr > 127).mean() * 100)


# ─────────────────────────────────────────────
# Coverage sifat bahosi
# ─────────────────────────────────────────────

def coverage_grade(coverage: float, expected_range: list, targets: dict, cloth_type: str) -> str:
    """
    Coverage foiziga qarab baho beradi:
      OK       — expected_range ichida
      WARN_LOW — expected_range dan past
      WARN_HIGH— expected_range dan baland
      FAIL_LOW — targets[cloth_type].min dan past (jiddiy muammo)
    """
    lo, hi = expected_range
    t = targets.get(cloth_type, {})
    t_min = t.get("min", 10)
    if coverage < t_min:
        return "FAIL_LOW"
    elif coverage < lo:
        return "WARN_LOW"
    elif coverage > hi:
        return "WARN_HIGH"
    else:
        return "OK"


# ─────────────────────────────────────────────
# CLIP Score (full mode uchun)
# ─────────────────────────────────────────────

_clip_model = None
_clip_processor = None

def load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor
    print("  [CLIP] Model yuklanmoqda (openai/clip-vit-base-patch32)...")
    from transformers import CLIPModel, CLIPProcessor
    import torch
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
    if torch.cuda.is_available():
        _clip_model = _clip_model.cuda()
    return _clip_model, _clip_processor


def compute_clip_score(garment_img: Image.Image, result_img: Image.Image) -> float:
    """
    Kiyim rasmi va natija rasmi orasidagi CLIP cosine similarity.
    Yuqoriroq = kiyim to'g'riroq kiydirilyapti.
    """
    import torch
    model, processor = load_clip()
    inputs = processor(
        images=[garment_img.convert("RGB"), result_img.convert("RGB")],
        return_tensors="pt",
        padding=True
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    score = (feats[0] * feats[1]).sum().item()
    return round(float(score), 4)


def resize_crop_pil(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    from PIL import ImageOps

    return ImageOps.fit(
        image.convert("RGB"),
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


# ─────────────────────────────────────────────
# Bitta juftlikni test qilish
# ─────────────────────────────────────────────

def eval_one_pair(pair: dict, args, automasker, pipeline,
                  mask_processor, run_output_dir: str) -> dict:
    """
    Bitta person+garment juftligi uchun metrikalarni hisoblaydi.
    Qaytaradi: dict (barcha metrikalar)
    """
    pid = pair["id"]
    tag = pair["tag"]
    cloth_type = pair["cloth_type"]
    expected_style = pair["expected_style"]
    expected_range = pair["expected_coverage_range"]

    person_path = pair["person"]
    garment_path = pair["garment"]

    result = {
        "id": pid,
        "tag": tag,
        "cloth_type": cloth_type,
        "expected_style": expected_style,
        "expected_coverage_range": expected_range,
        "priority": pair.get("priority", "medium"),
        "engine_name": args.engine,
        "review_status": pair.get("review_status", "NEEDS_HUMAN_REVIEW"),
        "human_rating": None,
        "failure_reason": None,
        "person_path": person_path,
        "garment_path": garment_path,
    }

    try:
        t_start = time.time()

        # Rasmlarni yuklash
        person_img = Image.open(person_path).convert("RGB")
        garment_img = Image.open(garment_path).convert("RGB")

        if args.engine == "identity_baseline":
            from tryon_engines.base import TryOnRequest
            from tryon_engines.baseline import IdentityBaselineEngine

            blank_mask = Image.new("L", (args.width, args.height), 0)
            result["detected_style"] = "not_applicable"
            result["style_correct"] = None
            result["covers_lower_legs"] = None
            result["style_debug"] = "identity_baseline skips style and mask inference"
            result["mask_time_s"] = 0.0
            result["coverage_pct"] = 0.0
            if run_output_dir:
                os.makedirs(run_output_dir, exist_ok=True)
                mask_path = os.path.join(run_output_dir, f"{pid}_{tag}_mask.png")
                blank_mask.save(mask_path)
                result["mask_path"] = mask_path

            if args.mode == "full":
                t_inf_start = time.time()
                engine_result = IdentityBaselineEngine().run(
                    TryOnRequest(
                        person_image=person_img,
                        garment_image=garment_img,
                        category=cloth_type,
                        seed=args.seed,
                        steps=args.num_inference_steps,
                        guidance_scale=2.5,
                        width=args.width,
                        height=args.height,
                    )
                )
                result_image = engine_result.image
                result["engine_name"] = engine_result.engine_name
                result["inference_time_s"] = round(time.time() - t_inf_start, 2)
                out_path = os.path.join(run_output_dir, f"{pid}_{tag}.png")
                result_image.save(out_path)
                result["output_path"] = out_path
                result["clip_score"] = compute_clip_score(garment_img, result_image)
            elif run_output_dir:
                preview_path = os.path.join(run_output_dir, f"{pid}_{tag}_preview.png")
                resize_crop_pil(person_img, (args.width, args.height)).save(preview_path)
                result["output_path"] = preview_path

            result["total_time_s"] = round(time.time() - t_start, 2)
            result["error"] = None
            return result

        from utils import infer_garment_style, resize_and_crop, resize_and_padding

        # Garment style aniqlash
        garment_style, covers_lower_legs, style_debug = infer_garment_style(
            garment_img, cloth_type
        )
        result["detected_style"] = garment_style
        result["style_correct"] = (garment_style == expected_style)
        result["covers_lower_legs"] = covers_lower_legs
        result["style_debug"] = style_debug

        # Resize
        person_resized = resize_and_crop(person_img, (args.width, args.height))
        garment_resized = resize_and_padding(garment_img, (args.width, args.height))

        # Mask yaratish
        t_mask_start = time.time()
        mask_result = automasker(
            person_resized,
            cloth_type,
            garment_style=garment_style,
            covers_lower_legs=covers_lower_legs,
        )
        mask = mask_result["mask"]
        mask_blurred = mask_processor.blur(mask, blur_factor=9)
        result["mask_time_s"] = round(time.time() - t_mask_start, 2)

        # Coverage hisoblash
        coverage = compute_coverage(mask_blurred)
        result["coverage_pct"] = round(coverage, 2)

        # Mask rasmini Drive'ga saqlash (fast mode'da ham)
        if run_output_dir:
            os.makedirs(run_output_dir, exist_ok=True)
            mask_path = os.path.join(run_output_dir, f"{pid}_{tag}_mask.png")
            mask_blurred.save(mask_path)
            result["mask_path"] = mask_path

        # Full mode: inference + CLIP
        if args.mode == "full" and pipeline is not None:
            import torch
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            t_inf_start = time.time()
            result_image = pipeline(
                image=person_resized,
                condition_image=garment_resized,
                mask=mask_blurred,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=2.5,
                generator=generator,
            )[0]
            result["inference_time_s"] = round(time.time() - t_inf_start, 2)

            # Natija rasmni saqlash
            out_path = os.path.join(run_output_dir, f"{pid}_{tag}.png")
            result_image.save(out_path)
            result["output_path"] = out_path

            # CLIP score
            result["clip_score"] = compute_clip_score(garment_img, result_image)

        result["total_time_s"] = round(time.time() - t_start, 2)
        result["error"] = None

    except Exception as e:
        result["error"] = str(e)
        print(f"   XATO: {e}")

    return result


def img_src(path: str | None, embed: bool = True) -> str:
    if not path:
        return ""
    if embed and os.path.exists(path):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{data}"
    return html.escape(path.replace("\\", "/"))


def generate_review_report(report_path: str, run_results: list, run_meta: dict):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    cards = []
    for r in run_results:
        status = r.get("review_status", "NEEDS_HUMAN_REVIEW")
        card_id = html.escape(r.get("id", "?"))
        error = r.get("error")
        output_html = ""
        if r.get("output_path"):
            output_html = f'<img src="{img_src(r.get("output_path"))}" alt="output">'
        else:
            output_html = '<div class="placeholder">No output in fast mode</div>'

        cards.append(f"""
        <article class="card {html.escape(status.lower())}" data-pair-id="{card_id}">
          <header>
            <h2>{html.escape(r.get("id", "?"))} - {html.escape(r.get("tag", ""))}</h2>
            <div class="badges">
              <span>{html.escape(r.get("cloth_type", "?"))}</span>
              <strong class="status-pill">{html.escape(status)}</strong>
            </div>
          </header>
          <div class="grid">
            <section><h3>Person</h3><img src="{img_src(r.get("person_path"))}" alt="person"></section>
            <section><h3>Garment</h3><img src="{img_src(r.get("garment_path"))}" alt="garment"></section>
            <section><h3>Mask</h3><img src="{img_src(r.get("mask_path"))}" alt="mask"></section>
            <section><h3>Output</h3>{output_html}</section>
          </div>
          <pre>{html.escape(json.dumps({
              "engine": r.get("engine_name"),
              "style": r.get("detected_style"),
              "expected_style": r.get("expected_style"),
              "style_correct": r.get("style_correct"),
              "coverage_pct": r.get("coverage_pct"),
              "coverage_grade": r.get("coverage_grade"),
              "clip_score": r.get("clip_score"),
              "error": error,
              "review_status": status,
              "human_rating": r.get("human_rating"),
              "failure_reason": r.get("failure_reason"),
          }, ensure_ascii=False, indent=2))}</pre>
          <div class="review" data-current="{html.escape(status)}">
            <button type="button" data-rating="GOOD">GOOD</button>
            <button type="button" data-rating="OK">OK</button>
            <button type="button" data-rating="BAD">BAD</button>
            <button type="button" data-rating="MODEL_FAIL">MODEL_FAIL</button>
            <button type="button" data-rating="MASK_FAIL">MASK_FAIL</button>
            <input type="text" placeholder="failure note (optional)" aria-label="failure note">
          </div>
        </article>
        """)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lookzi Benchmark Review - {html.escape(run_meta.get("timestamp", ""))}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111827; color: #e5e7eb; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .meta {{ color: #9ca3af; margin-bottom: 24px; }}
    .card {{ background: #1f2937; border: 1px solid #374151; border-radius: 8px; margin-bottom: 24px; padding: 16px; }}
    .card.reviewed {{ border-color: #22c55e; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; }}
    .badges {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    header span, .status-pill {{ background: #374151; border-radius: 999px; padding: 4px 10px; font-size: 13px; }}
    .status-pill {{ background: #4b5563; color: #f9fafb; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    section {{ min-width: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 8px 0; font-size: 13px; color: #cbd5e1; }}
    img {{ width: 100%; height: 360px; object-fit: contain; background: #020617; border-radius: 6px; }}
    pre {{ white-space: pre-wrap; background: #020617; padding: 12px; border-radius: 6px; overflow: auto; }}
    .placeholder {{ height: 360px; display: grid; place-items: center; background: #020617; color: #64748b; border-radius: 6px; }}
    .review {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .review button, #download-review {{ border: 0; border-radius: 6px; padding: 8px 10px; background: #374151; color: #e5e7eb; font-weight: 700; cursor: pointer; }}
    .review button.active {{ background: #22c55e; color: #052e16; }}
    .review input {{ min-width: 260px; flex: 1; border: 1px solid #4b5563; border-radius: 6px; padding: 8px 10px; background: #111827; color: #e5e7eb; }}
    #download-review {{ margin-bottom: 18px; background: #2563eb; }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} img, .placeholder {{ height: 280px; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Lookzi Benchmark Review</h1>
    <div class="meta">Commit {html.escape(str(run_meta.get("commit")))} | Mode {html.escape(str(run_meta.get("mode")))} | {html.escape(str(run_meta.get("timestamp")))}</div>
    <button id="download-review" type="button">Download review JSON</button>
    {''.join(cards)}
  </main>
  <script>
    const STORAGE_KEY = "lookzi_review_{html.escape(str(run_meta.get("timestamp")))}";

    function loadReviews() {{
      try {{
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
      }} catch (error) {{
        return {{}};
      }}
    }}

    function saveReviews(reviews) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews, null, 2));
    }}

    function applyReview(card, rating, note) {{
      const pairId = card.dataset.pairId;
      const reviews = loadReviews();
      reviews[pairId] = {{ human_rating: rating, failure_reason: note || null }};
      saveReviews(reviews);

      card.classList.add("reviewed");
      card.querySelector(".status-pill").textContent = rating;
      card.querySelectorAll(".review button").forEach((button) => {{
        button.classList.toggle("active", button.dataset.rating === rating);
      }});
    }}

    document.querySelectorAll(".card").forEach((card) => {{
      const pairId = card.dataset.pairId;
      const input = card.querySelector(".review input");
      const saved = loadReviews()[pairId];
      if (saved) {{
        input.value = saved.failure_reason || "";
        applyReview(card, saved.human_rating, input.value);
      }}
      card.querySelectorAll(".review button").forEach((button) => {{
        button.addEventListener("click", () => applyReview(card, button.dataset.rating, input.value));
      }});
      input.addEventListener("change", () => {{
        const active = card.querySelector(".review button.active");
        if (active) applyReview(card, active.dataset.rating, input.value);
      }});
    }});

    document.getElementById("download-review").addEventListener("click", () => {{
      const blob = new Blob([JSON.stringify(loadReviews(), null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "lookzi_human_review_{html.escape(str(run_meta.get("timestamp")))}.json";
      a.click();
      URL.revokeObjectURL(url);
    }});
  </script>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def category_status_summary(run_results: list) -> dict:
    summary = {}
    for r in run_results:
        category = r.get("cloth_type", "unknown")
        item = summary.setdefault(
            category,
            {"total": 0, "coverage_ok": 0, "style_ok": 0, "errors": 0, "status": "UNKNOWN"},
        )
        item["total"] += 1
        if r.get("coverage_grade") == "OK":
            item["coverage_ok"] += 1
        if r.get("style_correct"):
            item["style_ok"] += 1
        if r.get("error"):
            item["errors"] += 1

    for category, item in summary.items():
        total = max(1, item["total"])
        coverage_rate = item["coverage_ok"] / total
        style_rate = item["style_ok"] / total
        if item["errors"]:
            status = "BROKEN"
        elif coverage_rate >= 0.8 and style_rate >= 0.8:
            status = "PRODUCTION_CANDIDATE"
        elif coverage_rate >= 0.5:
            status = "NEEDS_REVIEW"
        else:
            status = "NOT_PRODUCTION_READY"
        item["coverage_rate"] = round(coverage_rate, 3)
        item["style_rate"] = round(style_rate, 3)
        item["status"] = status
    return summary


# ─────────────────────────────────────────────
# Delta hisobot (oldingi run bilan taqqos)
# ─────────────────────────────────────────────

def find_prev_run_results(all_results: list, pair_id: str) -> dict | None:
    """Oxirgi muvaffaqiyatli run'dan berilgan pair_id natijasini topadi."""
    # Oxirgi run (oxirgi element) dan boshlab qidiradi
    for run in reversed(all_results):
        for r in run.get("pair_results", []):
            if r.get("id") == pair_id and r.get("error") is None:
                return r
    return None


def delta_str(current, prev, higher_is_better=True, unit="", precision=2):
    """Ikkita qiymat o'rtasidagi farqni chiroyli chiqaradi."""
    if prev is None or current is None:
        return ""
    diff = current - prev
    if abs(diff) < 10 ** (-precision):
        return f"  [prev: {prev:.{precision}f}{unit}  Δ ≈0]"
    arrow = "↑" if diff > 0 else "↓"
    good = (diff > 0) == higher_is_better
    sign = "+" if diff > 0 else ""
    flag = "✅" if good else "⚠️"
    return f"  [prev: {prev:.{precision}f}{unit}  Δ {sign}{diff:.{precision}f} {arrow} {flag}]"


# ─────────────────────────────────────────────
# Hisobot chiqarish
# ─────────────────────────────────────────────

GRADE_ICON = {
    "OK": "✅",
    "WARN_LOW": "⚠️ PAST",
    "WARN_HIGH": "⚠️ BALAND",
    "FAIL_LOW": "❌ JIDDIY PAST",
    "UNKNOWN": "❓",
}


def print_report(run_results: list, all_prev: list, run_meta: dict, coverage_targets: dict):
    sep = "=" * 68
    thin = "-" * 68
    print(f"\n{sep}")
    print(f" LOOKZI MASK EVAL  —  {run_meta['timestamp']}")
    print(f" Commit: {run_meta['commit']}  |  Mode: {run_meta['mode']}")
    print(sep)

    total = len(run_results)
    style_ok = 0
    coverage_ok = 0
    clip_scores = []
    inf_times = []

    for r in run_results:
        pid = r["id"]
        tag = r["tag"]
        cloth_type = r["cloth_type"]
        prev = find_prev_run_results(all_prev, pid)

        print(f"\n #{pid} [{tag}]  →  cloth_type={cloth_type}")

        if r.get("error"):
            print(f"   ❌ XATO: {r['error']}")
            continue

        # Style
        style_icon = "✅" if r.get("style_correct") else "❌"
        prev_style = prev.get("detected_style") if prev else None
        style_change = ""
        if prev_style and prev_style != r.get("detected_style"):
            style_change = f"  [prev: {prev_style} → o'zgardi]"
        print(f"   Style:    {r.get('detected_style','?')} {style_icon}  (expected: {r['expected_style']}){style_change}")
        if r.get("style_correct"):
            style_ok += 1

        # Coverage
        cov = r.get("coverage_pct", 0)
        grade = r.get("coverage_grade", "UNKNOWN")
        grade_str = GRADE_ICON.get(grade, grade)
        prev_cov = prev.get("coverage_pct") if prev else None
        d_cov = delta_str(cov, prev_cov, higher_is_better=True, unit="%")
        t = coverage_targets.get(cloth_type, {})
        target_str = f"[maqsad: {r['expected_coverage_range'][0]}–{r['expected_coverage_range'][1]}%]"
        print(f"   Coverage: {cov:.1f}%  {grade_str}  {target_str}{d_cov}")
        if grade == "OK":
            coverage_ok += 1

        # Mask time
        print(f"   Mask vaqti: {r.get('mask_time_s', '?')}s")

        # Full mode metrikalar
        if "clip_score" in r:
            cs = r["clip_score"]
            clip_icon = "✅" if cs >= 0.20 else ("⚠️" if cs >= 0.15 else "❌")
            prev_cs = prev.get("clip_score") if prev else None
            d_cs = delta_str(cs, prev_cs, higher_is_better=True, precision=4)
            print(f"   CLIP:     {cs:.4f}  {clip_icon}{d_cs}")
            clip_scores.append(cs)

        if "inference_time_s" in r:
            it = r["inference_time_s"]
            prev_it = prev.get("inference_time_s") if prev else None
            d_it = delta_str(it, prev_it, higher_is_better=False, unit="s", precision=1)
            print(f"   Inference: {it:.1f}s{d_it}")
            inf_times.append(it)

        if r.get("output_path"):
            print(f"   📁 {r['output_path']}")

    # Umumiy
    print(f"\n{thin}")
    print(f" UMUMIY NATIJA  ({total} juftlik):")
    print(f"   Style to'g'ri : {style_ok}/{total}")
    print(f"   Coverage OK   : {coverage_ok}/{total}")
    if clip_scores:
        print(f"   Avg CLIP      : {sum(clip_scores)/len(clip_scores):.4f}")
    if inf_times:
        print(f"   Avg inference : {sum(inf_times)/len(inf_times):.1f}s")

    category_status = category_status_summary(run_results)
    if category_status:
        print("\n CATEGORY STATUS:")
        for category, item in category_status.items():
            print(
                f"   {category}: {item['status']} "
                f"(coverage_ok={item['coverage_ok']}/{item['total']}, "
                f"style_ok={item['style_ok']}/{item['total']})"
            )

    # Global delta (oxirgi run bilan)
    if all_prev:
        last_run = all_prev[-1]
        prev_style_ok = sum(1 for r in last_run.get("pair_results", []) if r.get("style_correct"))
        prev_cov_ok = sum(1 for r in last_run.get("pair_results", []) if r.get("coverage_grade") == "OK")
        prev_total = len(last_run.get("pair_results", []))
        print(f"\n OLDINGI RUN   ({last_run.get('timestamp','?')}):")
        print(f"   Style: {prev_style_ok}/{prev_total}  |  Coverage: {prev_cov_ok}/{prev_total}")

        ds = style_ok - prev_style_ok
        dc = coverage_ok - prev_cov_ok
        overall = "✅ YAXSHILANDI" if (ds + dc) > 0 else ("➡️ O'ZGARMADI" if (ds + dc) == 0 else "⚠️ YOMONLASHDI")
        print(f"   DELTA: Style {'+' if ds>=0 else ''}{ds}  |  Coverage {'+' if dc>=0 else ''}{dc}  →  {overall}")

    print(sep + "\n")


# ─────────────────────────────────────────────
# Asosiy funksiya
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    if args.device == "cpu":
        args.mixed_precision = "no"

    print(f"\n[Lookzi Eval] Device={args.device}  Mode={args.mode}  Engine={args.engine}  mixed_precision={args.mixed_precision}")

    # Pairs yuklab olish
    with open(args.pairs, "r", encoding="utf-8") as f:
        pairs_data = json.load(f)

    coverage_targets = pairs_data.get("coverage_targets", {})
    pairs = pairs_data["pairs"]
    if args.high_priority_only:
        pairs = [p for p in pairs if p.get("priority") == "high"]
    print(f"[Eval] {len(pairs)} juftlik test qilinadi...")

    # Drive logni o'qish
    all_prev = load_prev_results(args.drive_log)
    print(f"[Eval] Drive logda {len(all_prev)} ta oldingi run topildi.")

    repo_path = args.resume_path
    automasker = None
    mask_processor = None
    pipeline = None

    if args.engine == "catvton":
        # AutoMasker yuklash
        from diffusers.image_processor import VaeImageProcessor
        from model.cloth_masker import AutoMasker

        if not os.path.exists(repo_path):
            from huggingface_hub import snapshot_download
            repo_path = snapshot_download(repo_id=repo_path)

        print("[Eval] AutoMasker yuklanmoqda...")
        mask_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False,
            do_binarize=True, do_convert_grayscale=True
        )
        automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device=args.device,
        )

        # Pipeline (faqat full mode'da)
        if args.mode == "full":
            from model.pipeline import LookziPipeline
            from utils import init_weight_dtype
            print("[Eval] LookziPipeline yuklanmoqda...")
            pipeline = LookziPipeline(
                base_ckpt=args.base_model_path,
                attn_ckpt=repo_path,
                attn_ckpt_version="mix",
                weight_dtype=init_weight_dtype(args.mixed_precision),
                use_tf32=True,
                device=args.device,
                vae_ckpt=args.vae_model_path,
                skip_safety_check=True,
            )

    # Timestamp + output papka
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, args.engine, ts)

    # Har juftlik uchun eval
    run_results = []
    for i, pair in enumerate(pairs):
        pid = pair["id"]
        tag = pair["tag"]
        print(f"\n[{i+1}/{len(pairs)}] {pid} — {tag}")
        r = eval_one_pair(pair, args, automasker, pipeline,
                          mask_processor, run_output_dir)
        # Coverage grade uchun targets'ni inject qilamiz
        r["coverage_grade"] = coverage_grade(
            r.get("coverage_pct", 0),
            pair["expected_coverage_range"],
            coverage_targets,
            pair["cloth_type"],
        )
        run_results.append(r)
        # Har juftlikdan keyin qisqa chiqdi
        cov_g = GRADE_ICON.get(r.get("coverage_grade", "UNKNOWN"), "?")
        style_ok = "✅" if r.get("style_correct") else "❌"
        print(f"   → coverage={r.get('coverage_pct','?'):.1f}%  {cov_g} | style={r.get('detected_style','?')} {style_ok}")

    # Run metadatasi
    commit = get_git_commit()
    run_meta = {
        "timestamp": ts,
        "commit": commit,
        "mode": args.mode,
        "engine": args.engine,
        "device": args.device,
        "mixed_precision": args.mixed_precision,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps if args.mode == "full" else None,
        "pair_count": len(run_results),
        "pair_results": run_results,
        # Umumiy metrikalar
        "summary": {
            "style_correct": sum(1 for r in run_results if r.get("style_correct")),
            "coverage_ok": sum(1 for r in run_results if r.get("coverage_grade") == "OK"),
            "coverage_warn": sum(1 for r in run_results if "WARN" in r.get("coverage_grade", "")),
            "coverage_fail": sum(1 for r in run_results if "FAIL" in r.get("coverage_grade", "")),
            "avg_clip": round(
                sum(r["clip_score"] for r in run_results if "clip_score" in r) /
                max(1, sum(1 for r in run_results if "clip_score" in r)), 4
            ) if any("clip_score" in r for r in run_results) else None,
            "avg_inference_s": round(
                sum(r["inference_time_s"] for r in run_results if "inference_time_s" in r) /
                max(1, sum(1 for r in run_results if "inference_time_s" in r)), 1
            ) if any("inference_time_s" in r for r in run_results) else None,
            "category_status": category_status_summary(run_results),
        },
    }

    # To'liq hisobot chiqarish
    print_report(run_results, all_prev, run_meta, coverage_targets)

    report_path = args.review_report or os.path.join(run_output_dir, "review_report.html")
    generate_review_report(report_path, run_results, run_meta)
    print(f"[Eval] Review report: {report_path}")

    # Drive'ga saqlash
    all_prev.append(run_meta)
    save_results(args.drive_log, all_prev)
    print(f"[Eval] Natijalar saqlandi: {args.drive_log}")
    print(f"[Eval] Jami log: {len(all_prev)} run\n")


if __name__ == "__main__":
    main()

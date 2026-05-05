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
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Lookzi benchmark eval")
    p.add_argument("--mode", choices=["fast", "full"], default="fast",
                   help="fast=mask+style only; full=mask+style+inference+CLIP")
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


def save_results(log_path: str, all_results: list):
    """Yangilangan natijalarni Drive JSON fayliga yozadi."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


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


# ─────────────────────────────────────────────
# Bitta juftlikni test qilish
# ─────────────────────────────────────────────

def eval_one_pair(pair: dict, args, automasker, pipeline,
                  mask_processor, run_output_dir: str) -> dict:
    """
    Bitta person+garment juftligi uchun metrikalarni hisoblaydi.
    Qaytaradi: dict (barcha metrikalar)
    """
    import torch
    from utils import infer_garment_style, resize_and_crop, resize_and_padding

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
    }

    try:
        t_start = time.time()

        # Rasmlarni yuklash
        person_img = Image.open(person_path).convert("RGB")
        garment_img = Image.open(garment_path).convert("RGB")

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
        print(f"   ❌ Xato: {e}")

    return result


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
    import torch
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        args.mixed_precision = "no"

    print(f"\n[Lookzi Eval] Device={args.device}  Mode={args.mode}  mixed_precision={args.mixed_precision}")

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

    # AutoMasker yuklash
    from diffusers.image_processor import VaeImageProcessor
    from model.cloth_masker import AutoMasker

    repo_path = args.resume_path
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
    pipeline = None
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
    run_output_dir = os.path.join(args.output_dir, ts) if args.mode == "full" else None

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
        },
    }

    # To'liq hisobot chiqarish
    print_report(run_results, all_prev, run_meta, coverage_targets)

    # Drive'ga saqlash
    all_prev.append(run_meta)
    save_results(args.drive_log, all_prev)
    print(f"[Eval] Natijalar saqlandi: {args.drive_log}")
    print(f"[Eval] Jami log: {len(all_prev)} run\n")


if __name__ == "__main__":
    main()

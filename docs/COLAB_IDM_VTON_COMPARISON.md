# Colab IDM-VTON Comparison Workflow

This file is the simple path for comparing IDM-VTON against CatVTON.

## Goal

We want this table:

```text
CatVTON:   already tested
IDM-VTON:  C01-C08 outputs -> Lookzi review report -> human ratings
```

## Important Truth

IDM-VTON cannot be benchmarked until it produces output images.

Lookzi now supports importing those images, but IDM-VTON itself still has to run first.

Official sources:

- https://github.com/yisol/IDM-VTON
- https://huggingface.co/yisol/IDM-VTON

The official Gradio demo auto-mask path is upper-body oriented. It can be useful for a first smoke test, but it is not enough to prove lower/overall support.

## Cell 1 - Update Lookzi

```python
%cd /content/Lookzi
!git pull origin main
```

## Cell 2 - Export Lookzi Benchmark Inputs

```python
%cd /content/Lookzi
!python prepare_candidate_inputs.py \
  --pairs benchmark/catalog_pairs.json \
  --output_dir /content/drive/MyDrive/Lookzi/candidate_inputs
```

This creates:

```text
/content/drive/MyDrive/Lookzi/candidate_inputs/C01_person.png
/content/drive/MyDrive/Lookzi/candidate_inputs/C01_garment.png
...
/content/drive/MyDrive/Lookzi/candidate_inputs/manifest.json
```

## Cell 3 - Prepare IDM-VTON Repo

```python
%cd /content
!rm -rf /content/IDM-VTON
!git clone https://github.com/yisol/IDM-VTON.git /content/IDM-VTON
%cd /content/IDM-VTON
```

Install IDM-VTON dependencies according to the official repo. If this conflicts with the existing Lookzi runtime, use a fresh Colab runtime for IDM-VTON and keep Drive mounted.

## Cell 4 - Produce IDM-VTON Outputs

For each pair in:

```text
/content/drive/MyDrive/Lookzi/candidate_inputs/
```

Run IDM-VTON with:

```text
person:  C01_person.png
garment: C01_garment.png
save as: /content/drive/MyDrive/Lookzi/idm_vton_outputs/C01.png
```

Repeat for C01-C08.

Expected output folder:

```text
/content/drive/MyDrive/Lookzi/idm_vton_outputs/C01.png
/content/drive/MyDrive/Lookzi/idm_vton_outputs/C02.png
...
/content/drive/MyDrive/Lookzi/idm_vton_outputs/C08.png
```

If IDM-VTON only works well for upper-body in the official demo, save only the successful outputs and still import them. Missing pairs will show an error in the Lookzi report, which is useful evidence.

## Cell 5 - Import IDM Outputs Into Lookzi Benchmark

```python
%cd /content/Lookzi
!python eval_benchmark.py \
  --mode full \
  --engine external_outputs \
  --external_engine_name idm_vton \
  --external_output_dir /content/drive/MyDrive/Lookzi/idm_vton_outputs \
  --pairs benchmark/catalog_pairs.json \
  --device cuda \
  --mixed_precision fp16 \
  --width 768 \
  --height 1024 \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

## Cell 6 - Open Review Report

Use the path printed by Cell 5:

```python
%run display_report.py /content/drive/MyDrive/Lookzi/eval_logs/outputs/idm_vton/YOUR_TIMESTAMP/review_report.html
```

Then human-review it exactly like CatVTON.

## Decision Rule

IDM-VTON is useful only if it beats CatVTON on human review.

Current CatVTON baseline from 2026-05-06:

```text
1 GOOD / 6 BAD / 1 MASK_FAIL
```

If IDM-VTON cannot produce outputs for lower/overall, we mark it as an upper-body candidate only.

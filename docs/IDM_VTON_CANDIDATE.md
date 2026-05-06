# IDM-VTON Candidate Plan

## What We Know

Primary sources:

- Official code: https://github.com/yisol/IDM-VTON
- Official model: https://huggingface.co/yisol/IDM-VTON

The official repository supports dataset-style inference through `inference.py` and DressCode category inference through `inference_dc.py`.

The official Gradio demo is useful for smoke testing, but its auto-mask path is hard-coded to `upper_body` in `gradio_demo/app.py`. That means the demo path is not enough for Lookzi's lower and overall benchmark.

## Decision

Do not wire IDM-VTON into Lookzi as a production engine until it can produce real outputs for the benchmark pairs.

First, run IDM-VTON externally and save outputs with pair ids:

```text
C01.png
C02.png
C03.png
...
C08.png
```

Then import those outputs into the Lookzi benchmark using:

```bash
python eval_benchmark.py \
  --mode full \
  --engine external_outputs \
  --external_engine_name idm_vton \
  --external_output_dir /path/to/idm_outputs \
  --pairs benchmark/catalog_pairs.json \
  --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json \
  --output_dir /content/drive/MyDrive/Lookzi/eval_logs/outputs
```

This produces the same review gallery format used for CatVTON.

## Gate

IDM-VTON should be considered useful only if human review beats both:

- CatVTON on the same pair set.
- `identity_baseline`.

If IDM-VTON only works for upper-body cases, it should be marked as an upper-body candidate, not a full Lookzi engine.

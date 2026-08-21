#!/usr/bin/env python3
"""Find the wording that actually lands on Central Asia.

The roster says "Central Asian" and the generator keeps returning faces that
read as East or Southeast Asian. That is a training-distribution problem, not a
prompt bug: in most image corpora "Asian" is overwhelmingly East Asian, and
"Central Asian" is rare enough that the model falls back to the nearest thing it
knows well.

Z-Image-Turbo is CFG-distilled and runs at guidance_scale=0.0, so there is no
negative prompt to push back with. The only lever is a more specific positive
description, and which description works is an empirical question about this
model's vocabulary -- not something to reason out in advance.

So: hold the face fixed, vary only the ethnicity phrase, keep the seeds aligned
across phrasings. Same seed and same phrasing length differences aside, any
difference in the output is attributable to the wording. Look at the grid and
pick the column that is right.

    /opt/zimage-venv/bin/python elements/ethnicity_probe.py
    /opt/zimage-venv/bin/python elements/ethnicity_probe.py --seeds 6
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elements.catalog import REALISM_PERSON  # noqa: E402

# Each entry is a candidate for the `appearance` slot in hero_prompt().
#
# They escalate: a bare demographic label, then nationalities the model may have
# seen captioned, then a nationality plus the physical features that actually
# separate a Turkic Central Asian face from an East Asian one. If the bare label
# already works the rest is wasted effort; if only the last one works, the
# roster needs feature descriptions baked in permanently.
PHRASINGS = [
    ("baseline",   "Central Asian"),
    ("uzbek",      "Uzbek"),
    ("kazakh",     "Kazakh"),
    ("tajik",      "Tajik"),
    ("uz_place",   "Uzbek from Tashkent, Turkic Central Asian"),
    ("uz_feature", "Uzbek Central Asian, Turkic and Persian features -- a broad "
                   "face with high wide cheekbones, a low flat nose bridge, a "
                   "slight epicanthic fold, deep-set dark brown eyes and thick "
                   "straight brows"),
]

# Two subjects, because ethnicity cues do not always transfer between genders --
# a phrase that reads Central Asian on a woman can read generically Asian on a
# man, and the roster needs both.
SUBJECTS = [
    dict(key="f", gender="woman", pronoun="her", age="mid 20s", build="average",
         skin="medium olive skin", hair="shoulder-length dark brown hair",
         detail="a round face and full lips",
         clothes="a plain fitted grey t-shirt and dark straight jeans"),
    dict(key="m", gender="man", pronoun="his", age="early 30s", build="average",
         skin="tan skin", hair="short dark hair with a trimmed beard",
         detail="a square face and heavy brows",
         clothes="a plain grey t-shirt and dark straight jeans"),
]


def build_prompt(subj, appearance):
    return (f"photorealistic photograph of a {appearance} {subj['gender']} "
            f"in {subj['pronoun']} {subj['age']}, {subj['build']} build, "
            f"{subj['skin']}, {subj['hair']}, {subj['detail']}, "
            f"wearing {subj['clothes']}, "
            f"upper body and head, front view facing camera, "
            f"soft diffused daylight, plain light grey studio backdrop, "
            f"neutral expression, {REALISM_PERSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/ethnicity_probe")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = []
    for subj in SUBJECTS:
        for label, appearance in PHRASINGS:
            for s in range(args.seeds):
                # Name sorts subject -> phrasing -> seed, so the viewer shows
                # every wording for one seed side by side.
                name = f"{subj['key']}__{label}__s{s}.png"
                jobs.append((os.path.join(args.out, name),
                             build_prompt(subj, appearance),
                             args.seed + s))

    print(f"{len(jobs)} images: {len(SUBJECTS)} subjects x "
          f"{len(PHRASINGS)} phrasings x {args.seeds} seeds")
    for label, appearance in PHRASINGS:
        print(f"  {label:12} {appearance[:80]}")
    if args.dry_run:
        return 0

    import torch
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo",
                                          torch_dtype=torch.bfloat16).to("cuda")
    made = 0
    for path, prompt, seed in jobs:
        if os.path.exists(path):
            continue
        t = time.time()
        img = pipe(prompt=prompt, height=1024, width=768, num_inference_steps=9,
                   guidance_scale=0.0,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        img.save(path)
        made += 1
        print(f"  {os.path.basename(path):34} {time.time() - t:5.1f}s", flush=True)

    print(f"\n{made} made -> {args.out}")
    print("Review:  bash go.sh view /workspace/ethnicity_probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

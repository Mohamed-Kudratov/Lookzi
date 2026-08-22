#!/usr/bin/env python
"""Two-stage identity generation: pick one face, then vary *that* face.

The problem this solves. Z-Image-Turbo is text-to-image, so every call invents a
new person. Thirty prompts for "Central Asian woman, mid 20s" return thirty
different women who merely fit the description -- and an identity LoRA needs
thirty pictures of *one* woman. Generating the dataset directly from text cannot
work, no matter how detailed the prompt.

So it is done in two stages, with two models we already run:

  stage 1   Z-Image-Turbo, text only     -> candidate faces, pick one "hero"
  stage 2   Qwen-Image-Edit-2509, with   -> the same face at every angle,
            the hero as a reference         distance and light in the grid
  stage 3   train the LoRA on stage 2    -> identity finally locked

Stage 2 is the try-on model doing what it already does: multi-reference,
instruction-driven editing. "Show this person, full body, three-quarter view,
hard light" is the same operation as putting a garment on someone.

    # stage 1 -- 12 candidates for one roster entry
    python elements/hero.py candidates --face f_cauz_20s_hijab --n 12

    # look, choose, then stage 2 -- the full coverage grid from that hero
    python elements/hero.py variations --face f_cauz_20s_hijab --hero 003

Stage 1 needs the Z-Image venv; stage 2 needs the system interpreter and the
pinned diffusers fork. They cannot share one, which is why this is two
subcommands rather than one script that does both.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timing import phase, record  # noqa: E402
from catalog import (ANGLES, article, feature_clause, BACKGROUNDS_TRAIN, CLOTHING_MODEST, CLOTHING_NEUTRAL,
                     DISTANCE_MIX, EXPRESSION_MIX, LIGHTING, REALISM_PERSON,
                     ROSTER)  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _limit_threads():
    """Match the thread pools to the cgroup CPU quota, not to /proc/cpuinfo.

    RunPod's host has 128 cores and the container is allowed 13.6 of them, but
    nproc reports the host figure, so torch starts 128 OpenMP threads against a
    tenth of that quota. Every parallel tensor op then spends its time in futex
    contention rather than work.

    It is not a small effect. Converting the 810 MB Lightning LoRA -- pure CPU
    tensor work -- ran for over twenty minutes at roughly 0.6 cores of useful
    throughput before this, with 196 threads alive and the main thread parked in
    futex_do_wait. It looked exactly like a deadlock and was diagnosed as one
    twice.

    Set before torch is imported anywhere, because OMP reads its environment at
    library load and ignores changes afterwards.
    """
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota == "max":
            return
        n = max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


_limit_threads()


def face_by_id(face_id):
    for f in ROSTER:
        if f["id"] == face_id:
            return f
    raise SystemExit(f"unknown face {face_id!r}; known: {', '.join(f['id'] for f in ROSTER)}")


def hero_prompt(face):
    """A neutral, well-lit, unambiguous reference shot.

    Deliberately plain: front on, soft light, plain backdrop. This image is the
    source of truth for the identity, so nothing in it should be hard to read --
    dramatic light or an extreme angle hides the very features stage 2 has to
    carry over.

    **Full body, not waist-up.** The first version framed from the waist up, and
    the fuller-build model then came back looking horizontally stretched rather
    than genuinely fuller. Aspect ratio was not the cause -- input and output
    were both exactly 0.667. The reference simply contained no lower body, so
    when a full-body variation was asked for the model had to invent the
    proportions, and approximated "fuller" by widening. No instruction can
    supply information the reference does not carry.
    """
    pronoun = "her" if face["gender"] == "woman" else "his"
    clothes = (CLOTHING_MODEST if face["modest"] else CLOTHING_NEUTRAL)[0]
    return (f"photorealistic photograph of {article(face)} {face['gender']} "
            f"in {pronoun} {face['age']}, {face['build']} build, "
            f"{feature_clause(face)}{face['skin']}, "
            f"{face['hair']}, {face['detail']}, wearing {clothes}, "
            f"full body from head to feet, standing straight, whole figure in "
            f"frame with the feet visible, front view facing camera, "
            f"soft diffused daylight, plain light grey studio backdrop, "
            f"neutral expression, {REALISM_PERSON}")


def _faces_from(arg):
    """--face accepts one id, a comma-separated list, or 'all'."""
    if arg.strip() == "all":
        return list(ROSTER)
    return [face_by_id(f.strip()) for f in arg.split(",") if f.strip()]


def cmd_candidates(args):
    faces = _faces_from(args.face)
    for face in faces:
        os.makedirs(os.path.join(args.out, "heroes", face["id"]), exist_ok=True)
        print(f"{face['id']:20} {hero_prompt(face)[:110]}...")
    if args.dry_run:
        return 0

    import torch
    try:
        from diffusers import ZImagePipeline
    except ImportError:
        print("needs the Z-Image venv:\n"
              "  bash elements/setup_zimage.sh\n"
              "  /opt/zimage-venv/bin/python elements/hero.py candidates ...",
              file=sys.stderr)
        return 3

    # Prefer the bf16 copy if elements/save_bf16.py has been run: same weights,
    # half the bytes off the volume, and no cast on load.
    from save_bf16 import resolved_model_path
    model = resolved_model_path()
    with phase("zimage load", model):
        pipe = ZImagePipeline.from_pretrained(model,
                                              torch_dtype=torch.bfloat16).to("cuda")
    # One model load covers every face asked for. Loading costs minutes and a
    # candidate costs three seconds, so per-face invocations spend nearly all
    # their time reloading the same weights.
    for face in faces:
        out = os.path.join(args.out, "heroes", face["id"])
        prompt = hero_prompt(face)
        for i in range(args.n):
            path = os.path.join(out, "%03d.png" % i)
            if os.path.exists(path) and not args.redo:
                continue
            t = time.time()
            img = pipe(prompt=prompt, height=1152, width=768, num_inference_steps=9,
                       guidance_scale=0.0,
                       generator=torch.Generator("cuda").manual_seed(args.seed + i)).images[0]
            img.save(path)
            print(f"  {face['id']:20} {i:03d}  {time.time() - t:.1f}s", flush=True)

    print("\nLook at them, pick one per face, then run `variations`.")
    return 0


def variation_specs(per_face=30):
    """The same coverage grid catalog.py uses, as instructions rather than prompts."""
    specs = []
    for i in range(per_face):
        specs.append({
            "index": i,
            "angle": ANGLES[i % len(ANGLES)],
            "distance": DISTANCE_MIX[i % len(DISTANCE_MIX)],
            "lighting": LIGHTING[i % len(LIGHTING)],
            "background": BACKGROUNDS_TRAIN[i % len(BACKGROUNDS_TRAIN)],
            "expression": EXPRESSION_MIX[i % len(EXPRESSION_MIX)],
        })
    return specs


def read_picks(path):
    """`face_id index` per line, blank lines and # comments ignored.

    Curation is an input to the build, not a shell argument. Ten pairs already
    make a command line long enough that it has to be typed on one line to
    survive being pasted into a pod shell, and every rerun retypes it. In a file
    it is reviewed, corrected, and versioned like anything else.
    """
    picks = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace("=", " ").split()
            if len(parts) != 2:
                raise SystemExit(f"{path}:{n}: expected `face_id index`, got {line!r}")
            picks[parts[0]] = parts[1]
    if not picks:
        raise SystemExit(f"{path}: no picks")
    return picks


def hero_index_map(arg, faces):
    """--hero accepts one index for every face, `face=index` pairs, or @file.

    Curation does not produce one answer. Each face is judged on its own six
    candidates, so the winning index differs per face -- and a single --hero
    value forces either one run per face, each paying the model load again, or
    a rename dance on the pod. Pairs let all of them go in one pass.

    Unlisted faces are an error rather than a silent default: falling back to
    000 would quietly generate thirty variations of a candidate nobody chose,
    and that is only visible after twelve minutes of GPU time.
    """
    arg = arg.strip()
    if arg.startswith("@"):
        picks = read_picks(arg[1:])
    elif "=" not in arg:
        return {f["id"]: arg for f in faces}
    else:
        picks = {}
        for part in arg.replace(" ", "").split(","):
            if not part:
                continue
            fid, _, idx = part.partition("=")
            picks[fid] = idx
    unknown = sorted(set(picks) - {f["id"] for f in faces})
    if unknown:
        raise SystemExit(f"--hero names faces not in --face: {unknown}")
    missing = sorted({f["id"] for f in faces} - set(picks))
    if missing:
        raise SystemExit("--hero has no pick for: " + ", ".join(missing))
    return picks


def cmd_variations(args):
    # A picks file already names the faces. Repeating them in --face is a second
    # list to keep in sync, and the two disagreeing is a silent no-op or a
    # crash after the model has loaded.
    if args.face is None and args.hero.startswith("@"):
        args.face = ",".join(read_picks(args.hero[1:]))
    if args.face is None:
        raise SystemExit("--face is required unless --hero is a @picks file")
    faces = _faces_from(args.face)
    specs = variation_specs(args.per_face)
    if args.indices:
        wanted = {int(x) for x in args.indices.replace(" ", "").split(",") if x != ""}
        specs = [s for s in specs if s["index"] in wanted]
        if not specs:
            raise SystemExit(f"no spec indices matched {sorted(wanted)}")

    picks = hero_index_map(args.hero, faces)
    jobs = []
    for face in faces:
        hero_path = os.path.join(args.out, "heroes", face["id"],
                                 f"{picks[face['id']]}.png")
        jobs.append((face, hero_path))
        print(f"{face['id']:20} hero={hero_path}")
    print(f"making: {len(specs)} variations x {len(faces)} faces\n")

    # Checked after the summary so --dry-run can review the grid before any
    # hero exists.
    if args.dry_run:
        for s in specs[:4]:
            print(f"  {s['index']:03d}  {s['distance']}, {s['angle']}, {s['lighting']}")
        print("  ...")
        return 0
    missing = [p for _, p in jobs if not os.path.exists(p)]
    if missing:
        raise SystemExit("no hero at:\n  " + "\n  ".join(missing) +
                         "\nrun `candidates` first")

    out = os.path.join(args.out, "models")
    os.makedirs(out, exist_ok=True)

    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    # The stock edit pipeline, not LayeringVTONPipeline. Ours hardcodes the
    # try-on instruction -- "...and change the pose of the person in the first
    # image to the pose in the third image" -- which fights an instruction that
    # is trying to set the pose in words. The base model is a general
    # multi-reference editor and that is what this stage needs.
    # The transformer is loaded on its own, then handed to the pipeline.
    #
    # Three placements were tried on the pod and only this one survives:
    #
    #   .to("cuda")            from_pretrained materialises all 57.7 GB in host
    #                          RAM and then copies it across. Over the network
    #                          volume that read pattern stalls: the process sits
    #                          in D state with 5.5 GB placed and never recovers.
    #   device_map="balanced"  loads fine, but accelerate installs dispatch
    #                          hooks on every module, and PEFT injecting an
    #                          adapter into hooked modules deadlocks -- 55 GB
    #                          resident, 0% utilisation, futex_do_wait, twenty
    #                          minutes with no progress.
    #   device_map={"": 0}     rejected at pipeline level with ValueError; a
    #                          pipeline is not a model and takes only the
    #                          named strategies.
    #
    # But the dict form is accepted by the *model* loader, which is what
    # pipeline.py has always used and what has never hung. So every large
    # component is loaded that way first and the pipeline is assembled around
    # them.
    #
    # Both of them, not just the transformer. Placing only the transformer got
    # 40 GB onto the GPU and then stalled again in D state on the text encoder
    # -- 8.3B parameters is not "small enough for the plain path", and the
    # stall is about the read pattern, not the size.
    model_id = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509")
    from diffusers import QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration

    with phase("transformer load", "device_map"):
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
            device_map={"": 0})
    with phase("text encoder load", "device_map"):
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
            device_map={"": 0})
    with phase("pipeline assemble"):
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=torch.bfloat16)
    # The VAE is 0.2 GB and loads either way; the two already placed would be
    # copied a second time.
    pipe.vae.to("cuda")
    print("  pipeline ready", flush=True)

    steps, cfg = 40, 4.0
    if args.lightning:
        # Not pipe.load_lora_weights(): on a 20B pipeline already placed by
        # device_map it hangs indefinitely -- the model loads to 55 GB and then
        # sits there, with the LoRA already cached locally so nothing is being
        # downloaded. Reproduced twice.
        #
        # The PEFT path in pipeline.py drives the transformer directly and has
        # run many times, so use that. Same weights, same result, no hook
        # rewriting across a device_map'd model.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        from diffusers.utils import convert_unet_state_dict_to_peft
        from pipeline import (LIGHTNING_REPO, LIGHTNING_WEIGHTS,
                              _lora_config_from_state_dict, _strip_lora_prefix)

        _t_lightning = time.time()
        print("  lightning: fetching", flush=True)
        path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_WEIGHTS[args.lightning])
        from localfile import cached_local
        path = cached_local(path)
        print("  lightning: reading", flush=True)
        sd = QwenImageEditPlusPipeline.lora_state_dict(load_file(path))
        sd, prefix = _strip_lora_prefix(sd)
        sd = convert_unet_state_dict_to_peft(sd)
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        lcfg = _lora_config_from_state_dict(sd)
        print(f"  lightning: rank {lcfg.r}, {len(lcfg.target_modules)} module types"
              f" (prefix {prefix!r})", flush=True)
        print("  lightning: injecting adapter", flush=True)
        pipe.transformer.add_adapter(lcfg, adapter_name="lightning")
        print("  lightning: loading weights", flush=True)
        set_peft_model_state_dict(pipe.transformer, sd, adapter_name="lightning")
        record("lightning adapter", time.time() - _t_lightning,
               f"{args.lightning}-step")
        print(f"  lightning: ready ({time.time() - _t_lightning:.1f}s)", flush=True)
        steps, cfg = args.lightning, 1.0
    print(f"  sampling: {steps} steps, cfg {cfg}\n", flush=True)

    fields = ["id", "angle", "distance", "lighting", "background",
              "expression", "seconds", "error"]
    ok = failed = 0
    _t_run = time.time()

    for face, hero_path in jobs:
      _t_face = time.time()
      clothes = (CLOTHING_MODEST if face["modest"] else CLOTHING_NEUTRAL)
      hero = Image.open(hero_path).convert("RGB")
      print(f"\n--- {face['id']} ---", flush=True)

      # A partial run must not wipe the log of the full one. Read what is
      # there, replace only the rows this run touches, write it all back.
      log_path = os.path.join(out, f"variations__{face['id']}.csv")
      existing = {}
      if os.path.exists(log_path):
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    existing[row["id"]] = row

      for s in specs:
        path = os.path.join(out, "%s__%03d.png" % (face["id"], s["index"]))
        if os.path.exists(path) and not args.redo:
            continue
        outfit = clothes[s["index"] % len(clothes)]
        # Naming the attributes to preserve, not just "the same person".
        #
        # The first pass said only "same face, skin tone and hair" and left the
        # rest to the reference. Review found mild but consistent drift in
        # exactly what went unnamed: a fuller build came back slightly slimmed,
        # and hair shifted. The face, which *was* named, held.
        #
        # This does not contradict leaving the description out -- these values
        # are the ones the hero was generated from, so restating them reinforces
        # the reference instead of competing with it. Only invented detail would
        # fight it.
        preserve = (f"the same face, {face['skin']}, {face['hair']}, "
                    f"{face['detail']}, the same {face['build']} body build")
        instruction = (f"Keep the same person: {preserve}. "
                       f"Do not slim, reshape or beautify the body or face. "
                       f"Show them {s['distance']}, {s['angle']}, {s['lighting']}, "
                       f"{s['background']}, {s['expression']} expression, "
                       f"wearing {outfit}. Photorealistic, natural skin texture.")
        t = time.time()
        try:
            img = pipe(
                image=[hero],
                prompt=instruction,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                generator=torch.Generator("cuda").manual_seed(args.seed + s["index"]),
            ).images[0]
            img.save(path)
            elapsed, err, ok = time.time() - t, "", ok + 1
            print(f"  {s['index']:03d}  {elapsed:5.1f}s  {s['distance']}, {s['angle'][:28]}")
        except Exception as exc:
            elapsed, err, failed = time.time() - t, f"{type(exc).__name__}: {exc}", failed + 1
            print(f"  {s['index']:03d}  FAILED  {err}", file=sys.stderr)
        existing[os.path.basename(path)] = dict(
            id=os.path.basename(path), angle=s["angle"], distance=s["distance"],
            lighting=s["lighting"], background=s["background"],
            expression=s["expression"], seconds=round(elapsed, 1), error=err)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for k in sorted(existing):
                wr.writerow({c: existing[k].get(c, "") for c in fields})

      record("face", time.time() - _t_face, face["id"])

    elapsed_run = time.time() - _t_run
    record("generate", elapsed_run, f"{ok} images, {failed} failed")
    per = elapsed_run / ok if ok else 0
    print(f"\n{ok} made, {failed} failed in {elapsed_run / 60:.1f} min "
          f"({per:.1f}s each) -> {out}")
    print(f"Now curate:  python elements/curate.py --group {face['id']}")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("candidates", help="stage 1: text -> candidate faces")
    c.add_argument("--face", required=True)
    c.add_argument("--n", type=int, default=12)
    c.add_argument("--out", default=os.path.join(HERE, "out"))
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--redo", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(fn=cmd_candidates)

    v = sub.add_parser("variations", help="stage 2: hero -> the coverage grid")
    v.add_argument("--face", default=None,
               help="ids, or omitted when --hero is a @picks file")
    v.add_argument("--hero", default="000",
               help="one index for all faces (003), or per-face "
                    "pairs (f_cauz_20s_avg=002,m_slav_30s_avg=000)")
    v.add_argument("--per-face", type=int, default=30)
    v.add_argument("--out", default=os.path.join(HERE, "out"))
    v.add_argument("--lora", default="./weights")
    v.add_argument("--lightning", type=int, choices=[4, 8], default=8)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--indices", default=None,
                   help="regenerate only these grid positions, e.g. 3,11,23,27 -- "
                        "used with --redo when one axis value turns out bad")
    v.add_argument("--redo", action="store_true")
    v.add_argument("--dry-run", action="store_true")
    v.set_defaults(fn=cmd_variations)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

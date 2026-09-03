#!/usr/bin/env python3
"""Making a model that belongs to one seller, behind HTTP, on the pod.

    /opt/zimage-venv/bin/python -m service.zimage_server

A second process on the same machine as pod_server, and it has to be a second
process: Z-Image needs diffusers from source, which wants huggingface_hub 1.x,
while the try-on stack's transformers caps it below 1.0. Every attempt to find
one set of versions that satisfies both failed. Two interpreters settle it, and
80 GB of VRAM holds both models with room to spare -- 15.8 for the quantised
try-on model, about 19 for this one.

Bound to loopback for the same reason as pod_server: the pod has a public
address, so a service on 0.0.0.0 here is a service on the internet.

Port 8011, not 8001. RunPod's own nginx listens on 8001 on this image, and it
answers /health by proxying somewhere -- so the panel asked "is the model maker
up?", got a cheerful yes from nginx relaying the try-on server, and never
started this one. Two tools were dead and everything said it was fine.

This is the half of the product that listens to words. The try-on model reads
the garment image and ignores the text entirely (docs/CONTROLS.md); this one is
text to image and does nothing else, which is why "make a new model" can offer
choices at all and try-on cannot.
"""
import io
import os
import sys
import threading
import time

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "elements"))

# The bf16 copy if elements/save_bf16.py has been run: the published repository
# is fp32 and 30.6 GB, so loading it read twice the bytes off the volume and
# threw half away on arrival. 584 s against 1 s.
MODEL_PATH = os.environ.get("ZIMAGE_PATH", "")
STEPS = int(os.environ.get("ZIMAGE_STEPS", "9"))
WIDTH = int(os.environ.get("ZIMAGE_W", "768"))
HEIGHT = int(os.environ.get("ZIMAGE_H", "1152"))

# Guidance stays at zero and is not offered. Z-Image Turbo is distilled for it;
# measured on the pod at 1.5 the picture darkens, at 3.0 the blacks crush, and
# at 5.0 it burns out entirely -- and every step above zero costs 6.4s against
# 3.8s for no gain. The prompt is followed perfectly well at zero: asked for a
# blond man it produced one.
SCENE_GUIDANCE = 0.0

app = FastAPI(title="Lookzi model maker")

_pipe = None
_load_seconds = None
_error = None
_gpu = threading.Lock()
_stats = {"served": 0, "failed": 0, "seconds": 0.0}


def _resolve_path():
    if MODEL_PATH:
        return MODEL_PATH
    try:
        from save_bf16 import resolved_model_path
        return resolved_model_path()
    except Exception:                                         # noqa: BLE001
        return "Tongyi-MAI/Z-Image-Turbo"


def load():
    global _pipe, _load_seconds, _error
    import torch
    from diffusers import ZImagePipeline
    path = _resolve_path()
    t = time.time()
    print(f"[zimage] loading {path}", flush=True)
    try:
        _pipe = ZImagePipeline.from_pretrained(
            path, torch_dtype=torch.bfloat16).to("cuda")
        _load_seconds = round(time.time() - t, 1)
        print(f"[zimage] ready in {_load_seconds}s", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        _error = f"{type(exc).__name__}: {exc}"
        print(f"[zimage] failed to load: {_error}", flush=True)


@app.on_event("startup")
def _startup():
    threading.Thread(target=load, daemon=True).start()


@app.get("/health")
def health():
    return {"ready": _pipe is not None, "error": _error,
            "model": _resolve_path(), "steps": STEPS,
            "load_seconds": _load_seconds, "busy": _gpu.locked(),
            "served": _stats["served"], "failed": _stats["failed"],
            "mean_seconds": round(_stats["seconds"] / _stats["served"], 2)
            if _stats["served"] else None}


# What a made-to-order model needs whatever was asked for. hero_prompt carries
# its own version of this; a customer writing in their own words gets it added,
# because "a very slim Uzbek girl" is a description of a person and not of a
# photograph, and the try-on stage that may follow needs the whole figure.
CREATE_FRAMING = ("full body from head to feet, standing straight, whole figure "
                  "in frame with the feet visible, front view facing camera, "
                  "soft diffused daylight, plain light grey studio backdrop, "
                  "neutral expression, photorealistic, sharp focus")

# Asked for a person and told nothing about clothes, the model picks the least
# it can get away with: "white skin, uzbek girl, very slim" came back in a
# leotard, barefoot. Nobody wrote that and nobody wants it. hero_prompt has
# always dressed the roster from CLOTHING_NEUTRAL for exactly this reason; a
# written description was the one path that skipped it.
#
# Both halves named, because "everyday clothes" was not enough: it produced a
# top and no trousers, and "very slim" in the description pulls hard toward
# showing the figure. This is the shape CLOTHING_NEUTRAL uses for the roster,
# which has never had the problem.
#
# Naming garments is what the sleeveless attempt did wrong, so the difference
# matters: "sleeveless" is an unusual attribute and it leaked into everything
# else the person wore. A plain t-shirt and plain trousers are the default a
# photograph would have anyway, and they come last, so "in a long red coat"
# still comes back in a long red coat. Measured, not assumed.
CLOTHED = "wearing a plain t-shirt and plain trousers and flat shoes"


@app.post("/create")
def create(gender: str = Form("woman"), age: str = Form("20s"),
           build: str = Form("average"), look: str = Form("uzbek"),
           modest: str = Form("false"), seed: int = Form(0),
           prompt: str = Form("")):
    """One model, made from a handful of choices -- or from your own words.

    The choices are the ones a seller has an opinion about, and everything they
    do not choose is varied by seed, so asking twice gives two people rather
    than the same person twice.

    A written description replaces the choices when there is one. The studio
    offers the box, so it has to reach here: it was being recorded with the job
    and never sent, and a customer who wrote "white skin, uzbek girl, very
    slim" got whatever the five dropdowns happened to say instead.
    """
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    if _pipe is None:
        raise HTTPException(503, "the model is still loading")

    import torch
    from catalog import new_face
    from hero import hero_prompt

    face = new_face(gender=gender, age=age, build=build, look=look,
                    modest=str(modest).lower() in ("1", "true", "yes"),
                    seed=int(seed))
    written = (prompt or "").strip()
    if len(written) > 600:
        raise HTTPException(413, "that description is longer than the model reads")
    full = (f"{written}, {CLOTHED}, {CREATE_FRAMING}" if written
            else hero_prompt(face))

    started = time.time()
    with _gpu:
        try:
            img = _pipe(prompt=full, height=HEIGHT, width=WIDTH,
                        num_inference_steps=STEPS, guidance_scale=SCENE_GUIDANCE,
                        generator=torch.Generator("cuda").manual_seed(int(seed))
                        ).images[0]
        except Exception as exc:                              # noqa: BLE001
            _stats["failed"] += 1
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(
        content=buf.getvalue(), media_type="image/png",
        # The prompt comes back with the picture. A model made to order is a
        # thing the seller may want again, and the only way to make the same
        # person twice is to know what was asked for.
        headers={"X-Seconds": str(elapsed), "X-Width": str(img.width),
                 "X-Height": str(img.height), "X-Face": face["id"],
                 "X-Prompt": prompt[:900].replace("\n", " ")})


# What every scene needs whatever the customer wrote. The try-on stage that
# follows needs a whole person, front on, with the feet in frame: it cannot
# dress a figure the picture does not contain, and "no instruction can supply
# information the reference does not carry" is written in hero.py for the same
# reason. Appended rather than prepended, so the customer's own words lead.
SCENE_FRAMING = ("full body from head to feet, whole figure in frame with the "
                 "feet visible, the person close to the camera and filling most "
                 "of the frame, standing, facing the camera, "
                 "photorealistic, sharp focus, natural light")

# Same reason as CLOTHED above, and it matters more here: whatever this
# person is wearing, the try-on stage puts the garment on over it, so a
# figure that arrives in cycling shorts stays in cycling shorts below the
# waist. Plain clothes are the ones a garment sits on without argument.
# SCENE_FRAMING + CLOTHED, in that order, so the customer's words lead.

# There was a "wearing a plain fitted sleeveless top" here and it is gone.
#
# It was aimed at the layering fault -- the try-on stage puts a garment on over
# what the person already wears, and when the new sleeves are shorter the old
# ones show. A sleeveless base has nothing to poke out from under it, and in
# isolation it worked.
#
# In use it did not. "Sleeveless" spread to the whole outfit: a prompt asking
# for a woman in a bright studio came back in cycling shorts and bare feet, and
# it did that every time. The fault it was guarding against is intermittent --
# four deliberate attempts to reproduce it all came back clean. Trading an
# occasional fault for a constant one is a bad trade, and this was one.
#
# The layering fault is still open. docs/CONTROLS.md records what has been
# tried, and re-dressing the roster rather than instructing the prompt is the
# next thing worth trying.



@app.post("/prompt")
def from_prompt(prompt: str = Form(...), seed: int = Form(0),
                framing: str = Form("1")):
    """A person and a scene, from the customer's own words.

    This is what the try-on model cannot do. It reads the garment image and
    ignores text entirely, so a prompt box in front of it would be a control
    that does nothing (docs/CONTROLS.md). Put the prompt in front of *this*
    model instead, and hand what it makes to the try-on stage as the person.
    """
    if _error:
        raise HTTPException(503, f"the model did not load: {_error}")
    if _pipe is None:
        raise HTTPException(503, "the model is still loading")
    text = (prompt or "").strip()
    if not text:
        raise HTTPException(400, "write something for it to make")
    if len(text) > 800:
        raise HTTPException(413, "that prompt is longer than the model reads")

    import torch
    full = f"{text}, {CLOTHED}, {SCENE_FRAMING}" if framing != "0" else text
    started = time.time()
    with _gpu:
        try:
            img = _pipe(prompt=full, height=HEIGHT, width=WIDTH,
                        num_inference_steps=STEPS, guidance_scale=SCENE_GUIDANCE,
                        generator=torch.Generator("cuda").manual_seed(int(seed))
                        ).images[0]
        except Exception as exc:                              # noqa: BLE001
            _stats["failed"] += 1
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    elapsed = round(time.time() - started, 2)
    _stats["served"] += 1
    _stats["seconds"] += elapsed

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"X-Seconds": str(elapsed), "X-Width": str(img.width),
                             "X-Height": str(img.height),
                             "X-Prompt": full[:900].replace("\n", " ")})


def main():
    import uvicorn
    port = int(os.environ.get("ZIMAGE_PORT", "8011"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What kind of garment is in this photograph: a top, a bottom, or a one-piece.

The packshot needs to know. Asked to retouch a skirt photographed spread on the
floor, the model gave back a sundress with shoulder straps -- a category error,
not a detail one: a buyer who orders a dress and receives a skirt returns it.
Telling the model "this photograph shows a skirt, the result must still be a
skirt" fixed it on the first attempt. Forbidding it -- "do not add straps" --
made it worse, because a generative model reads the noun and not the negation.

So the fix needs the category, and asking the seller for it on every upload is
a button they will get wrong or ignore. This works it out instead.

CLIP zero-shot, and on the CPU. The card already holds two models with three
gigabytes spare, and this is a small classifier that runs in well under a
second on a processor -- there is no reason for it to compete for VRAM.

Whether it is good enough is a question with an answer rather than an opinion:
the hundred garments in `test products/` are filed by category, so the folder
names are ground truth. See tools/garment_type_eval.py. If it cannot manage,
the fallback is the button.
"""
import os

MODEL = os.environ.get("GARMENT_TYPE_MODEL", "openai/clip-vit-base-patch32")

# Several wordings per class and the best match wins, rather than one phrase
# per class averaged. A blouse and a padded jacket are both tops and they do
# not sit near each other; asking whether a picture is nearer to *any* top
# beats asking whether it is near the average of all tops.
PHRASES = {
    "tops": [
        "a photo of a t-shirt", "a photo of a shirt", "a photo of a blouse",
        "a photo of a jacket", "a photo of a sweater", "a photo of a hoodie",
        "a photo of a cardigan", "a photo of a coat",
    ],
    "bottoms": [
        "a photo of a skirt", "a photo of trousers", "a photo of jeans",
        "a photo of shorts", "a photo of leggings",
        "a photo of a pair of trousers laid flat",
    ],
    "one-pieces": [
        "a photo of a dress", "a photo of a jumpsuit",
        "a photo of a romper", "a photo of an evening gown",
    ],
}

# What each class becomes in the instruction. Positive and specific, because
# that is the form the model follows.
SENTENCE = {
    "tops": "This photograph shows an upper-body garment. The result must "
            "still be that garment: nothing below the waist, no skirt, no "
            "trousers.",
    "bottoms": "This photograph shows a lower-body garment such as a skirt or "
               "trousers. The result must still be that garment: nothing above "
               "the waistband, no straps, no bodice, no sleeves.",
    "one-pieces": "This photograph shows a one-piece garment such as a dress. "
                  "The result must still be one piece.",
}

_model = None
_proc = None
_text = None


def _load():
    global _model, _proc, _text
    if _model is not None:
        return
    import torch
    from transformers import CLIPModel, CLIPProcessor
    _model = CLIPModel.from_pretrained(MODEL).eval()
    _proc = CLIPProcessor.from_pretrained(MODEL)
    flat = [(k, p) for k, ps in PHRASES.items() for p in ps]
    with torch.no_grad():
        toks = _proc(text=[p for _, p in flat], return_tensors="pt", padding=True)
        emb = _model.get_text_features(**toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    _text = ([k for k, _ in flat], emb)


def classify(img):
    """Returns the class, the margin over the runner-up, and every score.

    The margin matters more than the winner. A picture the classifier is torn
    between is one where a wrong sentence in the instruction would do damage,
    and it is better to say nothing then than to guess.
    """
    import torch
    _load()
    labels, temb = _text
    with torch.no_grad():
        inp = _proc(images=img.convert("RGB"), return_tensors="pt")
        emb = _model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sim = (emb @ temb.T)[0]

    best = {}
    for label, score in zip(labels, sim.tolist()):
        if score > best.get(label, -1e9):
            best[label] = score
    order = sorted(best.items(), key=lambda kv: -kv[1])
    top, second = order[0], order[1]
    return {"kind": top[0], "margin": round(top[1] - second[1], 4),
            "scores": {k: round(v, 4) for k, v in best.items()}}


def sentence_for(img, min_margin=0.01):
    """The line to put in front of the retouch instruction, or nothing.

    Nothing is a real answer. A confident wrong category is worse than no
    category: it tells the model to turn a dress into a skirt.
    """
    got = classify(img)
    if got["margin"] < min_margin:
        return "", got
    return SENTENCE[got["kind"]], got

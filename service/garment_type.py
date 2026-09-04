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
# The noun matters and the class does not. Told "this photograph shows a
# lower-body garment such as a skirt or trousers", the model still returned a
# sundress; told "this photograph shows a skirt", it returned a skirt. A
# generative model answers a concrete noun and shrugs at a category. So the
# winning phrase is kept, not only the class it belongs to.
PHRASES = {
    "tops": [
        ("a t-shirt", "a photo of a t-shirt"), ("a shirt", "a photo of a shirt"),
        ("a blouse", "a photo of a blouse"), ("a jacket", "a photo of a jacket"),
        ("a sweater", "a photo of a sweater"), ("a hoodie", "a photo of a hoodie"),
        ("a cardigan", "a photo of a cardigan"), ("a coat", "a photo of a coat"),
    ],
    "bottoms": [
        ("a skirt", "a photo of a skirt"), ("trousers", "a photo of trousers"),
        ("jeans", "a photo of jeans"), ("shorts", "a photo of shorts"),
        ("leggings", "a photo of leggings"),
        ("trousers", "a photo of a pair of trousers laid flat"),
    ],
    "one-pieces": [
        ("a dress", "a photo of a dress"), ("a jumpsuit", "a photo of a jumpsuit"),
        ("a romper", "a photo of a romper"),
        ("an evening gown", "a photo of an evening gown"),
    ],
}

# What follows the noun. Positive and specific: the form the model follows.
RULE = {
    "tops": "The result must still be {noun} and nothing else: no skirt, no "
            "trousers, nothing below the waist.",
    "bottoms": "The result must still be {noun} and nothing else: nothing above "
               "the waistband, no straps, no bodice, no sleeves.",
    "one-pieces": "The result must still be {noun}, one piece, the same length.",
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
    flat = [(k, noun, phrase) for k, ps in PHRASES.items() for noun, phrase in ps]
    with torch.no_grad():
        toks = _proc(text=[p for _, _, p in flat], return_tensors="pt",
                     padding=True)
        emb = _model.get_text_features(**toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    _text = ([(k, noun) for k, noun, _ in flat], emb)


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
    for (label, noun), score in zip(labels, sim.tolist()):
        if score > best.get(label, (-1e9, ""))[0]:
            best[label] = (score, noun)
    order = sorted(best.items(), key=lambda kv: -kv[1][0])
    top, second = order[0], order[1]
    return {"kind": top[0], "noun": top[1][1],
            "margin": round(top[1][0] - second[1][0], 4),
            "scores": {k: round(v[0], 4) for k, v in best.items()}}


def sentence_for(img, min_margin=0.01):
    """The line to put in front of the retouch instruction, or nothing.

    Nothing is a real answer. A confident wrong category is worse than no
    category: it tells the model to turn a dress into a skirt.
    """
    got = classify(img)
    if got["margin"] < min_margin:
        return "", got
    noun = got["noun"]
    return (f"This photograph shows {noun}. "
            + RULE[got["kind"]].format(noun=noun)), got


# What the seller said, in the model's words.
#
# The classifier is 94% right and the person holding the garment is always
# right, so when they have answered, their answer wins and this file's guessing
# is skipped. But the class alone is not what the model responds to: told "a
# lower-body garment", it still returned a sundress; told "a skirt", it
# returned a skirt. A generative model answers a concrete noun and shrugs at a
# category.
#
# So the work is split at the seam where each side is strong. The seller gives
# the category, which is the part a classifier gets wrong across categories --
# a skirt read as a dress. The classifier picks the noun inside that category,
# which is the part it is good at and where a mistake is survivable: a pair of
# jeans called trousers still comes back as trousers.
KINDS = {"upper": "tops", "lower": "bottoms", "overall": "one-pieces"}


def noun_within(img, kind):
    """The best-fitting noun from one category, never leaving it.

    Falls back to the category's first noun if CLIP cannot be loaded, because a
    plain "a skirt" is worth far more than no sentence at all -- and the whole
    point of the seller answering is that the sentence gets written.
    """
    cls = KINDS.get(kind, kind)
    if cls not in PHRASES:
        return None
    try:
        import torch
        _load()
        labels, temb = _text
        with torch.no_grad():
            inp = _proc(images=img.convert("RGB"), return_tensors="pt")
            emb = _model.get_image_features(**inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            sim = (emb @ temb.T)[0]
        best, noun = -1e9, None
        for (label, n), score in zip(labels, sim.tolist()):
            if label == cls and score > best:
                best, noun = score, n
        return noun
    except Exception:                                         # noqa: BLE001
        return PHRASES[cls][0][0]


def sentence_for_kind(kind, img):
    """The line for a category the seller chose. No margin, no abstaining.

    `sentence_for` abstains when it is unsure, because a confident wrong guess
    would turn a dress into a skirt. There is nothing to be unsure about here.
    """
    cls = KINDS.get(kind, kind)
    if cls not in RULE:
        return "", {"kind": cls, "noun": None, "margin": None}
    noun = noun_within(img, cls) or PHRASES[cls][0][0]
    return (f"This photograph shows {noun}. " + RULE[cls].format(noun=noun),
            {"kind": cls, "noun": noun, "margin": None})

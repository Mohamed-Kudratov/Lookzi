#!/usr/bin/env python3
"""Is the garment that came out the garment that went in?

A generative packshot is worth having and it is worth being afraid of. It can
straighten a dress that was hanging crooked, invent the neckline the hanger was
covering, and put the whole thing on white -- and it can just as easily give
back a *different* dress: sleeves a little shorter, a print regenerated rather
than reproduced, a button quietly gone. In a catalogue that is not a cosmetic
problem. It is a return, and a seller who stops trusting the tool.

So nothing generative ships without this in front of it. The rule the product
follows is not "retry with another seed and hope"; it is:

    generate -> measure -> if it drifted, retry once
                        -> if it drifted again, hand back the plain cut-out

The customer's worst case is then today's output, which is honest, rather than
a garment that does not exist.

What it is compared against
---------------------------

The cut-out, not the photograph. The first version of this compared the
generated packshot to the raw phone photo and it did not work at all: a
faithful cut-out of a dress scored *worse* than the same dress rendered onto
an invented woman. Two reasons, and both are instructive.

The garment's outline in a phone photo is exactly what we do not have -- it is
the thing segmentation is for -- so it was guessed from the border colour, and
on a wardrobe door the guess takes in half the wall. And the two pictures were
framed differently, so even the parts that were garment were not the same
parts.

The cut-out fixes both. It carries the garment's own pixels and its own
outline, it is free, and it is already the fallback the gate hands back when a
generation is rejected. So it is the reference, and everything is cropped to
the garment before anything is measured.

What is measured, and why these three
-------------------------------------

**colour** catches the loudest failure and the commonest: navy coming back
grey, a print's palette shifting. Compared as a distribution, so a garment
photographed under a yellow bulb and corrected is not punished for the
correction.

**print** catches the failure that is easy to miss and expensive to ship: a
dense pattern that has been *redrawn* instead of copied. It looks right in a
thumbnail and wrong beside the real garment. Measured from the statistics of
local detail, and at two scales, because a regenerated print usually gets the
density roughly right and the scale slightly wrong.

**shape** catches sleeve length, hem length and neckline. It is the measure
that must be read differently depending on what was asked for, and that is the
point of `mode`: a cut-out may not change the silhouette at all, while a
ghost-mannequin packshot changes it deliberately -- straightening the garment
is the whole job. Judging both by one number would either forbid the work or
wave through a dress that came back a different length.

Numpy and Pillow only, so this runs in the web tier, on the pod, and on a
laptop with no GPU. These are proxies for a judgement a person makes in a
second, and they are chosen because each one names a failure that has actually
been seen. An embedding metric -- DINOv2 or CLIP over the garment crop -- is
the natural fourth and needs a GPU; it belongs here once there is one, and it
should be validated against these rather than replacing them, because a single
opaque number tells you that something changed and not what.
"""
import numpy as np
from PIL import Image

# Anything this far from the paper it was placed on is the garment. Generous:
# a white shirt on a white ground is a genuinely hard case and is better
# handled by passing a real mask than by tuning this.
GROUND_DISTANCE = 40.0
# The size everything is compared at. Large enough that a print survives,
# small enough that a hundred comparisons take seconds.
WORK = (320, 427)


def _load(src):
    if isinstance(src, Image.Image):
        im = src
    else:
        im = Image.open(src)
    if im.mode == "RGBA":
        # A cut-out carries its own mask and it is better than anything
        # guessed from the colours.
        rgba = np.asarray(im.resize(WORK, Image.LANCZOS)).astype(np.float32)
        return rgba[..., :3], rgba[..., 3] > 128
    im = im.convert("RGB")
    return np.asarray(im.resize(WORK, Image.LANCZOS)).astype(np.float32), None


def _mask(rgb, given):
    """Which pixels are the garment.

    A cut-out says so itself. A photograph does not, so the ground is taken
    from the border -- where the subject almost never is -- and everything
    well clear of it is the garment. On a cluttered wall this is rough, and
    rough is enough for a comparison of distributions.
    """
    if given is not None:
        return given
    h, w = rgb.shape[:2]
    m = max(2, int(min(h, w) * 0.06))
    edge = np.concatenate([rgb[:m].reshape(-1, 3), rgb[-m:].reshape(-1, 3),
                           rgb[:, :m].reshape(-1, 3), rgb[:, -m:].reshape(-1, 3)])
    ground = edge.mean(axis=0)
    return np.linalg.norm(rgb - ground, axis=2) > GROUND_DISTANCE


def _colour_hist(rgb, mask, bins=6):
    px = rgb[mask]
    if len(px) < 50:
        return None
    idx = np.clip((px / 256.0 * bins).astype(int), 0, bins - 1)
    flat = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
    h = np.bincount(flat, minlength=bins ** 3).astype(np.float32)
    return h / h.sum()


def _detail(rgb, mask, step=4):
    """How busy the surface is, as a distribution, at two scales.

    A print that was redrawn rather than reproduced tends to land close on
    average and wrong in the spread: the same amount of pattern, arranged with
    a different grain. Two scales, because getting the density right and the
    size of the motif wrong is the usual shape of the mistake.
    """
    grey = rgb.mean(axis=2)
    out = []
    for scale in (1, 2):
        g = grey[::scale, ::scale]
        m = mask[::scale, ::scale]
        h, w = g.shape
        h, w = h - h % step, w - w % step
        if h < step or w < step:
            return None
        tiles = g[:h, :w].reshape(h // step, step, w // step, step)
        cover = m[:h, :w].reshape(h // step, step, w // step, step)
        keep = cover.mean(axis=(1, 3)) > 0.6
        if keep.sum() < 20:
            return None
        sd = tiles.std(axis=(1, 3))[keep]
        # A distribution rather than a mean: the mean alone cannot tell a
        # plain navy dress from one covered in small navy leaves.
        hist, _ = np.histogram(sd, bins=24, range=(0, 60), density=True)
        out.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(out)


def _silhouette(mask, size=64):
    """The outline, normalised, so position and scale do not count as change."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return None
    box = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.asarray(Image.fromarray((box * 255).astype(np.uint8))
                      .resize((size, size), Image.BILINEAR)) > 127


def _crop_to_garment(rgb, mask):
    """Frame both pictures on the garment, so padding is not a difference.

    Without this the same dress cut out at 768x1024 and generated at 1024x1024
    compares as two different garments, because a third of one image is white
    margin and the comparison is over the whole frame.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return rgb, mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub_rgb = Image.fromarray(rgb[y0:y1, x0:x1].astype(np.uint8)).resize(
        WORK, Image.LANCZOS)
    sub_m = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(
        WORK, Image.BILINEAR)
    return np.asarray(sub_rgb).astype(np.float32), np.asarray(sub_m) > 127


def compare(reference, candidate):
    """Three numbers, each 0 (identical) to 1 (nothing in common).

    `reference` is the cut-out of the garment; `candidate` is what a
    generative pass produced from the same photograph.
    """
    a_rgb, a_alpha = _load(reference)
    b_rgb, b_alpha = _load(candidate)
    a_m, b_m = _mask(a_rgb, a_alpha), _mask(b_rgb, b_alpha)
    a_rgb, a_m = _crop_to_garment(a_rgb, a_m)
    b_rgb, b_m = _crop_to_garment(b_rgb, b_m)

    out = {}
    ha, hb = _colour_hist(a_rgb, a_m), _colour_hist(b_rgb, b_m)
    out["colour"] = (round(float(np.abs(ha - hb).sum() / 2), 3)
                     if ha is not None and hb is not None else None)

    da, db = _detail(a_rgb, a_m), _detail(b_rgb, b_m)
    out["print"] = (round(float(np.abs(da - db).sum() / 2 / 2), 3)
                    if da is not None and db is not None else None)

    sa, sb = _silhouette(a_m), _silhouette(b_m)
    if sa is not None and sb is not None:
        union = (sa | sb).sum()
        out["shape"] = (round(1.0 - float((sa & sb).sum()) / float(union), 3)
                        if union else None)
    else:
        out["shape"] = None
    return out


# Where each measure stops being acceptable. Set from what has actually been
# seen rather than from taste, and they are deliberately different per mode:
#
#   cut-out       nothing may change. The pixels are the garment's own.
#   packshot      the silhouette is allowed to change, because straightening
#                 the garment is what was asked for. Colour and print are not.
#
# These are a starting point and they are meant to be re-set once there are
# twenty judged examples. A threshold nobody has checked against a human eye
# is a number pretending to be a decision.
# Measured on forty of our own garments: a cut-out against itself scores 0.000
# on all three, and against a different garment 0.833 / 0.362 / 0.303 at the
# median. So colour separates cleanly and the other two do not separate two
# plain garments at all -- which is expected, since two black t-shirts have the
# same silhouette and the same texture. They earn their place by catching
# specific failures, not by telling garments apart.
#
# These are provisional and they are the weakest part of this file. Setting
# them properly needs twenty generated packshots of garments we know, judged by
# eye, and that needs a GPU. Until then they are wide enough to catch a garment
# that has clearly changed and no narrower.
LIMITS = {
    "cutout":   {"colour": 0.10, "print": 0.15, "shape": 0.10},
    "packshot": {"colour": 0.45, "print": 0.50, "shape": None},
}


def verdict(scores, mode="packshot"):
    """keep, retry, or give back the cut-out.

    Returns the decision and what drove it, because "it drifted" is not
    something to tell a seller and "the print does not match" is.
    """
    limits = LIMITS.get(mode, LIMITS["packshot"])
    failed = [k for k, cap in limits.items()
              if cap is not None and scores.get(k) is not None
              and scores[k] > cap]
    return {"ok": not failed, "failed": failed,
            "why": "" if not failed else
                   ", ".join(f"{k} {scores[k]} over {limits[k]}" for k in failed)}


# ---------------------------------------------------------------------------
# is it a packshot at all
#
# Fidelity and composition are different questions and one does not answer the
# other. The clearest failure so far scored 0.24 on all three measures above --
# mild, nearly passing -- and it was a photograph of a woman wearing the dress.
# The garment really was the same garment; what was wrong was everything else.
#
# Only half of this is built, and the half that is not is worth recording.
#
# Detecting the person by skin colour was tried and does not work. It found
# both offending images easily (0.33 and 0.30 of the subject in skin tones
# against 0.04 for a clean cut-out) and then rejected fifty of our own hundred
# cut-outs, because beige, tan and khaki garments are skin-coloured and no
# threshold separates a camel coat from an arm. The number could have been
# moved until the sample passed; that is fitting a measure to its test set.
#
# The right instrument is a person detector, and there is already one on the
# pod: DWPose ships a YOLOX person model and the try-on pipeline loads it for
# every job. That is the next thing to wire in, and it needs the GPU, so it
# waits for one.
#
# What is here is the ground check, which does work: 0.0 across every cut-out
# we have made, 43.8 on a phone photograph of a dress against a wall.


def looks_like_a_packshot(candidate, ground_limit=12.0):
    """Whether what came back sits on clean ground.

    Half a check. It cannot yet tell you a person walked into the frame -- see
    the note above -- so it must not be read as one.
    """
    rgb, alpha = _load(candidate)
    h, w = rgb.shape[:2]
    edge_w = max(2, int(min(h, w) * 0.06))
    edge = np.concatenate([rgb[:edge_w].reshape(-1, 3), rgb[-edge_w:].reshape(-1, 3),
                           rgb[:, :edge_w].reshape(-1, 3), rgb[:, -edge_w:].reshape(-1, 3)])
    ground_sd = float(edge.std())
    return {"ok": ground_sd <= ground_limit, "ground_sd": round(ground_sd, 1),
            "why": "" if ground_sd <= ground_limit else
                   f"ground {ground_sd:.1f} -- the background is not clean"}


# ---------------------------------------------------------------------------
# did the hanger come with it
#
# The commonest fault in the cut-outs we ship, and the one a seller notices
# first: the background goes and the coat hanger stays, because a hanger is a
# salient object and the model was trained to find salient objects.
#
# It has a shape, and the shape is what makes it measurable without another
# model. A hanger is a narrow thing sitting above a wide thing: a few rows of
# hook and bar, maybe a tenth of the garment's width, directly over shoulders
# that are the widest part of the picture. So: find the shoulders, look above
# them, and measure how much is up there.


def cut_quality(cutout):
    """What is above the shoulders, and how clean the edge is."""
    rgb, alpha = _load(cutout)
    mask = _mask(rgb, alpha)
    rows = mask.sum(axis=1)
    if rows.max() < 5:
        return {"hanger": None, "why": "nothing was kept"}
    top = np.argmax(rows > 0)
    bottom = len(rows) - np.argmax(rows[::-1] > 0)
    height = max(bottom - top, 1)
    widest = float(rows.max())

    # The shoulders: the first row, coming down, that is a real fraction of
    # the widest part. Everything between the top of the mask and there is
    # either a hanger or a collar, and a collar does not reach the top of the
    # frame on its own.
    shoulder = top
    for y in range(top, bottom):
        if rows[y] >= 0.35 * widest:
            shoulder = y
            break
    above = rows[top:shoulder]

    # Narrow-above-the-shoulders is not the same as a hanger, and looking at
    # the six highest scores showed it: three were hangers and three were a
    # turtleneck collar and two camisoles with spaghetti straps. All narrow,
    # all above the shoulders, and two of those three are the garment.
    #
    # What separates them is colour. A strap is cut from the same cloth; a
    # hanger is wood, wire or plastic and belongs to nobody. So the region
    # above the shoulders is compared with the body of the garment, and a
    # score only counts as a hanger when what is up there is a different
    # colour from what is below.
    foreign = 0.0
    if len(above) and shoulder > top:
        body = mask.copy()
        body[:shoulder] = False
        head = mask.copy()
        head[shoulder:] = False
        hb, hh = _colour_hist(rgb, body), _colour_hist(rgb, head)
        if hb is not None and hh is not None:
            foreign = float(np.abs(hb - hh).sum() / 2)

    tall = float(len(above)) / height
    return {
        # The one to read. Height above the shoulders multiplied by how
        # foreign the colour up there is, because either alone is wrong:
        # height alone ranks a turtleneck and a camisole's straps as hangers,
        # and colour alone fires on any garment with a contrast collar.
        #
        # Checked by eye at the top of the ranking: the six highest are two
        # metal hooks, a green plastic hanger, a white one, a wooden one and a
        # polo hung by its collar. Six for six.
        "hanger": round(tall * foreign, 3),
        "above": round(tall, 3),
        "foreign": round(foreign, 3),
        "width": round(float(above.mean() / widest), 3) if len(above) else 0.0,
        "kept": round(float(mask.mean()), 3),
    }

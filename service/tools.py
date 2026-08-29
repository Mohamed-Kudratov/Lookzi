#!/usr/bin/env python3
"""What the product can do, in one place.

This list decides three things that must never disagree: the buttons the bot
offers, the cards the web app draws, and what the API will accept. It lived in
the bot, and a web front end with its own copy would have drifted from it
inside a week -- a tool switched on in one place and missing in the other, or
priced differently in each.

The front end reads it from `GET /tools` rather than hardcoding it, so adding a
tool here makes it appear everywhere without a second edit.

`needs` is also the script for the conversation: what to ask for, in the order
it is asked. A new tool is an entry in this dictionary, not another branch in
the bot or another form in the browser.
"""

TOOLS = {
    "product-to-model": dict(
        label="Product → Model",
        blurb="A flat photo of the garment, worn by one of our models.",
        needs=["garment", "model"], cost=1, ready=True),
    "virtual-try-on": dict(
        label="Try it on me",
        blurb="Your own photo, wearing the garment you send.",
        needs=["person", "garment"], cost=1, ready=True),
    "model-swap": dict(
        label="Change the model",
        blurb="Your photo, same clothes and pose, a different person wearing them.",
        needs=["person", "model"], cost=1, ready=True),
    "product-in-scene": dict(
        label="Put it in a scene",
        blurb="Describe the shot you want. We make the person, then dress them "
              "in your garment.",
        needs=["garment", "prompt"], cost=2, ready=True),
    "packshot": dict(
        label="Packshot",
        blurb="A clean catalogue cut-out of the garment on its own.",
        needs=["garment"], cost=1, ready=True),
    "model-creation": dict(
        label="Make a new model",
        blurb="A model that belongs to you alone.",
        needs=["look", "prompt?"], cost=4, ready=True),
    "try-on-v2": dict(
        label="New engine (test)",
        blurb="The same job as Product → Model, through a different model: "
              "FASHN VTON 1.5, open weights, Apache-2.0. Here so the two can be "
              "compared on the same garment.",
        needs=["garment", "model", "category"], cost=1, ready=True),
    "short-video": dict(
        label="Short video",
        blurb="Five or ten seconds of motion from a finished image.",
        needs=[], cost=3, ready=False),
}

# What each input is called when it is asked for.
ASK = {
    "garment": "a photo of the <b>garment</b>, laid flat",
    "person": "a photo of the <b>person</b> — full body, facing the camera",
}

# The mode selector is gone. It asked the customer which part of the body the
# garment covers, and the answer changed nothing: the try-on model reads the
# garment image and ignores the text entirely -- measured against an accurate
# description, a deliberately wrong one, and none at all, at both guidance
# settings. See docs/CONTROLS.md.
#
# pipeline.MODE_INSTRUCTION still understands the vocabulary, so a caller that
# knows what it wants can still pass one. The product does not ask, because a
# control that does nothing is a promise the interface cannot keep.


def public():
    """The list as a client should see it, ready to render.

    Ordered so the tools that work come first: a customer scanning a menu
    should meet what they can use before what they cannot.
    """
    out = []
    for tid, t in TOOLS.items():
        out.append({"id": tid, "label": t["label"], "blurb": t["blurb"],
                    "needs": t["needs"], "cost": t["cost"], "ready": t["ready"]})
    out.sort(key=lambda t: (not t["ready"],))
    return out


def ready(tool_id):
    t = TOOLS.get(tool_id)
    return bool(t and t["ready"])

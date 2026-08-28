# What a customer can actually control

Measured on an A100 with the 4-bit checkpoint, the Layering VTON LoRA and the
Lightning 8-step adapter — the configuration the product ships.

## The finding

**The try-on model reads the garment image. It does not read the text.**

Same person, same garment image, same seed, four descriptions:

| Description | Result |
|---|---|
| `"a light gray turtleneck sweater"` — accurate | the sweater in the photo |
| `"a bright red hooded sweatshirt"` — deliberately wrong | the sweater in the photo |
| `"the garment"` — vague | the sweater in the photo |
| *(empty)* | the sweater in the photo |

Not similar. The same, to the eye, at 340 px wide side by side.

The obvious suspect was guidance. Lightning is distilled for `true_cfg_scale=1.0`
— one pass, no negative branch — and with no classifier-free guidance there is
nothing pulling the result toward the words. So the wrong description was run
again at CFG 4.0, which doubles the time to 32.3 s and turns guidance back on.

It made no difference. The sweater is still the sweater.

## What that costs us

Two controls in the product were doing nothing, and one of them had been
doing nothing in two different ways.

**The mode selector.** The customer picks Upper body, Lower body or Full
outfit. Those words never reached the model at all — `apply_mode` understood
only `add` and `swap`, so all three fell through to an unchanged description.
That is fixed: each mode now produces a sentence the model was trained on. It
still changes almost nothing, because the sentence is text and the text is
ignored. Trousers sent as "Upper body" and as "Lower body" come back the same.

**The description box.** Never mattered.

Both are worse than useless while they remain visible. A control that does
nothing is not a harmless extra: it is a promise, and the first customer who
notices that "Lower body" and "Upper body" give the same picture has learned
that the interface lies. Better to ask for nothing and do one thing well.

## What it means for the product

**A free-text front door is not available to us.** FASHN opens with "What will
you create today?" because their model answers text. Copying that box would be
selling a control we do not have, and it would fail on the first attempt.

**Our tools split in two, and only one half listens to words:**

| | Driven by | Text control |
|---|---|---|
| Try it on me | the garment photo | none |
| Product → Model | the garment photo | none |
| Change the model | the model photo | none |
| Make a new model | a prompt (Z-Image) | **full** |
| Backgrounds, scenes | a prompt | **full** — not built |

Z-Image-Turbo generated the entire roster from text descriptions, so text
control exists in our stack. It lives in the generation stage, not the try-on
stage. Anything we want a customer to describe in words has to be built there.

**The honest proposition is narrow and strong.** We put a garment photograph on
a person, in 16 seconds, for about six tenths of a cent, with models who look
like the customers' customers. That is what a seller in Tashkent needs and it
is what we do well. Adding an inert prompt box would not widen it; it would
just make the narrow thing look unreliable.

## What was not tested

- Whether a longer, more specific prompt at 40 undistilled steps behaves
  differently. The Lightning adapter was active throughout, and it is distilled
  for 8 steps; a fair test of the undistilled path needs a separate load.
- Whether text affects anything other than the garment — pose, background,
  lighting. Every test here changed only the garment noun.

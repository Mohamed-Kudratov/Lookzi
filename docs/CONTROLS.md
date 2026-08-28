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

## The try-on layers, and words do not change that

Reported twice: a polo comes back worn over the shirt the model already had on,
the old sleeves showing below the new ones. It is intermittent -- four
deliberate attempts to reproduce it all came back clean -- but when it happens
it is unmistakable.

Three things were tried on the pod.

**Wording.** Four phrasings of the same job: nothing at all, `mode=upper` which
becomes "swap the top for the garment", an explicit "swap the top for the
sweater", and an explicit "add the sweater". Four identical images, all showing
the base layer under the new garment. The model reads the garment image and
ignores the text, which this document already measured four other ways.

**Clearing the torso first.** Paint the chest a flat grey before the try-on, so
there is nothing to layer over. Worse: the model treats the gap as damage to
repair, invents a different shirt with pockets and rolled sleeves, and does not
apply the garment at all. This LoRA composites onto a clothed person; it does
not fill holes.

**A minimal base layer.** A model wearing a fitted sleeveless top takes a
garment cleanly every time, because there is nothing that can poke out from
under it. This is what the competitor's model images do, and it is why theirs
never show the problem.

So the scene tool now builds its person in a sleeveless top. It removes the
failure mode rather than fixing a mechanism anybody has been able to observe on
demand, and that distinction is worth keeping in mind.

Two consequences worth acting on:

- The roster wears white tees with sleeves and a hem. A short-sleeved garment
  over one of those is the exact case that fails. Re-dressing the thirteen in
  fitted sleeveless tops -- via the try-on pipeline itself, which preserves the
  face -- would close it for product-to-model too.
- Layering is right for outerwear and wrong for base layers. It is a capability
  as much as a defect, and "put this jacket over what they are wearing" is a
  thing a seller wants.

## What was not tested

- Whether a longer, more specific prompt at 40 undistilled steps behaves
  differently. The Lightning adapter was active throughout, and it is distilled
  for 8 steps; a fair test of the undistilled path needs a separate load.
- Whether text affects anything other than the garment — pose, background,
  lighting. Every test here changed only the garment noun.

## A phone photograph of a dress on a hanger — 2026-08-28

Reported with pictures: a blue strappy dress, photographed on a hanger against a
wardrobe in ordinary room light, sent through "product to model". The result put
the dress **over** the model's white t-shirt, the sleeves and neckline of the
tee plainly visible under the straps.

That is the layering fault again, and this is the first time it has arrived with
a reproducible input attached. Two things distinguish this case from the four
attempts that came back clean:

- **The garment has no sleeves at all.** Every clean reproduction used a polo or
  a sweater — a garment whose own sleeves cover the base layer's. A strappy
  dress covers nothing above the chest, so whatever the model already wore is
  still in the picture.
- **The photograph is poor.** Phone camera, mixed indoor light, a patterned
  wardrobe door behind, the hanger in frame. Whether that matters is untested.

Which of the two is doing the damage is exactly the question, and the answer
decides the fix. If it is the sleeves, re-dressing the roster in fitted
sleeveless tops closes it — the idea already written above, and the one that
must not be attempted through the scene prompt again. If it is the photograph,
the fix is a cut-out step before the try-on, which is work we half have in
packshot.

Not investigated yet. Deferred deliberately: the simple faults come first.

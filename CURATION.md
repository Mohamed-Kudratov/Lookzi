# Cleaning the data — what it is and why it decides everything

You can do this yourself. You should, in fact: the roster is a taste decision,
and taste does not delegate well. What follows is the actual mechanics.

## The one thing to understand

**A LoRA does not learn the face. It learns whatever is constant across your
images.**

That sentence is the whole reason cleaning matters. The model cannot tell which
constant you meant. It has thirty pictures and it finds what they share — and
whatever that turns out to be becomes "the thing".

So:

| If every image has... | The model concludes |
|---|---|
| the same grey studio background | grey studio is part of this person |
| a front-facing pose | this person only exists from the front |
| the same soft window light | this face is wrong under any other light |
| the same crop | this person is always framed at the chest |
| three shots of a slightly different face | this person is a blur of two people |

None of these produce an error. Training succeeds, the file is written, and the
failure only appears later when you ask for the face in a new pose and it falls
apart. **The model learns your mistakes exactly as reliably as it learns the
face.**

Cleaning is therefore not "removing bad images" in a vague sense. It is two
specific jobs: **removing constants you did not intend**, and **adding variety
you did intend**.

## Doing it: an identity LoRA, start to finish

### 1. Over-generate

Make 60–100 candidates of the face, not 30. Z-Image-Turbo is sub-second, so this
costs minutes. You are going to throw most away, and you want the luxury of
being harsh.

### 2. Cull on identity first

Look only at the parts AI drift changes and a human eye reads unconsciously:

- **eye spacing** and eye shape
- **nose bridge and tip**
- **jaw line** and chin width
- **ear shape** — the most reliable tell, and the one people never look at
- **hairline**

If an image feels like the person's sibling rather than the person, reject it.
Do not keep it because it is a nice photo. One near-miss in twenty images has
five per cent influence on the result, and five per cent of a wrong face is
visible.

### 3. Then cull on variety — and count it

This is the step people skip. Go through what survived and check coverage:

| Axis | You want |
|---|---|
| Angle | front, 3/4 left, 3/4 right, slight up, slight down |
| Distance | headshot, half body, full body |
| Lighting | soft, hard, side, backlit |
| Background | several, deliberately different |
| Expression | neutral and a slight smile at minimum |

If a row is empty, **go back and generate for it specifically**. A dataset that
is 30 front-facing shots in a studio is worse than 18 images that cover the
grid, no matter how good each one looks alone.

### 4. Reject on defects

Hands, ears, teeth, eyes, jewellery, text. AI artifacts here are subtle in a
thumbnail and obvious at full size. View at 100%.

### 5. Land on 20–30

**Twenty excellent beats forty mixed.** Every image is a vote, and a bad vote
counts as much as a good one.

Keep the rejects in a separate folder rather than deleting them. After three
faces you will recognise your own failure patterns, and that is worth more than
the images.

### 6. Caption what varies, never what stays

This is the technique that most people get backwards.

Each image needs a caption. Use a unique trigger token for the identity, then
describe **only what changes between images**:

```
good:  sks_model01, three-quarter view, hard side light, full body, white wall
good:  sks_model01, front view, soft light, headshot, outdoor blurred street

bad:   sks_model01, a beautiful woman with brown hair and green eyes, front view
```

The bad one is bad because "brown hair, green eyes" appears in every caption. The
model then attaches the face to those words instead of to the trigger, and your
trigger token ends up carrying nothing. **Describe the variable, let the trigger
carry the constant.**

### 7. Train, then validate on what you did not train

~20 minutes on an A100. Then generate the face in poses, lighting and settings
that were **not** in the training set.

- Identity holds in new conditions → ready
- Identity holds only in the trained poses → dataset was too narrow, go back to
  step 3
- Face looks over-sharp or "burnt" → trained too long or too hard, retrain
- Background follows the face around → step 3 again, vary backgrounds

Every failure mode on that list is a **dataset** problem. None of them are fixed
by a better model, more GPU, or more steps. That is why this work is curation
rather than engineering, and why you doing it yourself is the right call.

## Why we need clean data at all — the commercial version

Beyond the mechanics, there is a business reason.

A brand shooting forty products needs the same model across all forty. If
identity drifts by three per cent per image, nobody notices any single image and
everybody notices the catalogue. That inconsistency is exactly what "Consistent
Models" is sold as a separate product to fix — and it is bought or lost at the
dataset stage, weeks before any customer sees anything.

Your own `SUCCESS_CRITERIA.md` already names the bar: *"Show the result to
someone who does not know it is AI. If they do not ask what kind of photo it is,
you passed."* Nothing downstream rescues a dataset that fails that test.

## Time

For twenty roster faces: two to three days of focused work, most of it looking
at images rather than operating anything. Training itself is a script and runs
unattended.

That is the real cost of the roster. Eleven dollars of GPU, and your attention.

## What would make this faster

A curation tool: candidates in a grid, keep/reject on a keypress, coverage
counters for angle and lighting that tell you what is still missing, and an
export straight into the training format. The contact-sheet code in
`stress_test.py` is most of the rendering already.

Worth building before the roster rather than after — it is the difference
between three days and one.

# Roster specification

What models to build, and exactly what images each one needs. Written before
generating anything, because generating first and specifying later is how you
end up with sixty pictures of the same face at the same angle.

## Two things people conflate

**Fashion poses are not capture angles.**

The identity dataset does not need a model standing with a hand on the hip. It
needs the *same person* seen from enough angles and lights that the LoRA learns
the person rather than the photograph. Fashion poses come later, at generation
time, from the pose reference — that is what DWPose is for.

So: vary the camera, not the choreography.

## Part 1 — who is in the roster

Sized for the market being sold to, not the market Western tools were built
for. Thirteen faces for phase one.

### Women — 9 (fashion e-commerce skews female, and so should this)

| # | Appearance | Age | Build |
|---|---|---|---|
| 1 | Central Asian | 20s | slim |
| 2 | Central Asian | 20s | average |
| 3 | Central Asian | 30s | average |
| 4 | Central Asian | 35–45 | fuller |
| 5 | Central Asian | 50+ | average |
| 6 | **Central Asian, hijab** | 20s | average |
| 7 | **Central Asian, hijab** | 35–45 | average |
| 8 | Slavic / European | 20s | slim |
| 9 | Slavic / European | 30s | average |

### Men — 4

| # | Appearance | Age | Build |
|---|---|---|---|
| 10 | Central Asian | 20s | slim |
| 11 | Central Asian | 30s | average |
| 12 | Central Asian | 45+ | average |
| 13 | Slavic | 30s | average |

### The three choices worth defending

**Hijab models are not optional here.** A large share of buyers in this market
dress modestly, and effectively no Western tool offers a model who does. A
seller of modest wear currently has nothing. Two faces cover it, and it is the
single most differentiated entry in the table.

**Older and fuller models are in from the start.** Every competitor's roster is
20-something and slim. Buyers over 35 and above a size 12 are a large,
well-funded, and completely unrepresented segment — and a shopper converts when
the model reads as someone like them, which is the entire commercial argument
for on-model imagery in the first place.

**Children are deliberately absent.** Generating photorealistic AI children for
commercial clothing use is legally and reputationally hazardous, and content
classifiers — ours and every platform's — will and should block it. Sell
kidswear with flat-lay and packshot imagery instead. This is not a limitation to
work around later; it is a line to keep.

## Part 2 — what each face needs

Target **24–30 kept images** per face, from 60–100 generated.

### Coverage grid — every row must be filled

| Axis | Values | Minimum |
|---|---|---|
| **Angle** | front, ¾ left, ¾ right, near-profile, slight above, slight below | ≥2 each |
| **Distance** | headshot, half body, **full body** | ≥3 / ≥8 / ≥10 |
| **Lighting** | soft diffuse, hard directional, side, rim/backlit | ≥3 each |
| **Background** | at least 5 distinct | — |
| **Expression** | neutral, slight smile | ≥60% neutral |

### Why the distance distribution is skewed

Most identity-LoRA guides are written for portraits and tell you to shoot
headshots. **Do the opposite here.** These models wear clothes, so the LoRA has
to lock the *body* — proportions, shoulders, waist, height — not only the face.
A roster trained on headshots produces a recognisable face on an inconsistent
body, and the inconsistency shows the moment two garments are shot on the same
"model".

Hence at least ten full-body and eight half-body images out of thirty.

### Vary the clothing — deliberately

This is the trap specific to a fashion roster. If all thirty training images
show the model in a red dress, the LoRA learns the red dress as part of the
identity, and it will fight every garment you try to put on her afterwards.

Put the model in **different, plain, neutral clothing** across the set — simple
tops and trousers in varied muted colours. Never anything with a print, logo or
strong silhouette.

### Hands and feet

Include a few images with hands visible and unoccluded. Hands are where
generation fails most visibly, and a dataset that never shows them gives the
model nothing to hold onto.

## Part 3 — the file layout

Papkalar va bitta CSV. No database needed at this stage.

```
roster/
  face01_cauz_f_20s_slim/
    raw/           generated candidates, untouched
    keep/          the 24-30 that will train
    reject/        kept on purpose - you learn your own failure patterns
    dataset.csv
    captions.txt
```

`dataset.csv`, one row per generated image:

```csv
file,status,angle,distance,lighting,background,expression,notes
img_007.png,keep,three_quarter_left,full_body,hard,street,neutral,
img_008.png,reject,front,half_body,soft,studio,smile,ears differ
```

The four middle columns are not decoration. They are what the coverage counter
reads to tell you that ¾-right is still empty — and an empty cell there is
exactly what breaks the LoRA later.

## Part 4 — the prompt template

Consistent structure across candidates, varying only the axes:

```
photorealistic full-body photograph of {appearance}, {age}, {build},
wearing plain {colour} {garment}, {angle}, {lighting} lighting,
{background} background, {expression} expression,
natural skin texture, visible pores, shot on 85mm, shallow depth of field
```

The tail matters. `natural skin texture, visible pores` and a real focal length
are what push Z-Image-Turbo away from the plastic look — the thing FASHN puts a
whole section of its marketing page against.

## Effort

| | |
|---|---|
| Generation | minutes per face (sub-second each) |
| Curation | **1.5–2 hours per face** — the real cost |
| Training | ~20 min per face, unattended |
| 13 faces | ~$8 of GPU, 2–3 days of your attention |

The GPU is free in comparison. The attention is the product.

# FASHN AI — what they actually ship

Read from their own pages and API docs, August 2026. Numbers are theirs unless
marked as ours.

## The product surface

Nine API endpoints, and the app is a front end over the same models:

| Endpoint | What it does | Their latency |
|---|---|---|
| `tryon-max` | garment onto a person, up to 4K | 10–55s |
| `product-to-model` | flat-lay or ghost mannequin → on-model | 10–55s |
| `model-swap` | replace the person, keep garment, pose, light, background | 10–55s |
| `model-create` | generate a model with chosen demographics | 10–55s |
| `face-to-model` | a real face → a fashion model photo | 10–55s |
| `image-to-video` | motion clip from one still, with camera moves | 1–3 min |
| `edit` | pose, background, detail refinement | 10–55s |
| `reframe` | recompose, change aspect ratio | 10–55s |
| `background-remove` | cutout with transparency | 1–3s |

The app adds Brush Edit, 4K Upscaling, Face Swap, Change Aspect Ratio. SDKs in
TypeScript and Python. API outputs expire after three days.

## The API shape

`POST /v1/run` with `model_name` and an `inputs` object, then poll the returned
`id`. Shared parameters across endpoints: `resolution` (`1k`/`2k`/`4k`),
`generation_mode` (`fast`/`balanced`/`quality`), `seed` (default 42),
`num_images` (1–4), `output_format`, `return_base64`. Limits: 30 MiB per image,
minimum 15×15 px, aspect ratio between 1:16 and 16:1.

Worth copying: seed defaulting to a fixed value rather than random, so a result
is reproducible by default. Ours already does this.

## What it costs a customer

| Plan | Price | Credits |
|---|---|---|
| Free | — | 10 credits, once |
| Basic | $19/mo | 200 |
| Pro | $49/mo | 750 + 50 daily |
| Agency | $99/mo | 1 500 + 100 daily |

Top-ups at $0.10/credit, 100 minimum, valid 12 months. Per image:

| Mode | 1k | 2k | 4k |
|---|---|---|---|
| fast | 1 | 2 | 3 |
| balanced | 2 | 3 | 4 |
| quality | 3 | 4 | 5 |

`face_reference` adds 3 credits. So a 4K quality image with a locked face is
8 credits — about **$0.76** at the Basic rate, or $0.80 on top-ups.

Our measured cost at 8-step Lightning is **$0.0061** an image, at 512×896.
That is not the same product as their 4K, and the comparison is only honest
once we match resolution.

## Consistent Models — how it actually works

This is the finding that matters most, because it is the tool they position as
best-in-class and the one our roster competes with directly.

**They do not train anything.** The mechanism is a Face Reference: one image
anchors identity, and every tool accepts it. Their words: *"Face Reference
anchors identity to a single face across all tools"*, and it *"adapts to new
hairstyles, expressions, and lighting while keeping the core likeness"*. It
costs 3 extra credits and adds about 20 seconds.

Their own guidance admits the weakness. The blog tells customers to *"vary one
or two elements at a time and keep the core attributes steady"* and warns off
*"drastically different lighting styles or extreme camera angles"* — which is
the known failure mode of zero-shot identity adapters. Identity survives small
changes and drifts under large ones.

Uploading your own face is **Agency only**, $99/month. Everyone below that
picks from a shared gallery of FASHN Faces — so two competing brands on the
Basic plan can be advertising with the same model.

## Where they are genuinely ahead

- Nine shipped endpoints against our one and a half.
- 4K output; we generate at 512×896 and have not built upscaling.
- Video shipped; ours is unstarted.
- Documented API, two SDKs, team seats, three-day retention policy.
- Testimonials, logos, funding, and a team.

None of that is close. Treat it as the reference implementation, not as
something to catch up with feature by feature.

## Where they cannot follow

- **Telegram.** They are web and API only. The sellers in this market run their
  businesses inside Telegram, and a bot is not a smaller version of the web app
  — it is the whole product for that customer.
- **Local payment.** International cards only. A seller in Tashkent frequently
  cannot pay $19 a month to a foreign processor at all. Price is irrelevant to
  someone who cannot complete the transaction.
- **Language.** No Uzbek, no Russian.
- **Who the models look like.** Their roster is generic and global; ours is
  Central Asian across ages 20 to 50. Modest wear — hijab, long sleeves, full
  coverage — is a category they do not address and a large part of this market.
- **Exclusivity at the bottom of the range.** Their own model costs $99/month.
  Ours can be included far lower, because a roster we generate ourselves has no
  per-customer marginal cost.

## The one technical opening

Their consistency is zero-shot and drifts under exactly the conditions a
catalogue needs — many angles, many lighting setups, one person. A LoRA trained
per roster member does not drift, because identity is in the weights rather
than inferred at inference time.

We measured our current two-stage method at **0.684** mean ArcFace similarity
to the hero across 300 variations, 299 of 300 above the same-person threshold.
That is already strong without training. Training would push it higher and,
more usefully, make it *measurable against a claim they cannot match*.

Before that is worth anything, the roster has to be distinct: our own
measurement found four colliding pairs, three of them faces promoted from the
same slot. A roster whose members a matcher confuses fails the same way theirs
does, for a different reason.

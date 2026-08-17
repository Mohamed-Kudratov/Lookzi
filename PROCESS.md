# How it actually works, step by step

`ARCHITECTURE.md` describes the parts. This describes what happens, in order,
and when. Timings are measured on an A100 80GB unless marked otherwise.

## Three kinds of work

Everything in this product is one of three things, and confusing them is what
makes people overestimate the cost:

| | When | How often | Cost |
|---|---|---|---|
| **A. Our one-time work** | before launch | once | weeks of engineering |
| **B. Per-customer setup** | when a customer joins | once per face | ~20 min GPU |
| **C. Per-request** | every click | thousands/day | **14 seconds** |

Only C repeats at scale, and C is now measured at $0.0061 an image. A and B feel
expensive because they involve training, but they happen once.

---

## A. Our one-time work: task LoRAs

This is the part that makes the tools exist at all.

The base model can already put a garment on a person, because that is what the
layering LoRA we run was trained for. It cannot yet turn a flat-lay into an
on-model shot, or strip a person out to leave a packshot. Each of those needs a
**task LoRA** — trained once by us, shipped with the product, used by every
customer.

### What training a task LoRA actually involves

**1. Collect paired data.** This is the hard part, and it is data work, not ML
work. For Product to Model you need thousands of pairs: *this flat-lay* ↔ *that
same garment worn by a model*. For Packshot, the reverse. The pairs must be the
same garment, or the model learns to invent.

Where pairs come from, in order of practicality:
- e-commerce catalogues that publish both a packshot and an on-model shot of
  the same SKU — most retailers do
- generate them: run the try-on we already have to create the on-model side
  from a flat-lay, keep the good ones. Synthetic pairs are weaker but cheap.
- shoot a few hundred yourself for the categories that matter most

**2. Caption every pair** with a consistent instruction, because the base model
is instruction-driven. Sloppy captions are the most common cause of a LoRA that
"almost works".

**3. Train.** Rank 32–64 on the attention projections, the same shape as the
LoRA already in `weights/`. Hours to a day on one A100, depending on dataset
size.

**4. Evaluate against held-out pairs**, not against the training set. This is
where `eval/metrics.py` and the garment-fidelity check earn their place.

**5. Ship it** as a file. Each task LoRA is ~90 MB and loads in about a second.

Budget one to three weeks per task LoRA, most of it spent on data rather than
training. Four tools need one, so this is the bulk of A.

---

## B. Per-customer setup: identity LoRAs

Different thing entirely, despite also being called LoRA training. Small,
frequent, automatable.

### Building our own roster (once, before launch)

**1. Generate candidates.** Z-Image-Turbo, sub-second per image, so hundreds of
faces in minutes. Prompt for the demographics your sellers actually sell to.

**2. Choose.** Pick 10–20 for the roster. This is a taste decision, not a
technical one, and it deserves the most human attention of anything in this
document — the roster is the asset.

**3. Expand each chosen face** to 20–30 images: different angles, lighting,
expressions, distances. Same person throughout. Variety here is what makes the
LoRA generalise instead of memorising one photo.

**4. Curate.** Drop any image where identity drifted. Twenty good images beat
forty inconsistent ones.

**5. Train.** ~20 minutes on an A100 per face. Output: a ~90 MB file.

**6. Validate.** Generate the face in poses and lighting that were *not* in the
training set. If identity holds, it is ready; if it only works in the trained
poses, the dataset was too narrow.

**7. Register** as an `element` with `status=ready` and its `lora_path`.

Twenty faces ≈ 7 GPU-hours ≈ **$11**, plus the curation time, which is the real
cost. Done once.

### A brand building their own signature model

Same pipeline, triggered by the customer:

1. They upload 15–30 photos of one face, or pick a generated one
2. `POST /v1/elements` → `POST /v1/elements/{id}/train` → a job
3. A burst worker starts, trains ~20 min, writes the LoRA to storage
4. Automatic validation pass
5. `status: ready` — now usable in every tool
6. Charged as a one-off, and this is the paid tier that creates the lock-in

### What goes wrong

| Symptom | Cause |
|---|---|
| face drifts across poses | too few images, or too similar to each other |
| background bleeds into the face | training images shot in one place |
| identity only works at one angle | no variety in the dataset |
| face looks "burnt", over-sharp | learning rate too high, or trained too long |

Every one of these is a **dataset** problem, not a model problem. That is good
news: it is fixable by curation, which does not require an ML engineer.

---

## C. Per-request: what happens on every click

### Virtual Try-On

```
1. seller uploads person + garment            (upload)
2. API validates, reserves credits, creates job   ~50 ms
3. worker picks it up
4. DWPose extracts the pose                    ~2 s   (CPU)
5. pad all three to 512x896                    ~0.1 s
6. VAE encodes person, garment, pose           ~0.5 s
7. text encoder builds the instruction embeds  ~3 s
8. transformer denoises, 8 steps at CFG 1.0    ~11 s  <-- the whole cost
9. VAE decodes                                 ~0.5 s
10. safety + quality checks                    ~0.3 s
11. store, charge credits, notify              ~50 ms
                                        total  ~18 s
```

Step 8 is 80% of it, and it is what Lightning cut from 95 s to 11 s.

Step 7 is cached: rerunning the same inputs with a different seed skips it.

### Product to Model

Identical, except step 4 is skipped (no person to pose-detect), the product-to-
model LoRA is swapped in at step 8 (~1 s), and if the seller chose a roster
model its identity LoRA is stacked alongside. **Two adapters, one base model,
no reload** — this is why one warm GPU serves every tool.

### Model Swap

Adds segmentation before step 6, and compositing after step 9: the garment and
background pixels are copied from the original rather than regenerated. The
model only ever touches inside the person mask. Background change is then zero
by construction rather than by hope.

### Short Video

Different worker entirely.

```
1. seller picks a finished image + a motion preset
2. job routed to the video queue
3. burst worker starts, loads Wan 2.2            ~1-2 min
4. generates 120 frames (5 s at 24 fps)          minutes
5. first-vs-last frame product check
6. store, charge, notify
```

Bursty and expensive, so it scales to zero and is priced separately. Nobody
expects a video instantly, which is what makes the cold start acceptable here
and unacceptable for images.

---

## Where the time and money actually go

| | one-time | per customer | per request |
|---|---|---|---|
| task LoRAs (4 tools) | 4–12 weeks | — | — |
| our roster (20 faces) | $11 + curation | — | — |
| customer's own face | — | ~$0.55 | — |
| image | — | — | **$0.0061** |
| video (5 s) | — | — | ~$0.15–0.30 |

The per-request number is the only one that scales with success, and it is now
seven times below the market price of $0.04. That is the whole point of the
Lightning work.

## The order to build it in

1. **Ship try-on on the existing LoRA.** It works today. No new training.
2. **Build the roster.** Unlocks the model-dependent half of every other tool.
3. **Train the Product to Model task LoRA.** The tool sellers pay for repeatedly.
4. **Then Packshot, then Model Swap** — same pattern, each cheaper than the last
   because the pipeline is already there.
5. **Video last.** Separate worker, separate price, no dependency on the rest.

Nothing above requires research. It requires data collection, curation, and
patience with evaluation — which is why the earlier estimate of two engineers
and four to five months holds, and why the third person on the team should have
taste in fashion imagery rather than a PhD.

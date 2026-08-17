# System architecture

The models are maybe a third of this product. What follows is the rest.

## The two decisions everything else follows from

**1. One warm base model, LoRAs swapped per request.**

Six of the seven tools are `Qwen-Image-Edit-2509` with different references and a
different adapter. LoRAs are ~90 MB and load in about a second; the base takes
three minutes. So: one GPU stays warm and every image tool routes through it.
Six separate models would mean six warm GPUs — roughly $7,000/month instead of
$1,200, for the same output.

**2. Everything is a job.**

Video takes minutes. LoRA training takes twenty. Even a fast image is seconds.
Nothing here fits a synchronous HTTP request, so there is no synchronous path at
all — every tool creates a job, the client subscribes to it, the worker writes
results back. Building this on day one costs a week; retrofitting it costs a
rewrite.

## Services

```
                    ┌──────────────┐
   web / API keys → │  API gateway │ ← auth, quota, validation
                    └──────┬───────┘
                           │ creates job
                    ┌──────▼───────┐
                    │  agent router│ ← NL + images → (tool, params)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  job queue   │
                    └──┬───┬───┬───┘
          ┌────────────┘   │   └────────────┐
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │ image worker│  │ light worker│  │ burst worker│
   │  WARM       │  │  on demand  │  │ video/train │
   │ Qwen+LoRAs  │  │ Z-Image,    │  │ Wan 2.2,    │
   │ tools 1,2,4 │  │ try-off     │  │ LoRA train  │
   └─────────────┘  └─────────────┘  └─────────────┘
                           │
                 Postgres + pgvector · Redis · S3/R2
```

Only the image worker is always on. Everything else is bursty and should scale
from queue depth to zero.

## Data model

Five tables carry the product. `org_id` is on every row and enforced with
row-level security, not application code — multi-tenant leaks happen in the one
query somebody forgot to filter.

```sql
organizations   id, name, plan, credit_balance
users           id, org_id, email, role
api_keys        id, org_id, key_hash, scopes, last_used_at   -- hash only

elements        id, org_id, kind, name, status, lora_path,
                embedding vector(768), assets jsonb, created_at
  -- kind: face | model | background | product | preset
  -- status: draft | training | ready | failed
  -- THE registry. Every tool takes an optional element_id.

jobs            id, org_id, tool, status, params jsonb,
                input_asset_ids[], output_asset_ids[],
                gpu_seconds, credits_charged, worker_id,
                created_at, started_at, finished_at, error

assets          id, org_id, storage_key, kind, width, height,
                sha256, source_job_id, created_at

credit_ledger   id, org_id, delta, reason, job_id, created_at   -- append only
```

Two notes that matter later.

`elements` is what turns seven scripts into one product. A face created in Model
Creation is the same row that Product Photography reads as a face reference and
Model Swap applies to an existing photo. Without it you have seven tools; with
it you have a platform, and a reason for customers not to leave.

`credit_ledger` is append-only and never updated. Balance is derived. Every
argument about billing is settled by replaying the ledger.

## API

Uniform across tools, because the tools are uniform underneath:

```
POST /v1/jobs              { tool, params, inputs[], element_id? } → job
GET  /v1/jobs/{id}                                                 → status
GET  /v1/jobs/{id}/events                          → SSE stream
POST /v1/elements          { kind, name, assets[] }                → element
POST /v1/elements/{id}/train                       → job (LoRA)
GET  /v1/elements                                  → list
POST /v1/assets            multipart                               → asset
POST /v1/agent             { text, inputs[] }      → { tool, params } (no exec)
```

`/v1/agent` returns the routing decision without executing it, so the UI can
show the user which tool it picked and let them correct it before spending
credits. Never silently spend someone's money on a guess.

Idempotency keys on `POST /v1/jobs`. Retries on a flaky mobile connection must
not bill twice.

## GPU orchestration — the expensive part

Measured today: the base model is 57.7 GB and takes ~3 minutes to load from a
network volume. That single fact shapes the whole scaling strategy.

- **Do not autoscale the warm worker on demand.** A cold start is three minutes;
  nobody waits. Keep N warm, scale N on *sustained* queue depth, drain slowly.
- **Weights live on a shared network volume**, not baked into images. A new
  worker loads from disk rather than re-downloading 57.7 GB.
- **LoRAs are cached in RAM** on each worker, keyed by element_id. Swapping is
  ~1s; loading from object storage the first time is ~2s.
- **Burst workers scale to zero.** Video and training are bursty by nature and
  users already accept a wait for them.

Queue routing is by tool, not round-robin. A video job must never land on the
image worker and block six interactive requests for four minutes.

## Metering

Every job records `gpu_seconds` and converts to credits at a per-tool rate.
Charge on **completion**, refund on failure, and hold a reservation at
submission so a user cannot queue a hundred jobs against a balance of five.

Rates come from measurement, not guesses. Today: an image is ~$0.066 of GPU and
must reach under $0.01; a five-second video will be twenty to fifty times an
image. Price video separately and visibly — it is the tool most likely to be
abused if it looks free.

## Security

**Tenant isolation** through Postgres RLS. Every asset served through short-TTL
signed URLs, never a public bucket.

**API keys** stored as hashes with a visible prefix for identification. Scoped
per tool so a leaked key from a storefront integration cannot train LoRAs.

**Faces are biometric data.** This is the part of the system that carries real
legal weight — collect explicit consent for any uploaded face, support deletion
that actually deletes (including trained LoRA weights and derived embeddings),
and keep face assets in a separate bucket with its own retention policy. A
customer asking to be forgotten must not leave their face inside a LoRA.

**Content safety is not optional** for a product that puts clothes on people.
Classify on input and on output; block minors and NSFW outright. Log every
block. This protects the company more than any other single control here.

**Rate limits** per org and per key, with a separate lower ceiling on training
and video.

## UI

One workspace, as observed on the competitor: a prompt box with attachments at
the centre, and a right-hand panel that changes with the detected tool. The user
should not have to know the tool exists.

- **Agent-first**: type what you want, the router picks the tool, the UI shows
  which one and lets you override. Suggestion chips for the common intents.
- **Show intermediate output.** The existing pipeline already yields the padded
  inputs before sampling starts — that pattern should hold everywhere. A user
  who sees their inputs processed at second two will wait fifteen seconds. One
  staring at a spinner will not.
- **Gallery is the hub**, not an archive. Every output is the input to the next
  tool: hover a still and send it to video, to model swap, to packshot.
- **Elements library** as a first-class screen — this is where the customer's
  accumulated value lives, and it should feel like it.

## What is not a model problem

Two parts of the plan should not be built with a diffusion model:

**Sizing** — body measurement estimation (SMPL/SMPL-X fitting) from a photo plus
garment size charts. CPU, milliseconds. Closest published work: `FitVTON`.

**Style recommendation** — CLIP or SigLIP embeddings in pgvector, ranked by what
actually sold. The `elements.embedding` column is already there for it.

Both get better with usage data, which is why they are worth owning. The
generative models do not — anyone can rent those.

## Build order

**Stage 1 — skeleton (2–3 weeks).** Postgres, jobs, queue, storage, auth,
credits, one worker running the try-on we already have. No new models. The point
is that everything after this is adding a handler, not adding infrastructure.

**Stage 2 — cost (1 week).** Lightning, measured with `sweep_steps.py`. Under a
cent per image or the rest does not matter.

**Stage 3 — elements + faces (2–3 weeks).** Registry, Z-Image-Turbo model
creation, LoRA training worker. This unlocks four tools at once.

**Stage 4 — the studio tools (4–6 weeks).** Product to Model, Packshot, Model
Swap. All handlers on the existing worker.

**Stage 5 — agent + UI polish (2–3 weeks).** The routing layer and the single
workspace. Do this after the tools exist, not before — the router needs
something to route to.

**Stage 6 — video (2–3 weeks).** Wan 2.2 on a burst worker, priced separately.

Sizing and style recommendation run in parallel from Stage 3, by a different
person, on a different stack.

Roughly four to five months to a professional product with two engineers, and
that assumes the model work stays integration rather than research. It will, if
the scope holds.

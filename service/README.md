# The service

Three pieces that scale independently, because they have nothing in common but
a table.

```
web tier            queue (Postgres)          workers
──────────────      ──────────────────        ─────────────────────
FastAPI, no GPU  →  jobs, credits, results ←  fake_worker  (laptop)
$10 CPU box         the customer's history    gpu_worker   (A100)
never restarts      survives every restart    come and go freely
```

The web tier holds no model weights. It writes jobs and reads results; workers
elsewhere do the work. That is what lets the site stay up while every GPU is
asleep, and what lets the always-on half cost ten dollars a month instead of a
thousand.

## Run it locally

```bash
docker compose up --build
```

Then <http://localhost:8000/docs>.

Four containers stand in for four production services: Postgres for the managed
database, MinIO for Cloudflare R2, the web tier as itself, and `fake_worker`
where the GPU pool will be. Nothing in that stack imports torch, so it builds
in under a minute on a laptop.

**On Windows this needs WSL2.** Docker Desktop's Linux engine will not start
without it — the symptom is a 500 from the engine on every command. Install it
once, with a restart:

```powershell
wsl --install
```

## Run it without Docker

```bash
psql "$DATABASE_URL" -f service/db/001_initial.sql
python -m service.seed_models
uvicorn service.app:app --reload &
python -m service.fake_worker
```

## The fake worker is the point

`fake_worker.py` satisfies the same contract as `gpu_worker.py` by sleeping for
14.3 seconds — the measured time of a real job — and returning an image marked
PLACEHOLDER. Everything except the model can therefore be built and corrected
on a laptop: sign-in, credits, the queue, retries, refunds, history, Telegram,
the web app.

It sleeps for the real duration on purpose. An interface designed against an
instant reply is an interface designed against something that will never
happen.

```bash
FAKE_SECONDS=0.2 python -m service.fake_worker    # fast, for tests
FAKE_FAIL_RATE=0.2 python -m service.fake_worker  # exercise retries and refunds
```

## Writing a worker

A handler takes the job and returns where it put the result:

```python
def handle(job):
    ...
    return {"object_key": key, "kind": "image",
            "width": 768, "height": 1024, "seconds": 14.3}

Worker(handle, tools=["product-to-model"], name="gpu-1").run()
```

Raising is how failure is reported. The loop decides whether that means a
retry or a refund, so a handler never has to think about either.

`tools` is how pools are kept apart. A video worker holds a different model
resident and must never pick up a try-on job, or it would reload weights
between every pair of jobs.

## Things that are load-bearing

**`FOR UPDATE SKIP LOCKED`.** Without it two workers reading at the same moment
either block on each other, serialising the whole fleet through one row, or
both take the same job and the customer pays twice for one image.

**Charge and queue in one transaction.** Split apart, a crash in the gap takes
the credit and produces nothing.

**Refund only on the final failure.** Refunding each attempt and re-charging
each retry puts several pairs of entries in the ledger for one image, and the
history stops being readable.

**`release_stale`.** A pod reclaimed mid-job leaves its rows in `running` with
nobody working on them. We have watched that happen repeatedly. Whichever
worker notices first puts them back.

**Presigned uploads.** A 20 MB photo routed through the web tier is web-tier
load for no benefit, and it is the first thing to fall over under a crowd.

## Environment

| | |
|---|---|
| `DATABASE_URL` | `postgresql://lookzi:lookzi@localhost:5432/lookzi` |
| `S3_ENDPOINT` · `S3_KEY` · `S3_SECRET` · `S3_BUCKET` | MinIO locally, R2 in production |
| `TRIAL_CREDITS` | granted on first contact, default 20 |
| `WORKER_TOOLS` | comma-separated; empty means any tool |
| `JOB_STALE_AFTER` · `JOB_MAX_ATTEMPTS` | `15 minutes`, `3` |

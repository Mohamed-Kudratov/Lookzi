# Running it

Nothing here needs a GPU.

## Once, on Windows

Docker Desktop's Linux engine runs on WSL2, and WSL2 needs the Virtual Machine
Platform feature. In an **Administrator** PowerShell:

```powershell
wsl --install --no-distribution
```

Then **restart the computer**. The feature does not take effect until the
machine reboots — until then every Docker command returns a bare 500, and
`wsl --status` claims virtualisation is off even when the firmware has it on.

## Once, per checkout

Copy the template and fill in the bot token:

```bash
cp .env.example .env
```

`.env` is what Docker reads. `.env.example` is only the shape — editing it does
nothing, and it is committed, so a token written there would end up in the
repository.

Get the token from [@BotFather](https://t.me/BotFather). If it is ever pasted
into a chat, an email or a screenshot, revoke it with `/revoke` and generate a
new one; anyone holding it can act as your bot.

## Up

```bash
docker compose up --build
```

Five containers: Postgres, MinIO, the web tier, a stub worker, the bot. No CUDA
in any of them, so it builds in about a minute.

Then load the roster into the database:

```bash
docker compose exec web python -m service.seed_models
```

| | |
|---|---|
| API and docs | <http://localhost:8000/docs> |
| Storage console | <http://localhost:9000> — `lookzi` / `lookzi-dev-secret` |
| Bot | send it a photo |

## Checking it

```bash
python tests/test_service_logic.py     # no database needed
python tests/test_flow.py              # needs Postgres; skips without it
```

The first covers rules that fail silently — a batch key that never groups, a
priority that puts the free tier first. The second covers the ones that only
appear against a real database: that charging is atomic, that three workers
claiming at once take three different jobs, that a refund cannot be taken
twice.

## What you will see

The image comes back marked **PLACEHOLDER**. That is deliberate: no GPU is
involved. Everything around it is real — the credit is charged, the queue is
ordered, a failure refunds, the history is written.

Swap `fake_worker` for `gpu_worker` on a machine with a card and the same flow
produces real images. Nothing else changes.

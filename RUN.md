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

Then load the roster, and give each model a photograph:

```bash
docker compose exec web python -m service.seed_models
```

```bash
docker compose exec web python -m service.seed_heroes --placeholder assets
```

The second command assigns stand-in images so a model can be chosen locally.
The roster's own photographs live on the pod; when it is running, point at
them instead with `--dir /workspace/elements_out/heroes`.

| | |
|---|---|
| API and docs | <http://localhost:8080/docs> |
| Storage console | <http://localhost:9001> — `lookzi` / `lookzi-dev-secret` |
| Database | `localhost:5433` |
| Bot | send it a photo |

The ports are deliberately not the obvious ones. A developer machine often
already runs Postgres on 5432 or something on 8000, and when it does, Docker
wins only the IPv6 address — so `127.0.0.1` reaches the other program instead.
The symptom is not a clear conflict: the database answers and rejects the
password, and the web port returns a 404 from a server that looks like ours.

## Checking it

```bash
python tests/test_service_logic.py                     # no database needed
DATABASE_URL=postgresql://lookzi:lookzi@127.0.0.1:5433/lookzi python tests/test_flow.py
API_BASE=http://127.0.0.1:8080 python tests/smoke_api.py
```

The first covers rules that fail silently — a batch key that never groups, a
priority that puts the free tier first. The second covers what only appears
against a real database: that charging is atomic, that three workers claiming
at once take three different jobs, that a refund cannot be taken twice. The
third puts one job through the door a customer uses — HTTP, a presigned
upload, a worker in another container, a signed link back.

## What you will see

The image comes back marked **PLACEHOLDER**. That is deliberate: no GPU is
involved. Everything around it is real — the credit is charged, the queue is
ordered, a failure refunds, the history is written.

Swap `fake_worker` for `gpu_worker` on a machine with a card and the same flow
produces real images. Nothing else changes.

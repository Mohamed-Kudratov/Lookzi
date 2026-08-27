#!/usr/bin/env python3
"""The web tier. It never touches the model.

The first version of this file loaded 55 GB into the same process that served
HTTP. That works for one person on one pod and for nothing else: the web tier
could not restart without a ten-minute reload, could not run on a cheap CPU
box, and could not be more than one machine.

Now it holds no weights at all. It writes jobs to Postgres and reads results
back; workers on other machines do the work. Two consequences follow, and they
are the whole point:

  the site stays up when every GPU is asleep, off, or reclaimed
  the site costs about ten dollars a month instead of a thousand

    uvicorn service.app:app --host 0.0.0.0 --port 8000
"""
import os
import uuid

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import accounts
from . import queue as q
from . import storage
from . import tools as tool_registry

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Lookzi", version="0.2")


def db():
    conn = q.connect()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# identity
#
# The MVP signs people in through Telegram, which supplies an id with every
# message, so there is no password to check here. The header is read straight
# into an identity lookup; email and password arrive later as a second row in
# the same table rather than a change to any of this. See docs/AUTH.md.

def current_user(conn=Depends(db),
                 x_telegram_id: str = Header(default=None),
                 x_client_id: str = Header(default=None)):
    """Whoever is asking, by whichever identity they have.

    The bot passes a Telegram id. A browser has none, so the web app mints one
    and keeps it in local storage -- which is not authentication and is not
    pretending to be: there is nothing to protect until there is something to
    pay for, and inventing a login before then would be four decisions asked
    of someone who has not seen the product work yet. See docs/AUTH.md.
    """
    kind, value = (("telegram", x_telegram_id) if x_telegram_id
                   else ("web", x_client_id))
    if not value:
        raise HTTPException(401, "no identity")
    row = accounts.identify(conn, kind, value)
    if row["blocked_at"]:
        raise HTTPException(403, "account blocked")
    return row


# ---------------------------------------------------------------------------
# health and catalogue

@app.get("/health")
def health(conn=Depends(db)):
    """Reports the queue, and says plainly whether any worker is alive.

    A customer waiting on an image is owed the difference between "busy" and
    "nothing is running", and so is whoever is on call.
    """
    d = q.depth(conn)
    workers = q.alive(conn)
    return {"ok": True, "queued": d["queued"], "running": d["running"],
            "workers": len(workers),
            "worker_names": [w["name"] for w in workers],
            # Kept for older clients; the count is what mattered to them.
            "workers_seen": len(workers)}


@app.get("/tools")
def tools():
    """What the product can do, as the client should render it.

    Served rather than hardcoded in the page, so switching a tool on is one
    edit in service/tools.py and not three in three places that then disagree.
    """
    return tool_registry.public()


@app.get("/modes")
def modes():
    return [{"id": m, "label": label} for m, label in tool_registry.MODES]


@app.get("/models")
def models(conn=Depends(db)):
    rows = conn.execute(
        """SELECT id, display_name, age, gender, ethnicity, build, modest,
                  hero_key, hero_is_placeholder
             FROM models
            WHERE duplicate_of IS NULL
            ORDER BY gender, age""").fetchall()
    for r in rows:
        r["preview"] = storage.presigned_get(r.pop("hero_key")) if r["hero_key"] else None
    return rows


# ---------------------------------------------------------------------------
# uploads
#
# The browser and the bot upload straight to object storage with a signed link.
# Routing a 20 MB photo through the web tier would make every upload web-tier
# load, for no benefit, and it is the first thing that falls over under a crowd.

class UploadRequest(BaseModel):
    kind: str = Field("garment", pattern="^(garment|person)$")
    content_type: str = "image/png"


@app.post("/uploads")
def create_upload(req: UploadRequest, user=Depends(current_user)):
    key = storage.key_for(f"uploads/{req.kind}", user["id"])
    return {"key": key,
            "url": storage.presigned_put(key, content_type=req.content_type)}


# ---------------------------------------------------------------------------
# jobs

class JobRequest(BaseModel):
    tool: str = "product-to-model"
    garment_key: str
    person_key: str | None = None
    model_id: str | None = None
    mode: str = Field("upper", pattern="^(upper|lower|overall)$")
    description: str = ""
    seed: int = 42
    idem_key: str | None = None


@app.post("/jobs", status_code=201)
def create_job(req: JobRequest, conn=Depends(db), user=Depends(current_user)):
    if not tool_registry.ready(req.tool):
        raise HTTPException(400, f"{req.tool} is not available yet")
    if not req.person_key and not req.model_id:
        raise HTTPException(400, "send either person_key or model_id")

    person_key = req.person_key
    if not person_key:
        row = conn.execute(
            "SELECT hero_key FROM models WHERE id = %s AND duplicate_of IS NULL",
            (req.model_id,)).fetchone()
        # Two different problems, and they were reported as one. A model that
        # does not exist is a client mistake; a model whose photograph has not
        # been uploaded yet is ours, and saying "unknown model" about it sends
        # whoever is debugging to look in entirely the wrong place.
        if row is None:
            raise HTTPException(404, f"unknown model {req.model_id}")
        if not row["hero_key"]:
            raise HTTPException(
                503, f"model {req.model_id} has no photograph yet — "
                     "run service.seed_heroes")
        person_key = row["hero_key"]

    params = {"person_key": person_key, "garment_key": req.garment_key,
              "mode": req.mode, "description": req.description, "seed": req.seed,
              "width": 768, "height": 1024, "steps": 8}
    try:
        job, charged = q.submit(
            conn, user["id"], req.tool, params, model_id=req.model_id,
            cost=tool_registry.TOOLS[req.tool]["cost"],
            priority=_priority(user), idem_key=req.idem_key)
    except q.InsufficientCredit as exc:
        raise HTTPException(402, f"{exc.have} credits left, this costs {exc.need}")

    return {"job_id": str(job["id"]), "status": job["status"], "charged": charged}


def _priority(user):
    """Lower runs first. Paid tiers jump the queue.

    This is also what makes an always-warm worker pay for itself: the free tier
    fills the troughs the paid tier leaves behind, so the card is never idle
    and never in a paying customer's way.
    """
    return {"trial": 200, "seller": 100, "brand": 50}.get(user["plan"], 200)


@app.get("/jobs/{job_id}")
def job_status(job_id: uuid.UUID, conn=Depends(db), user=Depends(current_user)):
    job = q.status(conn, job_id)
    if job is None or job["user_id"] != user["id"]:
        raise HTTPException(404, "unknown job")
    out = {"job_id": str(job["id"]), "status": job["status"],
           "tool": job["tool"], "model_id": job["model_id"],
           "credits": job["credits_cost"], "created_at": job["created_at"],
           "seconds": job.get("seconds"), "error": job.get("error")}
    if job["status"] == "queued":
        out["position"] = job.get("position")
    if job.get("object_key"):
        out["result_url"] = storage.presigned_get(job["object_key"])
    return out


@app.get("/jobs")
def list_jobs(conn=Depends(db), user=Depends(current_user), limit: int = 50):
    rows = q.history(conn, user["id"], limit=min(limit, 200))
    for r in rows:
        r["job_id"] = str(r.pop("id"))
        k = r.pop("object_key", None)
        r["result_url"] = storage.presigned_get(k) if k else None
    return rows


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: uuid.UUID, conn=Depends(db), user=Depends(current_user)):
    outcome = q.cancel(conn, job_id, user_id=user["id"])
    if outcome is None:
        raise HTTPException(404, "unknown job")
    if outcome == "too late":
        # 409, not an error page: the request was well formed and the answer is
        # simply that the GPU time is already being spent.
        raise HTTPException(409, "already running")
    fresh = conn.execute("SELECT credits FROM users WHERE id = %s",
                         (user["id"],)).fetchone()
    return {"status": "cancelled", "credits": fresh["credits"]}


@app.get("/me")
def me(user=Depends(current_user)):
    return {"id": user["id"], "plan": user["plan"], "credits": user["credits"]}


# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(HERE, "static", "index.html")
    if not os.path.exists(p):
        return RedirectResponse("/docs")
    with open(p, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.exception_handler(psycopg.errors.UndefinedTable)
def no_schema(request, exc):
    """A missing table means the migration has not run.

    Saying so beats a 500 with a stack trace, which is what this looked like
    the first time and cost twenty minutes.
    """
    return JSONResponse(
        {"error": "database not migrated",
         "hint": "psql $DATABASE_URL -f service/db/001_initial.sql"},
        status_code=503)

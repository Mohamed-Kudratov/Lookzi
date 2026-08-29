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
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               StreamingResponse)
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
def tools(conn=Depends(db)):
    """What the product can do, as the client should render it.

    Served rather than hardcoded in the page, so switching a tool on is one
    edit in service/tools.py and not three in three places that then disagree.
    """
    out = tool_registry.public()
    # How long each tool has actually been taking, from its own last twenty
    # results. The studio used to count down from a figure measured once on a
    # different checkpoint; a tool that has never run says nothing rather than
    # guessing.
    rows = conn.execute(
        """SELECT tool, percentile_cont(0.5) WITHIN GROUP (ORDER BY seconds) AS med
             FROM (SELECT j.tool, r.seconds,
                          row_number() OVER (PARTITION BY j.tool
                                             ORDER BY j.finished_at DESC) AS n
                     FROM results r JOIN jobs j ON j.id = r.job_id
                    WHERE r.seconds IS NOT NULL) t
            WHERE n <= 20
            GROUP BY tool""").fetchall()
    median = {r["tool"]: float(r["med"]) for r in rows if r["med"] is not None}
    for t in out:
        t["typical_seconds"] = round(median[t["id"]], 1) if t["id"] in median else None
    return out


@app.get("/models")
def models(conn=Depends(db), user=Depends(current_user)):
    """The shared roster, plus whatever this account made for itself.

    exclusive_to has been on the table since the beginning and nothing read it,
    so a model made to order was a picture in the history and nothing more --
    the tool promised "a model that belongs to you alone" and then offered no
    way to use them again.
    """
    rows = conn.execute(
        """SELECT id, display_name, age, gender, ethnicity, build, modest,
                  hero_key, hero_is_placeholder, exclusive_to
             FROM models
            WHERE duplicate_of IS NULL
              AND (exclusive_to IS NULL OR exclusive_to = %s)
            ORDER BY exclusive_to NULLS LAST, gender, age""",
        (user["id"],)).fetchall()
    for r in rows:
        r["preview"] = storage.presigned_get(r.pop("hero_key")) if r["hero_key"] else None
        r["mine"] = r.pop("exclusive_to") is not None
    return rows


AGE_YEARS = {"20s": 24, "30s": 34, "40s": 44, "50s": 54}


class KeepModel(BaseModel):
    job_id: uuid.UUID
    display_name: str = Field("", max_length=40)


@app.post("/models/keep")
def keep_model(req: KeepModel, conn=Depends(db), user=Depends(current_user)):
    """Turn a made-to-order model into one this account can use again.

    The job already holds everything the roster needs -- who was asked for, and
    the picture that came back -- so nothing is re-generated and nothing is
    copied in storage: the result key becomes the hero key, and the same object
    is now the model's photograph.
    """
    row = conn.execute(
        """SELECT j.tool, j.params, j.user_id, r.object_key
             FROM jobs j LEFT JOIN results r ON r.job_id = j.id
            WHERE j.id = %s""", (req.job_id,)).fetchone()
    if row is None or row["user_id"] != user["id"]:
        raise HTTPException(404, "unknown job")
    if row["tool"] != "model-creation":
        raise HTTPException(400, "only a model you made can be kept as a model")
    if not row["object_key"]:
        raise HTTPException(409, "that job has no picture yet")

    p = row["params"] or {}
    # A short id from the job, so keeping the same one twice is the same row
    # rather than a second identical model in the picker.
    mid = "own_" + str(req.job_id).replace("-", "")[:10]
    gender = p.get("gender") or "woman"
    name = (req.display_name or "").strip() or ("My model " + mid[-4:])
    conn.execute(
        """INSERT INTO models (id, display_name, age, gender, ethnicity, build,
                               modest, hero_key, exclusive_to)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO UPDATE
                   SET display_name = EXCLUDED.display_name,
                       hero_key = EXCLUDED.hero_key""",
        (mid, name, AGE_YEARS.get(p.get("age"), 28), gender,
         p.get("look") or "uzbek", p.get("build") or "average",
         str(p.get("modest")).lower() in ("1", "true", "yes"),
         row["object_key"], user["id"]))
    return {"id": mid, "display_name": name}


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
    # Optional here and checked against the tool's own list of inputs below.
    # It used to be required, which quietly meant every tool was a try-on: a
    # packshot needs no model and no person, and was rejected by a rule written
    # for a different tool.
    garment_key: str | None = None
    person_key: str | None = None
    model_id: str | None = None
    # Defaults to nothing rather than "upper", which is an active mistake for a
    # pair of trousers -- "swap the top for the trousers". The model ignores
    # the text either way, so neutral is the safer of two irrelevancies.
    mode: str | None = Field(None, pattern="^(upper|lower|overall|layer)$")
    description: str = ""
    seed: int = 42
    idem_key: str | None = None
    # Making a model asks for a handful of choices instead of a photograph. The
    # rest of the face varies by seed, so asking twice gives two people.
    gender: str | None = Field(None, pattern="^(woman|man)$")
    age: str | None = Field(None, pattern="^(20s|30s|40s|50s)$")
    build: str | None = Field(None, pattern="^(slim|average|fuller)$")
    look: str | None = Field(None, pattern="^(uzbek|kazakh|tajik|slavic)$")
    modest: bool = False
    # What the customer wants the picture to be. Only the scene tool reads it:
    # the try-on model ignores text entirely, so offering it elsewhere would be
    # a control that does nothing. See docs/CONTROLS.md.
    prompt: str | None = Field(None, max_length=800)
    # Which part of the body the garment covers. Ours ignores the equivalent --
    # measured five ways in docs/CONTROLS.md -- but the new engine reads it, and
    # it is the difference between a dress worn as a dress and a dress worn as a
    # tunic over the trousers the model already had on.
    category: str | None = Field(None, pattern="^(tops|bottoms|one-pieces)$")


@app.post("/jobs", status_code=201)
def create_job(req: JobRequest, conn=Depends(db), user=Depends(current_user)):
    if not tool_registry.ready(req.tool):
        raise HTTPException(400, f"{req.tool} is not available yet")

    # What is required comes from the tool, not from here. Three tools shared
    # one hardcoded rule and the fourth could not be submitted at all.
    needs = [n for n in tool_registry.TOOLS[req.tool]["needs"]
             # A trailing "?" marks an input the tool will take but does not
             # require -- making a model accepts your own words instead of the
             # choices, and an empty box should not block a job.
             if not n.endswith("?")]

    # Photographs are counted, not matched to a field. "Change the model" wants
    # a person and a model, and the person the customer uploads is what carries
    # the clothes -- so it arrives as the garment, and person_key is deliberately
    # empty for the model to fill. Checking that field by name rejected every
    # such job while the studio was sending exactly what it should.
    photos_wanted = sum(1 for n in needs if n in ("garment", "person"))
    photos_sent = len([k for k in (req.garment_key, req.person_key) if k])
    missing = []
    if photos_sent < photos_wanted:
        missing.append(f"{photos_wanted} photo" + ("s" if photos_wanted > 1 else ""))
    if "model" in needs and not req.model_id:
        missing.append("a model")
    if "prompt" in needs and not (req.prompt or "").strip():
        missing.append("a description")
    if missing:
        raise HTTPException(400, f"{req.tool} needs {' and '.join(missing)}")

    # When a model was chosen, the model is who wears it. Preferring an
    # uploaded photo here put the customer inside their own photograph.
    person_key = None if "model" in needs else req.person_key
    if not person_key and req.model_id:
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

    # Whatever was uploaded is the garment. For "change the model" that upload
    # is a person wearing clothes, and the clothes are the point of it.
    params = {"person_key": person_key,
              "garment_key": req.garment_key or req.person_key,
              "mode": req.mode, "description": req.description, "seed": req.seed,
              "width": 768, "height": 1024, "steps": 8,
              "gender": req.gender or "woman", "age": req.age or "20s",
              "build": req.build or "average", "look": req.look or "uzbek",
              "modest": "true" if req.modest else "false",
              "prompt": (req.prompt or "").strip(),
              "category": req.category or "tops"}
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


# ---------------------------------------------------------------------------
# review
#
# Every garment anybody has tried is already kept -- nothing deletes uploads,
# and each job records the key it used -- but kept is not the same as visible.
# This is the page that makes the archive worth having: what went in, what came
# out, and how long it took, for every job, ours and customers' alike.
#
# It reads across all accounts, so it is not a customer feature and must not
# become reachable by accident. Set REVIEW_KEY and pass it, or leave it unset
# and the route only answers from a private address.

REVIEW_KEY = os.environ.get("REVIEW_KEY", "")
# The same ceiling the pod enforces, so a file that would be refused there
# is refused here rather than after it has been carried twice.
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))


def _may_review(request, key):
    if REVIEW_KEY:
        return key == REVIEW_KEY
    host = (request.client.host if request.client else "") or ""
    return (host.startswith(("127.", "10.", "192.168.", "172.")) or host == "::1"
            or host == "localhost")


@app.get("/api/review")
def review_data(request: Request, conn=Depends(db), limit: int = 60,
                key: str = ""):
    if not _may_review(request, key):
        raise HTTPException(404, "not found")
    rows = conn.execute(
        """SELECT j.id, j.tool, j.model_id, j.status, j.created_at,
                  j.params, r.object_key, r.seconds, r.width, r.height,
                  m.hero_key
             FROM jobs j
             LEFT JOIN results r ON r.job_id = j.id
             LEFT JOIN models m ON m.id = j.model_id
            ORDER BY j.created_at DESC
            LIMIT %s""", (min(limit, 200),)).fetchall()
    out = []
    for row in rows:
        params = row.pop("params") or {}
        row["job_id"] = str(row.pop("id"))
        row["garment"] = (storage.presigned_get(params["garment_key"])
                          if params.get("garment_key") else None)
        # "product to model" resolves the chosen roster model into person_key
        # before it queues, so the person and the model are the same file. Shown
        # as two frames it read as two inputs, and the card flipped between one
        # picture and itself.
        person_key = params.get("person_key")
        row["person"] = (storage.presigned_get(person_key)
                         if person_key and person_key != row.get("hero_key")
                         else None)
        # The roster model a job used is one of the pictures that made the
        # result, and until now the only trace of it was a name in small blue
        # type. "product to model" has two inputs and was showing one.
        hero = row.pop("hero_key", None)
        row["model"] = storage.presigned_get(hero) if hero else None
        row["prompt"] = (params.get("prompt") or "").strip() or None
        k = row.pop("object_key", None)
        row["result"] = storage.presigned_get(k) if k else None
        out.append(row)
    return out


@app.delete("/api/review/{job_id}")
def review_delete(job_id: uuid.UUID, request: Request, conn=Depends(db),
                  key: str = ""):
    """Remove one job from the archive, picture and all.

    The page used to promise that nothing here is ever deleted, and that was a
    reasonable promise for an archive nobody could edit. It is the wrong one
    for a working gallery: a failed experiment kept forever is noise in the one
    place we go to judge quality.

    The stored object goes with the row. Keeping the bytes after removing the
    only reference to them is not caution, it is a bill.
    """
    if not _may_review(request, key):
        raise HTTPException(404, "not found")
    row = conn.execute("SELECT object_key FROM results WHERE job_id = %s",
                       (job_id,)).fetchone()
    gone = conn.execute("DELETE FROM jobs WHERE id = %s RETURNING id",
                        (job_id,)).fetchone()
    if gone is None:
        raise HTTPException(404, "unknown job")
    if row and row["object_key"]:
        try:
            storage.delete(row["object_key"])
        except Exception as exc:                              # noqa: BLE001
            # The row is already gone; say so rather than failing the request
            # and leaving the caller thinking nothing happened.
            print(f"[review] deleted job {job_id} but not its object: {exc}",
                  flush=True)
    return {"deleted": str(job_id)}


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request, key: str = ""):
    if not _may_review(request, key):
        raise HTTPException(404, "not found")
    path = os.path.join(HERE, "static", "review.html")
    if not os.path.exists(path):
        raise HTTPException(500, "review.html is missing")
    with open(path, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


# ---------------------------------------------------------------------------
# files
#
# Only mounted when S3_PROXY is on, which is when the studio is reachable from
# somewhere other than this machine. A signed link names a host; the host it
# names is one only this machine can reach, so from anywhere else every picture
# is broken and every upload fails. These two routes make the app the address
# instead, and then there is no address to configure.

@app.get("/files/{key:path}")
def get_file(key: str):
    if not storage.PROXY:
        raise HTTPException(404, "not found")
    try:
        obj = storage.client().get_object(Bucket=storage.BUCKET, Key=key)
    except Exception:                                         # noqa: BLE001
        raise HTTPException(404, "no such file")
    return StreamingResponse(
        obj["Body"], media_type=obj.get("ContentType") or "image/png",
        # Keys carry a uuid, so a link is not guessable and may be cached.
        headers={"Cache-Control": "private, max-age=3600"})


@app.put("/files/{key:path}")
async def put_file(key: str, request: Request):
    if not storage.PROXY:
        raise HTTPException(404, "not found")
    # Only where an upload slot was just issued. Without this the route is a
    # writable bucket on the open internet, which is a different product.
    if not key.startswith("uploads/"):
        raise HTTPException(403, "that is not an upload key")
    body = await request.body()
    if len(body) > MAX_UPLOAD:
        raise HTTPException(413, f"larger than {MAX_UPLOAD} bytes")
    storage.client().put_object(
        Bucket=storage.BUCKET, Key=key, Body=body,
        ContentType=request.headers.get("Content-Type", "image/png"))
    return {"key": key}


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

#!/usr/bin/env python3
"""The try-on service: one warm model, a queue, and an HTTP interface.

Everything before this ran as scripts on a pod. A product cannot: the model
takes five to ten minutes to load and 55 GB of VRAM to hold, so it has to be
loaded once and kept, and requests have to wait their turn rather than each
paying the load again.

That single constraint decides the shape. One process owns the GPU. One worker
thread pulls from a queue, because two requests sampling at once on one card is
slower than doing them in order and risks an OOM that kills both. HTTP handlers
never touch the model -- they enqueue and return a job id, since a request that
blocks for eleven seconds behind four others in front of it times out somewhere
in between.

Two products, one endpoint. Virtual try-on puts a garment on a customer's own
photo; product-to-model puts it on a roster model. They are the same operation
with a different person image, which is the architectural bet in ROADMAP.md --
so the API takes either `person` (an upload) or `model_id` (a roster member),
and nothing downstream knows the difference.

    uvicorn service.app:app --host 0.0.0.0 --port 7860
"""
import io
import os
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from queue import Queue

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "elements"))

OUT_DIR = os.environ.get("SERVICE_OUT", "/workspace/service_out")
ELEMENTS_OUT = os.environ.get("ELEMENTS_OUT", "/workspace/elements_out")
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image-Edit-2509")
LORA_DIR = os.environ.get("LORA_DIR", os.path.join(ROOT, "weights"))
LIGHTNING = int(os.environ.get("LIGHTNING", "8"))
# Finished jobs are kept so a client that polls late still gets its result, but
# not forever -- each entry pins a PNG and the pod's disk is not large.
MAX_JOBS = int(os.environ.get("MAX_JOBS", "500"))

app = FastAPI(title="Lookzi")

_state = {"ready": False, "error": None, "load_seconds": None, "started": time.time()}
_pipe = None
_queue = Queue()
_jobs = OrderedDict()
_lock = threading.Lock()


def _hero_path(face_id):
    """The chosen candidate for a roster member, whichever index it was.

    picks.txt holds the choice, but a face can exist on disk without being in
    picks yet -- during curation that is the normal state, not an error.
    """
    from hero import read_picks
    try:
        idx = read_picks(os.path.join(ROOT, "elements", "picks.txt")).get(face_id)
    except SystemExit:
        idx = None
    if idx:
        p = os.path.join(ELEMENTS_OUT, "heroes", face_id, f"{idx}.png")
        if os.path.exists(p):
            return p
    d = os.path.join(ELEMENTS_OUT, "heroes", face_id)
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if name.endswith(".png"):
                return os.path.join(d, name)
    return None


def _catalogue():
    """Selectable roster members that actually have an image on disk.

    Both conditions matter. A member hidden by DUPLICATE_OF is a measured
    collision and must not be offered; a member with no hero has never been
    generated and would fail at request time rather than at listing time.
    """
    from catalog import selectable_roster
    out = []
    for f in selectable_roster():
        p = _hero_path(f["id"])
        if not p:
            continue
        out.append({
            "id": f["id"],
            "label": f"{f['appearance']}, {f['age']}, {f['build']} build",
            "gender": f["gender"],
            "age": f["age"],
            "modest": f["modest"],
            "preview": f"/models/{f['id']}/preview",
        })
    return out


def _load():
    t = time.time()
    try:
        from pipeline import LayeringVTONPipeline
        global _pipe
        _pipe = LayeringVTONPipeline(MODEL_PATH, LORA_DIR, lightning=LIGHTNING)
        _state["load_seconds"] = round(time.time() - t, 1)
        _state["ready"] = True
        print(f"[service] model ready in {_state['load_seconds']}s", flush=True)
    except Exception as exc:
        # Recorded rather than raised: the process must stay up to report why it
        # is unusable. A container that exits on a load failure restarts and
        # fails again, five minutes at a time, with the reason in a log nobody
        # is reading.
        _state["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


def _worker():
    from utils import process_inputs
    while True:
        job_id = _queue.get()
        with _lock:
            job = _jobs.get(job_id)
        if job is None:
            _queue.task_done()
            continue
        job["status"] = "running"
        job["started"] = time.time()
        try:
            person = Image.open(job["person_path"]).convert("RGB")
            garment = Image.open(job["garment_path"]).convert("RGB")
            pp, pg, ppose = process_inputs(person, garment, None)
            result = _pipe(person_img=pp, garment_img=pg, pose_img=ppose,
                           description=job["description"], mode=job["mode"],
                           seed=job["seed"])
            out = os.path.join(OUT_DIR, f"{job_id}.png")
            result.save(out)
            job["result"] = out
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job["seconds"] = round(time.time() - job["started"], 1)
            _queue.task_done()


@app.on_event("startup")
def _startup():
    os.makedirs(OUT_DIR, exist_ok=True)
    threading.Thread(target=_load, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()


@app.get("/health")
def health():
    return {
        "ready": _state["ready"],
        "error": _state["error"],
        "load_seconds": _state["load_seconds"],
        "uptime_seconds": round(time.time() - _state["started"], 1),
        "queued": _queue.qsize(),
        "models": len(_catalogue()),
    }


@app.get("/models")
def models():
    return _catalogue()


@app.get("/models/{face_id}/preview")
def preview(face_id: str):
    p = _hero_path(face_id)
    if not p:
        raise HTTPException(404, f"no image for {face_id}")
    return FileResponse(p, media_type="image/png")


@app.post("/tryon")
async def tryon(garment: UploadFile = File(...),
                person: UploadFile = File(None),
                model_id: str = Form(None),
                mode: str = Form("upper"),
                description: str = Form(""),
                seed: int = Form(42)):
    if not _state["ready"]:
        raise HTTPException(503, _state["error"] or "model still loading")
    if person is None and not model_id:
        raise HTTPException(400, "send either a person image or a model_id")

    job_id = uuid.uuid4().hex[:12]
    gp = os.path.join(OUT_DIR, f"{job_id}_garment.png")
    Image.open(io.BytesIO(await garment.read())).convert("RGB").save(gp)

    if person is not None:
        pp = os.path.join(OUT_DIR, f"{job_id}_person.png")
        Image.open(io.BytesIO(await person.read())).convert("RGB").save(pp)
        source = "upload"
    else:
        pp = _hero_path(model_id)
        if not pp:
            raise HTTPException(404, f"unknown model {model_id}")
        source = model_id

    job = {"id": job_id, "status": "queued", "person_path": pp, "garment_path": gp,
           "mode": mode, "description": description or "the garment", "seed": seed,
           "source": source, "queued_at": time.time()}
    with _lock:
        _jobs[job_id] = job
        while len(_jobs) > MAX_JOBS:
            _jobs.popitem(last=False)
    _queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "position": _queue.qsize()}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    public = {k: v for k, v in job.items()
              if k not in ("person_path", "garment_path", "result")}
    public["result"] = f"/jobs/{job_id}/result" if job.get("result") else None
    return public


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None or not job.get("result"):
        raise HTTPException(404, "no result")
    return FileResponse(job["result"], media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(HERE, "static", "index.html")
    if not os.path.exists(p):
        return JSONResponse({"error": "no UI installed"}, status_code=404)
    with open(p, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())

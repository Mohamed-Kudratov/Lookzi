#!/usr/bin/env python3
"""The product, on photographs it has never seen.

    python -m bench.stress --list
    python -m bench.stress --tool packshot
    python -m bench.stress                    # everything, in order

The material is in `test products/`: real listing photographs of real garments
on beds, hangers, wooden doors and tiled floors, under whatever light the room
had. It is the third tier -- the one no public dataset carries, because curated
collections are curated -- and it is the tier that decides whether this product
works, since it is what a seller in Tashkent actually has.

Two things about that folder are worth knowing before reading any score:

  the men's garments are 3024x4032 straight off a phone; the women's are
  400x533 and 24 KB. Fifty-six times fewer pixels. A men-versus-women
  difference in the results may be measuring the compression and not the tool,
  so the two are always reported apart

  `man/overall` holds street photographs of men wearing outfits, not garments
  laid out to be photographed. It is not a product image and it is left out of
  the garment runs

The folder names are ground truth for something else we want: upper, lower and
overall are exactly the tops / bottoms / one-pieces a model has to be told, so
this set also measures whether that could be worked out rather than asked.

Results are written after every job, so an hour-long run that dies at minute
fifty is still an hour of results.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STOCK = os.path.join(ROOT, "test products")
BASE = os.environ.get("LOOKZI_BASE", "http://127.0.0.1:8080")
CLIENT = os.environ.get("BENCH_CLIENT", "web-dc705b51-600c-4812-9230-fd7520f4a7fa")
HDR = {"X-Client-Id": CLIENT, "Content-Type": "application/json"}
SEED = 20260829
# Bumped by hand when a run is retried, so retries are new jobs.
ATTEMPT = os.environ.get("BENCH_ATTEMPT", "1")

# folder -> (gender, the category the garment actually is)
GARMENTS = [("man/upper", "man", "tops"),
            ("man/lower", "man", "bottoms"),
            ("Woman/Upper", "woman", "tops"),
            ("Woman/lower", "woman", "bottoms"),
            ("Woman/overall", "woman", "one-pieces")]
MODELS = [("man/models", "man"), ("Woman/models", "woman")]
EXT = (".jpg", ".jpeg", ".png", ".webp")


def listing(folder):
    d = os.path.join(STOCK, folder)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(EXT)]


def call(path, payload=None, timeout=40):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 method="POST" if data else "GET", headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return {"_error": exc.code, "_body": exc.read().decode()[:300]}
    except urllib.error.URLError as exc:
        return {"_error": 0, "_body": str(exc.reason)[:300]}


def upload(path, kind):
    """One upload per file for the whole run, not one per job.

    A hundred garments through three tools is three hundred jobs and would
    otherwise be three hundred uploads of the same fifty pictures.
    """
    ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    u = call("/uploads", {"kind": kind, "content_type": ctype})
    if u.get("_error"):
        raise RuntimeError(f"upload slot refused: {u['_body']}")
    with open(path, "rb") as fh:
        body = fh.read()
    req = urllib.request.Request(u["url"], data=body, method="PUT",
                                 headers={"Content-Type": ctype})
    urllib.request.urlopen(req, timeout=180)
    return u["key"]


def wait(job_id, budget=900):
    started = time.time()
    while time.time() - started < budget:
        s = call(f"/jobs/{job_id}")
        if s.get("_error"):
            return s
        if s.get("status") in ("done", "failed", "cancelled"):
            return s
        time.sleep(3)
    return {"status": "timeout"}


def build_cases(keys):
    """Every job, in the order it will run.

    Cheapest first, so the first minutes already say something: a packshot is
    two tenths of a second and a try-on is half a minute.
    """
    # keys is empty under --list, which runs before anything is uploaded.
    models = {g: [(p, keys.get(p, "")) for p in listing(f)] for f, g in MODELS}
    out = []
    for folder, gender, category in GARMENTS:
        for i, path in enumerate(listing(folder)):
            name = f"{folder.replace('/', '_')}/{os.path.basename(path)}"
            out.append(dict(id=f"pack/{name}", tool="packshot",
                            garment_key=keys.get(path, ""), category=category,
                            gender=gender, source=path))
    for folder, gender, category in GARMENTS:
        pool = models[gender]
        for i, path in enumerate(listing(folder)):
            if not pool:
                continue
            mpath, mkey = pool[i % len(pool)]
            name = f"{folder.replace('/', '_')}/{os.path.basename(path)}"
            out.append(dict(id=f"tryon/{name}", tool="virtual-try-on",
                            garment_key=keys.get(path, ""), person_key=mkey,
                            category=category, gender=gender, source=path,
                            model_source=mpath))
    return out


def _refuse_a_second_runner(outdir):
    """One runner per run directory.

    Three of these ran at once, because two `pkill` calls did not reach the
    Windows processes they were aimed at. Each rewrote the whole results file
    from its own view every job, so the file was whichever wrote last -- and
    all three queued jobs against one GPU, which is why a thirty-second job
    started taking fifty-five.

    The lock holds a pid. A stale one from a killed run is stepped over rather
    than becoming a thing to clear by hand at four in the morning.
    """
    lock = os.path.join(outdir, "RUNNING")
    if os.path.exists(lock):
        with open(lock, encoding="utf-8") as fh:
            pid = fh.read().strip()
        alive = False
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True, timeout=30)
            alive = pid in (out.stdout or "")
        except Exception:                                     # noqa: BLE001
            alive = os.path.exists(f"/proc/{pid}")
        if alive:
            print(f"  another runner is already going (pid {pid}). Stop it "
                  "first, or use --out with a different name.")
            raise SystemExit(2)
        print(f"  a lock from pid {pid} was left behind; it is gone, carrying on")
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    return lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", help="only this tool")
    ap.add_argument("--limit", type=int, default=0, help="first N jobs only")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="stress")
    args = ap.parse_args()

    outdir = os.path.join(HERE, "runs", args.out)
    os.makedirs(os.path.join(outdir, "img"), exist_ok=True)
    results_path = os.path.join(outdir, "results.json")
    lock = None if args.list else _refuse_a_second_runner(outdir)

    # Uploaded once and remembered, so a re-run after a failure does not send
    # fifty megabytes again.
    keys_path = os.path.join(outdir, "keys.json")
    keys = {}
    if os.path.exists(keys_path):
        with open(keys_path, encoding="utf-8") as fh:
            keys = json.load(fh)

    wanted = []
    for folder, _, _ in GARMENTS:
        wanted += listing(folder)
    for folder, _ in MODELS:
        wanted += listing(folder)
    todo_up = [p for p in wanted if p not in keys]
    if todo_up and not args.list:
        print(f"  uploading {len(todo_up)} files", flush=True)
        for n, p in enumerate(todo_up, 1):
            kind = "person" if "/models" in p.replace("\\", "/") else "garment"
            try:
                keys[p] = upload(p, kind)
            except Exception as exc:                          # noqa: BLE001
                print(f"    {os.path.basename(p)}: {exc}", flush=True)
            if n % 25 == 0 or n == len(todo_up):
                with open(keys_path, "w", encoding="utf-8") as fh:
                    json.dump(keys, fh, indent=1)
                print(f"    {n}/{len(todo_up)}", flush=True)
        with open(keys_path, "w", encoding="utf-8") as fh:
            json.dump(keys, fh, indent=1)

    cases = [c for c in build_cases(keys) if not args.tool or c["tool"] == args.tool]
    if args.limit:
        cases = cases[:args.limit]
    if args.list:
        for c in cases:
            print(f"  {c['tool']:16} {c['category']:11} {c['id']}")
        print(f"  {len(cases)} jobs")
        return 0

    done = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        done = {r["id"]: r for r in prev.get("rows", [])
                if r.get("status") == "done"}
        if done:
            print(f"  {len(done)} already done, skipping those", flush=True)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    rows = list(done.values())
    todo = [c for c in cases if c["id"] not in done]
    print(f"  {len(todo)} jobs -> bench/runs/{args.out}  (commit {commit})",
          flush=True)

    started_all = time.time()
    for n, case in enumerate(todo, 1):
        # The attempt is part of the key. A fixed one meant a retry handed
        # back the job that had already failed -- instantly, with no seconds
        # and the same error -- because that is exactly what idempotency is
        # for. Eighteen jobs "failed" a second time without being run.
        body = {"tool": case["tool"], "seed": SEED,
                "idem_key": f"stress-{SEED}-{case['id']}-{ATTEMPT}"}
        for k in ("garment_key", "person_key", "model_id", "prompt"):
            if case.get(k):
                body[k] = case[k]
        j = call("/jobs", body)
        if j.get("_error"):
            row = {**case, "status": "rejected", "error": j["_body"]}
        else:
            st = wait(j["job_id"])
            row = {**case, "job_id": j.get("job_id"), "status": st.get("status"),
                   "seconds": st.get("seconds"), "error": st.get("error")}
            if st.get("result_url"):
                fn = case["id"].replace("/", "_") + ".png"
                try:
                    with urllib.request.urlopen(st["result_url"], timeout=180) as r:
                        with open(os.path.join(outdir, "img", fn), "wb") as fh:
                            fh.write(r.read())
                    row["image"] = fn
                except Exception as exc:                      # noqa: BLE001
                    row["error"] = f"download failed: {exc}"
        rows.append(row)
        # Written every time. An hour-long run that dies at minute fifty is
        # still an hour of results.
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump({"run": args.out, "commit": commit, "seed": SEED,
                       "rows": rows}, fh, indent=1)
        rate = (time.time() - started_all) / n
        print(f"  {n:3}/{len(todo)} {case['id'][:44]:46} {row['status']:8} "
              f"{row.get('seconds')}s  eta {int(rate * (len(todo) - n) / 60)}m",
              flush=True)

    ok = sum(1 for r in rows if r["status"] == "done")
    print(f"  {ok}/{len(rows)} done -> {results_path}")
    if lock and os.path.exists(lock):
        os.remove(lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

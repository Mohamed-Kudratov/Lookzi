#!/usr/bin/env python3
"""Every tool, on the same garments, every time.

    python -m bench.run                 # the whole set
    python -m bench.run --tool packshot # one tool
    python -m bench.run --list          # what would run

Why this exists: until now "better" was a word. We changed the scene framing
and I said it was better; we put a second engine beside the first and I said it
won. Both were true, and neither was measured -- they rested on four pictures I
happened to look at. A claim about quality that cannot be repeated next week on
the same inputs is an opinion wearing a lab coat.

So: a frozen set of garments, drawn from what customers actually uploaded, run
through the real API rather than straight at the pod -- because the studio, the
queue, the bridge and the routing are part of what can be wrong, and a harness
that skips them measures a system nobody uses.

Results land in bench/runs/<stamp>/ with the git commit that produced them.
Nothing is overwritten, so two runs can always be put side by side.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("LOOKZI_BASE", "http://127.0.0.1:8080")
# The account the benchmark runs as. It has to be one that is never blocked
# halfway through: a run that stops at job forty because the ledger ran out is
# a wasted half hour and a table with holes in it.
CLIENT = os.environ.get("BENCH_CLIENT", "web-dc705b51-600c-4812-9230-fd7520f4a7fa")
HDR = {"X-Client-Id": CLIENT, "Content-Type": "application/json"}

# Seeds are fixed. Two runs of the same set must differ because the code
# changed, not because the dice did.
SEED = 20260829

SCENES = ["a woman walking through a sunlit market street",
          "a man standing in a bright modern office"]
FACES = ["uzbek woman in her 20s, long dark hair, slim",
         "uzbek man in his 30s, short dark hair, athletic",
         "kazakh woman in her 40s, fuller build"]


def load_set():
    with open(os.path.join(HERE, "set_keys.json"), encoding="utf-8") as fh:
        return json.load(fh)


def call(path, payload=None, timeout=30):
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


def cases(s):
    """Every job the benchmark runs, named so a row can be found again.

    Two models rather than the whole roster: the question here is whether a
    tool works, not whether it works on all fourteen people, and thirteen more
    of each case would turn half an hour into six hours for very little.
    """
    woman, man = "f_cauz_20s_avg", "m_cauz_20s_slim"
    out = []
    for name, g in s["garments"].items():
        for who, model in (("w", woman), ("m", man)):
            out.append(dict(id=f"p2m/{name}/{who}", tool="product-to-model",
                            garment_key=g["key"], model_id=model))
            out.append(dict(id=f"v2/{name}/{who}", tool="try-on-v2",
                            garment_key=g["key"], model_id=model,
                            category=g["category"]))
    for pname, p in s["people"].items():
        for gname in ("polo", "knit", "dress_blue"):
            out.append(dict(id=f"tryon/{gname}/{pname}", tool="virtual-try-on",
                            garment_key=s["garments"][gname]["key"],
                            person_key=p["key"]))
        out.append(dict(id=f"swap/{pname}", tool="model-swap",
                        garment_key=p["key"], model_id=woman))
    for name in ("polo", "jacket", "dress_blue", "dress_print", "dress_navy",
                 "dress_strip"):
        out.append(dict(id=f"pack/{name}", tool="packshot",
                        garment_key=s["garments"][name]["key"]))
    for name in ("polo", "dress_blue", "dress_print"):
        for i, scene in enumerate(SCENES[:1]):
            out.append(dict(id=f"scene/{name}", tool="product-in-scene",
                            garment_key=s["garments"][name]["key"], prompt=scene))
    for i, face in enumerate(FACES):
        out.append(dict(id=f"make/{i}", tool="model-creation", prompt=face))
    return out


def submit(case):
    body = {"tool": case["tool"], "seed": SEED,
            "idem_key": f"bench-{SEED}-{case['id']}-{time.time():.0f}"}
    for k in ("garment_key", "person_key", "model_id", "prompt", "category"):
        if case.get(k):
            body[k] = case[k]
    return call("/jobs", body)


def wait(job_id, budget=600):
    started = time.time()
    while time.time() - started < budget:
        s = call(f"/jobs/{job_id}")
        if s.get("_error"):
            return s
        if s.get("status") in ("done", "failed", "cancelled"):
            return s
        time.sleep(3)
    return {"status": "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", help="only this tool")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    s = load_set()
    todo = cases(s)
    if args.tool:
        todo = [c for c in todo if c["tool"] == args.tool]
    if args.list:
        for c in todo:
            print(f"  {c['tool']:18} {c['id']}")
        print(f"  {len(todo)} jobs")
        return 0

    stamp = args.out or time.strftime("%Y%m%d-%H%M")
    outdir = os.path.join(HERE, "runs", stamp)
    os.makedirs(os.path.join(outdir, "img"), exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    rows = []
    print(f"  {len(todo)} jobs -> bench/runs/{stamp}  (commit {commit})", flush=True)
    for n, case in enumerate(todo, 1):
        j = submit(case)
        if j.get("_error"):
            rows.append({**case, "status": "rejected", "error": j["_body"]})
            print(f"  {n:3}/{len(todo)} {case['id']:26} REJECTED {j['_body'][:60]}",
                  flush=True)
            continue
        st = wait(j["job_id"])
        row = {**case, "job_id": j.get("job_id"), "status": st.get("status"),
               "seconds": st.get("seconds"), "error": st.get("error")}
        if st.get("result_url"):
            name = case["id"].replace("/", "_") + ".png"
            try:
                with urllib.request.urlopen(st["result_url"], timeout=120) as r:
                    open(os.path.join(outdir, "img", name), "wb").write(r.read())
                row["image"] = name
            except Exception as exc:                          # noqa: BLE001
                row["error"] = f"download failed: {exc}"
        rows.append(row)
        print(f"  {n:3}/{len(todo)} {case['id']:26} {row['status']:8} "
              f"{row.get('seconds')}s", flush=True)

    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"stamp": stamp, "commit": commit, "seed": SEED,
                   "base": BASE, "rows": rows}, fh, indent=1)
    done = sum(1 for r in rows if r["status"] == "done")
    print(f"  {done}/{len(rows)} finished -> bench/runs/{stamp}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

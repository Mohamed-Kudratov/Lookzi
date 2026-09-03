#!/usr/bin/env python3
"""Which cut-out model should the packshot use?

    python -m bench.cutters --list
    python -m bench.cutters                     # every model, every garment
    python -m bench.cutters --models birefnet-general,u2net

The packshot has shipped on u2net since the beginning. u2net is from 2020 and
was trained to find *salient objects*, which is why it keeps the coat hanger:
a hanger is a salient object. Measured on the hundred real listing photographs
in `test products/`, fifty-nine of a hundred cut-outs come back with something
narrow and foreign-coloured above the shoulders.

rembg -- already installed on the pod -- carries the BiRefNet family, which is
the current state of the art for this and is what the competitor ships too. So
this comparison costs a download and no new code.

It runs against the pod directly rather than through the queue: a cut-out is
two tenths of a second and the queue would be most of the wall clock.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STOCK = os.path.join(ROOT, "test products")
POD = os.environ.get("POD_BASE", "http://127.0.0.1:18000")

# In the order they are worth trying. u2net first so the baseline is measured
# by the same code on the same day as everything it is compared with.
MODELS = ["u2net", "isnet-general-use", "birefnet-general-lite",
          "birefnet-general", "birefnet-dis", "birefnet-massive",
          "u2net_cloth_seg"]

FOLDERS = ["man/upper", "man/lower", "Woman/Upper", "Woman/lower",
           "Woman/overall"]
EXT = (".jpg", ".jpeg", ".png", ".webp")


def listing(folder, limit=0):
    d = os.path.join(STOCK, folder)
    if not os.path.isdir(d):
        return []
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d))
          if f.lower().endswith(EXT)]
    return fs[:limit] if limit else fs


def cut(path, model, timeout=300):
    """One garment through one cutter, straight at the pod."""
    boundary = "----lz"
    with open(path, "rb") as fh:
        data = fh.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
            f"{model}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"garment\";"
             f" filename=\"g.png\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        POD + "/packshot", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), round(time.time() - started, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--per-folder", type=int, default=0,
                    help="only the first N garments of each folder")
    ap.add_argument("--out", default="cutters")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    garments = []
    for f in FOLDERS:
        garments += listing(f, args.per_folder)
    if args.list:
        print(f"  {len(models)} models x {len(garments)} garments = "
              f"{len(models) * len(garments)} cut-outs")
        for m in models:
            print("   ", m)
        return 0

    outdir = os.path.join(HERE, "runs", args.out)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    results_path = os.path.join(outdir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as fh:
            rows = json.load(fh).get("rows", [])
    done = {(r["model"], r["source"]) for r in rows if r.get("image")}

    for model in models:
        os.makedirs(os.path.join(outdir, model), exist_ok=True)
        first = True
        for n, path in enumerate(garments, 1):
            if (model, path) in done:
                continue
            name = os.path.basename(path) + ".png"
            try:
                png, secs = cut(path, model)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:160]
                print(f"  {model:22} {name[:28]:30} HTTP {exc.code} {detail}",
                      flush=True)
                rows.append({"model": model, "source": path, "error": detail})
                continue
            except Exception as exc:                          # noqa: BLE001
                print(f"  {model:22} {name[:28]:30} {type(exc).__name__}",
                      flush=True)
                rows.append({"model": model, "source": path,
                             "error": type(exc).__name__})
                continue
            with open(os.path.join(outdir, model, name), "wb") as fh:
                fh.write(png)
            rows.append({"model": model, "source": path,
                         "image": os.path.join(model, name), "seconds": secs})
            if first:
                # The first call of a model pays for its download and load;
                # every figure after it is the real one.
                print(f"  {model}: first cut {secs}s (loading), "
                      f"{len(garments)} to go", flush=True)
                first = False
            if n % 20 == 0:
                with open(results_path, "w", encoding="utf-8") as fh:
                    json.dump({"rows": rows}, fh, indent=1)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, indent=1)
        got = [r for r in rows if r["model"] == model and r.get("image")]
        if got:
            secs = sorted(r["seconds"] for r in got)[len(got) // 2]
            print(f"  {model:22} {len(got):3} cut, median {secs}s", flush=True)
    print(f"  -> bench/runs/{args.out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

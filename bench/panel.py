#!/usr/bin/env python3
"""The VTON panel: twelve garments, one run, the same twelve every time.

    python bench/panel.py            # run it and write bench/panel_<stamp>/

Why this exists. Every fix today was judged on whichever garment happened to
be at hand, and twice that produced a fix that was announced and then failed
on the next garment: routing bottoms to the instruct engine looked solved on
five navy skirts and returned a striped rainbow one for a denim mini. A panel
that does not change between runs is the only way to tell a real improvement
from a different picture.

Each garment goes through the packshot first. That is what the product tells a
seller to do, and it removes the thing that makes a seller's own photograph
hard to judge -- the bed, the wooden floor, the shadow -- so what is left to
look at is the try-on's own work.

The number at the end is not a verdict. It measures one thing: whether the
garment on the model is the colour of the garment that was sent. That catches
the failure that cost the most today -- a garment replaced by an invented one
-- and it is blind to fit, to placement, and to the model's own clothes
showing through. Those still need eyes, which is what the contact sheet is for.
"""
import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

BASE = os.environ.get("LOOKZI_BASE", "http://127.0.0.1:8080")
CLIENT = os.environ.get("LOOKZI_CLIENT", "web-dc705b51-600c-4812-9230-fd7520f4a7fa")
HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "panel.txt")

# man/* garments on a man, Woman/* on a woman. Dressing a woman in a man's
# jacket measures the model's tolerance for a silly request rather than
# anything about the garment.
MODE = {"tops": "upper", "bottoms": "lower", "one-pieces": "overall"}


def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method="POST" if data else "GET",
        headers={"X-Client-Id": CLIENT, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read() or b"{}")


def upload(path):
    slot = api("/uploads", {"kind": "garment", "content_type": "image/png"})
    urllib.request.urlopen(urllib.request.Request(
        BASE + slot["url"], data=open(path, "rb").read(), method="PUT",
        headers={"Content-Type": "image/png"}), timeout=300)
    return slot["key"]


def run(body, label, timeout=900):
    started = time.time()
    job = api("/jobs", body)
    while time.time() - started < timeout:
        state = api("/jobs/" + job["job_id"])
        if state["status"] in ("done", "failed"):
            print(f"  {label:30} {state['status']:6} {time.time()-started:5.1f}s "
                  f"{(state.get('error') or '')[:50]}", flush=True)
            return state
        time.sleep(3)
    print(f"  {label:30} timed out", flush=True)
    return {"status": "failed"}


def fetch(url):
    return Image.open(io.BytesIO(
        urllib.request.urlopen(BASE + url, timeout=300).read())).convert("RGB")


def garment_colour(im, mask=None):
    """The average colour of the garment, ignoring near-white ground.

    Hue and saturation rather than RGB: the try-on relights the garment, and
    a navy skirt photographed under a bulb and rendered under studio light is
    the same navy at two brightnesses. Comparing RGB called that a change.
    """
    a = np.asarray(im).astype(float) / 255.0
    if mask is None:
        mx, mn = a.max(axis=2), a.min(axis=2)
        mask = ~((mx > 0.88) & ((mx - mn) < 0.10))       # not near-white
    if mask.sum() < 200:
        return None
    px = a[mask]
    mx, mn = px.max(axis=1), px.min(axis=1)
    chroma = mx - mn
    weight = chroma * mx + 1e-6                          # colourful and lit
    return {"r": float((px[:, 0] * weight).sum() / weight.sum()),
            "g": float((px[:, 1] * weight).sum() / weight.sum()),
            "b": float((px[:, 2] * weight).sum() / weight.sum()),
            "chroma": float((chroma * mx).sum() / mx.sum()),
            "value": float(mx.mean())}


def changed_region(before, after):
    """Where the try-on altered the picture -- which is where the garment is."""
    b = np.asarray(before.resize(after.size, Image.LANCZOS)).astype(float)
    a = np.asarray(after).astype(float)
    d = np.abs(a - b).mean(axis=2)
    return d > max(18.0, float(np.percentile(d, 70)))


def colour_gap(one, two):
    """0 is the same colour, 1 is nothing alike."""
    if not one or not two:
        return None
    dr = one["r"] - two["r"]
    dg = one["g"] - two["g"]
    db = one["b"] - two["b"]
    return round(float(min(1.0, (dr * dr + dg * dg + db * db) ** 0.5)), 3)


def main():
    rows = [l.split("\t") for l in open(PANEL).read().splitlines() if l.strip()]
    stamp = datetime.now().strftime("%m%d-%H%M")
    out_dir = os.path.join(HERE, f"panel_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    models = api("/models")
    woman = [m for m in models if m["gender"] == "woman" and m.get("preview")][0]
    man = [m for m in models if m["gender"] == "man" and m.get("preview")][0]
    print(f"  models: {woman['display_name']} / {man['display_name']}", flush=True)

    report = []
    for i, (group, path) in enumerate(rows, 1):
        name = os.path.basename(path)[:8]
        model = man if path.startswith("man") else woman
        mode = MODE[group]
        key = upload(path)
        n = int(time.time() * 1000)

        shot = run({"tool": "packshot", "garment_key": key, "mode": mode,
                    "seed": 13, "idem_key": f"panel-p-{n}"}, f"{i:2d} {name} packshot")
        if shot["status"] != "done":
            report.append({"n": i, "group": group, "file": name, "stage": "packshot"})
            continue
        shot_key = shot["results"][0]["key"]
        shot_im = fetch(shot["results"][0]["url"])
        shot_im.save(os.path.join(out_dir, f"{i:02d}_packshot.png"))

        worn = run({"tool": "product-to-model", "garment_key": shot_key,
                    "mode": mode, "model_id": model["id"], "seed": 7,
                    "idem_key": f"panel-t-{n}"}, f"{i:2d} {name} try-on")
        if worn["status"] != "done":
            report.append({"n": i, "group": group, "file": name, "stage": "try-on"})
            continue
        worn_im = fetch(worn["results"][0]["url"])
        worn_im.save(os.path.join(out_dir, f"{i:02d}_worn.png"))

        model_im = fetch(model["preview"])
        gap = colour_gap(garment_colour(shot_im),
                         garment_colour(worn_im, changed_region(model_im, worn_im)))
        report.append({"n": i, "group": group, "file": name, "stage": "ok",
                       "colour_gap": gap, "size": list(worn_im.size),
                       "seconds": worn["results"][0].get("seconds")})
        print(f"      colour gap {gap}", flush=True)

    open(os.path.join(out_dir, "report.json"), "w").write(json.dumps(report, indent=1))
    sheet(out_dir, rows, report)
    ok = [r for r in report if r["stage"] == "ok" and r.get("colour_gap") is not None]
    if ok:
        gaps = sorted(r["colour_gap"] for r in ok)
        print(f"\n  {len(ok)} of {len(rows)} completed. colour gap: "
              f"best {gaps[0]}, median {gaps[len(gaps)//2]}, worst {gaps[-1]}",
              flush=True)
    print(f"  {out_dir}", flush=True)


def sheet(out_dir, rows, report):
    """Three across per garment: what was sent, the packshot, the model."""
    H, pad, bar = 210, 6, 13
    lines = []
    for r in report:
        if r["stage"] != "ok":
            continue
        i = r["n"]
        trio = [Image.open(rows[i - 1][1]).convert("RGB"),
                Image.open(os.path.join(out_dir, f"{i:02d}_packshot.png")),
                Image.open(os.path.join(out_dir, f"{i:02d}_worn.png"))]
        trio = [t.resize((int(t.width * H / t.height), H), Image.LANCZOS)
                for t in trio]
        lines.append((f"{i:02d} {r['group']}  gap {r.get('colour_gap')}", trio))
    if not lines:
        return
    per = 2
    rows_n = (len(lines) + per - 1) // per
    unit = max(sum(t.width for t in ts) + pad * 3 for _, ts in lines)
    w = unit * per + pad
    h = rows_n * (H + bar + pad) + pad
    s = Image.new("RGB", (w, h), (238, 238, 241))
    d = ImageDraw.Draw(s)
    for k, (label, ts) in enumerate(lines):
        x0 = pad + (k % per) * unit
        y0 = pad + (k // per) * (H + bar + pad)
        d.text((x0 + 2, y0), label, fill=(40, 40, 48))
        x = x0
        for t in ts:
            s.paste(t, (x, y0 + bar)); x += t.width + pad
    s.save(os.path.join(out_dir, "sheet.jpg"), quality=86)


if __name__ == "__main__":
    sys.exit(main())

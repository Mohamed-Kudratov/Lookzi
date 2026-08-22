#!/usr/bin/env python3
"""Record how long each phase actually took.

Every session so far has re-answered "how long does this take" from memory or
from scrollback, and scrollback does not survive a pod. The numbers matter for
planning -- whether the roster finishes tonight, whether a load is worth
optimising, whether something has silently got slower -- so they are appended to
a file on the volume, which outlives the container.

    from timing import phase

    with phase("model load"):
        pipe = load()

Prints as it goes and appends one CSV row per phase. Failures are recorded too,
with the exception type: a phase that took twenty minutes and then raised is the
most useful row in the file, and the one most likely to be lost.
"""
import csv
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

LOG = os.environ.get("TIMING_LOG", "/workspace/timings.csv")
FIELDS = ["when", "run", "phase", "seconds", "detail", "error"]

# One id per process, so rows from concurrent or interleaved runs stay separable.
RUN = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")


def record(name, seconds, detail="", error=""):
    row = {"when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "run": RUN, "phase": name, "seconds": round(seconds, 1),
           "detail": detail, "error": error}
    try:
        os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
        new = not os.path.exists(LOG)
        with open(LOG, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    except OSError as exc:
        # Never let bookkeeping kill a run that is otherwise working.
        print(f"  [timing] could not write {LOG}: {exc}", flush=True)
    return row


@contextmanager
def phase(name, detail=""):
    print(f"  [{name}] start", flush=True)
    t = time.time()
    try:
        yield
    except BaseException as exc:
        record(name, time.time() - t, detail, type(exc).__name__)
        print(f"  [{name}] FAILED after {time.time() - t:.1f}s "
              f"({type(exc).__name__})", flush=True)
        raise
    record(name, time.time() - t, detail)
    print(f"  [{name}] {time.time() - t:.1f}s", flush=True)


def summary(path=LOG, run=None):
    """Totals per phase, newest run by default."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    run = run or rows[-1]["run"]
    return [r for r in rows if r["run"] == run]


if __name__ == "__main__":
    import sys
    rows = summary(run=sys.argv[1] if len(sys.argv) > 1 else None)
    if not rows:
        raise SystemExit(f"no timings in {LOG}")
    print(f"run {rows[0]['run']}")
    total = 0.0
    for r in rows:
        total += float(r["seconds"])
        flag = f"  FAILED {r['error']}" if r["error"] else ""
        print(f"  {r['phase']:24} {float(r['seconds']):8.1f}s  {r['detail']}{flag}")
    print(f"  {'total':24} {total:8.1f}s  ({total / 60:.1f} min)")

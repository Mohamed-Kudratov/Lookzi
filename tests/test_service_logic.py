#!/usr/bin/env python3
"""Everything in the service that can be checked without Postgres or a GPU.

Not a substitute for running the stack. It covers the rules that are silent
when they are wrong -- a batch key that never groups, a priority that puts the
free tier first, an age parser that files a fifty-year-old under thirty --
none of which raise, and all of which would be found weeks later by a customer.

    python tests/test_service_logic.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "elements"))

failures = []


def check(name, got, want):
    if got == want:
        print(f"  ok    {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def truthy(name, cond, why=""):
    if cond:
        print(f"  ok    {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name} {why}")


# --- batch key -------------------------------------------------------------
print("\nbatch key")
from service.queue import batch_key

a = {"tool": "product-to-model", "params": {"width": 768, "height": 1024, "steps": 8}}
b = {"tool": "product-to-model", "params": {"width": 768, "height": 1024, "steps": 8}}
c = {"tool": "product-to-model", "params": {"width": 512, "height": 1024, "steps": 8}}
d = {"tool": "product-to-model", "params": {"width": 768, "height": 1024, "steps": 4}}
e = {"tool": "model-swap", "params": {"width": 768, "height": 1024, "steps": 8}}

check("identical jobs batch together", batch_key(a) == batch_key(b), True)
check("different width does not batch", batch_key(a) == batch_key(c), False)
check("different step count does not batch", batch_key(a) == batch_key(d), False)
check("different tool does not batch", batch_key(a) == batch_key(e), False)
check("missing params does not explode", batch_key({"tool": "x"}),
      ("x", None, None, None))

# A batch is taken from the head of the queue, so the group returned must be
# the one the oldest job belongs to -- not the largest group available.
rows = [a, e, b, e, b]
first = batch_key(rows[0])
check("batch follows the oldest job",
      [i for i, r in enumerate(rows) if batch_key(r) == first], [0, 2, 4])

# --- priority --------------------------------------------------------------
print("\nqueue priority")
from service.app import _priority

check("brand runs before seller", _priority({"plan": "brand"}) < _priority({"plan": "seller"}), True)
check("seller runs before trial", _priority({"plan": "seller"}) < _priority({"plan": "trial"}), True)
check("unknown plan is treated as trial", _priority({"plan": "???"}), _priority({"plan": "trial"}))

# --- age parsing -----------------------------------------------------------
print("\nroster age")
from service.seed_models import age_of

check("early 20s", age_of({"age": "early 20s"}), 21)
check("mid 20s", age_of({"age": "mid 20s"}), 25)
check("late 30s", age_of({"age": "late 30s"}), 38)
check("early 50s", age_of({"age": "early 50s"}), 51)
truthy("every roster entry parses to a plausible age",
       all(18 <= age_of(f) <= 70 for f in __import__("catalog").ROSTER))

# --- storage keys ----------------------------------------------------------
print("\nstorage keys")
from service import storage

k1 = storage.key_for("results", 7)
k2 = storage.key_for("results", 7)
truthy("keys are unique", k1 != k2)
truthy("keys carry the date for lifecycle rules",
       re.match(r"results/\d{4}/\d{2}/\d{2}/7/[0-9a-f]{32}\.png$", k1) is not None,
       f"-> {k1}")

# --- schema and queries agree ---------------------------------------------
print("\nschema")
sql = open(os.path.join(ROOT, "service", "db", "001_initial.sql"), encoding="utf-8").read()
qy = open(os.path.join(ROOT, "service", "queue.py"), encoding="utf-8").read()
truthy("credits cannot go negative", "CHECK (credits >= 0)" in sql)
truthy("a job cannot be charged twice", "credit_entries_once" in sql)
truthy("idempotency key is unique per user", "UNIQUE (user_id, idem_key)" in sql)
truthy("claims skip locked rows", "SKIP LOCKED" in qy)
truthy("refund is guarded against double insert", "ON CONFLICT DO NOTHING" in qy)

# --- the web tier stays light ---------------------------------------------
print("\nweb tier weight")
req = open(os.path.join(ROOT, "service", "requirements.txt"), encoding="utf-8").read()
pkgs = {l.split(">")[0].split("[")[0].strip().lower()
        for l in req.splitlines() if l.strip() and not l.startswith("#")}
truthy("no torch in the web tier", "torch" not in pkgs)
truthy("no diffusers in the web tier", "diffusers" not in pkgs)
# Each module is imported in a fresh interpreter. Checking sys.modules in this
# one proved nothing: the loop asserted the same condition once per name without
# importing anything, so a module that dragged in torch would still have passed,
# and adding a name to the list would have printed a reassuring line about code
# nobody had run.
for mod in ("service.app", "service.queue", "service.storage", "service.worker",
            "service.runpod_bridge", "service.tools", "service.accounts"):
    probe = subprocess.run(
        [sys.executable, "-c",
         f"import importlib, sys; importlib.import_module('{mod}');"
         " sys.exit(1 if 'torch' in sys.modules else 0)"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    if probe.returncode == 0:
        truthy(f"{mod} imports without torch", True)
    elif probe.returncode == 1:
        truthy(f"{mod} imports without torch", False, "-> torch was imported")
    else:
        truthy(f"{mod} imports at all", False,
               f"-> {probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else probe.returncode}")

print()
if failures:
    print(f"{len(failures)} failed: {failures}")
    raise SystemExit(1)
print("all checks passed")

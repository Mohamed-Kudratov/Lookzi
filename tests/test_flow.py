#!/usr/bin/env python3
"""The whole flow against a real Postgres, without a GPU.

The logic tests cover rules that can be checked in isolation. These cover the
ones that only appear when a database is actually there: whether a transaction
really is atomic, whether two workers really do take different jobs, whether a
refund really is idempotent under a unique index.

Skips itself, loudly, when there is no database to talk to.

    docker compose up -d db storage
    python tests/test_flow.py
"""
import os
import sys
import threading
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from service import queue as q  # noqa: E402

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


def reachable():
    try:
        c = q.connect()
        c.execute("SELECT 1").fetchone()
        c.close()
        return True
    except Exception as exc:
        print(f"no database at {q.DSN.split('@')[-1]}: {type(exc).__name__}")
        print("start one with:  docker compose up -d db")
        return False


def fresh_user(conn, credits=10):
    """A throwaway account per test, so one test cannot spend another's credit."""
    with conn.transaction():
        u = conn.execute("INSERT INTO users (credits) VALUES (%s) RETURNING *",
                         (credits,)).fetchone()
        conn.execute(
            """INSERT INTO identities (user_id, kind, value, verified_at)
               VALUES (%s, 'telegram', %s, now())""",
            (u["id"], f"test-{uuid.uuid4().hex[:12]}"))
    return u


def balance(conn, user_id):
    return conn.execute("SELECT credits FROM users WHERE id = %s",
                        (user_id,)).fetchone()["credits"]


def ledger_total(conn, user_id):
    """The ledger is the truth; users.credits is a cache of it.

    Comparing the two is the check that matters -- if they ever disagree, a
    customer has been charged for something no row can account for.
    """
    row = conn.execute(
        "SELECT coalesce(sum(delta), 0) AS n FROM credit_entries WHERE user_id = %s",
        (user_id,)).fetchone()
    return row["n"]


PARAMS = {"width": 768, "height": 1024, "steps": 8, "garment_key": "k", "person_key": "p"}

def new_tool():
    """A tool name nothing else will ever claim, unique per section.

    Two things would otherwise interfere. A worker running in the background
    claims anything it is not told to skip -- that is what happened the first
    time this ran against live containers. And each section leaves its own
    jobs queued behind it, so a later section claiming "the next job" gets an
    earlier section's instead, and measures the wrong thing.

    A fresh name per section makes every claim in this file provably about the
    jobs that section just created.
    """
    return "test-" + uuid.uuid4().hex[:10]


def main():
    if not reachable():
        return 0

    conn = q.connect()
    print("\ncharging")
    tool = new_tool()
    u = fresh_user(conn, credits=2)
    job, charged = q.submit(conn, u["id"], tool, PARAMS)
    check("a job costs a credit", balance(conn, u["id"]), 1)
    check("the charge is reported", charged, True)
    check("the ledger agrees with the balance", ledger_total(conn, u["id"]), -1)

    print("\nidempotency")
    key = f"idem-{uuid.uuid4().hex[:8]}"
    j1, c1 = q.submit(conn, u["id"], tool, PARAMS, idem_key=key)
    j2, c2 = q.submit(conn, u["id"], tool, PARAMS, idem_key=key)
    check("the same request makes one job", j1["id"], j2["id"])
    check("and charges once", (c1, c2), (True, False))
    check("balance reflects one charge", balance(conn, u["id"]), 0)

    print("\ncredit floor")
    try:
        q.submit(conn, u["id"], tool, PARAMS)
        failures.append("empty balance is refused")
        print("  FAIL  empty balance is refused -- it went through")
    except q.InsufficientCredit as exc:
        check("empty balance is refused", (exc.have, exc.need), (0, 1))

    print("\nclaiming")
    tool = new_tool()
    u2 = fresh_user(conn, credits=5)
    made = [q.submit(conn, u2["id"], tool, PARAMS)[0]["id"]
            for _ in range(3)]
    claimed = []

    def take():
        c = q.connect()
        try:
            j = q.claim(c, tools=[tool])
            if j:
                claimed.append(j["id"])
        finally:
            c.close()

    # Three connections claiming at once is the case SKIP LOCKED exists for.
    # Without it these either serialise or collide; with it each gets its own.
    threads = [threading.Thread(target=take) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    check("three workers take three different jobs", len(set(claimed)), 3)
    truthy("all of them came from the queue", set(claimed) <= set(made))

    print("\nfinishing")
    q.finish(conn, claimed[0], "results/test.png", width=768, height=1024, seconds=14.3)
    st = q.status(conn, claimed[0])
    check("status is done", st["status"], "done")
    check("the result is attached", st["object_key"], "results/test.png")

    print("\nfailure, retry and refund")
    tool = new_tool()
    u3 = fresh_user(conn, credits=3)
    j, _ = q.submit(conn, u3["id"], tool, PARAMS)
    check("charged on submit", balance(conn, u3["id"]), 2)

    outcomes = []
    for _ in range(q.MAX_ATTEMPTS + 1):
        claimed_job = q.claim(conn, tools=[tool])
        if claimed_job is None or claimed_job["id"] != j["id"]:
            continue
        outcomes.append(q.fail(conn, j["id"], "simulated"))
    truthy("the test claimed its own job", bool(outcomes),
           "-- something else took it first")
    if outcomes:
        check("retried before giving up", outcomes[:-1],
              ["requeued"] * (len(outcomes) - 1))
        check("finally failed", outcomes[-1], "failed")
    check("credit came back", balance(conn, u3["id"]), 3)
    check("ledger nets to zero after a refund", ledger_total(conn, u3["id"]), 0)

    # A second refund must be a no-op, not free credit.
    q.fail(conn, j["id"], "simulated again")
    check("a second refund changes nothing", balance(conn, u3["id"]), 3)

    print("\nstale jobs")
    tool = new_tool()
    u4 = fresh_user(conn, credits=2)
    j4, _ = q.submit(conn, u4["id"], tool, PARAMS)
    q.claim(conn, tools=[tool])
    conn.execute("UPDATE jobs SET claimed_at = now() - INTERVAL '1 day' WHERE id = %s",
                 (j4["id"],))
    released = q.release_stale(conn)
    truthy("a job abandoned by a dead worker is requeued", j4["id"] in released)
    check("and is queued again", q.status(conn, j4["id"])["status"], "queued")

    print("\nbatching")
    tool = new_tool()
    u5 = fresh_user(conn, credits=10)
    for w in (768, 768, 512, 768):
        p = dict(PARAMS, width=w)
        q.submit(conn, u5["id"], tool, p)
    batch = q.claim_batch(conn, 4, tools=[tool])
    widths = {(b["params"] or {}).get("width") for b in batch}
    check("a batch is one shape only", len(widths), 1)
    truthy("and it follows the oldest job", widths == {768}, f"-> {widths}")

    conn.close()
    print()
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

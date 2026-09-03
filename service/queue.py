#!/usr/bin/env python3
"""The job queue, in Postgres.

It was an in-memory `Queue()` inside the web process. That has three faults
that only appear once there is more than one user: a restart loses every queued
job, no second worker can ever see the queue, and the customer has no history
because nothing was written down.

A table fixes all three at once, and the queue becomes the history.

The claim is the only interesting part:

    SELECT ... FOR UPDATE SKIP LOCKED

Without SKIP LOCKED, two workers reading at the same moment either block on
each other -- serialising the whole fleet through one row -- or both take the
same job and the customer is charged twice for one image. SKIP LOCKED tells
Postgres to step over rows another transaction already holds, so every worker
walks away with a different job and none of them wait.

Nothing here imports torch or touches a GPU. It is the same code whether the
worker on the other end is a real A100 or the stub in fake_worker.py.
"""
import json
import os
import socket
import uuid

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("DATABASE_URL", "postgresql://lookzi:lookzi@localhost:5432/lookzi")

# A worker that dies mid-job leaves the row in 'running' for ever. Anything
# claimed longer ago than this is assumed dead and returned to the queue.
STALE_AFTER = os.environ.get("JOB_STALE_AFTER", "15 minutes")
MAX_ATTEMPTS = int(os.environ.get("JOB_MAX_ATTEMPTS", "3"))

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


def connect():
    """Autocommit, with explicit transactions where they are needed.

    Not a style preference -- the alternative is broken. Without autocommit,
    any bare conn.execute() outside a transaction block opens a transaction
    that never closes, and every later `with conn.transaction()` becomes a
    savepoint inside it rather than a transaction of its own. Nothing commits.

    That is not theoretical: the web tier reads a user, then submits a job, on
    the same connection. Under the old setting the read opened the transaction,
    the submit wrote into it, the request returned a job id, the connection
    closed, and Postgres rolled the whole thing back. The customer would have
    been charged for a job no worker could see, and then not charged either,
    because the charge went with it. Silent, and total.

    With autocommit, a bare execute commits as it goes and every
    `with conn.transaction()` is a real transaction that commits at the end of
    the block -- which is what the rest of this module assumes.
    """
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


# ---------------------------------------------------------------------------
# writing

def submit(conn, user_id, tool, params, model_id=None, cost=1,
           priority=100, idem_key=None):
    """Charge the customer and queue the job, or neither.

    One transaction on purpose. Charging in one statement and inserting in
    another leaves a window where a crash takes the credit and produces no
    image, which is the failure a customer notices and never forgives.

    Returns (job, charged). `charged` is False when an identical request has
    already been queued -- a retried HTTP call, a double tap in Telegram -- in
    which case the original job comes back and no second credit is taken.
    """
    with conn.transaction():
        if idem_key:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE user_id = %s AND idem_key = %s",
                (user_id, idem_key)).fetchone()
            if existing:
                return existing, False

        # Locking the user row first serialises concurrent submissions by the
        # same person, so two requests cannot both read a balance of 1 and both
        # decide they can afford it.
        row = conn.execute(
            "SELECT credits, plan FROM users WHERE id = %s FOR UPDATE",
            (user_id,)).fetchone()
        if row is None:
            raise LookupError(f"no such user: {user_id}")

        # An account on the unlimited plan is never charged and never blocked.
        # It exists for whoever is testing the product: being stopped by your
        # own credit ledger halfway through checking whether a tool works is a
        # waste of a person and of a rented card.
        #
        # The job still records what it would have cost, so the review page and
        # any pricing question can still be answered from the history. Nothing
        # is written to the ledger, because nothing moved.
        free = row["plan"] == "unlimited"
        if not free and row["credits"] < cost:
            raise InsufficientCredit(row["credits"], cost)

        job = conn.execute(
            """INSERT INTO jobs (user_id, tool, model_id, params, credits_cost,
                                 priority, idem_key)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *""",
            (user_id, tool, model_id, json.dumps(params), cost, priority,
             idem_key)).fetchone()

        if free:
            return job, False

        conn.execute("UPDATE users SET credits = credits - %s WHERE id = %s",
                     (cost, user_id))
        conn.execute(
            """INSERT INTO credit_entries (user_id, delta, reason, job_id)
               VALUES (%s, %s, 'job', %s)""",
            (user_id, -cost, job["id"]))
        return job, True


class InsufficientCredit(Exception):
    def __init__(self, have, need):
        super().__init__(f"{have} credits, needs {need}")
        self.have, self.need = have, need


# ---------------------------------------------------------------------------
# reading, by workers

def claim(conn, tools=None, worker_id=WORKER_ID):
    """Take one job, or return None.

    Ordered by priority then age, so a paid tier can jump the queue by carrying
    a lower number while everything within a tier stays first-come.

    `tools` lets a worker pool restrict itself -- the video pool should never
    pick up a try-on job, because the two need different models resident.
    """
    # One statement, with the lock taken inside a subquery.
    #
    # The obvious form -- SELECT ... ORDER BY ... FOR UPDATE SKIP LOCKED
    # LIMIT 1, then UPDATE -- is wrong, and wrong silently. Postgres sorts
    # before it locks, so every session picks the same first row; the ones
    # that lose the race skip it and return nothing rather than moving to the
    # next row, because LIMIT has already been applied. Three workers polling
    # a queue of three jobs came back with one job and two empty hands, which
    # in production is a fleet where only one worker ever does anything.
    #
    # Putting the lock in a subquery lets each session settle on its own row
    # before the outer statement claims it.
    with conn.transaction():
        return conn.execute(
            """UPDATE jobs SET status = 'running', claimed_by = %s,
                               claimed_at = now(), started_at = now(),
                               attempts = attempts + 1
                WHERE id = (
                      SELECT id FROM jobs
                       WHERE status = 'queued'
                         AND (%s::text[] IS NULL OR tool = ANY(%s::text[]))
                       ORDER BY priority, created_at
                       FOR UPDATE SKIP LOCKED
                       LIMIT 1)
             RETURNING *""",
            (worker_id, tools, tools)).fetchone()


def batch_key(job):
    """What makes two jobs runnable in the same forward pass.

    Shapes cannot be stacked into one tensor, and mixing step counts would
    give some images more denoising than they were charged for. Tool is in the
    key because two tools may want different adapters resident.

    Lifted out of the query so it can be tested without a database, which
    matters: this is the rule the whole batching gain rests on, and it is
    silent when it is wrong -- a bad key does not crash, it just never batches.
    """
    p = job.get("params") or {}
    return (job.get("tool"), p.get("width"), p.get("height"), p.get("steps"))


def claim_batch(conn, size, tools=None, worker_id=WORKER_ID):
    """Take up to `size` jobs that can run in one forward pass.

    Batching is what makes one GPU worth roughly twice as much: a single image
    leaves most of the card idle, and four together take about twice as long
    rather than four times.

    The catch is that a batch has to be uniform -- same resolution, same step
    count -- so jobs are grouped by that signature and only the first group is
    returned. Mixing shapes into one tensor is not possible, and mixing step
    counts would give some images more denoising than they were charged for.
    """
    # Same shape as claim(), for the same reason: the rows are locked inside
    # the subquery, then claimed by the outer UPDATE. Candidates are drawn
    # wider than the batch because they still have to be filtered down to one
    # uniform shape afterwards.
    with conn.transaction():
        rows = conn.execute(
            """UPDATE jobs SET status = 'running', claimed_by = %s,
                               claimed_at = now(), started_at = now(),
                               attempts = attempts + 1
                WHERE id IN (
                      SELECT id FROM jobs
                       WHERE status = 'queued'
                         AND (%s::text[] IS NULL OR tool = ANY(%s::text[]))
                       ORDER BY priority, created_at
                       FOR UPDATE SKIP LOCKED
                       LIMIT %s)
             RETURNING *""",
            (worker_id, tools, tools, size * 4)).fetchall()
        if not rows:
            return []

        # The claim already happened, so anything outside the chosen shape has
        # to go back. Releasing rather than holding it keeps a worker from
        # sitting on jobs another pool could be running.
        rows.sort(key=lambda r: (r["priority"], r["created_at"]))
        first = batch_key(rows[0])
        batch = [r for r in rows if batch_key(r) == first][:size]
        keep = {r["id"] for r in batch}
        spare = [r["id"] for r in rows if r["id"] not in keep]
        if spare:
            conn.execute(
                """UPDATE jobs SET status = 'queued', claimed_by = NULL,
                                   claimed_at = NULL, started_at = NULL,
                                   attempts = attempts - 1
                    WHERE id = ANY(%s)""", (spare,))
        return batch


# ---------------------------------------------------------------------------
# finishing

def finish(conn, job_id, object_key, kind="image", width=None, height=None,
           seconds=None, variant=None, notes=None, extras=()):
    """Store what the job produced and mark it done.

    `extras` is for a tool that produces more than one picture and lets the
    customer choose. The packshot is the case: a generative pass and the plain
    cut-out of the same garment, both kept, because a gate that silently
    swapped one for the other would hide the choice from the person who has
    the garment in their hand.

    The first one is the primary and is what older clients see as the result;
    the rest are labelled and returned beside it.
    """
    import json as _json
    with conn.transaction():
        rows = [(object_key, kind, width, height, seconds, variant, notes)]
        rows += [(e["object_key"], e.get("kind", "image"), e.get("width"),
                  e.get("height"), e.get("seconds"), e.get("variant"),
                  e.get("notes")) for e in extras]
        for key, k, w, h, secs, var, note in rows:
            conn.execute(
                """INSERT INTO results (job_id, object_key, kind, width, height,
                                        seconds, variant, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (job_id, COALESCE(variant, '')) DO UPDATE
                       SET object_key = EXCLUDED.object_key,
                           width = EXCLUDED.width, height = EXCLUDED.height,
                           seconds = EXCLUDED.seconds, notes = EXCLUDED.notes""",
                (job_id, key, k, w, h, secs, var,
                 _json.dumps(note) if note is not None else None))
        conn.execute(
            "UPDATE jobs SET status = 'done', finished_at = now() WHERE id = %s",
            (job_id,))


def fail(conn, job_id, error, refund=True):
    """Mark a job failed, retrying it if it has attempts left.

    Credit comes back only when the job is finally given up on. Refunding on
    every attempt and re-charging on every retry would put two entries in the
    ledger for one image and make the history unreadable.
    """
    with conn.transaction():
        job = conn.execute("SELECT * FROM jobs WHERE id = %s FOR UPDATE",
                           (job_id,)).fetchone()
        if job is None:
            return None
        if job["attempts"] < MAX_ATTEMPTS:
            conn.execute(
                """UPDATE jobs SET status = 'queued', claimed_by = NULL,
                                   claimed_at = NULL, error = %s
                    WHERE id = %s""", (error[:2000], job_id))
            return "requeued"

        conn.execute(
            """UPDATE jobs SET status = 'failed', finished_at = now(), error = %s
                WHERE id = %s""", (error[:2000], job_id))
        if refund:
            # The balance moves only if the ledger entry was actually written.
            #
            # The unique index already stopped a second refund from being
            # recorded, but the UPDATE beneath it ran regardless -- so calling
            # fail() twice left one refund in the ledger and two in the
            # balance. The two then disagreed permanently, and the balance is
            # the number the customer spends. RETURNING makes the insert say
            # whether it happened, and the balance follows it.
            # Refunded only if it was charged. An unlimited account pays
            # nothing and writes no ledger entry, so refunding one would mint
            # credits out of a failure -- the more jobs it failed, the richer
            # it would get. The ledger is the record of what happened, so the
            # ledger is what decides.
            paid = conn.execute(
                "SELECT 1 FROM credit_entries WHERE job_id = %s AND reason = 'job'",
                (job_id,)).fetchone()
            if not paid:
                return "given up"
            written = conn.execute(
                """INSERT INTO credit_entries (user_id, delta, reason, job_id)
                   VALUES (%s, %s, 'refund', %s)
                   ON CONFLICT DO NOTHING
                RETURNING id""",
                (job["user_id"], job["credits_cost"], job_id)).fetchone()
            if written:
                conn.execute("UPDATE users SET credits = credits + %s WHERE id = %s",
                             (job["credits_cost"], job["user_id"]))
        return "failed"


def cancel(conn, job_id, user_id=None):
    """Withdraw a job that has not started, and give the credit back.

    Only while it is still queued. Once a worker has claimed it the GPU time is
    already being spent, and stopping it mid-generation would cost the same as
    letting it finish -- so the honest answer at that point is that it is too
    late, not a cancellation that silently does nothing.

    Returns 'cancelled', 'too late', or None when the job is not theirs. The
    user_id check is not decoration: the job id travels in a Telegram callback,
    and a callback can be replayed by whoever received it.
    """
    with conn.transaction():
        job = conn.execute("SELECT * FROM jobs WHERE id = %s FOR UPDATE",
                           (job_id,)).fetchone()
        if job is None or (user_id is not None and job["user_id"] != user_id):
            return None
        if job["status"] != "queued":
            return "too late"

        conn.execute(
            "UPDATE jobs SET status = 'cancelled', finished_at = now() WHERE id = %s",
            (job_id,))
        # Same guard as a refund after failure: the ledger decides whether the
        # balance moves, so a cancel that arrives twice pays once.
        # Refunded only if it was charged; see fail().
        paid = conn.execute(
            "SELECT 1 FROM credit_entries WHERE job_id = %s AND reason = 'job'",
            (job_id,)).fetchone()
        if not paid:
            return "cancelled"

        written = conn.execute(
            """INSERT INTO credit_entries (user_id, delta, reason, job_id)
               VALUES (%s, %s, 'refund', %s)
               ON CONFLICT DO NOTHING
            RETURNING id""",
            (job["user_id"], job["credits_cost"], job_id)).fetchone()
        if written:
            conn.execute("UPDATE users SET credits = credits + %s WHERE id = %s",
                         (job["credits_cost"], job["user_id"]))
        return "cancelled"


def heartbeat(conn, name, tools, done=0, failed=0):
    """Say this worker is alive, and what it handles.

    Liveness used to be inferred from recently claimed jobs, which made an
    idle worker indistinguishable from no worker -- the studio told customers
    nothing was running while one sat there waiting. Saying so directly also
    answers the question scaling asks first: what is already up.
    """
    conn.execute(
        """INSERT INTO workers (name, tools, last_seen, jobs_done, jobs_failed)
           VALUES (%s, %s, now(), %s, %s)
           ON CONFLICT (name) DO UPDATE SET
             tools = EXCLUDED.tools, last_seen = now(),
             jobs_done = EXCLUDED.jobs_done, jobs_failed = EXCLUDED.jobs_failed""",
        (name, tools, done, failed))


def alive(conn, within="60 seconds"):
    """Workers that have checked in recently, newest first."""
    return conn.execute(
        f"""SELECT name, tools, jobs_done, jobs_failed,
                   extract(epoch FROM now() - last_seen)::int AS seconds_ago
              FROM workers
             WHERE last_seen > now() - INTERVAL '{within}'
             ORDER BY last_seen DESC""").fetchall()


def release_stale(conn):
    """Return jobs held by workers that are no longer alive.

    A pod reclaimed mid-job leaves its rows in 'running' with nobody working
    on them. Without this they sit there until someone notices, and the
    customer waits for an image that will never arrive.
    """
    with conn.transaction():
        rows = conn.execute(
            f"""UPDATE jobs SET status = 'queued', claimed_by = NULL, claimed_at = NULL
                 WHERE status = 'running'
                   AND claimed_at < now() - INTERVAL '{STALE_AFTER}'
                 RETURNING id""").fetchall()
        return [r["id"] for r in rows]


# ---------------------------------------------------------------------------
# reading, by the app

def status(conn, job_id):
    # The primary result is the one with no variant, or the earliest if every
    # row is labelled -- so a tool that produces two pictures still has one
    # answer for a client that only knows how to show one.
    job = conn.execute(
        """SELECT j.*, r.object_key, r.kind, r.seconds
             FROM jobs j LEFT JOIN LATERAL (
                    SELECT * FROM results WHERE job_id = j.id
                     ORDER BY (variant IS NOT NULL), id LIMIT 1) r ON true
            WHERE j.id = %s""", (job_id,)).fetchone()
    if job is not None:
        job["results"] = conn.execute(
            """SELECT object_key, kind, width, height, seconds, variant, notes
                 FROM results WHERE job_id = %s
                ORDER BY (variant IS NOT NULL), id""", (job_id,)).fetchall()
    if job and job["status"] == "queued":
        job["position"] = conn.execute(
            """SELECT count(*) AS n FROM jobs
                WHERE status = 'queued'
                  AND (priority, created_at) < (%s, %s)""",
            (job["priority"], job["created_at"])).fetchone()["n"] + 1
    return job


def depth(conn, tools=None):
    return conn.execute(
        """SELECT
             count(*) FILTER (WHERE status = 'queued')  AS queued,
             count(*) FILTER (WHERE status = 'running') AS running
           FROM jobs
          WHERE (%s::text[] IS NULL OR tool = ANY(%s::text[]))""",
        (tools, tools)).fetchone()


def history(conn, user_id, limit=50):
    return conn.execute(
        """SELECT j.id, j.tool, j.model_id, j.status, j.credits_cost,
                  j.created_at, j.finished_at, r.object_key, r.seconds
             FROM jobs j LEFT JOIN results r ON r.job_id = j.id
            WHERE j.user_id = %s
            ORDER BY j.created_at DESC LIMIT %s""",
        (user_id, limit)).fetchall()

#!/usr/bin/env python3
"""The contract between the queue and whatever runs the model.

A worker is a loop: claim a job, run it, record the result. What "run it" means
is the only part that differs between a stub on a laptop and an A100 on
RunPod -- so that is the only part injected, and everything else is shared.

Writing the contract down before the GPU work is what makes the GPU optional
during development. fake_worker.py satisfies it by sleeping; gpu_worker.py
satisfies it by loading 15.8 GB and sampling. The web tier cannot tell them
apart, which means the whole product can be built, demonstrated and corrected
without spending anything on a card.

    python -m service.fake_worker          # no GPU
    python -m service.gpu_worker           # the real thing

A handler receives the job dict and returns a dict:

    {"object_key": str, "kind": "image"|"video",
     "width": int, "height": int, "seconds": float}

Raising is how a handler reports failure; the loop decides whether that means
a retry or a refund.
"""
import os
import signal
import time
import traceback

from . import queue as q

POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "1.0"))
# Stale jobs are swept by whichever worker gets there first, not by a separate
# service. One fewer thing to deploy, and a fleet of one still sweeps.
SWEEP_EVERY = float(os.environ.get("WORKER_SWEEP_SECONDS", "60"))


def tools_from_env():
    """The tool list this worker handles, or None for all of them.

    Read here rather than in each worker, because it was read in one and
    forgotten in the other: the stub then claimed every job in the table,
    which is what a video worker would do to try-on jobs in production.
    """
    raw = os.environ.get("WORKER_TOOLS", "")
    return [t.strip() for t in raw.split(",") if t.strip()] or None


class Worker:
    def __init__(self, handler, tools=..., name=None, batch_size=1):
        self.handler = handler
        # Ellipsis, not None: None is a meaningful value here -- "claim
        # anything" -- and has to stay distinguishable from "not specified".
        self.tools = tools_from_env() if tools is ... else tools
        self.name = name or q.WORKER_ID
        self.batch_size = batch_size
        self.stopping = False
        self.done = 0
        self.failed = 0

    # A pod being reclaimed sends SIGTERM and then waits. Finishing the job in
    # hand before exiting is the difference between a customer getting their
    # image and getting a retry.
    def install_signals(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._stop)
            except (ValueError, OSError):
                pass  # not the main thread, or a platform without it

    def _stop(self, *_):
        if self.stopping:
            raise KeyboardInterrupt("second signal: exiting now")
        print(f"[{self.name}] stopping after the current job", flush=True)
        self.stopping = True

    def run(self):
        self.install_signals()
        print(f"[{self.name}] polling for {self.tools or 'any tool'}", flush=True)
        conn = q.connect()
        last_sweep = 0.0
        try:
            while not self.stopping:
                now = time.time()
                if now - last_sweep > SWEEP_EVERY:
                    released = q.release_stale(conn)
                    if released:
                        print(f"[{self.name}] released {len(released)} stale job(s)",
                              flush=True)
                    last_sweep = now

                job = q.claim(conn, tools=self.tools, worker_id=self.name)
                if job is None:
                    time.sleep(POLL_SECONDS)
                    continue
                self.run_one(conn, job)
        finally:
            conn.close()
            print(f"[{self.name}] stopped · {self.done} done, {self.failed} failed",
                  flush=True)

    def run_one(self, conn, job):
        started = time.time()
        print(f"[{self.name}] {job['id']} {job['tool']} start", flush=True)
        try:
            out = self.handler(job)
            elapsed = round(time.time() - started, 2)
            q.finish(conn, job["id"], out["object_key"],
                     kind=out.get("kind", "image"),
                     width=out.get("width"), height=out.get("height"),
                     seconds=out.get("seconds", elapsed))
            self.done += 1
            print(f"[{self.name}] {job['id']} done in {elapsed}s", flush=True)
        except Exception as exc:
            # Never let one bad job stop the loop. A malformed request, a
            # corrupt upload or a transient failure would otherwise take the
            # worker down and, with it, everybody else's queue.
            self.failed += 1
            traceback.print_exc()
            outcome = q.fail(conn, job["id"], f"{type(exc).__name__}: {exc}")
            print(f"[{self.name}] {job['id']} {outcome}", flush=True)

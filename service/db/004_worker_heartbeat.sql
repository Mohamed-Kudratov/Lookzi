-- Workers say they are alive, rather than being inferred from their work.
--
-- Liveness was read from jobs claimed in the last two minutes, so an idle but
-- perfectly healthy worker was indistinguishable from no worker at all -- the
-- studio told a customer "No worker running" while one sat there waiting for
-- something to do. It also matters for scaling: deciding whether to add
-- capacity needs to know what capacity is already up.

CREATE TABLE IF NOT EXISTS workers (
    name        TEXT PRIMARY KEY,
    tools       TEXT[],
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    jobs_done   INTEGER NOT NULL DEFAULT 0,
    jobs_failed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS workers_alive ON workers (last_seen DESC);

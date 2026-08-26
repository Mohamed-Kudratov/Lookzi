-- Lookzi, first migration.
--
-- Two decisions are baked in here and both are cheap now and expensive later.
--
-- Identities are their own table rather than columns on users. The MVP signs
-- people in through Telegram, which hands us an id with every message; email
-- and password come after the MVP proves the product. Adding them then is an
-- insert against the same user rather than a migration of every row and every
-- query that touched an `email` column. See docs/AUTH.md.
--
-- The job queue is a table rather than a process's memory. In memory it dies
-- with the process -- a restart loses every queued job, and no second worker
-- can ever see it. In a table, work survives, several workers draw from the
-- same queue, and the queue is also the history the customer reads.

BEGIN;

CREATE TABLE users (
    id           BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    plan         TEXT         NOT NULL DEFAULT 'trial',
    -- Credits are an integer count, never a float. A balance that can be
    -- 4.999999 is a balance that eventually lets someone generate for free.
    credits      INTEGER      NOT NULL DEFAULT 20 CHECK (credits >= 0),
    locale       TEXT         NOT NULL DEFAULT 'en',
    blocked_at   TIMESTAMPTZ
);

CREATE TABLE identities (
    id           BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id      BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind         TEXT         NOT NULL CHECK (kind IN ('telegram','email','phone')),
    -- Lower-cased before insert for email; the raw numeric id for telegram.
    value        TEXT         NOT NULL,
    verified_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (kind, value)
);
CREATE INDEX identities_user ON identities (user_id);

-- The roster a customer can choose from. Mirrors elements/catalog.py, which
-- stays the source of truth for generation; this table exists so a job can
-- reference a model by a stable id and so the app can list them without
-- importing Python.
CREATE TABLE models (
    id            TEXT        PRIMARY KEY,          -- f_cauz_20s_avg
    display_name  TEXT        NOT NULL,             -- Nigora
    age           INTEGER     NOT NULL,
    gender        TEXT        NOT NULL CHECK (gender IN ('woman','man')),
    ethnicity     TEXT        NOT NULL,
    build         TEXT        NOT NULL,
    modest        BOOLEAN     NOT NULL DEFAULT false,
    exclusive_to  BIGINT      REFERENCES users(id) ON DELETE SET NULL,
    hero_key      TEXT,                             -- object storage key
    -- A model measured as colliding with another is kept but not offered.
    -- See eval/roster_distinctness.py.
    duplicate_of  TEXT        REFERENCES models(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX models_selectable ON models (id) WHERE duplicate_of IS NULL;

CREATE TABLE jobs (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tool          TEXT        NOT NULL,
    model_id      TEXT        REFERENCES models(id),
    -- Everything the worker needs, and nothing it does not: input keys, mode,
    -- seed, steps. Kept as one document because the shape differs per tool and
    -- a column per tool would be a migration every time a tool is added.
    params        JSONB       NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued','running','done','failed','cancelled')),
    priority      SMALLINT    NOT NULL DEFAULT 100,   -- lower runs first
    credits_cost  INTEGER     NOT NULL DEFAULT 1,
    attempts      SMALLINT    NOT NULL DEFAULT 0,
    -- Set when a worker claims the job, so a crashed worker's jobs can be
    -- found and released rather than sitting 'running' for ever.
    claimed_by    TEXT,
    claimed_at    TIMESTAMPTZ,
    -- The same request sent twice -- a retried HTTP call, a double tap in
    -- Telegram -- must produce one job and charge once.
    idem_key      TEXT,
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    UNIQUE (user_id, idem_key)
);

-- The index the claim query rides on. Partial, because only queued rows are
-- ever scanned for work and the table will be mostly finished jobs.
CREATE INDEX jobs_claimable ON jobs (priority, created_at)
    WHERE status = 'queued';
CREATE INDEX jobs_user_recent ON jobs (user_id, created_at DESC);
-- Finding jobs stranded by a worker that died mid-run.
CREATE INDEX jobs_stale ON jobs (claimed_at) WHERE status = 'running';

CREATE TABLE results (
    id           BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id       UUID         NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    -- Object storage key, never a path on a pod: the pod's disk is wiped every
    -- time it stops.
    object_key   TEXT         NOT NULL,
    kind         TEXT         NOT NULL DEFAULT 'image' CHECK (kind IN ('image','video')),
    width        INTEGER,
    height       INTEGER,
    seconds      NUMERIC(6,2),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX results_job ON results (job_id);

-- Every movement of credit, append-only. The balance on users is a cache of
-- this ledger; the ledger is what can be audited when someone says they were
-- charged twice.
CREATE TABLE credit_entries (
    id           BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id      BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    delta        INTEGER      NOT NULL,
    reason       TEXT         NOT NULL,      -- job | refund | topup | grant
    job_id       UUID         REFERENCES jobs(id) ON DELETE SET NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX credit_entries_user ON credit_entries (user_id, created_at DESC);
-- A job may be charged once and refunded once, never twice.
CREATE UNIQUE INDEX credit_entries_once ON credit_entries (job_id, reason)
    WHERE job_id IS NOT NULL;

COMMIT;

-- Telegram: conversation state, and delivery that survives a restart.

BEGIN;

-- Where each chat is in the flow. In memory this would be lost on every
-- deploy, and the seller would be asked to send their photo again -- the one
-- thing guaranteed to make them stop using it.
CREATE TABLE bot_state (
    chat_id     BIGINT       PRIMARY KEY,
    user_id     BIGINT       REFERENCES users(id) ON DELETE CASCADE,
    step        TEXT         NOT NULL DEFAULT 'idle',
    data        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Delivery is driven by the table, not by whichever coroutine happened to
-- submit the job. A bot that restarts between "queued" and "done" must still
-- send the image, and a job must never be sent twice.
ALTER TABLE jobs ADD COLUMN delivered_at TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN chat_id      BIGINT;

CREATE INDEX jobs_undelivered ON jobs (finished_at)
    WHERE status IN ('done','failed') AND chat_id IS NOT NULL AND delivered_at IS NULL;

COMMIT;

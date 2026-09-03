-- A job may have more than one result, and the customer chooses between them.
--
-- The packshot is the reason. A generative pass can straighten a dress that
-- was hanging crooked and fill in the neckline a hanger was covering -- and it
-- can also hand back a garment with shorter sleeves than the one in the seller's
-- hand. Silently swapping in a cut-out when that happens hides the choice from
-- the person who knows the garment; so both are produced, both are kept, and
-- the seller picks. The gate's opinion goes with them as a label rather than
-- as a decision.
--
-- The table already allowed several rows per job -- job_id has no unique
-- constraint -- so nothing here changes what is stored. What was missing was a
-- way to say which is which.
ALTER TABLE results ADD COLUMN IF NOT EXISTS variant TEXT;

-- What the fidelity gate thought of it, kept beside the picture rather than
-- recomputed: the scores are cheap but the judgement is what was shown to the
-- seller, and a number that changes after the fact is not a record.
ALTER TABLE results ADD COLUMN IF NOT EXISTS notes JSONB;

-- One row per variant per job. Asking for the same variant twice is a retry,
-- not a second result to choose from.
CREATE UNIQUE INDEX IF NOT EXISTS results_job_variant
    ON results (job_id, COALESCE(variant, ''));

-- What kind of garment a photograph holds, asked once and remembered.
--
-- The seller is the one who knows. A skirt sent through try-on came back worn
-- on the model's torso as a top, and no wording fixes that: the try-on model
-- reads the garment image and ignores text -- measured five ways, see
-- docs/CONTROLS.md. So the category is not an instruction to the model. It is
-- what lets the studio say "this tool wears garments on the torso" before the
-- seller spends thirty seconds and a credit finding out.
--
-- It is also the better answer for the packshot, which does read it. The
-- classifier gets the category right 94% of the time; the person holding the
-- garment gets it right always.
--
-- Keyed by the object rather than the job, because it has to survive the trip
-- through the studio. A packshot's result becomes a try-on's input, and asking
-- the same question again at every hop is how a one-tap answer turns into a
-- nuisance the seller learns to click past.
CREATE TABLE IF NOT EXISTS garment_kinds (
    object_key  TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('upper', 'lower', 'overall')),
    -- 'seller' when they chose it, 'inherited' when it came from the picture
    -- this one was made from. Kept apart so a guess never overwrites an answer.
    source      TEXT NOT NULL DEFAULT 'seller',
    user_id     BIGINT REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS garment_kinds_user ON garment_kinds (user_id);

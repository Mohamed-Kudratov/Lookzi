-- Mark which photographs are stand-ins.
--
-- Local development seeds the roster from a handful of sample images, so one
-- photograph ends up under several names -- a man's name over a woman's
-- picture, the same face three times. That is fine for exercising the flow and
-- indefensible to show anyone, and the difference has to be visible in the
-- interface rather than remembered.

ALTER TABLE models ADD COLUMN IF NOT EXISTS hero_is_placeholder BOOLEAN NOT NULL DEFAULT false;

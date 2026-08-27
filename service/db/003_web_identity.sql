-- Let a browser be an identity too.
--
-- The MVP signs people in through Telegram, which supplies an id with every
-- message. A browser supplies nothing, so the web app mints one and keeps it
-- locally -- enough to carry a credit balance and a history while there are no
-- accounts and nothing to pay for.
--
-- It is deliberately its own kind rather than a fake telegram row. When email
-- sign-up arrives, a person who used both the bot and the browser has two
-- identities and one account, which is what the table was shaped for.

ALTER TABLE identities DROP CONSTRAINT IF EXISTS identities_kind_check;
ALTER TABLE identities ADD CONSTRAINT identities_kind_check
    CHECK (kind IN ('telegram', 'email', 'phone', 'web'));

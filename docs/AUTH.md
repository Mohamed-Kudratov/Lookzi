# Accounts — decided, deferred

The full account system below is **not built for the MVP**. It is written down
now because the schema has to leave room for it, and because a decision made
once and recorded is cheaper than the same decision made three times.

Build it when the MVP has shown that sellers actually use the product.

## What was decided

Sign-up, by email and password:

1. The person enters their email.
2. We send a **six-digit code** to that address.
3. They enter the code, then set **first name, last name, username, password**.
4. The account is created.

Forgotten password:

1. They enter their email.
2. We send a **link**, not a code — a link cannot be read out over the phone to
   someone pretending to be support, and it carries a single-use token.
3. The link opens a form where they set a new password.

Sign-in is by password. Passwordless sign-in -- a link mailed on every login --
was proposed and rejected: these accounts hold credit balances, and handing out
a link that logs someone straight in was judged too much to put in an inbox.

One fact to carry forward rather than rediscover. Password and magic link share
the same worst case: whoever controls the inbox controls the account, because
password reset runs through that inbox either way. The lever that actually
protects a balance is a second factor, not the choice between them. Add TOTP
when money is real.

## What the MVP does instead

Telegram already identifies the person. A bot receives `telegram_user_id` with
every message — no password, no email, no confirmation code, no domain, no
deliverability problem. Identity is free.

So the MVP has no accounts to build. A row keyed on the Telegram id carries the
credit balance and the history, and that is enough to answer the only question
the MVP exists to answer.

## What the schema must allow for, from the first migration

The mistake to avoid is an `email` column on `users`. Adding SMS or Telegram
later then means a migration of every row and every query that touched it.

Two tables instead:

```
users        the person: credits, plan, settings, created_at
identities   one row per way of signing in, many per user
             (kind: telegram | email | phone, value, verified_at)
```

The MVP writes one `identities` row of kind `telegram`. Email sign-up later
writes a second row of kind `email` against the same user, and the two are the
same account. Nothing migrates.

Columns the deferred design needs, added when it is built rather than now:
`password_hash`, `username` (unique, case-insensitive), `first_name`,
`last_name`. Verification codes and reset tokens live in their own short-lived
table with an expiry, never on `users`.

## The parts that are not code

These take longer than the code and none of them can start on the day the code
is ready:

- a **domain**, so mail comes from the product rather than a free inbox
- **SPF and DKIM** on that domain, or the six-digit code lands in spam
- deliverability tested against **Gmail and Mail.ru** specifically — Mail.ru is
  still widely used across the CIS and filters hard
- a **legal entity**, which local payment processors require before they will
  talk to you

Start the legal entity first when the time comes. It is the longest path and it
blocks revenue, not features.

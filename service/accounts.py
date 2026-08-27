#!/usr/bin/env python3
"""Finding, or creating, the account behind an identity.

One function, because there were two -- one in the bot and one in the web tier
-- and only one of them was safe. The web page opens by asking five endpoints
at once, and for a visitor with no account yet every one of those requests
looked up the identity, found nothing, and tried to create it. One won and the
rest returned 500s, so a first visit showed no credits and no history.

Sign-up stays implicit. Telegram has already established who somebody is, and
a browser that has not paid for anything has nothing worth protecting yet, so
asking either of them to register would be asking for a decision before they
have seen the product work. See docs/AUTH.md.
"""
import os


class _AlreadyExists(Exception):
    """Another request created this identity while we were creating it."""


def find(conn, kind, value):
    return conn.execute(
        """SELECT u.* FROM users u JOIN identities i ON i.user_id = u.id
            WHERE i.kind = %s AND i.value = %s""", (kind, value)).fetchone()


def identify(conn, kind, value):
    """The account for this identity, creating one the first time.

    Safe against several requests racing for the same new identity: the insert
    is conflict-guarded, and losing the race rolls back the half-made account
    rather than leaving a user row with nothing attached to it.
    """
    row = find(conn, kind, value)
    if row is not None:
        return row

    grant = int(os.environ.get("TRIAL_CREDITS", "20"))
    try:
        with conn.transaction():
            user = conn.execute(
                "INSERT INTO users (credits) VALUES (0) RETURNING *").fetchone()
            claimed = conn.execute(
                """INSERT INTO identities (user_id, kind, value, verified_at)
                   VALUES (%s, %s, %s, now())
                   ON CONFLICT (kind, value) DO NOTHING
                RETURNING user_id""", (user["id"], kind, value)).fetchone()
            if claimed is None:
                # Somebody else got there first. Raising unwinds the whole
                # block, which is the point: the users row we just inserted
                # must not survive as an account nobody can reach.
                raise _AlreadyExists

            # Trial credit goes through the ledger, not straight into the
            # column, so the balance and its history agree from the first row.
            conn.execute(
                """INSERT INTO credit_entries (user_id, delta, reason)
                   VALUES (%s, %s, 'grant')""", (user["id"], grant))
            conn.execute("UPDATE users SET credits = %s WHERE id = %s",
                         (grant, user["id"]))
            user["credits"] = grant
            return user
    except _AlreadyExists:
        return find(conn, kind, value)

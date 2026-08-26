#!/usr/bin/env python3
"""The Telegram bot. For this market it is the product, not an accessory.

Sellers here already run their businesses inside Telegram. Asking them to open
a browser, create an account and remember a password to try something is asking
for four decisions before they have seen anything work.

It also solves two problems the web app has to solve the hard way. Telegram
supplies an identity with every message, so the MVP needs no sign-up at all.
And a cold GPU stops being a problem worth hiding: the seller sends a photo,
puts the phone down, and the notification finds them. Seven minutes is nothing
when you were not sitting there watching a spinner.

    TELEGRAM_BOT_TOKEN=... python -m service.bot

The token is read from the environment and never logged. Anyone holding it can
impersonate the bot; if it is ever pasted anywhere it should be revoked with
/revoke in BotFather and replaced.
"""
import io
import json
import os
import time
import traceback

import httpx

from . import queue as q
from . import storage

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
API = f"https://api.telegram.org/bot{TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{TOKEN}"

POLL_TIMEOUT = 25          # long poll: one request parked, not a hot loop
DELIVER_EVERY = 2.0        # seconds between sweeps for finished work
MODES = [("upper", "Upper body"), ("lower", "Lower body"), ("overall", "Full outfit")]


# ---------------------------------------------------------------------------
# Telegram plumbing

def call(method, **params):
    r = httpx.post(f"{API}/{method}", json=params, timeout=POLL_TIMEOUT + 10)
    body = r.json()
    if not body.get("ok"):
        # The token never appears here: `method` and the error are enough to
        # diagnose, and the URL would carry the secret into the log.
        raise RuntimeError(f"{method}: {body.get('description')}")
    return body["result"]


def send(chat_id, text, buttons=None):
    kw = {}
    if buttons:
        kw["reply_markup"] = {"inline_keyboard": buttons}
    return call("sendMessage", chat_id=chat_id, text=text,
                parse_mode="HTML", **kw)


def send_photo(chat_id, data, caption="", buttons=None):
    files = {"photo": ("result.png", data, "image/png")}
    payload = {"chat_id": str(chat_id), "caption": caption, "parse_mode": "HTML"}
    if buttons:
        payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})
    r = httpx.post(f"{API}/sendPhoto", data=payload, files=files, timeout=60)
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"sendPhoto: {body.get('description')}")
    return body["result"]


def download(file_id):
    path = call("getFile", file_id=file_id)["file_path"]
    r = httpx.get(f"{FILE_API}/{path}", timeout=60)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# state

def identify(conn, tg_user):
    """Find or create the account behind a Telegram id.

    Sign-up is implicit. Telegram has already established who this is, so
    asking them to register would be asking them to do it twice.
    """
    row = conn.execute(
        """SELECT u.* FROM users u JOIN identities i ON i.user_id = u.id
            WHERE i.kind = 'telegram' AND i.value = %s""",
        (str(tg_user["id"]),)).fetchone()
    if row:
        return row
    with conn.transaction():
        user = conn.execute(
            "INSERT INTO users (credits) VALUES (0) RETURNING *").fetchone()
        conn.execute(
            """INSERT INTO identities (user_id, kind, value, verified_at)
               VALUES (%s, 'telegram', %s, now())""",
            (user["id"], str(tg_user["id"])))
        grant = int(os.environ.get("TRIAL_CREDITS", "20"))
        conn.execute(
            "INSERT INTO credit_entries (user_id, delta, reason) VALUES (%s, %s, 'grant')",
            (user["id"], grant))
        conn.execute("UPDATE users SET credits = %s WHERE id = %s", (grant, user["id"]))
        user["credits"] = grant
    return user


def get_state(conn, chat_id):
    row = conn.execute("SELECT * FROM bot_state WHERE chat_id = %s",
                       (chat_id,)).fetchone()
    return row or {"chat_id": chat_id, "step": "idle", "data": {}}


def set_state(conn, chat_id, user_id, step, data):
    with conn.transaction():
        conn.execute(
            """INSERT INTO bot_state (chat_id, user_id, step, data, updated_at)
               VALUES (%s, %s, %s, %s, now())
               ON CONFLICT (chat_id) DO UPDATE SET
                 user_id = EXCLUDED.user_id, step = EXCLUDED.step,
                 data = EXCLUDED.data, updated_at = now()""",
            (chat_id, user_id, step, json.dumps(data)))


def model_keyboard(conn):
    """Models by name, two to a row. Never an internal id.

    Choosing who models your product is casting, not querying a table.
    """
    rows = conn.execute(
        """SELECT id, display_name, age, gender FROM models
            WHERE duplicate_of IS NULL AND hero_key IS NOT NULL
            ORDER BY gender, age""").fetchall()
    if not rows:
        return None
    buttons, row = [], []
    for m in rows:
        row.append({"text": f"{m['display_name']} · {m['age']}",
                    "callback_data": f"model:{m['id']}"})
        if len(row) == 2:
            buttons.append(row); row = []
    if row:
        buttons.append(row)
    return buttons


# ---------------------------------------------------------------------------
# conversation

HELP = (
    "<b>Lookzi</b> — put your product on a model.\n\n"
    "Send a photo of the garment, laid flat. Choose who wears it and which part "
    "of the body it covers, and the finished image comes back here.\n\n"
    "/credits — what you have left\n"
    "/models — who is available\n"
    "/help — this message"
)


def on_message(conn, msg):
    chat_id = msg["chat"]["id"]
    user = identify(conn, msg["from"])
    text = (msg.get("text") or "").strip()

    if text.startswith("/start") or text.startswith("/help"):
        send(chat_id, HELP)
        send(chat_id, f"You have <b>{user['credits']}</b> credits.")
        return
    if text.startswith("/credits"):
        send(chat_id, f"<b>{user['credits']}</b> credits.")
        return
    if text.startswith("/models"):
        kb = model_keyboard(conn)
        send(chat_id, "Choose a model, then send the garment photo."
             if kb else "No models are available yet.", kb)
        return

    if msg.get("photo"):
        if user["credits"] < 1:
            send(chat_id, "You are out of credits.")
            return
        # Telegram sends several sizes; the last is the largest.
        data = download(msg["photo"][-1]["file_id"])
        key = storage.key_for("uploads/garment", user["id"], ext="jpg")
        storage.put_bytes(key, data, content_type="image/jpeg")

        kb = model_keyboard(conn)
        if not kb:
            send(chat_id, "No models are available yet.")
            return
        set_state(conn, chat_id, user["id"], "await_model", {"garment_key": key})
        send(chat_id, "Got it. Who should wear this?", kb)
        return

    if msg.get("document"):
        send(chat_id, "Send it as a photo rather than a file, so I can read it.")
        return

    send(chat_id, "Send a photo of the garment, or /help.")


def on_callback(conn, cb):
    chat_id = cb["message"]["chat"]["id"]
    user = identify(conn, cb["from"])
    state = get_state(conn, chat_id)
    data = dict(state["data"] or {})
    value = cb.get("data", "")
    # Acknowledge first: Telegram shows a spinner on the button until this
    # returns, and the work below can take a moment.
    call("answerCallbackQuery", callback_query_id=cb["id"])

    if value.startswith("model:"):
        if not data.get("garment_key"):
            send(chat_id, "Send the garment photo first.")
            return
        data["model_id"] = value.split(":", 1)[1]
        set_state(conn, chat_id, user["id"], "await_mode", data)
        send(chat_id, "Which part of the body does it cover?",
             [[{"text": label, "callback_data": f"mode:{m}"} for m, label in MODES]])
        return

    if value.startswith("mode:"):
        if not (data.get("garment_key") and data.get("model_id")):
            send(chat_id, "Start again: send the garment photo.")
            return
        data["mode"] = value.split(":", 1)[1]
        submit(conn, chat_id, user, data)
        set_state(conn, chat_id, user["id"], "idle", {})
        return

    if value == "again":
        send(chat_id, "Send the next garment photo.")
        return


def submit(conn, chat_id, user, data):
    row = conn.execute("SELECT hero_key FROM models WHERE id = %s",
                       (data["model_id"],)).fetchone()
    if not row or not row["hero_key"]:
        send(chat_id, "That model has no reference image yet.")
        return

    params = {"person_key": row["hero_key"], "garment_key": data["garment_key"],
              "mode": data["mode"], "description": "the garment", "seed": 42,
              "width": 768, "height": 1024, "steps": 8}
    try:
        job, _ = q.submit(conn, user["id"], "product-to-model", params,
                          model_id=data["model_id"], cost=1,
                          priority={"trial": 200, "seller": 100, "brand": 50}
                          .get(user["plan"], 200))
    except q.InsufficientCredit as exc:
        send(chat_id, f"You have {exc.have} credits and this costs {exc.need}.")
        return

    conn.execute("UPDATE jobs SET chat_id = %s WHERE id = %s", (chat_id, job["id"]))
    conn.commit()

    d = q.depth(conn)
    ahead = max(0, d["queued"] - 1)
    # An honest wait, with the reason attached. A bare "processing…" is what
    # makes a seven-minute cold start feel broken instead of merely slow.
    if ahead:
        when = f"{ahead} ahead of you"
    else:
        when = "starting now"
    send(chat_id,
         f"<b>Queued</b> — {when}.\n"
         "You can close Telegram; the image arrives here when it is done.")


# ---------------------------------------------------------------------------
# delivery
#
# Driven by the table rather than by the coroutine that submitted the job, so a
# bot restarted mid-generation still delivers, and delivered_at makes sure it
# never delivers twice.

def deliver(conn):
    rows = conn.execute(
        """SELECT j.id, j.chat_id, j.status, j.error, r.object_key, r.seconds
             FROM jobs j LEFT JOIN results r ON r.job_id = j.id
            WHERE j.status IN ('done','failed')
              AND j.chat_id IS NOT NULL AND j.delivered_at IS NULL
            ORDER BY j.finished_at LIMIT 10""").fetchall()
    for job in rows:
        try:
            if job["status"] == "done" and job["object_key"]:
                send_photo(job["chat_id"], storage.get_bytes(job["object_key"]),
                           caption=f"Ready — {job['seconds']}s",
                           buttons=[[{"text": "Another garment",
                                      "callback_data": "again"}]])
            else:
                send(job["chat_id"],
                     "That one failed and the credit has been returned. "
                     "Try again, or send a different photo.")
            conn.execute("UPDATE jobs SET delivered_at = now() WHERE id = %s",
                         (job["id"],))
            conn.commit()
        except Exception:
            # A chat the user blocked, or a transient API failure. Leaving
            # delivered_at unset retries on the next sweep; it must not stop
            # the others in this batch.
            traceback.print_exc()
            conn.rollback()


# ---------------------------------------------------------------------------

def main():
    if not TOKEN:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "Get one from @BotFather and put it in .env — never in a message.")
    storage.ensure_bucket()
    me = call("getMe")
    print(f"[bot] @{me['username']} listening", flush=True)

    conn = q.connect()
    offset = None
    last_deliver = 0.0
    try:
        while True:
            try:
                updates = call("getUpdates", offset=offset, timeout=POLL_TIMEOUT,
                               allowed_updates=["message", "callback_query"])
            except Exception:
                traceback.print_exc()
                time.sleep(3)
                continue

            for u in updates:
                offset = u["update_id"] + 1
                try:
                    if "message" in u:
                        on_message(conn, u["message"])
                    elif "callback_query" in u:
                        on_callback(conn, u["callback_query"])
                except Exception:
                    # One malformed update must not take the bot down for
                    # everyone else waiting on it.
                    traceback.print_exc()
                    conn.rollback()

            if time.time() - last_deliver > DELIVER_EVERY:
                try:
                    deliver(conn)
                except Exception:
                    traceback.print_exc()
                    conn.rollback()
                last_deliver = time.time()
    finally:
        conn.close()


if __name__ == "__main__":
    main()

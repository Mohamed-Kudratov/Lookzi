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
import hashlib
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

# Bump when the roster sheet's drawing changes. The cache key is built from the
# roster, so a change to the layout alone produced no new key and the old
# picture kept coming back -- which looked like the edit had not taken.
SHEET_VERSION = 2


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


def model_sheet(conn):
    """One picture of the whole roster, numbered, cached in object storage.

    Names and ages alone are not a choice -- somebody picking who models their
    product is casting, and casting is done by looking. Thirteen separate photo
    messages would bury the chat, and Telegram albums cannot carry a button per
    item, so the roster goes out as a single numbered grid and the buttons
    refer to the numbers on it.

    Cached under a key derived from the roster itself, so it is drawn once and
    redrawn the moment a model is added, removed or re-photographed.
    """
    from PIL import Image, ImageDraw

    models = selectable(conn)
    if not models:
        return None, []

    stamp = hashlib.sha1(
        (f"v{SHEET_VERSION}|" + "|".join(
            f"{m['id']}:{m['hero_key']}:{m['hero_is_placeholder']}"
            for m in models)).encode()
    ).hexdigest()[:16]
    key = f"sheets/roster-{stamp}.jpg"
    try:
        return storage.get_bytes(key), models
    except Exception:
        pass  # not drawn yet, or the cache was cleared

    cols = 4
    rows = (len(models) + cols - 1) // cols
    W, H, BAR = 200, 266, 30
    sheet = Image.new("RGB", (cols * W, rows * (H + BAR)), "#16181C")
    draw = ImageDraw.Draw(sheet)

    for i, m in enumerate(models):
        x, y = (i % cols) * W, (i // cols) * (H + BAR)
        try:
            photo = Image.open(io.BytesIO(storage.get_bytes(m["hero_key"])))
            photo = photo.convert("RGB").resize((W, H))
            sheet.paste(photo, (x, y))
        except Exception:
            draw.rectangle([x, y, x + W, y + H], fill="#22262E")
        # The number is what the button says, so it has to be readable against
        # any photograph -- hence the solid chip rather than bare text.
        draw.rectangle([x + 6, y + 6, x + 34, y + 30], fill="#16181CDD")
        draw.text((x + 15, y + 12), str(i + 1), fill="#F2F1ED")
        draw.text((x + 8, y + H + 8),
                  f"{m['display_name']} · {m['age']}", fill="#B4B9C0")
        if m["hero_is_placeholder"]:
            # Said on the picture, not in a note underneath it. A stand-in
            # photograph that looks like the roster is worse than no picture:
            # it puts a man's name over a woman's face and reads as carelessness.
            draw.rectangle([x, y + H - 26, x + W, y + H], fill="#B4573DEE")
            # Plain ASCII: the default bitmap font has no em-dash and draws
            # a hollow box instead, which reads as a rendering fault
            # rather than a warning.
            draw.text((x + 8, y + H - 19), "SAMPLE - not this model",
                      fill="#F7F6F3")

    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=82, optimize=True)
    data = buf.getvalue()
    storage.put_bytes(key, data, content_type="image/jpeg")
    return data, models


def selectable(conn):
    return conn.execute(
        """SELECT id, display_name, age, gender, hero_key, hero_is_placeholder
             FROM models
            WHERE duplicate_of IS NULL AND hero_key IS NOT NULL
            ORDER BY gender, age""").fetchall()


def model_buttons(models):
    """Numbered to match the grid, four to a row.

    The number carries the meaning and the name confirms it, so a mistap is
    visible before it costs a credit.
    """
    out, row = [], []
    for i, m in enumerate(models, 1):
        row.append({"text": f"{i} · {m['display_name']}",
                    "callback_data": f"model:{m['id']}"})
        if len(row) == 2:
            out.append(row); row = []
    if row:
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# what the bot can do
#
# The tool is chosen first, because it decides what to ask for next. Asking for
# a garment photo before knowing the job meant every conversation started the
# same way and only one of the tools could ever be reached.

TOOLS = {
    "product-to-model": dict(
        label="Product → Model",
        blurb="A flat photo of the garment, worn by one of our models.",
        needs=["garment", "model", "mode"], cost=1, ready=True),
    "virtual-try-on": dict(
        label="Try it on me",
        blurb="Your own photo, wearing the garment you send.",
        needs=["person", "garment", "mode"], cost=1, ready=True),
    "model-swap": dict(
        label="Change the model",
        blurb="Your photo, same clothes and pose, a different person wearing them.",
        needs=["person", "model"], cost=1, ready=True),
    "packshot": dict(
        label="Packshot",
        blurb="A clean catalogue cut-out of the garment on its own.",
        needs=["garment"], cost=1, ready=False),
    "model-creation": dict(
        label="Make a new model",
        blurb="A model that belongs to you alone.",
        needs=[], cost=4, ready=False),
    "short-video": dict(
        label="Short video",
        blurb="Five or ten seconds of motion from a finished image.",
        needs=[], cost=3, ready=False),
}

# What to ask for, in the order it is asked.
ASK = {
    "garment": "Send a photo of the <b>garment</b>, laid flat.",
    "person":  "Send a photo of the <b>person</b> — full body, facing the camera.",
}


def tool_keyboard():
    out, row = [], []
    for tid, t in TOOLS.items():
        mark = "" if t["ready"] else " · soon"
        row.append({"text": t["label"] + mark, "callback_data": f"tool:{tid}"})
        if len(row) == 2:
            out.append(row); row = []
    if row:
        out.append(row)
    return out


HELP = (
    "<b>Lookzi</b> — product photography without the shoot.\n\n"
    "Pick what you want to do and I will ask for what I need.\n\n"
    "/start — begin again\n"
    "/credits — what you have left\n"
    "/models — see the roster"
)


def start(conn, chat_id, user):
    set_state(conn, chat_id, user["id"], "await_tool", {})
    send(chat_id, "<b>What would you like to do?</b>", tool_keyboard())


# On every prompt, so a wrong turn is one tap from being undone rather than a
# conversation the customer has to escape by guessing at a command. Mistakes
# here are normal: the wrong photograph, the wrong tool, a change of mind.
CANCEL = [{"text": "Cancel", "callback_data": "cancel"}]


def next_step(conn, chat_id, user, data):
    """Ask for whatever the chosen tool still needs, or run it.

    The tool's `needs` list is the script for the conversation, so adding a
    tool is a dictionary entry rather than another branch in here.
    """
    tool = TOOLS[data["tool"]]
    for need in tool["needs"]:
        if need in ("garment", "person") and not data.get(f"{need}_key"):
            set_state(conn, chat_id, user["id"], f"await_{need}", data)
            send(chat_id, ASK[need], [CANCEL])
            return
        if need == "model" and not data.get("model_id"):
            sheet, models = model_sheet(conn)
            if not models:
                send(chat_id, "No models are available yet.")
                return
            set_state(conn, chat_id, user["id"], "await_model", data)
            fake = sum(1 for m in models if m["hero_is_placeholder"])
            note = ("\n\n<i>The photographs are samples while the roster "
                    "loads — the names and faces do not match yet.</i>"
                    if fake else "")
            send_photo(chat_id, sheet,
                       caption="<b>Who should wear it?</b>\nTap a number." + note,
                       buttons=model_buttons(models) + [CANCEL])
            return
        if need == "mode" and not data.get("mode"):
            set_state(conn, chat_id, user["id"], "await_mode", data)
            send(chat_id, "Which part of the body does it cover?",
                 [[{"text": label, "callback_data": f"mode:{m}"} for m, label in MODES],
                  CANCEL])
            return
    confirm(conn, chat_id, user, data)


def confirm(conn, chat_id, user, data):
    """Show what is about to be made, and wait to be told to make it.

    Cancelling after the job was already queued turned out to be almost
    useless: answering the last question queued it immediately, a warm worker
    claimed it within a second, and the cancel button was already too late by
    the time it appeared. The moment worth interrupting is before the credit
    is spent, not after.

    The summary also catches the ordinary mistake -- the wrong model, the wrong
    part of the body -- while it still costs nothing to fix.
    """
    tool = TOOLS[data["tool"]]
    lines = [f"<b>{tool['label']}</b>"]

    if data.get("model_id"):
        row = conn.execute("SELECT display_name, age FROM models WHERE id = %s",
                           (data["model_id"],)).fetchone()
        if row:
            lines.append(f"Model — {row['display_name']}, {row['age']}")
    if data.get("person_key"):
        lines.append("Person — the photo you sent")
    if data.get("garment_key"):
        lines.append("Garment — the photo you sent")
    if data.get("mode"):
        lines.append("Covers — " + dict(MODES)[data["mode"]].lower())

    lines.append("")
    lines.append(f"Costs <b>{tool['cost']}</b> "
                 f"of your <b>{user['credits']}</b> credits.")

    # Three buttons, because stopping and starting again are different wishes.
    # Somebody who spots the wrong model wants to go back to the beginning;
    # somebody who has changed their mind entirely wants to be left alone, and
    # offering them a fresh menu is not an answer.
    set_state(conn, chat_id, user["id"], "await_confirm", data)
    send(chat_id, "\n".join(lines),
         [[{"text": f"Generate · {tool['cost']} credit", "callback_data": "go"}],
          [{"text": "Start over", "callback_data": "restart"},
           {"text": "Cancel", "callback_data": "cancel"}]])


def on_message(conn, msg):
    chat_id = msg["chat"]["id"]
    user = identify(conn, msg["from"])
    text = (msg.get("text") or "").strip()
    state = get_state(conn, chat_id)
    data = dict(state["data"] or {})

    if text.startswith("/start"):
        send(chat_id, HELP)
        start(conn, chat_id, user)
        return
    if text.startswith("/help"):
        send(chat_id, HELP)
        return
    if text.startswith("/cancel") or text.lower() in ("cancel", "stop"):
        set_state(conn, chat_id, user["id"], "idle", {})
        send(chat_id, "Stopped. Nothing was charged.\n"
                      "<i>/start whenever you want to begin again.</i>")
        return
    if text.startswith("/credits"):
        send(chat_id, f"<b>{user['credits']}</b> credits.")
        return
    if text.startswith("/models"):
        sheet, models = model_sheet(conn)
        if not models:
            send(chat_id, "No models are available yet.")
        else:
            send_photo(chat_id, sheet,
                       caption=f"<b>{len(models)} models.</b> "
                               "Start with /start to use one.")
        return

    if msg.get("photo"):
        if not data.get("tool"):
            send(chat_id, "First, tell me what to do with it.")
            start(conn, chat_id, user)
            return
        if user["credits"] < TOOLS[data["tool"]]["cost"]:
            send(chat_id, "You are out of credits.")
            return

        # Which slot this photo fills is decided by the step we are on, not by
        # what it looks like -- the same photograph is the garment in one tool
        # and the person in another.
        slot = "garment" if state["step"] == "await_garment" else (
               "person" if state["step"] == "await_person" else None)
        if slot is None:
            send(chat_id, "I was not expecting a photo just now. /start to begin again.")
            return

        # Telegram sends several sizes; the last is the largest.
        raw = download(msg["photo"][-1]["file_id"])
        key = storage.key_for(f"uploads/{slot}", user["id"], ext="jpg")
        storage.put_bytes(key, raw, content_type="image/jpeg")
        data[f"{slot}_key"] = key
        next_step(conn, chat_id, user, data)
        return

    if msg.get("document"):
        send(chat_id, "Send it as a photo rather than a file, so I can read it.")
        return

    send(chat_id, "Use /start to begin.")


def on_callback(conn, cb):
    chat_id = cb["message"]["chat"]["id"]
    user = identify(conn, cb["from"])
    state = get_state(conn, chat_id)
    data = dict(state["data"] or {})
    value = cb.get("data", "")
    # Acknowledge first: Telegram shows a spinner on the button until this
    # returns, and the work below can take a moment.
    call("answerCallbackQuery", callback_query_id=cb["id"])

    if value.startswith("tool:"):
        tid = value.split(":", 1)[1]
        tool = TOOLS.get(tid)
        if not tool:
            return
        if not tool["ready"]:
            send(chat_id, f"<b>{tool['label']}</b> is not switched on yet. "
                          "Everything else in the list works.")
            return
        send(chat_id, f"<b>{tool['label']}</b> — {tool['blurb']}")
        next_step(conn, chat_id, user, {"tool": tid})
        return

    # Cancelling has to work from any state, including one the bot has already
    # forgotten -- a tap on an old message is the most likely moment somebody
    # wants out.
    if value == "cancel":
        set_state(conn, chat_id, user["id"], "idle", {})
        send(chat_id, "Stopped. Nothing was charged.\n"
                      "<i>/start whenever you want to begin again.</i>")
        return

    if value == "restart":
        set_state(conn, chat_id, user["id"], "idle", {})
        start(conn, chat_id, user)
        return

    if value.startswith("drop:"):
        outcome = q.cancel(conn, value.split(":", 1)[1], user_id=user["id"])
        if outcome == "cancelled":
            fresh = identify(conn, cb["from"])
            send(chat_id, "Cancelled, and the credit is back. "
                          f"You have <b>{fresh['credits']}</b>.")
        elif outcome == "too late":
            send(chat_id, "Too late — it is already being generated. "
                          "It will arrive here shortly.")
        else:
            send(chat_id, "That job is not one of yours.")
        return

    if not data.get("tool"):
        send(chat_id, "That was from an older message. /start to begin again.")
        return

    if value.startswith("model:"):
        data["model_id"] = value.split(":", 1)[1]
        next_step(conn, chat_id, user, data)
        return

    if value.startswith("mode:"):
        data["mode"] = value.split(":", 1)[1]
        next_step(conn, chat_id, user, data)
        return

    if value == "go":
        if state["step"] != "await_confirm":
            # A second tap on the same button, or a tap on an old summary. The
            # first one already queued a job; a second must not queue another
            # and charge for it.
            send(chat_id, "That one is already on its way. /start for another.")
            return
        set_state(conn, chat_id, user["id"], "idle", {})
        submit(conn, chat_id, user, data)
        return

    if value == "again":
        start(conn, chat_id, user)
        return


def submit(conn, chat_id, user, data):
    """Turn a finished conversation into a job.

    The person in the picture comes from one of two places and the rest of the
    system does not care which: a roster model's photograph for
    product-to-model, or the customer's own upload for try-on. That is the same
    substitution the API makes, and the reason six products can share one
    resident model.
    """
    tool_id = data["tool"]
    tool = TOOLS[tool_id]

    person_key = data.get("person_key")
    if not person_key and data.get("model_id"):
        row = conn.execute("SELECT hero_key FROM models WHERE id = %s",
                           (data["model_id"],)).fetchone()
        if not row or not row["hero_key"]:
            send(chat_id, "That model has no photograph yet. Pick another.")
            return
        person_key = row["hero_key"]
    if not person_key:
        send(chat_id, "Something went missing. /start to begin again.")
        return

    params = {"person_key": person_key,
              "garment_key": data.get("garment_key", person_key),
              "mode": data.get("mode", "upper"),
              "description": "the garment", "seed": 42,
              "width": 768, "height": 1024, "steps": 8}
    try:
        job, _ = q.submit(conn, user["id"], tool_id, params,
                          model_id=data.get("model_id"), cost=tool["cost"],
                          priority={"trial": 200, "seller": 100, "brand": 50}
                          .get(user["plan"], 200))
    except q.InsufficientCredit as exc:
        send(chat_id, f"You have {exc.have} credits and this costs {exc.need}.")
        return

    conn.execute("UPDATE jobs SET chat_id = %s WHERE id = %s", (chat_id, job["id"]))

    d = q.depth(conn)
    ahead = max(0, d["queued"] - 1)
    # An honest wait, with the reason attached. A bare "processing…" is what
    # makes a seven-minute cold start feel broken instead of merely slow.
    if ahead:
        when = f"{ahead} ahead of you"
    else:
        when = "starting now"
    # The button stays useful only until a worker takes the job, which is
    # exactly when stopping is still free. Offering it is what makes pressing
    # Generate a low-stakes act rather than a commitment.
    send(chat_id,
         f"<b>Queued</b> — {when}.\n"
         "You can close Telegram; the image arrives here when it is done.",
         [[{"text": "Cancel this job", "callback_data": f"drop:{job['id']}"}]])


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
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Long polling times out whenever nobody has written to the
                # bot, which is most of the time. Printing a traceback for it
                # buried ten of them in the log overnight and would have hidden
                # a real failure among them. One line, and carry on.
                print(f"[bot] poll {type(exc).__name__}, retrying", flush=True)
                time.sleep(3)
                continue
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

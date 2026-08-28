#!/usr/bin/env python3
"""Put an account on a plan, or list who is on which.

    python -m service.set_plan --list
    python -m service.set_plan --web web-1f2e... --plan unlimited
    python -m service.set_plan --telegram 123456789 --plan unlimited
    python -m service.set_plan --all-web --plan unlimited     # a dev machine

`unlimited` is never charged and never blocked. It is for whoever is testing
the product: being stopped by your own credit ledger halfway through checking
whether a tool works wastes a person and a rented card at the same time.

Jobs on it still record what they would have cost, so /review and any pricing
question can still be answered from the history. Nothing reaches the ledger,
because nothing moved -- and queue.fail() refunds only what the ledger says was
charged, so a failing unlimited account cannot mint credits out of its own
failures.
"""
import argparse

from . import queue as q

PLANS = ("trial", "seller", "brand", "unlimited")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show every account")
    ap.add_argument("--web", help="the client id a browser generated")
    ap.add_argument("--telegram", help="a Telegram user id")
    ap.add_argument("--all-web", action="store_true",
                    help="every browser account on this machine, for development")
    ap.add_argument("--plan", choices=PLANS, default="unlimited")
    args = ap.parse_args()

    conn = q.connect()
    try:
        if args.list or not (args.web or args.telegram or args.all_web):
            rows = conn.execute(
                """SELECT u.id, u.plan, u.credits,
                          string_agg(i.kind || ':' || i.value, ', ') AS who,
                          (SELECT count(*) FROM jobs j WHERE j.user_id = u.id) AS jobs
                     FROM users u LEFT JOIN identities i ON i.user_id = u.id
                    GROUP BY u.id, u.plan, u.credits
                    ORDER BY jobs DESC""").fetchall()
            print(f"{'plan':10} {'credits':>8} {'jobs':>5}  who")
            for r in rows:
                print(f"{r['plan']:10} {r['credits']:>8} {r['jobs']:>5}  "
                      f"{(r['who'] or '-')[:70]}")
            return 0

        if args.all_web:
            rows = conn.execute(
                """UPDATE users SET plan = %s
                    WHERE id IN (SELECT user_id FROM identities WHERE kind = 'web')
                RETURNING id""", (args.plan,)).fetchall()
            print(f"{len(rows)} browser account(s) now on {args.plan}")
            return 0

        kind, value = ("web", args.web) if args.web else ("telegram", args.telegram)
        row = conn.execute(
            """UPDATE users SET plan = %s
                WHERE id = (SELECT user_id FROM identities
                             WHERE kind = %s AND value = %s)
            RETURNING id, plan, credits""", (args.plan, kind, value)).fetchone()
        if row is None:
            print(f"no account for {kind}:{value} -- open the studio once first, "
                  "which creates it")
            return 1
        print(f"{kind}:{value} is now on {row['plan']} ({row['credits']} credits "
              "left in the ledger, unused while unlimited)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Copy the roster from elements/catalog.py into the models table.

catalog.py stays the source of truth: it is what generation reads, and a
second hand-maintained list would drift from it within a week. This projects
it into a table so the web tier can list models without importing Python that
pulls in the generation stack.

Idempotent -- run it after every roster change.

    python -m service.seed_models
"""
import os
import sys

from . import queue as q

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Human names are a product decision, not a technical one: nobody casts
# "m_cauz_20s_slim" in a photoshoot. Unnamed entries fall back to the id so a
# new roster member appears rather than vanishing.
DISPLAY = {
    "f_cauz_20s_avg": "Nigora",   "f_cauz_20s_hijab": "Zilola",
    "f_cauz_20s_slim": "Dilnoza", "f_cauz_30s_avg": "Malika",
    "f_cauz_40s_full": "Gulnora", "f_cauz_40s_hijab": "Shahnoza",
    "f_cauz_50s_avg": "Kamola",   "f_slav_20s_slim": "Alina",
    "f_slav_30s_avg": "Marina",   "m_cauz_20s_slim": "Bekzod",
    "m_cauz_30s_avg": "Sardor",   "m_cauz_40s_avg": "Rustam",
    "m_slav_30s_avg": "Andrey",
}


def age_of(face):
    """"early 30s" -> 31. The table sorts and filters on a number."""
    text = face["age"]
    tens = next((int(t) for t in ("20", "30", "40", "50", "60") if t in text), 30)
    offset = {"early": 1, "mid": 5, "late": 8}
    for word, add in offset.items():
        if word in text:
            return tens + add
    return tens + 5


def main():
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "elements"))
    from catalog import ROSTER, DUPLICATE_OF

    conn = q.connect()
    try:
        with conn.transaction():
            for face in ROSTER:
                conn.execute(
                    """INSERT INTO models (id, display_name, age, gender,
                                           ethnicity, build, modest, duplicate_of)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO UPDATE SET
                         display_name = EXCLUDED.display_name,
                         age = EXCLUDED.age, gender = EXCLUDED.gender,
                         ethnicity = EXCLUDED.ethnicity, build = EXCLUDED.build,
                         modest = EXCLUDED.modest,
                         duplicate_of = EXCLUDED.duplicate_of""",
                    (face["id"], DISPLAY.get(face["id"], face["id"]),
                     age_of(face), face["gender"], face["ethnicity"],
                     face["build"], face["modest"],
                     DUPLICATE_OF.get(face["id"])))
        total = conn.execute("SELECT count(*) AS n FROM models").fetchone()["n"]
        shown = conn.execute(
            "SELECT count(*) AS n FROM models WHERE duplicate_of IS NULL"
        ).fetchone()["n"]
        print(f"{total} models, {shown} selectable ({total - shown} hidden as duplicates)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

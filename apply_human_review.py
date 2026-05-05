"""
Merge downloaded review_report.html ratings into the cumulative eval log.

Colab usage:
  python apply_human_review.py \
    --review_json /content/lookzi_human_review_20260505_173201.json \
    --drive_log /content/drive/MyDrive/Lookzi/eval_logs/results.json
"""

import argparse
import json
import os
from datetime import datetime


VALID_RATINGS = {"GOOD", "OK", "BAD", "MODEL_FAIL", "MASK_FAIL"}


def parse_args():
    parser = argparse.ArgumentParser(description="Apply Lookzi human review JSON")
    parser.add_argument("--review_json", required=True, help="Downloaded review JSON file")
    parser.add_argument(
        "--drive_log",
        default="/content/drive/MyDrive/Lookzi/eval_logs/results.json",
        help="Cumulative eval log to update",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional run timestamp. Default: latest run in drive_log",
    )
    return parser.parse_args()


def load_json(path, fallback):
    if not os.path.exists(path):
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def choose_run(runs, timestamp):
    if not runs:
        raise ValueError("drive_log is empty; run eval_benchmark.py first")
    if timestamp is None:
        return runs[-1]
    for run in runs:
        if run.get("timestamp") == timestamp:
            return run
    raise ValueError(f"run timestamp not found: {timestamp}")


def normalize_review(raw):
    if isinstance(raw, str):
        return {"human_rating": raw, "failure_reason": None}
    if not isinstance(raw, dict):
        return None
    return {
        "human_rating": raw.get("human_rating"),
        "failure_reason": raw.get("failure_reason"),
    }


def main():
    args = parse_args()
    reviews = load_json(args.review_json, {})
    runs = load_json(args.drive_log, [])
    if not isinstance(reviews, dict):
        raise ValueError("review_json must be an object keyed by pair id")
    if not isinstance(runs, list):
        raise ValueError("drive_log must be a list")

    run = choose_run(runs, args.timestamp)
    applied = 0
    skipped = []

    for result in run.get("pair_results", []):
        pair_id = result.get("id")
        if pair_id not in reviews:
            continue
        review = normalize_review(reviews[pair_id])
        if review is None:
            skipped.append(pair_id)
            continue
        rating = review.get("human_rating")
        if rating not in VALID_RATINGS:
            skipped.append(pair_id)
            continue
        result["human_rating"] = rating
        result["failure_reason"] = review.get("failure_reason")
        result["review_status"] = rating
        result["reviewed_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        applied += 1

    run["human_review_applied_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_json(args.drive_log, runs)

    print(f"[Review] Applied: {applied}")
    print(f"[Review] Updated log: {args.drive_log}")
    if skipped:
        print(f"[Review] Skipped invalid pair ids: {', '.join(skipped)}")


if __name__ == "__main__":
    main()

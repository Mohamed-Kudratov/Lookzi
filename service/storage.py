#!/usr/bin/env python3
"""Object storage for inputs and results.

Nothing generated may live on the machine that generated it. A GPU pod's disk
is wiped every time it stops, and it stops often -- reclaimed, migrated, or
simply scaled back to zero when the queue empties. An image written to local
disk is an image that disappears while the customer is still looking at the
link.

S3's API is the one every provider speaks, so the same code runs against
MinIO on a laptop, Cloudflare R2 in production, and AWS if it ever comes to
that. R2 is the one to choose: egress is free, and an image service pays for
egress more than for anything else.
"""
import os
import uuid
from datetime import datetime, timezone

import boto3
from botocore.config import Config

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
BUCKET = os.environ.get("S3_BUCKET", "lookzi")
REGION = os.environ.get("S3_REGION", "auto")
PUBLIC_BASE = os.environ.get("S3_PUBLIC_BASE", "")

_client = None


def client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            region_name=REGION,
            aws_access_key_id=os.environ.get("S3_KEY", "lookzi"),
            aws_secret_access_key=os.environ.get("S3_SECRET", "lookzi-dev-secret"),
            # Path style keeps MinIO working; R2 accepts it too. Virtual-host
            # style would need a wildcard DNS entry per bucket in development.
            config=Config(s3={"addressing_style": "path"},
                          retries={"max_attempts": 3, "mode": "standard"}),
        )
    return _client


def ensure_bucket():
    """Create the bucket if it is missing. Development only.

    In production the bucket is made once, by hand, with a lifecycle rule and
    the right permissions -- not by whichever process happens to boot first.
    """
    c = client()
    try:
        c.head_bucket(Bucket=BUCKET)
    except Exception:
        c.create_bucket(Bucket=BUCKET)


def key_for(kind, user_id, ext="png"):
    """A key that sorts by date and cannot collide.

    Date first because that is how storage is browsed, audited and expired --
    a lifecycle rule that deletes trial output after thirty days needs the date
    in the prefix. The uuid because two uploads in the same second are normal.
    """
    day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    return f"{kind}/{day}/{user_id}/{uuid.uuid4().hex}.{ext}"


def put_bytes(key, data, content_type="image/png"):
    client().put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
    return key


def get_bytes(key):
    return client().get_object(Bucket=BUCKET, Key=key)["Body"].read()


def presigned_get(key, seconds=3600):
    """A link that works without credentials and expires.

    Results are private: one seller's catalogue must not be readable by another
    just because the key was guessed. A signed link is handed out per request
    and dies on its own, which is safer than a public bucket and cheaper than
    proxying every image through the web tier.
    """
    if PUBLIC_BASE:
        return f"{PUBLIC_BASE.rstrip('/')}/{key}"
    return client().generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=seconds)


def presigned_put(key, seconds=900, content_type="image/png"):
    """A link the browser uploads to directly.

    The alternative is uploading through the web tier, which then holds a 20 MB
    file in memory and forwards it. That turns every upload into web-tier load
    for no benefit, and it is the first thing to fall over under a crowd.
    """
    return client().generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=seconds)

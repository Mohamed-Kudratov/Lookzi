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
# Presigned links are handed to a browser or a phone, so they must name an
# address that side can resolve. Inside the compose network the bucket is
# `storage:9000`, which means nothing outside it -- a link generated with the
# internal endpoint is a link nobody can open. In production both are the same
# public R2 hostname and this collapses to one value.
PUBLIC_ENDPOINT = os.environ.get("S3_PUBLIC_ENDPOINT", ENDPOINT)
# Hand out our own relative paths instead of signed links to the bucket.
#
# A signed link names a host, and the host it names is the one this machine
# can reach. That is fine on a laptop and useless the moment the studio is
# opened from somewhere else: the page loads and every picture is broken,
# because 127.0.0.1:9000 is the visitor's own machine and there is nothing
# there. Uploads fail the same way and more quietly.
#
# With this on, both directions go through the web app, which knows where the
# bucket is. It costs a copy through the web tier -- the thing the comment on
# presigned_put warns about -- and that is the right trade for a handful of
# people looking at a link. It would be the wrong one for a crowd.
PROXY = os.environ.get("S3_PROXY", "") not in ("", "0", "false")

_client = None
_public_client = None


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


def public_client():
    """A client whose signatures name the outside address.

    The signature covers the host, so a link cannot simply be string-replaced
    after the fact -- it has to be signed against the endpoint the caller will
    use.
    """
    global _public_client
    if PUBLIC_ENDPOINT == ENDPOINT:
        return client()
    if _public_client is None:
        _public_client = boto3.client(
            "s3",
            endpoint_url=PUBLIC_ENDPOINT,
            region_name=REGION,
            aws_access_key_id=os.environ.get("S3_KEY", "lookzi"),
            aws_secret_access_key=os.environ.get("S3_SECRET", "lookzi-dev-secret"),
            config=Config(s3={"addressing_style": "path"},
                          retries={"max_attempts": 3, "mode": "standard"}),
        )
    return _public_client


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


def delete(key):
    """Remove one object. Deleting what is not there is not an error here.

    S3 delete is idempotent by design, and the caller is usually removing a row
    and its picture together -- a picture that has already gone should not stop
    the row from going.
    """
    client().delete_object(Bucket=BUCKET, Key=key)


def presigned_get(key, seconds=3600):
    """A link that works without credentials and expires.

    Results are private: one seller's catalogue must not be readable by another
    just because the key was guessed. A signed link is handed out per request
    and dies on its own, which is safer than a public bucket and cheaper than
    proxying every image through the web tier.
    """
    if PROXY:
        return "/files/" + key
    if PUBLIC_BASE:
        return f"{PUBLIC_BASE.rstrip('/')}/{key}"
    return public_client().generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=seconds)


def presigned_put(key, seconds=900, content_type="image/png"):
    """A link the browser uploads to directly.

    The alternative is uploading through the web tier, which then holds a 20 MB
    file in memory and forwards it. That turns every upload into web-tier load
    for no benefit, and it is the first thing to fall over under a crowd.
    """
    if PROXY:
        return "/files/" + key
    return public_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=seconds)

#!/usr/bin/env bash
# Stop this pod, now.
#
#     bash tools/stop_pod.sh            # stop, keeping the pod and the volume
#     DRY_RUN=1 bash tools/stop_pod.sh  # check the key without stopping
#
# A pod bills for existing, not for working, so a run that finishes at 3am and
# idles until morning costs more than the run did. Two pods have now been paid
# for while doing nothing, which is the entire reason this exists.
#
# Stopping keeps the pod and the volume. Nothing is lost: the weights, the
# images and the repository are all on /workspace, which survives both stopping
# and the container disk being wiped on the next start.
#
# The credential. RunPod injects RUNPOD_API_KEY into every pod, but it is
# pod-scoped and answers 403 to anything account-level -- including stopping the
# pod it belongs to. So the account key has to be supplied. Put it on the volume
# from the RunPod console's web terminal, where the value never leaves your
# browser and the pod:
#
#   printf %s 'rpa_YOURKEY' > /workspace/.runpod_key && chmod 600 /workspace/.runpod_key
#
# Create one at runpod.io/console/user/settings with read/write permission.
#
# autostop.sh does the same stopping, but waits for a particular job to finish
# and watches for a marker only hero.py prints. This is the unconditional half,
# for a script that already knows it is done.
set -uo pipefail

KEY_FILE="${RUNPOD_KEY_FILE:-/workspace/.runpod_key}"
POD_ID="${RUNPOD_POD_ID:-}"

say() { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$1"; }

[ -n "$POD_ID" ] || { say "RUNPOD_POD_ID is not set; this is not a RunPod pod"; exit 2; }
[ -s "$KEY_FILE" ] || { say "no key at $KEY_FILE -- see the header of this file"; exit 2; }

KEY=$(tr -d ' \t\r\n' < "$KEY_FILE")
if [ "${#KEY}" -lt 30 ]; then
    # A pod id is about 14 characters and gets pasted here by mistake often
    # enough to be worth naming.
    say "the key in $KEY_FILE is ${#KEY} characters. An API key is around 50;"
    say "${#KEY} is the length of a pod id. Create a key at"
    say "runpod.io/console/user/settings."
    exit 3
fi

# Positive proof, not the absence of an error. Checking whether a response
# "looks like" a failure treats an empty body, a timeout and any unexpected
# shape as success -- and RunPod returns 200 with a null account for a rejected
# key, so the status code alone proves nothing either.
graphql() {
    curl -s --max-time 25 -w '\n%{http_code}' \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer $KEY" \
         -X POST https://api.runpod.io/graphql -d "$1"
}

verify() {
    local out code body
    out=$(graphql '{"query":"query { myself { id pods { id } } }"}') || return 1
    code=$(printf '%s' "$out" | tail -1)
    body=$(printf '%s' "$out" | sed '$d')
    [ "$code" = "200" ] || { echo "http $code"; return 1; }
    printf '%s' "$body" | POD="$POD_ID" python3 -c '
import json, os, sys
try:
    d = json.load(sys.stdin)
except Exception as exc:
    print(f"unparseable response: {exc}"); raise SystemExit(1)
if d.get("errors"):
    print(str(d["errors"])[:150]); raise SystemExit(1)
me = (d.get("data") or {}).get("myself")
if not me or not me.get("id"):
    print("no account id in the response -- the key was rejected"); raise SystemExit(1)
pods = [p["id"] for p in (me.get("pods") or [])]
if os.environ["POD"] not in pods:
    # Reading is not enough. A key that cannot see this pod cannot stop it, and
    # finding that out from the mutation is finding it out too late.
    print(f"the key cannot see pod {os.environ[\"POD\"]}"); raise SystemExit(1)
print("ok")
'
}

# One retry: a transient network failure is not a bad key, but one that
# persists is fatal either way.
if ! reason=$(verify); then
    sleep 8
    if ! reason=$(verify); then
        say "key rejected: $reason"
        exit 3
    fi
fi
say "key accepted for pod $POD_ID"

if [ "${DRY_RUN:-0}" = "1" ]; then
    say "dry run; not stopping"
    exit 0
fi

say "stopping"
out=$(graphql "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}")
code=$(printf '%s' "$out" | tail -1)
body=$(printf '%s' "$out" | sed '$d')
if [ "$code" != "200" ]; then
    say "stop failed: http $code -- $(printf '%s' "$body" | head -c 200)"
    exit 4
fi
printf '%s' "$body" | python3 -c '
import json, sys
d = json.load(sys.stdin)
if d.get("errors"):
    print(str(d["errors"])[:200]); raise SystemExit(1)
pod = (d.get("data") or {}).get("podStop") or {}
print(f"desiredStatus is now {pod.get(\"desiredStatus\")}")
' || { say "stop was not confirmed; check the console"; exit 4; }
say "stopped -- the volume and everything on it is kept"

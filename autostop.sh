#!/usr/bin/env bash
# Stop this pod as soon as the job it is watching finishes.
#
#   bash autostop.sh /workspace/s2.log
#
# A pod bills for existing, not for working. A run that finishes at 3am and
# then idles until morning costs more than the run did -- and the only way to
# stop the meter is to stop the pod, which needs a credential the pod does not
# have. RunPod injects RUNPOD_API_KEY, but it is pod-scoped and answers 403 to
# anything account-level, including stopping this pod.
#
# So the key has to be supplied. Put it in /workspace/.runpod_key from a shell
# on the pod -- the RunPod console's web terminal is the safe place, since the
# value then never leaves your browser and this machine:
#
#   printf %s 'rpa_YOURKEY' > /workspace/.runpod_key && chmod 600 /workspace/.runpod_key
#
# Create it at runpod.io/console/user/settings with read/write permission.
#
# Cancel at any time: pkill -f autostop.sh
#
# Stopping keeps the pod and the volume. Nothing generated is lost -- the images
# are on the volume, which survives both stopping and the container disk being
# wiped on the next start.

set -uo pipefail

LOG="${1:-/workspace/s2.log}"
KEY_FILE="${RUNPOD_KEY_FILE:-/workspace/.runpod_key}"
STATE="/workspace/autostop.state"
# The marker every hero.py run prints when it is done, success or not.
DONE_RE='made, [0-9]+ failed'

say() { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "$STATE"; }

if [ ! -s "$KEY_FILE" ]; then
    say "no key at $KEY_FILE -- refusing to watch a job it cannot stop"
    say "see the header of autostop.sh"
    exit 2
fi
KEY=$(tr -d ' \t\r\n' < "$KEY_FILE")

# Fail now, not in four hours. A key that cannot read cannot stop either, and
# the whole point is that nobody is awake to notice.
#
# Positive proof only. The first version asked whether the response *looked
# like* an error and treated everything else as success -- so an empty body, a
# timeout, or any unanticipated shape all passed. It cheerfully accepted a pod
# id pasted in place of a key and reported "key accepted", which is the exact
# failure it was written to prevent.
check_key() {
    local body code
    body=$(curl -s --max-time 20 -w '
%{http_code}' -H "Content-Type: application/json"         -H "Authorization: Bearer $KEY" -X POST https://api.runpod.io/graphql         -d '{"query":"query { myself { id pods { id } } }"}')
    code=$(printf '%s' "$body" | tail -1)
    body=$(printf '%s' "$body" | sed '$d')
    [ "$code" = "200" ] || { echo "http $code"; return 1; }
    # The account id must be present. UNAUTHORIZED returns 200 with a null
    # myself, so the status code alone proves nothing.
    printf '%s' "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as exc:
    print(f"unparseable response: {exc}"); raise SystemExit(1)
if d.get("errors"):
    print(str(d["errors"])[:150]); raise SystemExit(1)
me = (d.get("data") or {}).get("myself")
if not me or not me.get("id"):
    print("no account id in response"); raise SystemExit(1)
pods = [p["id"] for p in (me.get("pods") or [])]
import os
if os.environ["RUNPOD_POD_ID"] not in pods:
    # Reading is not enough -- the key must be able to see this pod, or the
    # stop mutation will fail on a key that passed every other check.
    print(f"key cannot see pod {os.environ['RUNPOD_POD_ID']}"); raise SystemExit(1)
print("ok")
' || return 1
}

if [ "${#KEY}" -lt 30 ]; then
    say "key in $KEY_FILE is ${#KEY} characters -- too short to be an API key."
    say "A RunPod API key is about 50 characters. ${#KEY} characters is the"
    say "length of a pod id; create a key at runpod.io/console/user/settings."
    exit 3
fi

# One retry, because a transient network failure is not a bad key -- but a
# failure that persists is treated as fatal either way.
if ! reason=$(check_key); then
    sleep 10
    if ! reason=$(check_key); then
        say "key rejected: $reason"
        say "not watching -- a key that cannot read this pod cannot stop it"
        exit 3
    fi
fi
say "key accepted; watching $LOG for a finished run"

while true; do
    if ! pgrep -f 'hero\.py' > /dev/null; then
        say "hero.py is gone"
        break
    fi
    if grep -aqE "$DONE_RE" "$LOG" 2>/dev/null; then
        say "run reported finished"
        break
    fi
    sleep 30
done

# Let the last image finish writing and the CSV close.
sleep 20
say "images on the volume: $(ls /workspace/elements_out/models/*.png 2>/dev/null | wc -l)"
grep -aE 'made, [0-9]+ failed' "$LOG" 2>/dev/null | tail -1 | tee -a "$STATE"

say "stopping pod $RUNPOD_POD_ID"
resp=$(curl -s --max-time 30 -H "Content-Type: application/json" \
    -H "Authorization: Bearer $KEY" -X POST https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}")
say "response: ${resp:0:200}"

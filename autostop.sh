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
probe=$(curl -s --max-time 20 -H "Content-Type: application/json" \
    -H "Authorization: Bearer $KEY" -X POST https://api.runpod.io/graphql \
    -d '{"query":"query { myself { id } }"}')
case "$probe" in
    *UNAUTHORIZED*|*"errors"*)
        say "key rejected by the API -- not watching. Response: ${probe:0:120}"
        exit 3 ;;
esac
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

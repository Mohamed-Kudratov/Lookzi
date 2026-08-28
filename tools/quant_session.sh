#!/usr/bin/env bash
# The whole 4-bit measurement, start to finish, as one detached run.
#
#     nohup bash tools/quant_session.sh > /workspace/session.log 2>&1 &
#
# Written as one chain because the last attempt was driven step by step from a
# laptop, and when that connection stopped the pod sat idle with the work half
# done. A pod bills for existing, so anything that needs a human between steps
# is a step that can cost a night. Everything here resumes: the download picks
# up where it stopped, and images already generated are kept.
#
# The order is not arbitrary. Downloading the 4-bit checkpoint while the bf16
# model is loading puts a 15.8 GB write and a 54 GB read on the same network
# volume, and last time the download slowed to a crawl -- 2.5 GB to 2.8 GB in
# the time the other job read fifty. So the download finishes first, alone.
set -uo pipefail

REPO=/workspace/lvton
PAIRS="${PAIRS:-20}"
OUT="${OUT:-/workspace/quant_eval}"

step() { printf '\n=== %s  (%s) ===\n' "$1" "$(date -u +%H:%M:%S)" ; }

cd "$REPO"

step "environment"
bash tools/setup_pod.sh || { echo "setup failed"; exit 1; }
# shellcheck disable=SC1091
. "$REPO/.podenv"

step "4-bit checkpoint"
python - <<'PY'
import time
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("ovedrive/Qwen-Image-Edit-2509-4bit")
print(f"  {p}\n  took {time.time() - t:.0f}s", flush=True)
PY
[ $? -eq 0 ] || { echo "download failed"; exit 1; }
du -sh "$HF_HOME/hub/models--ovedrive--Qwen-Image-Edit-2509-4bit"

step "bf16"
python eval/quantisation.py --pairs "$PAIRS" --out "$OUT" --only bf16

step "4-bit"
python eval/quantisation.py --pairs "$PAIRS" --out "$OUT" --only nf4

step "measuring"
# No --only, so this pass finds both sets on disk, skips generation entirely,
# and goes straight to scoring.
python eval/quantisation.py --pairs "$PAIRS" --out "$OUT"

step "done"
# Stop the meter if the key is there to do it with. RunPod injects a pod-scoped
# RUNPOD_API_KEY that answers 403 to anything account-level, including stopping
# this pod, so the account key has to be put on the volume by hand -- from the
# RunPod console's web terminal, where the value never leaves your browser and
# this machine:
#
#   printf %s 'rpa_YOURKEY' > /workspace/.runpod_key && chmod 600 /workspace/.runpod_key
#
# Without it this just says so and leaves the pod running, which is the safe
# failure: an idle pod costs money, but a pod stopped by surprise costs work.
if [ -x "$REPO/tools/stop_pod.sh" ] && [ -s /workspace/.runpod_key ]; then
    bash "$REPO/tools/stop_pod.sh" || echo "  could not stop; pod left running"
else
    echo "no /workspace/.runpod_key, so the pod is left running -- stop it yourself"
fi

#!/usr/bin/env bash
# Bring the service up on a fresh pod, with one command.
#
#   bash serve.sh
#
# A pod's container disk is wiped every time it stops, so a restart loses the
# Python environment, and the service that was answering on
# https://<POD_ID>-7860.proxy.runpod.net stops answering. The volume keeps the
# weights, the images and the code; everything else has to be put back.
#
# This does that in order and then stays out of the way: update the code,
# install whatever the container disk lost, start uvicorn detached so it
# survives the ssh session, and wait until /health says the model is resident.
#
# Idempotent. Running it against a pod that is already serving replaces the
# process rather than starting a second one on a port that is already taken --
# which is a real failure mode, because uvicorn exits on a bind error and
# leaves the previous process serving stale code.

set -uo pipefail

REPO="${REPO:-/workspace/lvton}"
PORT="${PORT:-7860}"
LOG="${SERVICE_LOG:-/workspace/svc.log}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.pip-cache}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

say() { printf '\n=== %s ===\n' "$1"; }

cd "$REPO" || { echo "no repo at $REPO" >&2; exit 2; }

say "code"
rm -f .git/index.lock
git fetch -q origin 2>/dev/null && git reset -q --hard origin/main 2>/dev/null
git log --oneline -1
# The bundled fork's repo root shadows the installed package -- it has no
# __init__.py, so the import yields an empty namespace package.
if [ -d "$REPO/diffusers" ]; then
    rm -rf "$REPO/diffusers_src"
    mv "$REPO/diffusers" "$REPO/diffusers_src"
fi

say "stack"
if python -c "from diffusers import QwenImageEditPlusPipeline" 2>/dev/null; then
    echo "already installed"
else
    bash start.sh --no-ui 2>&1 | grep -aE '^---|weights ok|ERROR' || true
fi
python -c "import fastapi, uvicorn, multipart" 2>/dev/null || \
    pip install -q fastapi "uvicorn[standard]" python-multipart

say "service"
# Anything already on the port has to go first. Gradio viewers from a previous
# session bind 7860 too, and uvicorn does not take the port from them -- it
# exits, leaving the old process answering with whatever it was serving.
pkill -9 -f 'uvicorn service.app' 2>/dev/null
pkill -9 -f view_sweep 2>/dev/null
sleep 3
nohup setsid python -m uvicorn service.app:app \
    --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 < /dev/null &

POD_URL="https://${RUNPOD_POD_ID:-POD_ID}-${PORT}.proxy.runpod.net"
echo "starting -> $LOG"
echo "url: $POD_URL"

# The model takes ten minutes from cold. Waiting here means the command returns
# when the service is actually usable, rather than when it has merely started
# and would 503 every request for the next ten minutes.
say "waiting for the model"
for i in $(seq 1 120); do
    h=$(curl -s --max-time 5 "http://localhost:$PORT/health" 2>/dev/null)
    case "$h" in
        *'"ready":true'*) echo "$h"; echo; echo "Ready: $POD_URL"; exit 0 ;;
        *'"error":"'*)    echo "$h"; echo; echo "Load FAILED -- see $LOG" >&2; exit 1 ;;
    esac
    [ $((i % 6)) -eq 0 ] && printf '  %ds\n' $((i * 10))
    sleep 10
done
echo "still not ready after 20 minutes -- see $LOG" >&2
exit 1

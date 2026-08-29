#!/usr/bin/env bash
# Start, stop or check the model server on a pod.
#
#     bash tools/pod_serve.sh start            # the try-on model
#     bash tools/pod_serve.sh start zimage     # the model maker
#     bash tools/pod_serve.sh status zimage
#     bash tools/pod_serve.sh stop
#
# This exists because `pkill -f pod_server` kills the shell that runs it. The
# pattern matches the ssh command line carrying the pkill, so the whole session
# dies -- including the server it was about to start. It looked like the server
# had crashed silently: no process, nothing listening, and nothing in the log.
#
# So the running process is found by the port it holds, which nothing else can
# accidentally match.
set -uo pipefail

REPO="${REPO:-/opt/lookzi}"

# Two servers, because Z-Image needs diffusers from source and the try-on stack
# cannot have it -- their huggingface_hub requirements do not overlap. Same
# machine, same card, two interpreters.
WHICH="${2:-tryon}"
if [ "$WHICH" = "zimage" ]; then
    PORT="${ZIMAGE_PORT:-8001}"
    LOG="${ZIMAGE_LOG:-/workspace/zimage_server.log}"
    PYTHON="${ZIMAGE_PYTHON:-/opt/zimage-venv/bin/python}"
    MODULE="service.zimage_server"
    MODEL="Z-Image-Turbo"
else
    PORT="${POD_SERVER_PORT:-8000}"
    LOG="${POD_SERVER_LOG:-/workspace/pod_server.log}"
    PYTHON="${POD_PYTHON:-python}"
    MODULE="service.pod_server"
    MODEL="${MODEL_PATH:-ovedrive/Qwen-Image-Edit-2509-4bit}"
fi

pid_on_port() {
    ss -tlnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p' \
        | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2
}

case "${1:-status}" in
start|restart)
    pid=$(pid_on_port)
    if [ -n "$pid" ]; then
        echo "stopping $pid"
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 20); do
            sleep 0.5
            [ -z "$(pid_on_port)" ] && break
        done
        [ -n "$(pid_on_port)" ] && { kill -9 "$(pid_on_port)" 2>/dev/null; sleep 1; }
    fi

    cd "$REPO" || exit 1
    # The same environment every process on this pod needs. The thread caps are
    # a correctness fix: the container gets 13.6 CPUs while /proc advertises the
    # host's 128, and a pool sized from the wrong number spends its life in the
    # scheduler.
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    export NUMEXPR_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
    export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_XET=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export MODEL_PATH="$MODEL" POD_SERVER_PORT="$PORT" ZIMAGE_PORT="$PORT"

    if [ ! -x "$PYTHON" ] && [ "$PYTHON" != "python" ]; then
        echo "no interpreter at $PYTHON -- run tools/setup_pod.sh --zimage" >&2
        exit 1
    fi
    setsid nohup "$PYTHON" -m "$MODULE" > "$LOG" 2>&1 < /dev/null &
    echo "starting $MODEL on :$PORT"

    # Wait for the port, not for the weights. The server listens while it
    # loads, on purpose -- otherwise every check during three minutes of
    # loading is a refused connection, which reads as "it did not start".
    for _ in $(seq 1 30); do
        sleep 1
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null; then
            curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo
            exit 0
        fi
    done
    echo "did not come up; last of $LOG:"
    tail -15 "$LOG"
    exit 1
    ;;
stop)
    pid=$(pid_on_port)
    [ -n "$pid" ] || { echo "nothing on :$PORT"; exit 0; }
    kill "$pid" && echo "stopped $pid"
    ;;
status)
    pid=$(pid_on_port)
    if [ -z "$pid" ]; then echo "nothing listening on :$PORT"; exit 1; fi
    echo "pid $pid"
    curl -s -m 8 "http://127.0.0.1:$PORT/health"; echo
    ;;
*)
    echo "usage: $0 start|stop|status [tryon|zimage]" >&2; exit 2 ;;
esac

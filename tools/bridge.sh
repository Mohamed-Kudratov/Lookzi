#!/usr/bin/env bash
# The bridge, started from what the panel already knows.
#
#     bash tools/bridge.sh
#
# The address comes from .env, which the panel writes at the end of every
# successful setup. Before this it lived in a file under /tmp that was edited
# by hand after each migration, and RunPod hands out a new address every time
# -- so the bridge would come back up pointing at a pod that no longer exists,
# and the failure looked like the pod being broken rather than the address
# being stale.
#
# Only one runs at a time. The worker refuses to start if the forwarded port
# is already held, which is the same rule the benchmark needed and did not
# have: two of these claim the same jobs and fight over the same tunnel.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [ ! -f .env ]; then
    echo "no .env here. Run the panel first, or write POD_SSH into it." >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a

if [ -z "${POD_SSH:-}" ]; then
    echo "POD_SSH is not set in .env. The panel writes it when a pod is ready;" >&2
    echo "otherwise it looks like: POD_SSH=\"root@1.2.3.4 -p 22022\"" >&2
    exit 1
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://lookzi:lookzi@127.0.0.1:5433/lookzi}"
export S3_ENDPOINT="${S3_ENDPOINT:-http://127.0.0.1:9000}"
export S3_PUBLIC_ENDPOINT="${S3_PUBLIC_ENDPOINT:-http://127.0.0.1:9000}"
export S3_KEY="${S3_KEY:-lookzi}"
export S3_SECRET="${S3_SECRET:-lookzi-dev-secret}"
export S3_BUCKET="${S3_BUCKET:-lookzi}"
# Every tool the product offers. A tool missing from this list is a tool whose
# jobs sit in the queue for ever while the studio says a worker is ready.
export WORKER_TOOLS="${WORKER_TOOLS:-product-to-model,virtual-try-on,model-swap,packshot,model-creation,product-in-scene,try-on-v2}"

echo "bridge -> ${POD_SSH}"
exec python -m service.tunnel_worker

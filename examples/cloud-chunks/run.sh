#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$PYTHONHOME/bin/python3.10" "$root/serving/examples/cloud-chunks/run.py" "$@"

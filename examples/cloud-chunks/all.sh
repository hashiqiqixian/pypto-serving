#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
for chunk in 80 128 512; do
    echo "BEGIN_OFFICIAL_CHUNK $chunk"
    bash "$here/run.sh" "$chunk"
done

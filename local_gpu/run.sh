#!/usr/bin/env bash
set -euo pipefail

script_dir="$(dirname -- "${BASH_SOURCE[0]}")"
exec bash "$script_dir/../scripts/run_gpu.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

python_bin="${MEETING_TRANSCRIBER_PYTHON:-python}"
cuda_libraries="$("$python_bin" - <<'PY'
from importlib.util import find_spec

paths = []
for module in ("nvidia.cublas.lib", "nvidia.cuda_nvrtc.lib", "nvidia.cudnn.lib"):
    try:
        spec = find_spec(module)
    except ModuleNotFoundError:
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Install the GPU dependencies with python -m pip install -e '.[gpu]'.")
    paths.extend(spec.submodule_search_locations)

print(":".join(paths))
PY
)"

export LD_LIBRARY_PATH="$cuda_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$python_bin" -u -m meeting_transcriber "$@" --device cuda

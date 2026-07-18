#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUB_LIB="$ROOT_DIR/third_party/lib/libittnotify_stub.so"

if [[ -f "$STUB_LIB" ]]; then
  export LD_PRELOAD="$STUB_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

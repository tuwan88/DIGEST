#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT_DIR/scripts/ittnotify_stub.c"
OUT_DIR="$ROOT_DIR/third_party/lib"
OUT_LIB="$OUT_DIR/libittnotify_stub.so"

mkdir -p "$OUT_DIR"
gcc -shared -fPIC -O2 "$SRC" -o "$OUT_LIB"
echo "built $OUT_LIB"

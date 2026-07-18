#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUB_LIB="$ROOT_DIR/third_party/lib/libittnotify_stub.so"

if [[ ! -f "$STUB_LIB" ]]; then
  echo "missing stub library: $STUB_LIB" >&2
  echo "build it first with: bash scripts/build_ittnotify_stub.sh" >&2
  exit 1
fi

source "$ROOT_DIR/scripts/activate_runtime.sh"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import dgl; print(dgl.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
python -c "import torch_scatter, torch_sparse; print('pyg ops ok')"

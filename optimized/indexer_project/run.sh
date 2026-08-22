#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
die() { echo "ERROR: $*" >&2; exit 1; }

[ -n "${ASCEND_HOME_PATH:-}" ] || source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH unset"

# 优先使用当前环境的 python
PY=$(which python3)
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PY" || die "cmake config failed"
make -j8 || die "make failed"
cd ..
echo "built build/libindexer_ops.so successfully!"

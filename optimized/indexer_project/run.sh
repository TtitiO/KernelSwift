#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
die() { echo "ERROR: $*" >&2; exit 1; }

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    # 版本化的 toolkit 优先（与 sparse_attention/run_huawei.sh 保持一致）
    [ -f /usr/local/Ascend/cann-8.5.2/set_env.sh ] && source /usr/local/Ascend/cann-8.5.2/set_env.sh
    [ -n "${ASCEND_HOME_PATH:-}" ] || source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH unset"

# 项目 venv 优先（含 torch/torch_npu）
PY=/home/tinglin/wksp/KernelSwift/.venv/bin/python
[ -x "$PY" ] || PY=$(which python3)
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PY" -DINDEXER_KERNEL_CXX_LINKAGE=ON || die "cmake config failed"
make -j8 || die "make failed"
cd ..
echo "built build/libindexer_ops.so successfully!"

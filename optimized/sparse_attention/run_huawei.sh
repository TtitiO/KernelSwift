set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
die() { echo "ERROR: $*" >&2; exit 1; }
[ -n "${ASCEND_HOME_PATH:-}" ] || source /usr/local/Ascend/cann-8.5.2/set_env.sh
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH unset"
# Use the project venv python (has torch/torch_npu).
PY=/home/tinglin/wksp/KernelSwift/.venv/bin/python
[ -x "$PY" ] || PY=python3
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PY" -DSPARSE_ATTN_KERNEL_CXX_LINKAGE=ON || die "cmake config failed"
make -j8 || die "make failed"
cd ..
echo "built build/libsparse_attn_ops.so"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS_DIR="${REPO_ROOT}/difflow3d/ops/pointnet2"
SRC_DIR="${OPS_DIR}/src"

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'PointNet2 ops: %s\n' "${OPS_DIR}"
printf 'Python: %s\n' "$(command -v python3)"
python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
PY

# Fail early with a useful error if an old THC-based source tree is present.
if grep -RInE 'THC/THC\.h|THCState' "${SRC_DIR}" --include='*.cpp' --include='*.cu' --include='*.h' >/tmp/difflow_legacy_pointnet2.txt; then
    echo 'ERROR: legacy THC API is still present in PointNet2 sources:' >&2
    cat /tmp/difflow_legacy_pointnet2.txt >&2
    exit 2
fi

cd "${OPS_DIR}"
rm -rf build pointnet2_cuda*.so
python3 setup.py build_ext --inplace

# Verify the extension from THIS checkout, not a stale /opt or site-packages copy.
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}:${OPS_DIR}" python3 - <<'PY'
from pathlib import Path
from difflow3d.ops.pointnet2 import pointnet2_utils

required = (
    'gaussian_softmax_recovery_wrapper',
    'gaussian_recovery_hash_build_wrapper',
    'gaussian_softmax_recovery_local_wrapper',
    'gaussian_softmax_recovery_local_track_aware_wrapper',
)

loaded = Path(pointnet2_utils.extension_path()).resolve()
expected_dir = (Path.cwd() / 'difflow3d' / 'ops' / 'pointnet2').resolve()
print('pointnet2_cuda:', loaded)
if loaded.parent != expected_dir:
    raise RuntimeError(
        f'Loaded stale pointnet2_cuda from {loaded}; expected an extension in {expected_dir}'
    )
missing = [name for name in required if not hasattr(pointnet2_utils.pointnet2, name)]
if missing:
    raise RuntimeError(f'Missing recovery CUDA symbols after rebuild: {missing}')
print('PointNet2 + dense-recovery CUDA extension build OK')
PY

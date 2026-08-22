#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${DIFFLOW_CONFIG:-configs/config.yaml}"

# Also works from an interactive DifFlow3D container.
if [[ -f /.dockerenv ]]; then
    cd "${REPO_ROOT}"
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        set +u
        source /opt/ros/humble/setup.bash
        set -u
    fi
    exec python3 scripts/test_voxel_difflow.py --config "${CONFIG}" "$@"
fi

# Host mode: create/reuse one development container and run the current
# bind-mounted checkout. Rebuild the native extension only when it is missing
# or older than its C++/CUDA sources; this adds no runtime cost to the algorithm.
# shellcheck source=_docker_common.sh
source "${REPO_ROOT}/scripts/_docker_common.sh"
ensure_pointnet2_extension

exec docker exec -it "${DIFFLOW_CONTAINER}" \
    bash -lc 'source /opt/ros/humble/setup.bash && cd /workspace && exec python3 scripts/test_voxel_difflow.py --config "$1" "${@:2}"' \
    bash "${CONFIG}" "$@"

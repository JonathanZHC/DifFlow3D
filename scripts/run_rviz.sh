#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVIZ_CONFIG="${DIFFLOW_RVIZ_CONFIG:-rviz/voxel_difflow.rviz}"

if [[ -f /.dockerenv ]]; then
    cd "${REPO_ROOT}"
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        set +u
        source /opt/ros/humble/setup.bash
        set -u
    fi
    exec rviz2 -d "${RVIZ_CONFIG}"
fi

# shellcheck source=_docker_common.sh
source "${REPO_ROOT}/scripts/_docker_common.sh"
if [[ -z "${DISPLAY:-}" ]]; then
    echo "ERROR: DISPLAY is not set; RViz needs an X11/Xwayland display." >&2
    exit 1
fi
ensure_container

exec docker exec -it \
    -e "DISPLAY=${DISPLAY}" \
    "${DIFFLOW_CONTAINER}" \
    bash -lc 'source /opt/ros/humble/setup.bash && cd /workspace && exec rviz2 -d "$1"' bash "${RVIZ_CONFIG}"

#!/usr/bin/env bash
set -euo pipefail

# Native ROS 2 Jazzy uses Python 3.12 and is compatible with Isaac Sim 6.
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # setup.bash is not guaranteed to be nounset-safe.
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export PYTHONPATH="/opt/DifFlow3D:/opt/DifFlow3D/pointnet2${PYTHONPATH:+:${PYTHONPATH}}"

exec "$@"
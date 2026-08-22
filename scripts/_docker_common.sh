#!/usr/bin/env bash
# Shared host-side Docker helpers. Source this file; do not run it directly.

DIFFLOW_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFLOW_IMAGE="${DIFFLOW_IMAGE:-difflow3d:latest}"
DIFFLOW_CONTAINER="${DIFFLOW_CONTAINER:-difflow3d}"

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker is not installed or not on PATH." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Docker daemon is not reachable by the current user." >&2
        exit 1
    fi
}

container_exists() {
    docker container inspect "${DIFFLOW_CONTAINER}" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker container inspect -f '{{.State.Running}}' "${DIFFLOW_CONTAINER}" 2>/dev/null || true)" == "true" ]]
}

workspace_mount_source() {
    docker container inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' \
        "${DIFFLOW_CONTAINER}" 2>/dev/null || true
}

ensure_container() {
    require_docker
    if ! docker image inspect "${DIFFLOW_IMAGE}" >/dev/null 2>&1; then
        echo "ERROR: Docker image ${DIFFLOW_IMAGE} does not exist." >&2
        echo "Build it first with: bash scripts/docker_build.sh" >&2
        exit 1
    fi

    if container_exists; then
        local mounted
        mounted="$(workspace_mount_source)"
        if [[ -z "${mounted}" ]]; then
            echo "ERROR: Existing container ${DIFFLOW_CONTAINER} does not bind-mount this checkout at /workspace." >&2
            echo "Remove it with: docker rm -f ${DIFFLOW_CONTAINER}" >&2
            echo "Then rerun this script." >&2
            exit 1
        fi
        if [[ "$(readlink -f "${mounted}")" != "$(readlink -f "${DIFFLOW_REPO_ROOT}")" ]]; then
            echo "ERROR: Existing container ${DIFFLOW_CONTAINER} mounts a different /workspace:" >&2
            echo "  ${mounted}" >&2
            echo "Remove/recreate it, or set DIFFLOW_CONTAINER to another name." >&2
            exit 1
        fi
        if ! container_running; then
            docker start "${DIFFLOW_CONTAINER}" >/dev/null
        fi
        return
    fi

    local -a args=(
        run -d
        --gpus all
        --ipc=host
        --network host
        --name "${DIFFLOW_CONTAINER}"
        --workdir /workspace
        --user "$(id -u):$(id -g)"
        -e HOME="/tmp/difflow3d-home-$(id -u)"
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e QT_X11_NO_MITSHM=1
        --mount "type=bind,src=${DIFFLOW_REPO_ROOT},dst=/workspace"
    )

    # Mount X11 resources at container creation even for timing-only runs, so
    # RViz can be launched later in the same container without recreation.
    if [[ -d /tmp/.X11-unix ]]; then
        args+=( -v /tmp/.X11-unix:/tmp/.X11-unix:rw )
        [[ -n "${DISPLAY:-}" ]] && args+=( -e "DISPLAY=${DISPLAY}" )
        local xauth="${XAUTHORITY:-${HOME}/.Xauthority}"
        if [[ -f "${xauth}" ]]; then
            args+=( -e XAUTHORITY=/tmp/difflow3d.xauth -v "${xauth}:/tmp/difflow3d.xauth:ro" )
        elif command -v xhost >/dev/null 2>&1; then
            # The container runs with the host UID/GID, so granting the local
            # host user is sufficient for X11 in the usual Xorg/Xwayland setup.
            xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || true
        fi
    fi

    args+=( "${DIFFLOW_IMAGE}" bash -lc 'mkdir -p "$HOME" && exec sleep infinity' )
    docker "${args[@]}" >/dev/null
    echo "Started ${DIFFLOW_CONTAINER} with ${DIFFLOW_REPO_ROOT} mounted at /workspace."
}

ensure_pointnet2_extension() {
    ensure_container
    docker exec "${DIFFLOW_CONTAINER}" bash -lc '
        set -euo pipefail
        cd /workspace
        shopt -s nullglob
        so_files=(difflow3d/ops/pointnet2/pointnet2_cuda*.so)
        rebuild=0
        if (( ${#so_files[@]} == 0 )); then
            rebuild=1
        else
            so="${so_files[0]}"
            while IFS= read -r source; do
                if [[ "${source}" -nt "${so}" ]]; then
                    rebuild=1
                    break
                fi
            done < <(find difflow3d/ops/pointnet2/src -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" \) -print)
            [[ difflow3d/ops/pointnet2/setup.py -nt "${so}" ]] && rebuild=1
        fi
        if (( rebuild )); then
            echo "PointNet2 extension missing/stale; rebuilding once..."
            bash scripts/build_pointnet2_ops.sh
        else
            echo "PointNet2 extension is up to date."
        fi
    '
}

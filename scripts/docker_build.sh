#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DIFFLOW_IMAGE:-difflow3d:latest}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH." >&2
    exit 1
fi

echo "Building ${IMAGE} from ${REPO_ROOT}"
docker build -t "${IMAGE}" "$@" "${REPO_ROOT}"

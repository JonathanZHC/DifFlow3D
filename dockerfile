# Isaac Sim 6.0.1 + ROS 2 Jazzy + RTX 5090/Blackwell DifFlow3D image.
#
# Required build-context files:
#   Dockerfile
#   isaacsim_multicamera_rgbd.py
#   test_difflow3d_dense_recovery.py
#
# This image:
#   - uses the official Isaac Sim 6.0.1 runtime;
#   - compiles the legacy DifFlow3D PointNet++ CUDA extension for Python 3.12,
#     PyTorch 2.11, CUDA 12.8 and Blackwell sm_120;
#   - installs ROS 2 Jazzy;
#   - keeps third-party Python packages isolated from Isaac Sim's bundled
#     Python environment;
#   - prepares persistent cache mount points for GUI/headless operation.

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1
ARG CUDA_BUILDER_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG TORCH_VERSION=2.11.0
ARG DIFFLOW_REPO_URL=https://github.com/IRMVLab/DifFlow3D.git
ARG DIFFLOW_REPO_REF=main
ARG TORCH_CUDA_ARCH_LIST=12.0

# =============================================================================
# Stage 1: build DifFlow3D PointNet++ against Python 3.12 / Torch 2.11 / CUDA 12.8
# =============================================================================
FROM ${CUDA_BUILDER_IMAGE} AS difflow-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION
ARG DIFFLOW_REPO_URL
ARG DIFFLOW_REPO_REF
ARG TORCH_CUDA_ARCH_LIST

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    MAX_JOBS=4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      ninja-build \
      python3 \
      python3-dev \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04's system pip is owned by apt. Use an isolated venv rather than
# attempting to upgrade/uninstall the system pip package.
ENV VIRTUAL_ENV=/opt/difflow-build-venv \
    PATH=/opt/difflow-build-venv/bin:${PATH}

RUN /usr/bin/python3 -m venv "${VIRTUAL_ENV}" \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
      torch==${TORCH_VERSION} \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python3 - <<'PY'
import sys
import torch

print("builder Python:", sys.version)
print("builder executable:", sys.executable)
print("builder torch:", torch.__version__)
print("builder torch CUDA:", torch.version.cuda)
PY

RUN git clone --recursive "${DIFFLOW_REPO_URL}" /opt/DifFlow3D \
    && cd /opt/DifFlow3D \
    && git checkout "${DIFFLOW_REPO_REF}" \
    && test -f pretrain_weights/model_difflow_355_0.0114.pth

# Apply API-only compatibility edits required by modern PyTorch/CUDA. The CUDA
# kernel mathematics and DifFlow3D model computations are not changed.
RUN python3 - <<'PY'
from pathlib import Path
import re

repo = Path("/opt/DifFlow3D")
src_dir = repo / "pointnet2" / "src"

for path in sorted(src_dir.glob("*")):
    if path.suffix not in {".cpp", ".cu", ".h", ".hpp"}:
        continue

    text = path.read_text(encoding="utf-8")
    original = text

    # THC was removed from modern PyTorch.
    text = re.sub(
        r'^\s*#include\s*[<"]THC/THC\.h[>"]\s*$\n?',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^\s*extern\s+THCState\s*\*\s*state\s*;\s*$\n?',
        "",
        text,
        flags=re.MULTILINE,
    )

    # Tensor::data<T>() and DeprecatedTypeProperties are obsolete.
    text = re.sub(
        r'\.data\s*<\s*([^>]+?)\s*>\s*\(\s*\)',
        r'.data_ptr<\1>()',
        text,
    )
    text = text.replace(".type().is_cuda()", ".is_cuda()")

    # Use the modern c10 CUDA stream API.
    if "getCurrentCUDAStream" in text:
        text = text.replace(
            "at::cuda::getCurrentCUDAStream()",
            "c10::cuda::getCurrentCUDAStream().stream()",
        )
        text = text.replace(
            "c10::cuda::getCurrentCUDAStream()",
            "c10::cuda::getCurrentCUDAStream().stream()",
        )
        text = text.replace(
            "c10::cuda::getCurrentCUDAStream().stream().stream()",
            "c10::cuda::getCurrentCUDAStream().stream()",
        )

        include = "#include <c10/cuda/CUDAStream.h>"
        if include not in text:
            lines = text.splitlines()
            insert_at = 0
            for index, line in enumerate(lines):
                if line.lstrip().startswith("#include"):
                    insert_at = index + 1
                elif insert_at:
                    break
            lines.insert(insert_at, include)
            text = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    if text != original:
        path.write_text(text, encoding="utf-8")
        print("patched", path.relative_to(repo))

joined = "\n".join(
    candidate.read_text(encoding="utf-8")
    for candidate in src_dir.glob("*")
    if candidate.suffix in {".cpp", ".cu", ".h", ".hpp"}
)

for forbidden in (
    "#include <THC/THC.h>",
    "extern THCState *state;",
    "at::cuda::getCurrentCUDAStream()",
    ".data<float>()",
    ".data<int>()",
):
    if forbidden in joined:
        raise RuntimeError(f"Unpatched legacy API remains: {forbidden}")
PY

RUN python3 -m pip install numpy==1.26.4 \
    && cd /opt/DifFlow3D/pointnet2 \
    && rm -rf build pointnet2_cuda*.so \
    && python3 setup.py build_ext --inplace \
    && cd /opt/DifFlow3D \
    && PYTHONPATH=/opt/DifFlow3D:/opt/DifFlow3D/pointnet2 python3 - <<'PY'
import sys
import torch
import pointnet2_cuda
from pointnet2 import pointnet2_utils

print("builder Python:", sys.executable)
print("builder torch:", torch.__version__)
print("builder torch CUDA:", torch.version.cuda)
print("pointnet2_cuda:", pointnet2_cuda.__file__)
print("pointnet2_utils:", pointnet2_utils.__file__)
print("PointNet++ builder import OK")
PY

# The custom extension has a DT_NEEDED entry for libcudart.so.12. Bundle only
# that runtime library; NVIDIA driver libraries are still supplied at run time
# by the NVIDIA Container Toolkit through --gpus all.
RUN python3 - <<'PY'
from pathlib import Path
import shutil

root = Path("/usr/local/cuda")
candidates = sorted(root.rglob("libcudart.so.12"))
if not candidates:
    raise FileNotFoundError("libcudart.so.12 was not found in the CUDA builder")

source = candidates[0].resolve()
destination_dir = Path("/opt/difflow-cuda-runtime/lib")
destination_dir.mkdir(parents=True, exist_ok=True)
destination = destination_dir / "libcudart.so.12"
shutil.copy2(source, destination)
print("bundled CUDA runtime:", source, "->", destination)
PY

# =============================================================================
# Stage 2: official Isaac Sim runtime
# =============================================================================
FROM ${ISAAC_SIM_IMAGE}

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION
ARG TORCH_CUDA_ARCH_LIST

# Do NOT set PYTHONPATH here. Isaac Sim's python.sh prepares its own bundled
# Python paths. Replacing PYTHONPATH globally can hide Isaac Sim's Torch and
# extension packages. Additional paths are registered later through a .pth file.
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    ISAAC_SIM_PATH=/isaac-sim \
    DIFFLOW_REPO=/opt/DifFlow3D \
    ROS_DISTRO=jazzy \
    ROS_DOMAIN_ID=100 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNICLIENT_HUB_MODE=disabled \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    HOME=/isaac-sim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ROS 2 Jazzy uses Python 3.12 on Ubuntu 24.04 and therefore matches Isaac Sim
# 6's Python ABI.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg2 \
      locales \
      lsb-release \
      software-properties-common \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libx11-6 \
      libxrandr2 \
      libxinerama1 \
      libxcursor1 \
      libxi6 \
      xauth \
      mesa-utils \
    && locale-gen en_US.UTF-8 \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-base \
      ros-jazzy-rmw-fastrtps-cpp \
      ros-jazzy-geometry-msgs \
      ros-jazzy-sensor-msgs \
      ros-jazzy-sensor-msgs-py \
      ros-jazzy-std-msgs \
      ros-jazzy-visualization-msgs \
      ros-jazzy-tf2-ros \
      ros-jazzy-tf2-tools \
      ros-jazzy-rviz2 \
      ros-jazzy-image-transport-plugins \
      python3-colcon-common-extensions \
      python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Isaac Sim's container does not guarantee that torch is importable from
# /isaac-sim/python.sh. Install the exact same PyTorch/CUDA wheel used by the
# builder into an isolated target directory. Keeping this directory separate
# prevents pip from modifying Kit's bundled extension packages.
RUN mkdir -p /opt/difflow-python-packages \
    && /isaac-sim/python.sh -m pip install \
      --no-cache-dir \
      --target /opt/difflow-python-packages \
      torch==${TORCH_VERSION} \
      --index-url https://download.pytorch.org/whl/cu128

# Install only the extra packages needed by the DifFlow3D scripts. Do not
# install another NumPy build here: Isaac Sim 6.0.1 already provides NumPy
# 2.3.x. Numba is intentionally omitted because DifFlow3D does not import it,
# and the older Numba 0.61.2 release rejects NumPy 2.3 at import time.
# --no-deps prevents pip from replacing Isaac Sim's bundled packages.
RUN /isaac-sim/python.sh -m pip install \
      --no-cache-dir \
      --no-deps \
      --target /opt/difflow-python-packages \
      scipy==1.15.3 \
      scikit-learn==1.6.1 \
      joblib==1.4.2 \
      threadpoolctl==3.5.0 \
      tqdm==4.67.1 \
      cffi==1.17.1 \
      pycparser==2.22 \
      pypng==0.20220715.0 \
      "ultralytics-thop>=2.0.18" \
      PyYAML==6.0.2 \
      Pillow==11.2.1

COPY --from=difflow-builder /opt/DifFlow3D /opt/DifFlow3D

# Supply the CUDA runtime required by pointnet2_cuda and register it with the
# system dynamic linker.
RUN mkdir -p /usr/local/lib/difflow-cuda

COPY --from=difflow-builder \
    /opt/difflow-cuda-runtime/lib/libcudart.so.12 \
    /usr/local/lib/difflow-cuda/libcudart.so.12

RUN printf '%s\n' \
      '/usr/local/lib/difflow-cuda' \
      '/opt/difflow-python-packages/torch/lib' \
      > /etc/ld.so.conf.d/difflow-runtime.conf \
    && ldconfig \
    && ldconfig -p | grep -F 'libcudart.so.12' \
    && test -f /opt/difflow-python-packages/torch/lib/libtorch_python.so

# Add paths at the END of Isaac Sim's normal sys.path. This preserves Isaac
# Sim's bundled Torch, numpy, packaging and extension packages while exposing
# DifFlow3D and the isolated dependency directory.
RUN /isaac-sim/python.sh - <<'PY'
from pathlib import Path
import site

site_dirs = site.getsitepackages()
if not site_dirs:
    raise RuntimeError("Isaac Sim site-packages directory was not found")

site_dir = Path(site_dirs[0])
pth = site_dir / "difflow3d_paths.pth"
pth.write_text(
    "import sys; "
    "sys.path.append('/opt/difflow-python-packages'); "
    "sys.path.append('/opt/DifFlow3D'); "
    "sys.path.append('/opt/DifFlow3D/pointnet2')\n",
    encoding="utf-8",
)

print("created:", pth)
print(pth.read_text(encoding="utf-8"))
PY

WORKDIR /workspace

COPY isaacsim_multicamera_rgbd.py /workspace/isaacsim_multicamera_rgbd.py
COPY test_difflow3d_dense_recovery.py /workspace/test_difflow3d_dense_recovery.py

# Create the entrypoint here so no separate entrypoint file is required.
RUN cat > /usr/local/bin/isaacsim-entrypoint <<'EOF_ENTRYPOINT'
#!/usr/bin/env bash
set -euo pipefail

# Native ROS 2 Jazzy uses Python 3.12. Source it without replacing Isaac Sim's
# own Python package paths; DifFlow3D paths are loaded through a .pth file.
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export OMNICLIENT_HUB_MODE="${OMNICLIENT_HUB_MODE:-disabled}"

exec "$@"
EOF_ENTRYPOINT

RUN chmod +x /usr/local/bin/isaacsim-entrypoint

# Inspect dynamic dependencies. The command is allowed to report NVIDIA driver
# libraries as missing during docker build because those are injected only at
# docker run --gpus all time.
RUN extension_path="$(find /opt/DifFlow3D/pointnet2 -maxdepth 1 \
      -name 'pointnet2_cuda*.so' -print -quit)" \
    && test -n "${extension_path}" \
    && echo "PointNet++ extension: ${extension_path}" \
    && ldd "${extension_path}" || true

# Validate the isolated PyTorch runtime and then DifFlow3D.
RUN /isaac-sim/python.sh - <<'PY'
import sys
import numpy
import torch

print("Isaac Python:", sys.version)
print("Isaac executable:", sys.executable)
print("Isaac numpy:", numpy.__version__)
print("Isaac torch:", torch.__version__)
print("Isaac torch CUDA:", torch.version.cuda)
print("CUDA available during docker build:", torch.cuda.is_available())

import scipy
import sklearn
import yaml
import PIL
import thop
import pointnet2_cuda
from pointnet2 import pointnet2_utils

print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("pointnet2_cuda:", pointnet2_cuda.__file__)
print("pointnet2_utils:", pointnet2_utils.__file__)
print("DifFlow3D PointNet++ import in Isaac Python OK")
PY

# Prepare only the directories that must be writable at run time.
#
# Do not recursively chown/chmod /opt/DifFlow3D or
# /opt/difflow-python-packages. They are immutable runtime dependencies and
# are already world-readable under the normal build umask. Recursive metadata
# changes over the large pip target can trigger overlayfs/BuildKit
# "Bad message" errors.
#
# Bind mounts inherit ownership from the host, so the corresponding host cache
# and output directories must also be writable by UID/GID 1234.
RUN install -d -o 1234 -g 1234 -m 0775 \
      /workspace \
      /workspace/camera_output \
      /isaac-sim/.cache \
      /isaac-sim/.cache/ov \
      /isaac-sim/.cache/warp \
      /isaac-sim/.nv \
      /isaac-sim/.nv/ComputeCache \
      /isaac-sim/.nvidia-omniverse \
      /isaac-sim/.nvidia-omniverse/logs \
      /isaac-sim/.nvidia-omniverse/config \
      /isaac-sim/.local \
      /isaac-sim/.local/share \
      /isaac-sim/.local/share/ov \
      /isaac-sim/.local/share/ov/data \
      /isaac-sim/.local/share/ov/pkg \
    && test -r /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
    && test -r /opt/difflow-python-packages/torch/__init__.py \
    && test -x /opt/ros/jazzy/bin/rviz2

USER 1234:1234

ENTRYPOINT ["/usr/local/bin/isaacsim-entrypoint"]
CMD ["/bin/bash"]
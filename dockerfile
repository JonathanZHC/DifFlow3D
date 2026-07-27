# RTX 5090 / Blackwell-compatible DifFlow3D image.
# Build context needs only:
#   Dockerfile
#   test_difflow3d_superquadrics.py

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble
ARG PYTORCH_VERSION=2.7.1
ARG DIFFLOW_REPO_URL=https://github.com/IRMVLab/DifFlow3D.git
ARG DIFFLOW_REPO_REF=main
ARG TORCH_CUDA_ARCH_LIST=12.0

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    DIFFLOW_REPO=/opt/DifFlow3D \
    ROS_DOMAIN_ID=100 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    PYTHONPATH=/opt/DifFlow3D:/opt/DifFlow3D/pointnet2:/opt/ros/humble/lib/python3.10/site-packages \
    LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/usr/local/cuda/lib64

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      gnupg2 \
      locales \
      lsb-release \
      ninja-build \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      libgl1 \
      libglib2.0-0 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 is used only by the test script's publishers. RViz can run on the host.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
      ros-${ROS_DISTRO}-geometry-msgs \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-sensor-msgs-py \
      ros-${ROS_DISTRO}-std-msgs \
      ros-${ROS_DISTRO}-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
      pip==25.1.1 setuptools==69.5.1 wheel==0.45.1 \
    && python3 -m pip install \
      torch==${PYTORCH_VERSION} \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install \
      numpy==1.26.4 \
      scipy==1.13.1 \
      scikit-learn==1.5.2 \
      numba==0.60.0 \
      tqdm==4.67.1 \
      cffi==1.17.1 \
      pypng==0.20220715.0 \
      thop==0.1.1.post2209072238 \
      PyYAML==6.0.2 \
      packaging==24.2

RUN git clone --recursive "${DIFFLOW_REPO_URL}" /opt/DifFlow3D \
    && cd /opt/DifFlow3D \
    && git checkout "${DIFFLOW_REPO_REF}" \
    && test -f pretrain_weights/model_difflow_355_0.0114.pth

# The upstream PointNet++ extension targets PyTorch 1.7/CUDA 11.0.
# Apply API-only compatibility edits for modern PyTorch/CUDA. The CUDA kernels
# and all model mathematics remain unchanged.
RUN python3 - <<'PY'
from pathlib import Path
import re

repo = Path('/opt/DifFlow3D')
src_dir = repo / 'pointnet2' / 'src'

for path in sorted(src_dir.glob('*')):
    if path.suffix not in {'.cpp', '.cu', '.h', '.hpp'}:
        continue

    text = path.read_text(encoding='utf-8')
    original = text

    # THC was removed from modern PyTorch.
    text = re.sub(
        r'^\s*#include\s*[<"]THC/THC\.h[>"]\s*$\n?',
        '',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^\s*extern\s+THCState\s*\*\s*state\s*;\s*$\n?',
        '',
        text,
        flags=re.MULTILINE,
    )

    # Tensor::data<T>() and DeprecatedTypeProperties are obsolete.
    text = re.sub(
        r'\.data\s*<\s*([^>]+?)\s*>\s*\(\s*\)',
        r'.data_ptr<\1>()',
        text,
    )
    text = text.replace('.type().is_cuda()', '.is_cuda()')

    # Modern PyTorch exposes current streams through c10::cuda.
    if 'getCurrentCUDAStream' in text:
        text = text.replace(
            'at::cuda::getCurrentCUDAStream()',
            'c10::cuda::getCurrentCUDAStream().stream()',
        )
        text = text.replace(
            'c10::cuda::getCurrentCUDAStream()',
            'c10::cuda::getCurrentCUDAStream().stream()',
        )
        text = text.replace(
            'c10::cuda::getCurrentCUDAStream().stream().stream()',
            'c10::cuda::getCurrentCUDAStream().stream()',
        )

        include = '#include <c10/cuda/CUDAStream.h>'
        if include not in text:
            lines = text.splitlines()
            insert_at = 0
            for i, line in enumerate(lines):
                if line.lstrip().startswith('#include'):
                    insert_at = i + 1
                elif insert_at:
                    break
            lines.insert(insert_at, include)
            text = '\n'.join(lines) + ('\n' if original.endswith('\n') else '')

    if text != original:
        path.write_text(text, encoding='utf-8')
        print('patched', path.relative_to(repo))

# Fail the Docker build early if any known old API survived.
joined = '\n'.join(
    candidate.read_text(encoding='utf-8')
    for candidate in src_dir.glob('*')
    if candidate.suffix in {'.cpp', '.cu', '.h', '.hpp'}
)
for forbidden in (
    '#include <THC/THC.h>',
    'extern THCState *state;',
    'at::cuda::getCurrentCUDAStream()',
    '.data<float>()',
    '.data<int>()',
):
    if forbidden in joined:
        raise RuntimeError(f'Unpatched legacy API remains: {forbidden}')
PY

RUN cd /opt/DifFlow3D/pointnet2 \
    && rm -rf build pointnet2_cuda*.so \
    && python3 setup.py build_ext --inplace \
    && python3 - <<'PY'
import torch
import pointnet2_cuda
from pointnet2 import pointnet2_utils

print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('pointnet2_cuda:', pointnet2_cuda.__file__)
print('PointNet++ import OK')
PY

WORKDIR /workspace

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
    && echo 'cd /workspace' >> /root/.bashrc

CMD ["/bin/bash"]
# RTX 5090 / Blackwell-compatible DifFlow3D inference image.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble
ARG PYTORCH_VERSION=2.8.0
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
    PYTHONPATH=/opt/DifFlow3D:/opt/DifFlow3D/difflow3d/ops/pointnet2:/opt/ros/humble/lib/python3.10/site-packages \
    LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/usr/local/cuda/lib64

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git gnupg2 locales lsb-release \
      ninja-build python3 python3-dev python3-pip python3-setuptools python3-wheel \
      libgl1 libglib2.0-0 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 is only needed by optional RViz/test publishers.
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
      ros-${ROS_DISTRO}-rviz2 \
      ros-${ROS_DISTRO}-rviz-default-plugins \
      xauth mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
      pip==25.1.1 setuptools==69.5.1 wheel==0.45.1 \
    && python3 -m pip install \
      torch==${PYTORCH_VERSION} \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install \
      numpy==1.26.4 scipy==1.13.1 PyYAML==6.0.2 packaging==24.2

COPY . /opt/DifFlow3D
RUN test -f /opt/DifFlow3D/checkpoints/model_difflow_355_0.0114.pth \
    && cd /opt/DifFlow3D \
    && bash scripts/build_pointnet2_ops.sh

WORKDIR /workspace

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
    && echo 'cd /workspace' >> /root/.bashrc

CMD ["/bin/bash"]

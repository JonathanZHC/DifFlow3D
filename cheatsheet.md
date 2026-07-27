# Build the docker:

docker build -t difflow3d-test .


# Run the docker:

docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  -e ROS_DOMAIN_ID=117 \
  -v "$PWD:/workspace" \
  difflow3d-test


# Run the script:

python3 test_difflow3d_superquadrics.py \
  --difflow-repo /workspace \
  --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
  --frames 300 \
  --sensor-hz 30 \
  --difflow-num-points 2048 \
  --difflow-iters 4 \
  --difflow-uncertainty 0.2 \
  --warmup 1 \
  --rviz 

Note: should choose difflow-iters = 2/4


# For Cuda profiler:

python3 test_difflow3d_superquadrics_profiled.py \
  --difflow-repo /workspace \
  --model-module model_difflow_profiled \
  --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
  --frames 100 \
  --sensor-hz 30 \
  --difflow-num-points 2048 \
  --difflow-iters 2 \
  --difflow-uncertainty 0.2 \
  --execution-backend cuda-graph \
  --cuda-graph-warmup 10 \
  --cuda-graph-no-fallback \
  --enable-tf32 \
  --warmup 1 \
  --rviz \
  --profile-cuda \
  --profile-only \
  --profile-output-dir ./profiles/difflow_2048_iters4 \
  --profile-warmup 10 \
  --profile-wait 1 \
  --profile-schedule-warmup 2 \
  --profile-active 5 \
  --profile-repeat 1 \
  --profile-row-limit 100

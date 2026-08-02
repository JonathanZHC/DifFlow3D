# Build the docker:

docker build -f Dockerfile -t difflow3d-isaacsim:6.0.1 .

# Run the docker:

docker run --rm -it \
  --name isaacscene-gui \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e OMNICLIENT_HUB_MODE=disabled \
  -e DISPLAY="$DISPLAY" \
  -e HOME=/isaac-sim \
  -e ROS_DOMAIN_ID=117 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw" \
  -v "$HOME/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$HOME/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "$HOME/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$HOME/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw" \
  -v "$HOME/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "$PWD/isaacscene:/workspace/isaacscene:rw" \
  -v "$PWD/camera_output:/workspace/camera_output:rw" \
  difflow3d-isaacsim:6.0.1 \
  /bin/bash

# Run isaacsim to generate the scene:

/isaac-sim/python.sh \
  /workspace/isaacscene/run_isaacsim.py \
    --scene dynamic \
    --width 640 \
    --height 480 \
    --camera-hz 30 \
    --pointcloud-hz 30 \
    --corrupt \
    --no-rgb-corruption

# Run Rviz for pcd visualization:

rviz2 -d /workspace/isaacscene/isaacscene.rviz

# Check RGB-D publication frequency

python3 \
  /workspace/isaacscene/python_image_rate_subscriber.py \
  --topic /camera_0/depth/image_raw \
  --reliability reliable \
  --expected-hz 30

# If cannot write to isaacscene.rviz:

mv isaacscene.rviz isaacscene.rviz.owner1003.backup
cp isaacscene.rviz.owner1003.backup isaacscene.rviz
chmod u+rw isaacscene.rviz





## The rest is for old pipeline

# Run the docker:

docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -e ROS_DOMAIN_ID=117 \
  -v "$PWD:/workspace" \
  difflow3d-test


# Run the script:

rviz2 -d test_scene_flow.rviz

python3 test_difflow3d_superquadrics.py \
  --difflow-repo /workspace \
  --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
  --frames 300 \
  --sensor-hz 30 \
  --difflow-num-points 4096 \
  --difflow-iters 4 \
  --difflow-uncertainty 0.2 \
  --cuda-graph-warmup 10 \
  --warmup 1 \
  --rviz 

Note: 
frames=300, sensor-hz=30, difflow-num-points=1024/2048, difflow-iters=4 
frames=300, sensor-hz=30, difflow-num-points=4096, difflow-iters=2/4
frames=100, sensor-hz=10, difflow-num-points=8192, difflow-iters=2 


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




With distance-based softmax:

python3 test_voxel_fps_difflow3d.py \
    --difflow-repo /workspace \
    --model-module model_difflow \
    --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
    --all-points 300000 \
    --voxel-resolution 0.020 \
    --enable-second-downsample \
    --second-voxel-resolution 0 \
    --second-candidate-ratio 2.5 \
    --fps-points 2048 \
    --difflow-iters 4 \
    --recovery-method softmax \
    --recovery-softmax-sigma 0.05 \
    --frames 300 \
    --sensor-hz 30 \
    --warmup 2 \
    --rviz 

Or with inverse-distance weighted sum:

python3 test_voxel_fps_difflow3d.py \
    --difflow-repo /workspace \
    --model-module model_difflow \
    --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
    --all-points 300000 \
    --voxel-resolution 0.010 \
    --enable-second-downsample \
    --second-voxel-resolution 0 \
    --second-candidate-ratio 2.5 \
    --fps-points 2048 \
    --difflow-iters 4 \
    --recovery-method inverse-distance \
    --recovery-idw-power 2.0 \
    --recovery-idw-epsilon 1e-5 \
    --recovery-chunk-size 4096 \
    --frames 300 \
    --sensor-hz 30 \
    --warmup 2 \
    --rviz 

For visualization:

rviz2 -d voxel_fps_difflow3d.rviz
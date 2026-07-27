docker build -t difflow3d-test .


docker run --rm -it \
  --gpus all \
  --network host \
  --ipc=host \
  -e ROS_DOMAIN_ID=117 \
  -v "$PWD:/workspace" \
  difflow3d-test


python3 test_difflow3d_superquadrics.py \
  --difflow-repo /opt/DifFlow3D \
  --checkpoint /opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth \
  --frames 100 \
  --sensor-hz 10 \
  --difflow-num-points 2048 \
  --difflow-iters 2 \
  --difflow-uncertainty 0.2 \
  --warmup 1 \
  --rviz 


should choose difflow-iters = 2/4
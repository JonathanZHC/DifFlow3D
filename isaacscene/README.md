# isaacscene

四个主模块：

- `scene_settings.py`：桌面和日常物体。
- `camera_settings.py`：两台相机、标定、RGB-D、点云和可选 corruption。
- `ros_camera_publisher.py`：ROS 2 Image、CameraInfo、PointCloud2、Pose、TF。
- `run_isaacsim.py`：启动 Isaac Sim 并组合前三个模块。

默认话题：

- `/camera_0/color/image_raw`
- `/camera_0/depth/image_raw`
- `/camera_0/camera_info`
- `/camera_0/points`
- `/camera_0/pose`
- `/camera_1/color/image_raw`
- `/camera_1/depth/image_raw`
- `/camera_1/camera_info`
- `/camera_1/points`
- `/camera_1/pose`
- `/cameras/fused_points`

静态 TF：

- `world -> camera_0_optical_frame`
- `world -> camera_1_optical_frame`


## RViz 可视化

主点云 `/cameras/fused_points` 自动表示当前主数据：
启用 `--corrupt` 时为损坏数据，否则为干净数据。

三维相机可视化：
- `/cameras/visualization`
- 相机机身、坐标轴、光轴、视锥体；
- 视锥底面使用当前 RGB 图像作为动态纹理。

启动：
```bash
rviz2 -d /workspace/isaacscene/isaacscene.rviz
```

推荐为避免增加仿真负载，将纹理平面设为 2 Hz：
```bash
--visualization-hz 2
```


## Isaac Sim 内部可视化

GUI 模式默认显示：

- `/World/Visualizations/PrimaryFusedPointCloud`
- `/World/Visualizations/Cameras/camera_0`
- `/World/Visualizations/Cameras/camera_1`
- 两个当前主 RGB 流的 Isaac UI 图像窗口

主数据选择：

- 使用 `--corrupt`：显示损坏点云与损坏 RGB；
- 不使用 `--corrupt`：显示干净点云与干净 RGB。

推荐参数：

```bash
--isaac-visualization-hz 5 \
--isaac-max-points 40000 \
--isaac-point-size 0.008 \
--isaac-frustum-depth 0.45
```

关闭全部 Isaac 内部可视化：

```bash
--no-isaac-visualization
```

只关闭相机图像窗口、保留三维点云和视锥：

```bash
--no-isaac-camera-windows
```

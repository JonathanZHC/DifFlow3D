# isaacscene: static and dynamic scene modes

The scene code now has exactly two mutually exclusive configurations selected
by one command-line flag.

## Scene selection

### Original static scene

```bash
/isaac-sim/python.sh /workspace/isaacscene/run_isaacsim.py \
  --scene static \
  --width 640 \
  --height 480 \
  --rgbd-hz 30 \
  --pointcloud-hz 2
```

This restores the original stationary tabletop objects:

- cereal box;
- food can;
- ordinary bottle;
- mug;
- apple;
- bowl.

`static` is the default, so omitting `--scene` gives the same result.

### Dynamic scene

```bash
/isaac-sim/python.sh /workspace/isaacscene/run_isaacsim.py \
  --scene dynamic \
  --width 640 \
  --height 480 \
  --rgbd-hz 30 \
  --pointcloud-hz 2 \
  --motion-speed-scale 1.0
```

The dynamic tabletop contains:

- a tall moving shelf/cart;
- a compound moving bottle;
- a floating drone-like object.

The original stationary tabletop objects are not created in dynamic mode, so
there is no mixed or duplicated scene.

## Code ownership

- `scene_settings.py`: scene-mode flag definitions, common ground/table/lights,
  and the original static objects.
- `moving_objects.py`: all dynamic geometry and motion trajectories.
- `run_isaacsim.py`: selects one mode with `--scene static|dynamic` and updates
  the motion controller only in dynamic mode.

The stale duplicate `dynamic_objects.py` has been removed.

The camera, corruption, ROS publication and GPU point-cloud code are unchanged.

# PyRoboVision Roadmap

**Current Version:** v1.1.0

## Vision

PyRoboVision provides autonomous driving perception and foundation model inference with multi-sensor fusion and real-time processing.

## Completed Milestones

✅ **v1.0** — Foundation & Perception
- Object detection (YOLO, R-CNN)
- Semantic segmentation
- Bird's eye view (BEV) projection
- Multi-sensor fusion (camera, Lidar, radar)
- Foundation model inference (SAM2, CLIP, Grounding DINO)

✅ **v1.1 (July 2026)** — Workflow Integration
- CLI: `pyrobovision process-video`, `detect`, `segment`, `bev`, `fuse`
- REST API (Port 8010) for automation
- Temporal, Airflow integration for AV pipelines
- Multi-frame processing API

## In Progress

⏳ **v1.2 (Aug 2026)** — Tracking & Prediction
- Multi-object tracking (MOT)
- Trajectory prediction
- Behavioral analysis
- Intent prediction

## Planned

📅 **v1.5 (Sep 2026)** — 3D Perception
- 3D object detection
- Depth estimation from monocular vision
- Occupancy grid generation
- LiDAR point cloud processing

📅 **v2.0 (Oct 2026)** — End-to-End Learning
- Imitation learning from demonstrations
- Reinforcement learning integration
- Behavior cloning models
- Safe learning guardrails

📅 **v2.5 (Q4 2026)** — Autonomous Driving
- Full autonomous driving stack
- Path planning integration
- Safety verification
- OTA update support

📅 **v3.0 (Q1 2027)** — Enterprise Deployment
- Edge deployment optimization
- Redundancy & failover
- Compliance & certification
- Fleet management API

## Integration Points

- **Perception:** YOLOv8, SAM2, CLIP, Grounding DINO
- **Sensors:** Camera, LiDAR, Radar, IMU
- **Frameworks:** PyTorch, TensorRT, ONNX
- **Platforms:** ROS2, Autoware, Apollo
- **Workflow:** Temporal, Airflow, Kubernetes

## Priority Features

1. **Tracking & Prediction** (Q3 2026) — Multi-object analysis
2. **3D Perception** (Q3 2026) — Depth & point clouds
3. **End-to-End Learning** (Q4 2026) — Neural networks
4. **Autonomous Driving** (Q4 2026) — Full autonomy stack

## Known Limitations

- GPU inference required (no CPU support yet)
- Frame rate limited by inference speed
- Multi-camera calibration required
- Weather/lighting variations affect accuracy

## Community

Contribute:
https://github.com/Mullassery/PyRoboVision/issues

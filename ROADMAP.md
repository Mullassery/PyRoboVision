# PyRoboVision Roadmap

**Current Version:** v1.2.0

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

✅ **v1.2 (Aug 2026)** — Tracking & Prediction
- Multi-object tracking (MOT) — Kalman filter + Hungarian algorithm
- Trajectory prediction — CV/CA models with ADE/FDE metrics
- Behavioral analysis — 8-class motion + 8-class behavior classification
- Intent prediction — 11-class intent with collision detection
- Uncertainty quantification — Covariance + confidence ellipses
- **124 tests, 70% coverage, <50ms latency per frame**

✅ **v1.5 (Sep 2026)** — 3D Perception
- Monocular depth estimation — Edge detection + median filtering
- 3D bounding boxes — Depth-to-3D conversion + point cloud-based
- Occupancy grid — BEV representation from points/bboxes
- LiDAR processing — Filtering, clustering, normal estimation
- Depth-LiDAR fusion for multi-sensor accuracy
- **64 tests, 88% coverage, real-time performance**

✅ **v2.0 (Oct 2026)** — End-to-End Learning & Optimization
- Imitation learning framework — Trajectory collection & augmentation
- Behavior cloning models — Supervised learning from demos
- Policy networks — Actor-critic with value head
- Safety validation — Constraint-based action correction
- Training infrastructure — Early stopping, logging, checkpoints
- Sensor fusion — IMU/GPS Kalman filter with WGS84→ENU transformation
- Model optimization — ONNX/TensorRT export, int8/int16/float16 quantization, inference profiling
- **79 tests, 82% coverage, <2ms latency per inference, 4-10x GPU speedup**

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

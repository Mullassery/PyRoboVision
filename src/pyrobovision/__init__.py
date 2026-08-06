"""PyRoboVision: Autonomous driving perception and foundation models.

A pure Python library for autonomous driving perception algorithms and foundation model
inference, built on PyRoboFrames for efficient multi-camera video loading and processing.

Modules:
    perception: 3D object detection, depth estimation, LiDAR processing
    tracking: Multi-object tracking with Kalman filtering
    prediction: Trajectory prediction and uncertainty estimation
    learning: Imitation learning, behavior cloning, safety-constrained policies
    fusion: Sensor fusion (IMU/GPS), model optimization
    behavior: Behavior analysis and pattern recognition
    intent: Intent prediction from motion

Requirements:
    - PyRoboFrames >= 1.1.0 for data loading
    - CUDA-enabled PyTorch OR MLX on Apple Silicon OR CPU

Performance:
    - Autonomous driving: 30 FPS @ 1080p on H100 GPU
    - Perception: 60 FPS @ 720p 3D detection on RTX 4090
    - Apple Silicon: Native MLX acceleration (ANE + GPU)
"""

from typing import Final

__version__: Final[str] = "2.1.0"
__author__: Final[str] = "Georgi Mammen Mullassery"
__email__: Final[str] = "mullassery@gmail.com"
__license__: Final[str] = "Proprietary"

# Perception: 3D detection, depth, occupancy
from .perception import (
    DepthEstimator,
    DepthMap,
    BBox3D,
    Box3DConverter,
    OccupancyGrid,
    OccupancyGridBuilder,
    LiDARProcessor,
    PointCloud,
)

# Tracking: Multi-object tracking
from .tracking import (
    MOTTracker,
    Track,
    ObjectAssociation,
    KalmanFilter,
)

# Prediction: Trajectory and intent prediction
from .prediction import (
    ConstantVelocityModel,
    ConstantAccelerationModel,
    TrajectoryPredictor,
    UncertaintyEstimator,
)

from .intent import (
    IntentPredictor,
    Intent,
)

# Learning: Imitation learning, behavior cloning, safety
from .learning import (
    ImitationLearner,
    DemonstrationDataset,
    BehaviorCloningModel,
    BCLoss,
    PolicyNetwork,
    PolicyOptimizer,
    SafetyValidator,
    ConstraintChecker,
    TrainingConfig,
    Trainer,
)

# Fusion: Sensor fusion and optimization
from .fusion import (
    SensorFusionEngine,
    IMUData,
    GPSData,
    FusionState,
    ModelOptimizer,
    QuantizationConfig,
)

# Behavior: Analysis and patterns
from .behavior import (
    BehaviorAnalyzer,
    BehaviorPattern,
    MotionPattern,
)

# MCP 2.0 Support (v2.1.0+) — Autonomous driving perception tools
from ._mcp_connector import PerceptionEngine

__all__: Final[list[str]] = [
    # Perception
    "DepthEstimator",
    "DepthMap",
    "BBox3D",
    "Box3DConverter",
    "OccupancyGrid",
    "OccupancyGridBuilder",
    "LiDARProcessor",
    "PointCloud",
    # Tracking
    "MOTTracker",
    "Track",
    "ObjectAssociation",
    "KalmanFilter",
    # Prediction
    "ConstantVelocityModel",
    "ConstantAccelerationModel",
    "TrajectoryPredictor",
    "UncertaintyEstimator",
    # Intent
    "IntentPredictor",
    "Intent",
    # Learning
    "ImitationLearner",
    "DemonstrationDataset",
    "BehaviorCloningModel",
    "BCLoss",
    "PolicyNetwork",
    "PolicyOptimizer",
    "SafetyValidator",
    "ConstraintChecker",
    "TrainingConfig",
    "Trainer",
    # Fusion
    "SensorFusionEngine",
    "IMUData",
    "GPSData",
    "FusionState",
    "ModelOptimizer",
    "QuantizationConfig",
    # Behavior
    "BehaviorAnalyzer",
    "BehaviorPattern",
    "MotionPattern",
    # MCP
    "PerceptionEngine",
]

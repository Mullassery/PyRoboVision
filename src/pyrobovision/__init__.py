"""PyRoboVision: Multi-object tracking and trajectory prediction toolkit.

A pure Python library (NumPy + SciPy only) built around a real, tested
multi-object tracking core: Kalman filter state estimation + Hungarian
algorithm association, plus constant-velocity/acceleration trajectory
prediction with uncertainty quantification. It also includes supporting
utilities for 3D perception (depth, LiDAR, occupancy grids), behavior/intent
classification, and simple imitation-learning/safety-constraint building
blocks.

What's real and tested here:
    tracking:   Multi-object tracking (Kalman filter + Hungarian association)
    prediction: Trajectory forecasting (CV/CA models) + uncertainty
    perception: Depth estimation (real MiDaS via optional torch dependency,
                or a clearly-labeled non-neural placeholder heuristic),
                3D bbox conversion, LiDAR point-cloud utilities, occupancy grids
    behavior / intent: Rule-based motion classification and intent heuristics
    learning / fusion: Imitation learning, behavior cloning, safety
                constraint validation, and IMU/GPS sensor fusion utilities

What this is NOT (be aware before relying on it for those things):
    - There is no object detector included (no YOLO/SAM/etc). You bring your
      own detections (bounding boxes) into the tracker.
    - No GPU-accelerated inference pipeline, no foundation-model (CLIP/SAM/
      Grounding DINO) integration.
    - Not a certified or safety-verified autonomous driving stack.

Only NumPy and SciPy are required to import this package. PyTorch is an
optional extra (`pip install "pyrobovision[depth]"`) needed only for real
MiDaS depth inference and ONNX/TensorRT model export utilities.
"""

from typing import Final

__version__: Final[str] = "3.0.0"
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
]

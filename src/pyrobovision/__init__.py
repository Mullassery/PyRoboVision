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

__version__: Final[str] = "3.3.1"
__author__: Final[str] = "Georgi Mammen Mullassery"
__email__: Final[str] = "mullassery@gmail.com"
__license__: Final[str] = "Apache-2.0"

# Perception: 3D detection, depth, occupancy
# Behavior: Analysis and patterns
from .behavior import (
    BehaviorAnalyzer,
    BehaviorPattern,
    MotionPattern,
)

# Fusion: Sensor fusion and optimization
from .fusion import (
    FusionState,
    GPSData,
    IMUData,
    ModelOptimizer,
    QuantizationConfig,
    SensorFusionEngine,
)

# Optional integrations: Supervision detections, Alpamayo reasoning-VLA outputs
from .integrations.alpamayo_adapter import (
    AlpamayoOutput,
    AlpamayoShapeError,
    AlpamayoTrajectory,
    AlpamayoTrajectoryAdapter,
    trajectory_errors,
)

# Tracking: Multi-object tracking
from .integrations.supervision_adapter import (
    SupervisionTrackerAdapter,
    SupervisionUnavailable,
    detections_from_supervision,
    tracks_to_supervision,
)
from .intent import (
    Intent,
    IntentPredictor,
)

# Learning: Imitation learning, behavior cloning, safety
from .learning import (
    BCLoss,
    BehaviorCloningModel,
    ConstraintChecker,
    DemonstrationDataset,
    ImitationLearner,
    PolicyNetwork,
    PolicyOptimizer,
    SafetyValidator,
    Trainer,
    TrainingConfig,
)
from .perception import (
    AdaptiveFrameConfig,
    AdaptiveFrameController,
    BBox3D,
    Box3DConverter,
    DepthEstimator,
    DepthMap,
    FrameDecision,
    LiDARProcessor,
    OccupancyGrid,
    OccupancyGridBuilder,
    PointCloud,
)

# Prediction: Trajectory and intent prediction
from .prediction import (
    ConstantAccelerationModel,
    ConstantVelocityModel,
    TrajectoryPredictor,
    UncertaintyEstimator,
)
from .tracking import (
    ByteTrackTracker,
    KalmanFilter,
    MOTTracker,
    ObjectAssociation,
    Track,
)

__all__: Final[list[str]] = [
    # Perception
    "AdaptiveFrameConfig",
    "AdaptiveFrameController",
    # Optional Alpamayo integration
    "AlpamayoOutput",
    "AlpamayoShapeError",
    "AlpamayoTrajectory",
    "AlpamayoTrajectoryAdapter",
    "BBox3D",
    "BCLoss",
    # Behavior
    "BehaviorAnalyzer",
    "BehaviorCloningModel",
    "BehaviorPattern",
    "Box3DConverter",
    "ByteTrackTracker",
    "ConstantAccelerationModel",
    # Prediction
    "ConstantVelocityModel",
    "ConstraintChecker",
    "DemonstrationDataset",
    "DepthEstimator",
    "DepthMap",
    "FrameDecision",
    "FusionState",
    "GPSData",
    "IMUData",
    # Learning
    "ImitationLearner",
    "Intent",
    # Intent
    "IntentPredictor",
    "KalmanFilter",
    "LiDARProcessor",
    # Tracking
    "MOTTracker",
    "ModelOptimizer",
    "MotionPattern",
    "ObjectAssociation",
    "OccupancyGrid",
    "OccupancyGridBuilder",
    "PointCloud",
    "PolicyNetwork",
    "PolicyOptimizer",
    "QuantizationConfig",
    "SafetyValidator",
    # Fusion
    "SensorFusionEngine",
    "SupervisionTrackerAdapter",
    "SupervisionUnavailable",
    "Track",
    "Trainer",
    "TrainingConfig",
    "TrajectoryPredictor",
    "UncertaintyEstimator",
    "detections_from_supervision",
    "tracks_to_supervision",
    "trajectory_errors",
]

"""Tests for multimodal sensor fusion (RGB + depth + IMU)."""

import numpy as np
import pyarrow as pa
import pytest

import pyroboframes as prf
from pyroboframes.dataframe import RoboticsDataFrame, TopicFrame
from pyroboframes.sensor_fusion import (
    MultimodalBatch,
    MultimodalDataFrame,
    SensorFusionConfig,
    create_humanoid_config,
)


def create_test_robotics_frame() -> RoboticsDataFrame:
    """Create a test RoboticsDataFrame with RGB, depth, and IMU data."""
    n_samples = 100
    n_points = 1000

    # Create RGB camera topic
    rgb_data = {
        "log_time": np.arange(n_samples) * 33_333_333,  # ~30 Hz (ns)
        "frame_id": np.zeros(n_samples),
        "height": np.full(n_samples, 480),
        "width": np.full(n_samples, 640),
    }
    rgb_table = pa.Table.from_pydict(rgb_data)
    rgb_frame = TopicFrame("/camera/rgb", rgb_table)

    # Create depth topic (point clouds)
    depth_times = np.arange(n_samples) * 33_333_333 + 16_666_666  # ~15 Hz, 50ms lag
    # Simulate point cloud as [n_samples, n_points, 3]
    depth_points = np.random.randn(n_samples, n_points, 3).astype(np.float32)
    depth_data = {
        "log_time": depth_times,
        "points_x": depth_points[:, :, 0],
        "points_y": depth_points[:, :, 1],
        "points_z": depth_points[:, :, 2],
    }
    # Flatten for Parquet
    depth_data_flat = {
        "log_time": depth_times,
        "num_points": np.full(n_samples, n_points),
    }
    depth_table = pa.Table.from_pydict(depth_data_flat)
    depth_frame = TopicFrame("/depth/camera", depth_table)

    # Create IMU topic
    imu_times = np.arange(n_samples * 10) * 3_333_333  # ~300 Hz
    n_imu = len(imu_times)
    imu_data = {
        "log_time": imu_times,
        "accel_x": np.random.randn(n_imu) * 0.1,
        "accel_y": np.random.randn(n_imu) * 0.1,
        "accel_z": np.random.randn(n_imu) * 0.1 + 9.81,  # gravity
        "gyro_x": np.random.randn(n_imu) * 0.01,
        "gyro_y": np.random.randn(n_imu) * 0.01,
        "gyro_z": np.random.randn(n_imu) * 0.01,
    }
    imu_table = pa.Table.from_pydict(imu_data)
    imu_frame = TopicFrame("/imu/data", imu_table)

    frames = {
        "/camera/rgb": rgb_frame,
        "/depth/camera": depth_frame,
        "/imu/data": imu_frame,
    }

    return RoboticsDataFrame(frames)


def test_sensor_fusion_config_creation():
    """Test creating sensor fusion configuration."""
    config = SensorFusionConfig(
        reference_topic="/camera/rgb",
        camera_topics=["/camera/rgb"],
        depth_topics=["/depth/camera"],
        imu_topics=["/imu/data"],
    )
    assert config.reference_topic == "/camera/rgb"
    assert len(config.camera_topics) == 1
    assert len(config.depth_topics) == 1
    assert len(config.imu_topics) == 1


def test_humanoid_config():
    """Test humanoid robot configuration."""
    config = create_humanoid_config()
    assert config.reference_topic == "/camera/front/rgb"
    assert len(config.camera_topics) == 3  # head, front, wrist
    assert len(config.depth_topics) == 1  # wrist
    assert len(config.imu_topics) == 1  # shoulder


def test_auto_detect_topics():
    """Test auto-detection of camera/depth/IMU topics."""
    df = create_test_robotics_frame()
    config = SensorFusionConfig()
    config.auto_detect(df)

    assert any("camera" in t or "image" in t.lower() for t in config.camera_topics)
    assert any("depth" in t or "point_cloud" in t for t in config.depth_topics)
    assert any("imu" in t for t in config.imu_topics)


def test_multimodal_batch_creation():
    """Test creating a multimodal batch."""
    data = {
        "log_time": np.arange(10),
        "camera.rgb.frame": np.random.rand(10, 480, 640, 3),
        "depth.camera.points": np.random.rand(10, 1000, 3),
        "imu.data.accel_x": np.random.randn(10),
    }
    batch = MultimodalBatch(data)

    assert len(batch) == 10
    assert len(batch.cameras) > 0
    assert len(batch.depth_streams) > 0
    assert len(batch.imu_streams) > 0


def test_multimodal_dataframe_creation():
    """Test creating a multimodal dataframe."""
    df = create_test_robotics_frame()
    mdf = MultimodalDataFrame(df)

    assert mdf.df is df
    assert mdf.config is not None
    assert len(mdf.config.camera_topics) > 0 or len(mdf.config.depth_topics) > 0


def test_align_multimodal():
    """Test time-aligning multimodal sensors."""
    df = create_test_robotics_frame()
    mdf = MultimodalDataFrame(df)
    batch = mdf.align_multimodal(reference_topic="/camera/rgb")

    assert isinstance(batch, MultimodalBatch)
    assert "log_time" in batch._data
    assert len(batch) > 0


def test_multimodal_batch_access():
    """Test accessing data in multimodal batch."""
    data = {
        "log_time": np.arange(10),
        "camera.rgb.data": np.ones((10, 3)),
        "imu.accel.x": np.ones(10),
    }
    batch = MultimodalBatch(data)

    assert "camera.rgb.data" in batch
    assert "log_time" in batch
    assert batch["camera.rgb.data"].shape == (10, 3)


def test_project_depth_to_image():
    """Test projecting depth to image plane."""
    df = create_test_robotics_frame()
    mdf = MultimodalDataFrame(df)
    batch = mdf.align_multimodal(reference_topic="/camera/rgb")

    # Create mock calibration
    calib = prf.CameraCalibration(
        name="rgb",
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )

    calibrations = {"camera.rgb": calib}
    enhanced = mdf.project_depth_to_image(batch, calibrations)

    assert isinstance(enhanced, MultimodalBatch)


def test_fuse_imu_with_vision():
    """Test IMU fusion with vision for motion compensation."""
    df = create_test_robotics_frame()
    mdf = MultimodalDataFrame(df)
    batch = mdf.align_multimodal(reference_topic="/camera/rgb")

    enhanced = mdf.fuse_imu_with_vision(batch)
    assert isinstance(enhanced, MultimodalBatch)

    # Should add motion confidence and rotation magnitude
    # (if IMU data was aligned)


def test_stack_batch_for_training():
    """Test stacking multimodal batch for training."""
    data = {
        "log_time": np.arange(10),
        "camera.rgb.frame": np.random.rand(10, 480, 640, 3),
        "imu.accel_x": np.random.randn(10),
    }
    batch = MultimodalBatch(data)
    mdf = MultimodalDataFrame(create_test_robotics_frame())

    output = mdf.stack_batch_for_training(batch, stack_frames=1)
    assert isinstance(output, dict)
    assert "log_time" in output


def test_multimodal_repr():
    """Test string representations."""
    batch = MultimodalBatch({"log_time": np.arange(10)})
    assert "MultimodalBatch" in repr(batch)

    df = create_test_robotics_frame()
    mdf = MultimodalDataFrame(df)
    assert "MultimodalDataFrame" in repr(mdf)


def test_tolerance_filtering():
    """Test timestamp tolerance filtering in alignment."""
    df = create_test_robotics_frame()
    config = SensorFusionConfig(
        reference_topic="/camera/rgb",
        tolerance_ns=10_000_000,  # 10ms tight tolerance
    )
    mdf = MultimodalDataFrame(df, config)

    batch = mdf.align_multimodal()
    # With tight tolerance, some samples should be NaN where other sensors lag too much
    assert isinstance(batch, MultimodalBatch)


def test_missing_topic_handling():
    """Test handling of missing topics."""
    df = create_test_robotics_frame()
    config = SensorFusionConfig(
        reference_topic="/camera/rgb",
        camera_topics=["/camera/nonexistent"],  # Doesn't exist
    )
    mdf = MultimodalDataFrame(df, config)

    batch = mdf.align_multimodal()
    assert isinstance(batch, MultimodalBatch)  # Should not crash


def test_invalid_reference_topic():
    """Test error handling for invalid reference topic."""
    df = create_test_robotics_frame()
    mdf = MultimodalDataFrame(df)

    with pytest.raises(KeyError):
        mdf.align_multimodal(reference_topic="/nonexistent/topic")


def test_multimodal_batch_dict_interface():
    """Test dict-like interface of MultimodalBatch."""
    data = {
        "log_time": np.arange(5),
        "camera.rgb": np.ones((5, 3)),
        "imu.accel": np.ones((5, 3)),
    }
    batch = MultimodalBatch(data)

    # Test __getitem__
    assert batch["log_time"].shape == (5,)

    # Test __contains__
    assert "camera.rgb" in batch
    assert "nonexistent" not in batch

    # Test keys()
    assert "log_time" in batch.keys()

    # Test len()
    assert len(batch) == 5

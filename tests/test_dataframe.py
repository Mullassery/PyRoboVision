"""Robotics DataFrame (P2): load converted output, slice by time, and time-align sensors."""

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyroboframes as prf


def _make_converted(path: str) -> None:
    """Write a converted-style directory: two topics + a metadata.json manifest."""
    os.makedirs(path, exist_ok=True)
    pq.write_table(
        pa.table({"log_time": [10, 20, 30], "x": [1.0, 2.0, 3.0]}),
        os.path.join(path, "state.parquet"),
    )
    pq.write_table(
        pa.table({"log_time": [12, 25], "a": [100.0, 200.0]}),
        os.path.join(path, "imu.parquet"),
    )
    metadata = {
        "format": "pyroboframes-columnar",
        "version": 1,
        "topics": [
            {"topic": "/state", "path": "state.parquet", "columns": {"x": "float64"}},
            {"topic": "/imu", "path": "imu.parquet", "columns": {"a": "float64"}},
        ],
    }
    with open(os.path.join(path, "metadata.json"), "w") as fh:
        json.dump(metadata, fh)


def test_load_and_per_topic_access(tmp_path):
    _make_converted(str(tmp_path))
    df = prf.RoboticsDataFrame.from_converted(str(tmp_path))

    assert set(df.topics) == {"/state", "/imu"}
    assert len(df) == 2
    assert "/state" in df

    state = df["/state"]
    assert len(state) == 3
    assert state.columns == ["x"]
    np.testing.assert_array_equal(state.log_time, [10, 20, 30])
    np.testing.assert_array_equal(state.column("x"), [1.0, 2.0, 3.0])

    assert df.time_range() == (10, 30)


def test_slice_filters_every_topic(tmp_path):
    _make_converted(str(tmp_path))
    df = prf.RoboticsDataFrame.from_converted(str(tmp_path))

    sub = df.slice(15, 30)  # state: log_time 20; imu: log_time 25
    np.testing.assert_array_equal(sub["/state"].log_time, [20])
    np.testing.assert_array_equal(sub["/imu"].log_time, [25])
    # Original frame is unchanged.
    assert len(df["/state"]) == 3


def test_align_as_of_join(tmp_path):
    _make_converted(str(tmp_path))
    df = prf.RoboticsDataFrame.from_converted(str(tmp_path))

    aligned = df.align("/state")  # reference timestamps 10, 20, 30
    np.testing.assert_array_equal(aligned.log_time, [10, 20, 30])
    np.testing.assert_array_equal(aligned["x"], [1.0, 2.0, 3.0])

    # imu backward as-of: t=10 -> nothing before 12 (NaN); t=20 -> 12 (100); t=30 -> 25 (200).
    imu = aligned["imu.a"]
    assert np.isnan(imu[0])
    np.testing.assert_array_equal(imu[1:], [100.0, 200.0])
    assert "imu.a" in aligned.columns


def test_align_tolerance_drops_stale_matches(tmp_path):
    _make_converted(str(tmp_path))
    df = prf.RoboticsDataFrame.from_converted(str(tmp_path))

    # imu matches: t=10 -> none; t=20 -> imu t=12 (dt=8); t=30 -> imu t=25 (dt=5).
    # tolerance=6 keeps only the dt=5 match.
    aligned = df.align("/state", tolerance=6)
    imu = aligned["imu.a"]
    assert np.isnan(imu[0])  # t=10, none before
    assert np.isnan(imu[1])  # t=20, dt=8 > 6 -> dropped
    assert imu[2] == 200.0  # t=30, dt=5 <= 6 -> kept


def test_resample_previous_and_linear(tmp_path):
    _make_converted(str(tmp_path))  # /state t[10,20,30] x[1,2,3]; /imu t[12,25] a[100,200]
    df = prf.RoboticsDataFrame.from_converted(str(tmp_path))

    prev = df.resample(period=10, start=10, end=30, method="previous")  # grid 10,20,30
    np.testing.assert_array_equal(prev.log_time, [10, 20, 30])
    np.testing.assert_array_equal(prev["state.x"], [1.0, 2.0, 3.0])
    imu = prev["imu.a"]
    assert np.isnan(imu[0])  # no imu sample at/before t=10
    np.testing.assert_array_equal(imu[1:], [100.0, 200.0])

    lin = df.resample(period=5, start=10, end=30, method="linear")  # grid 10,15,20,25,30
    np.testing.assert_allclose(lin["state.x"], [1.0, 1.5, 2.0, 2.5, 3.0])


def test_resample_nearest(tmp_path):
    _make_converted(str(tmp_path))
    df = prf.RoboticsDataFrame.from_converted(str(tmp_path))
    # imu samples at t=12 (100) and t=25 (200); nearest to grid 10,20,30.
    near = df.resample(period=10, start=10, end=30, method="nearest")["imu.a"]
    # 10->12 (d2); 20->25 (d5 < d8 to 12); 30->25 (d5).
    np.testing.assert_array_equal(near, [100.0, 200.0, 200.0])


def test_from_mcap_end_to_end(tmp_path):
    pytest.importorskip("mcap.writer")
    from test_mcap import _write_mcap  # reuse the MCAP fixture writer

    mcap_path = tmp_path / "run.mcap"
    _write_mcap(str(mcap_path))
    df = prf.RoboticsDataFrame.from_mcap(str(mcap_path), out_dir=str(tmp_path / "out"))

    assert df.topics == ["/state"]  # the protobuf /raw topic is skipped
    state = df["/state"]
    assert len(state) == 2
    assert "observation.state.0" in state.columns

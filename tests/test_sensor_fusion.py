import pytest
import numpy as np
from pyrobovision.fusion.sensor_fusion import SensorFusionEngine, IMUData, GPSData


class TestIMUData:
    def test_initialization(self):
        acc = np.array([0.1, 0.2, 9.8])
        gyro = np.array([0.01, 0.02, 0.03])

        imu = IMUData(timestamp=0.0, accelerometer=acc, gyroscope=gyro)

        assert imu.timestamp == 0.0
        assert np.allclose(imu.accelerometer, acc)
        assert np.allclose(imu.gyroscope, gyro)


class TestGPSData:
    def test_initialization(self):
        gps = GPSData(
            timestamp=0.0,
            latitude=37.7749,
            longitude=-122.4194,
            altitude=10.0,
            speed=5.0,
            heading=90.0,
            accuracy=2.0,
        )

        assert gps.latitude == 37.7749
        assert gps.accuracy == 2.0


class TestSensorFusionEngine:
    def test_initialization(self):
        engine = SensorFusionEngine(origin_lat=37.7749, origin_lon=-122.4194)

        assert engine.origin_lat == 37.7749
        assert np.allclose(engine.state.position, np.zeros(3))
        assert np.allclose(engine.state.velocity, np.zeros(3))

    def test_gps_to_local(self):
        engine = SensorFusionEngine(origin_lat=0.0, origin_lon=0.0, origin_alt=0.0)

        local_pos = engine._gps_to_local(0.01, 0.01, 10.0)

        assert local_pos[2] == 10.0
        assert abs(local_pos[0]) < 2000
        assert abs(local_pos[1]) < 2000

    def test_update_imu(self):
        engine = SensorFusionEngine()

        imu = IMUData(
            timestamp=0.0,
            accelerometer=np.array([0.0, 0.0, 9.8]),
            gyroscope=np.array([0.0, 0.0, 0.0]),
        )

        state = engine.update_imu(imu)

        assert state is not None
        assert state.position.shape == (3,)

    def test_update_imu_sequence(self):
        engine = SensorFusionEngine()

        for i in range(5):
            imu = IMUData(
                timestamp=float(i) * 0.1,
                accelerometer=np.array([1.0, 0.0, 9.8]),
                gyroscope=np.array([0.0, 0.0, 0.0]),
            )
            state = engine.update_imu(imu)

        assert len(engine.fusion_history) == 5
        assert state.velocity[0] > 0

    def test_update_gps(self):
        engine = SensorFusionEngine(origin_lat=0.0, origin_lon=0.0, origin_alt=0.0)

        gps = GPSData(
            timestamp=0.0,
            latitude=0.0,
            longitude=0.0,
            altitude=10.0,
            speed=5.0,
            heading=90.0,
            accuracy=2.0,
        )

        state = engine.update_gps(gps)

        assert state is not None
        assert state.position[2] > 0

    def test_gps_speed_update(self):
        engine = SensorFusionEngine()

        gps = GPSData(
            timestamp=0.0,
            latitude=0.0,
            longitude=0.0,
            altitude=0.0,
            speed=10.0,
            heading=0.0,
            accuracy=1.0,
        )

        state = engine.update_gps(gps)

        assert np.linalg.norm(state.velocity[:2]) > 0

    def test_get_state_vector(self):
        engine = SensorFusionEngine()

        imu = IMUData(
            timestamp=0.0,
            accelerometer=np.array([0.0, 0.0, 9.8]),
            gyroscope=np.array([0.0, 0.0, 0.0]),
        )
        engine.update_imu(imu)

        state_vector = engine.get_state_vector()

        assert state_vector.shape == (9,)

    def test_get_covariance(self):
        engine = SensorFusionEngine()

        cov = engine.get_covariance()

        assert cov.shape == (9, 9)
        assert np.all(np.isfinite(cov))

    def test_reset(self):
        engine = SensorFusionEngine()

        imu = IMUData(
            timestamp=0.0,
            accelerometer=np.array([0.0, 0.0, 9.8]),
            gyroscope=np.array([0.0, 0.0, 0.0]),
        )
        engine.update_imu(imu)

        assert len(engine.fusion_history) > 0

        engine.reset()

        assert len(engine.fusion_history) == 0
        assert np.allclose(engine.state.position, np.zeros(3))

    def test_get_statistics(self):
        engine = SensorFusionEngine()

        imu = IMUData(
            timestamp=0.0,
            accelerometer=np.array([0.0, 0.0, 9.8]),
            gyroscope=np.array([0.0, 0.0, 0.0]),
        )
        engine.update_imu(imu)

        stats = engine.get_statistics()

        assert stats["imu_updates"] == 1
        assert "position" in stats
        assert "velocity" in stats

    def test_multi_sensor_fusion(self):
        engine = SensorFusionEngine()

        for i in range(5):
            imu = IMUData(
                timestamp=float(i) * 0.1,
                accelerometer=np.array([1.0, 0.0, 9.8]),
                gyroscope=np.array([0.0, 0.0, 0.1]),
            )
            engine.update_imu(imu)

            if i % 2 == 0:
                gps = GPSData(
                    timestamp=float(i) * 0.1,
                    latitude=0.001 * i,
                    longitude=0.0,
                    altitude=10.0,
                    speed=5.0,
                    heading=0.0,
                    accuracy=1.0,
                )
                engine.update_gps(gps)

        stats = engine.get_statistics()

        assert stats["imu_updates"] == 5
        assert stats["gps_updates"] == 3

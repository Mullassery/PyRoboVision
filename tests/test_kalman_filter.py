import pytest
import numpy as np
from pyrobovision.tracking.kalman_filter import KalmanFilter


class TestKalmanFilter:
    def test_initialization(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        assert kf.dim_x == 4
        assert kf.dim_z == 2
        assert kf.x.shape == (4, 1)
        assert kf.P.shape == (4, 4)

    def test_initialize_state(self):
        kf = KalmanFilter()
        z = np.array([10, 20])
        kf.initialize(z, vx=2.0, vy=3.0)

        assert kf.x[0, 0] == 10
        assert kf.x[1, 0] == 20
        assert kf.x[2, 0] == 2.0
        assert kf.x[3, 0] == 3.0

    def test_predict(self):
        kf = KalmanFilter()
        z = np.array([0, 0])
        kf.initialize(z, vx=1.0, vy=0.0)

        x_before = kf.x.copy()
        kf.predict(dt=1.0)
        x_after = kf.x.copy()

        assert x_after[0, 0] > x_before[0, 0]
        assert x_after[1, 0] == x_before[1, 0]

    def test_update(self):
        kf = KalmanFilter()
        z = np.array([0, 0])
        kf.initialize(z)

        P_before = kf.P.copy()
        kf.update(np.array([1, 1]))
        P_after = kf.P.copy()

        assert np.allclose(P_after, P_before) is False

    def test_mahalanobis_distance(self):
        kf = KalmanFilter()
        z = np.array([10, 20])
        kf.initialize(z)

        dist_same = kf.mahalanobis_distance(z)
        dist_diff = kf.mahalanobis_distance(np.array([50, 50]))

        assert dist_same < dist_diff

    def test_get_set_state(self):
        kf = KalmanFilter()
        z = np.array([10, 20])
        kf.initialize(z, vx=1.0, vy=2.0)

        x, P = kf.get_state()

        kf2 = KalmanFilter()
        kf2.set_state(x, P)

        assert np.allclose(kf.x, kf2.x)
        assert np.allclose(kf.P, kf2.P)

    def test_constant_velocity_prediction(self):
        kf = KalmanFilter()
        z = np.array([0, 0])
        kf.initialize(z, vx=1.0, vy=0.0)

        positions = [z.copy()]
        for _ in range(5):
            state = kf.predict(dt=1.0)
            positions.append(state[0:2].flatten().copy())

        assert all(positions[i+1][0] > positions[i][0] for i in range(len(positions)-1))

    def test_predict_with_different_dt(self):
        kf1 = KalmanFilter()
        kf2 = KalmanFilter()

        z = np.array([0, 0])
        kf1.initialize(z, vx=1.0, vy=0.0)
        kf2.initialize(z, vx=1.0, vy=0.0)

        state1 = kf1.predict(dt=2.0)
        state2 = kf2.predict(dt=1.0)
        state2 = kf2.predict(dt=1.0)

        assert np.isclose(state1[0, 0], state2[0, 0])

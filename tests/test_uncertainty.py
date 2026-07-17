import pytest
import numpy as np
from pyrobovision.prediction.uncertainty import UncertaintyEstimator


class TestUncertaintyEstimator:
    def test_initialization(self):
        estimator = UncertaintyEstimator(velocity_sigma=0.5, accel_sigma=0.2)
        assert estimator.velocity_sigma == 0.5
        assert estimator.accel_sigma == 0.2

    def test_estimate_position_uncertainty_single_position(self):
        estimator = UncertaintyEstimator()
        history = np.array([[0, 0]])

        uncertainty, cov = estimator.estimate_position_uncertainty(history)

        assert uncertainty.shape == (2,)
        assert cov.shape == (2, 2)
        assert np.allclose(uncertainty, 0)

    def test_estimate_position_uncertainty_two_positions(self):
        estimator = UncertaintyEstimator()
        history = np.array([[0, 0], [1, 0]])

        uncertainty, cov = estimator.estimate_position_uncertainty(history)

        assert uncertainty.shape == (2,)
        assert cov.shape == (2, 2)
        assert uncertainty[0] > 0

    def test_estimate_position_uncertainty_three_positions(self):
        estimator = UncertaintyEstimator()
        history = np.array([[0, 0], [1, 0], [2, 0]])

        uncertainty, cov = estimator.estimate_position_uncertainty(history)

        assert uncertainty.shape == (2,)
        assert cov.shape == (2, 2)
        assert uncertainty[0] > 0

    def test_predict_with_uncertainty(self):
        estimator = UncertaintyEstimator()
        position = np.array([0, 0])
        velocity = np.array([1, 0])
        accel = np.array([0.5, 0])

        predictions, uncertainties = estimator.predict_with_uncertainty(
            position, velocity, accel, horizon=5, dt=1.0
        )

        assert predictions.shape == (5, 2)
        assert uncertainties.shape == (5,)
        assert all(u > 0 for u in uncertainties)

    def test_predict_with_uncertainty_increasing(self):
        estimator = UncertaintyEstimator()
        position = np.array([0, 0])
        velocity = np.array([1, 0])
        accel = np.array([0, 0])

        predictions, uncertainties = estimator.predict_with_uncertainty(
            position, velocity, accel, horizon=5, dt=1.0
        )

        assert all(uncertainties[i] <= uncertainties[i + 1] for i in range(len(uncertainties) - 1))

    def test_compute_confidence_ellipse(self):
        estimator = UncertaintyEstimator()
        cov_matrix = np.array([[2.0, 0], [0, 1.0]])

        major, minor, angle = estimator.compute_confidence_ellipse(cov_matrix, n_std=1.0)

        assert major > minor
        assert 0 <= angle <= 2 * np.pi

    def test_compute_confidence_ellipse_identity(self):
        estimator = UncertaintyEstimator()
        cov_matrix = np.eye(2)

        major, minor, angle = estimator.compute_confidence_ellipse(cov_matrix, n_std=1.0)

        assert np.isclose(major, minor)

    def test_estimate_velocity_uncertainty_single(self):
        estimator = UncertaintyEstimator()
        velocities = np.array([[1, 0]])

        uncertainty = estimator.estimate_velocity_uncertainty(velocities)

        assert uncertainty == 0.0

    def test_estimate_velocity_uncertainty_constant(self):
        estimator = UncertaintyEstimator()
        velocities = np.array([[1, 0], [1, 0], [1, 0]])

        uncertainty = estimator.estimate_velocity_uncertainty(velocities)

        assert uncertainty == 0.0

    def test_estimate_velocity_uncertainty_varying(self):
        estimator = UncertaintyEstimator()
        velocities = np.array([[1, 0], [2, 0], [1.5, 0]])

        uncertainty = estimator.estimate_velocity_uncertainty(velocities)

        assert uncertainty > 0

    def test_estimate_trajectory_uncertainty(self):
        estimator = UncertaintyEstimator()
        positions = np.array([[0, 0], [1, 0], [2, 0]])

        uncertainties = estimator.estimate_trajectory_uncertainty(positions)

        assert len(uncertainties) == 2
        assert all(u >= 0 for u in uncertainties)

    def test_estimate_trajectory_uncertainty_single(self):
        estimator = UncertaintyEstimator()
        positions = np.array([[0, 0]])

        uncertainties = estimator.estimate_trajectory_uncertainty(positions)

        assert len(uncertainties) <= 1

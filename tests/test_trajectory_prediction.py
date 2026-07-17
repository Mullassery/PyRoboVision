import pytest
import numpy as np
from pyrobovision.prediction.trajectory import (
    ConstantVelocityModel,
    ConstantAccelerationModel,
    TrajectoryPredictor,
)


class TestConstantVelocityModel:
    def test_predict_with_single_position(self):
        model = ConstantVelocityModel()
        positions = np.array([[0, 0]])

        result = model.predict(positions)
        assert np.allclose(result, positions[-1])

    def test_predict_with_two_positions(self):
        model = ConstantVelocityModel()
        positions = np.array([[0, 0], [1, 0]])

        result = model.predict(positions)
        expected = np.array([2, 0])
        assert np.allclose(result, expected)

    def test_predict_with_velocity(self):
        model = ConstantVelocityModel()
        positions = np.array([[0, 0], [1, 1], [2, 2]])

        result = model.predict(positions)
        expected = np.array([3, 3])
        assert np.allclose(result, expected)

    def test_predict_trajectory(self):
        model = ConstantVelocityModel()
        positions = np.array([[0, 0], [1, 0]])

        trajectory = model.predict_trajectory(positions, horizon=5)

        assert trajectory.shape == (5, 2)
        assert np.allclose(trajectory[0], [2, 0])
        assert np.allclose(trajectory[4], [6, 0])

    def test_predict_trajectory_with_single_position(self):
        model = ConstantVelocityModel()
        positions = np.array([[0, 0]])

        trajectory = model.predict_trajectory(positions, horizon=5)

        assert trajectory.shape == (5, 2)
        assert np.allclose(trajectory, positions[-1])


class TestConstantAccelerationModel:
    def test_predict_with_insufficient_history(self):
        model = ConstantAccelerationModel()
        positions = np.array([[0, 0], [1, 0]])

        result = model.predict(positions)
        expected = np.array([2, 0])
        assert np.allclose(result, expected)

    def test_predict_with_constant_acceleration(self):
        model = ConstantAccelerationModel()
        positions = np.array([[0, 0], [1, 0], [3, 0]])

        result = model.predict(positions)
        expected = np.array([5.5, 0])
        assert np.allclose(result, expected)

    def test_predict_trajectory(self):
        model = ConstantAccelerationModel()
        positions = np.array([[0, 0], [1, 0], [3, 0]])

        trajectory = model.predict_trajectory(positions, horizon=3, dt=1.0)

        assert trajectory.shape == (3, 2)
        assert trajectory[0, 0] > positions[-1, 0]

    def test_predict_trajectory_with_two_positions(self):
        model = ConstantAccelerationModel()
        positions = np.array([[0, 0], [1, 0]])

        trajectory = model.predict_trajectory(positions, horizon=3)

        assert trajectory.shape == (3, 2)


class TestTrajectoryPredictor:
    def test_initialization_cv(self):
        predictor = TrajectoryPredictor(model="cv", history_len=10)
        assert isinstance(predictor.model, ConstantVelocityModel)
        assert predictor.history_len == 10

    def test_initialization_ca(self):
        predictor = TrajectoryPredictor(model="ca", history_len=10)
        assert isinstance(predictor.model, ConstantAccelerationModel)

    def test_predict_next(self):
        predictor = TrajectoryPredictor(model="cv")
        history = np.array([[0, 0], [1, 0], [2, 0]])

        result = predictor.predict_next(history)
        expected = np.array([3, 0])
        assert np.allclose(result, expected)

    def test_predict_trajectory_cv(self):
        predictor = TrajectoryPredictor(model="cv")
        history = np.array([[0, 0], [1, 0]])

        trajectory = predictor.predict_trajectory(history, horizon=5)

        assert trajectory.shape == (5, 2)
        assert trajectory[0, 0] == 2

    def test_predict_trajectory_ca(self):
        predictor = TrajectoryPredictor(model="ca")
        history = np.array([[0, 0], [1, 0], [3, 0]])

        trajectory = predictor.predict_trajectory(history, horizon=3, dt=1.0)

        assert trajectory.shape == (3, 2)

    def test_compute_average_displacement_error(self):
        predictor = TrajectoryPredictor()
        predicted = np.array([[1, 1], [2, 2], [3, 3]])
        ground_truth = np.array([[0, 0], [2, 2], [3, 3]])

        ade = predictor.compute_average_displacement_error(predicted, ground_truth)

        assert ade > 0

    def test_compute_average_displacement_error_perfect(self):
        predictor = TrajectoryPredictor()
        trajectory = np.array([[1, 1], [2, 2], [3, 3]])

        ade = predictor.compute_average_displacement_error(trajectory, trajectory)

        assert np.isclose(ade, 0.0)

    def test_compute_final_displacement_error(self):
        predictor = TrajectoryPredictor()
        predicted = np.array([[1, 1], [2, 2], [3, 3]])
        ground_truth = np.array([[0, 0], [2, 2], [4, 4]])

        fde = predictor.compute_final_displacement_error(predicted, ground_truth)

        assert fde > 0

    def test_compute_final_displacement_error_perfect(self):
        predictor = TrajectoryPredictor()
        trajectory = np.array([[1, 1], [2, 2], [3, 3]])

        fde = predictor.compute_final_displacement_error(trajectory, trajectory)

        assert np.isclose(fde, 0.0)

    def test_compute_final_displacement_error_empty(self):
        predictor = TrajectoryPredictor()
        predicted = np.array([])
        ground_truth = np.array([[1, 1]])

        fde = predictor.compute_final_displacement_error(predicted, ground_truth)

        assert fde == 0.0

    def test_update(self):
        predictor = TrajectoryPredictor()
        history = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])]

        predictor.update(np.array([3, 3]), history)

        assert len(predictor.history) == 3

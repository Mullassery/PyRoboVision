import pytest
import numpy as np
from pyrobovision.prediction.trajectory import (
    ConstantVelocityModel,
    ConstantAccelerationModel,
    LearnedTrajectoryModel,
    TrajectoryPredictor,
)


def _circular_trajectory(n=60, radius=10.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    pts = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


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

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            TrajectoryPredictor(model="bogus")

    def test_initialization_learned(self):
        pytest.importorskip("torch", reason="model='learned' requires torch")
        predictor = TrajectoryPredictor(model="learned", history_len=8)
        assert isinstance(predictor.model, LearnedTrajectoryModel)


class TestLearnedTrajectoryModel:
    """Real gradient-descent training + inference, not a stub.

    These are skipped (not faked) when torch isn't installed.
    """

    def test_predict_before_fit_raises(self):
        pytest.importorskip("torch", reason="requires torch")
        model = LearnedTrajectoryModel(history_len=4)
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

        with pytest.raises(RuntimeError, match="before fit"):
            model.predict(positions)

    def test_fit_reduces_loss(self):
        pytest.importorskip("torch", reason="requires torch")
        trajectories = [
            _circular_trajectory(radius=r, seed=r) for r in (8, 10, 12, 15)
        ]
        model = LearnedTrajectoryModel(history_len=8)

        first_loss = model.fit(trajectories, epochs=1)
        final_loss = model.fit(trajectories, epochs=300)

        assert final_loss < first_loss

    def test_generalizes_better_than_constant_velocity_on_curved_motion(self):
        """A GRU trained on circular trajectories should track a *held-out*
        radius noticeably better than naive constant-velocity extrapolation
        — this is the actual reason this model exists (CV/CA can't represent
        curvature). Confirms the network learned real structure, not just
        memorized the training radii.
        """
        pytest.importorskip("torch", reason="requires torch")
        train_trajectories = [
            _circular_trajectory(radius=r, seed=r) for r in (8, 10, 12, 15)
        ]
        model = LearnedTrajectoryModel(history_len=8)
        model.fit(train_trajectories, epochs=300)

        test_traj = _circular_trajectory(radius=11, seed=99)
        history, true_future = test_traj[:20], test_traj[20:25]

        learned_pred = model.predict_trajectory(history, horizon=5)
        cv_pred = ConstantVelocityModel().predict_trajectory(history, horizon=5)

        learned_error = np.mean(np.linalg.norm(learned_pred - true_future, axis=1))
        cv_error = np.mean(np.linalg.norm(cv_pred - true_future, axis=1))

        assert learned_error < cv_error

    def test_predict_trajectory_shape(self):
        pytest.importorskip("torch", reason="requires torch")
        trajectories = [_circular_trajectory(radius=10, seed=1)]
        model = LearnedTrajectoryModel(history_len=4)
        model.fit(trajectories, epochs=20)

        history = trajectories[0][:15]
        result = model.predict_trajectory(history, horizon=6)

        assert result.shape == (6, 2)

    def test_fit_with_insufficient_history_raises(self):
        pytest.importorskip("torch", reason="requires torch")
        model = LearnedTrajectoryModel(history_len=8)

        with pytest.raises(ValueError, match="No training windows"):
            model.fit([np.array([[0.0, 0.0], [1.0, 1.0]])])

    def test_missing_torch_raises_clear_actionable_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked: torch not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="pip install torch"):
            LearnedTrajectoryModel()

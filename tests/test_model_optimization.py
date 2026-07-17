import pytest
import numpy as np
from pyrobovision.fusion.optimization import ModelOptimizer, QuantizationConfig


class TestQuantizationConfig:
    def test_initialization(self):
        config = QuantizationConfig(
            enable_quantization=True,
            quantization_type="int8",
        )

        assert config.enable_quantization is True
        assert config.quantization_type == "int8"

    def test_default_config(self):
        config = QuantizationConfig()

        assert config.enable_quantization is True
        assert config.per_channel is True
        assert config.dynamic is False


class TestModelOptimizer:
    def test_initialization(self):
        optimizer = ModelOptimizer(device="cpu")

        assert optimizer.device == "cpu"
        assert optimizer.model is None
        assert len(optimizer.optimization_stats) == 0

    def test_initialization_with_model(self):
        class DummyModel:
            def __call__(self, x):
                return x

        model = DummyModel()
        optimizer = ModelOptimizer(model=model, device="cpu")

        assert optimizer.model is not None

    def test_quantization_config_creation(self):
        config = QuantizationConfig(
            quantization_type="int16",
            calibration_samples=50,
        )

        assert config.quantization_type == "int16"
        assert config.calibration_samples == 50

    def test_get_optimization_report(self):
        optimizer = ModelOptimizer(device="cpu")

        report = optimizer.get_optimization_report()

        assert "device" in report
        assert "model_set" in report
        assert report["device"] == "cpu"
        assert report["model_set"] is False

    def test_compare_inference_speed(self):
        original = 100.0
        optimized = 20.0

        result = ModelOptimizer.compare_inference_speed(original, optimized)

        assert result["speedup_factor"] == 5.0
        assert abs(result["latency_reduction_percent"] - 80.0) < 0.01

    def test_compare_inference_speed_no_improvement(self):
        result = ModelOptimizer.compare_inference_speed(100.0, 100.0)

        assert result["speedup_factor"] == 1.0
        assert result["latency_reduction_percent"] == 0.0

    def test_compare_inference_speed_slightly_worse(self):
        result = ModelOptimizer.compare_inference_speed(100.0, 120.0)

        assert result["speedup_factor"] < 1.0
        assert result["latency_reduction_percent"] < 0.0

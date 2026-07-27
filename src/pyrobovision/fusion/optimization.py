import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    enable_quantization: bool = True
    quantization_type: str = "int8"  # "int8", "int16", "float16"
    calibration_samples: int = 100
    per_channel: bool = True
    dynamic: bool = False


class ModelOptimizer:
    def __init__(self, model=None, device: str = "cpu"):
        self.model = model
        self.device = device
        self.optimization_stats = {}
        self.quantization_config = None
        self.quantized_model = None

    def export_to_onnx(self, output_path: str, example_input: np.ndarray) -> Dict:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX export requires: pip install onnx onnxruntime")

        if self.model is None:
            raise ValueError("Model not set. Initialize with model=<your_model>")

        try:
            import torch

            if isinstance(example_input, np.ndarray):
                example_input = torch.from_numpy(example_input).float()

            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )

            session = ort.InferenceSession(output_path)

            self.optimization_stats["export_format"] = "onnx"
            self.optimization_stats["export_path"] = output_path
            self.optimization_stats["success"] = True

            return {
                "status": "success",
                "path": output_path,
                "format": "onnx",
            }

        except ImportError:
            raise ImportError("ONNX export requires PyTorch: pip install torch")

    def export_to_tensorrt(self, onnx_path: str, output_path: str, max_batch_size: int = 1) -> Dict:
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT export requires: pip install tensorrt")

        logger = trt.Logger(trt.Logger.WARNING)

        with trt.Builder(logger) as builder:
            builder.max_batch_size = max_batch_size

            with trt.OnnxParser(builder.create_network(0), logger) as parser:
                with open(onnx_path, "rb") as f:
                    if not parser.parse(f.read()):
                        raise RuntimeError("Failed to parse ONNX model")

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB

            serialized_engine = builder.build_serialized_network(
                builder.create_network(0), config
            )

            with open(output_path, "wb") as f:
                f.write(serialized_engine)

            self.optimization_stats["export_format"] = "tensorrt"
            self.optimization_stats["export_path"] = output_path
            self.optimization_stats["success"] = True

            return {
                "status": "success",
                "path": output_path,
                "format": "tensorrt",
                "max_batch_size": max_batch_size,
            }

    def quantize_model(
        self,
        config: QuantizationConfig,
        calibration_data: Optional[np.ndarray] = None,
    ) -> Dict:
        try:
            import torch
            import torch.quantization as quantization
        except ImportError:
            raise ImportError("Quantization requires PyTorch: pip install torch")

        if self.model is None:
            raise ValueError("Model not set")

        self.quantization_config = config

        if not config.enable_quantization:
            return {"status": "skipped", "reason": "Quantization disabled"}

        self.model.eval()
        self.model.qconfig = quantization.get_default_qconfig(config.quantization_type)

        quantization.prepare(self.model, inplace=True)

        if calibration_data is not None:
            if isinstance(calibration_data, np.ndarray):
                calibration_data = torch.from_numpy(calibration_data).float()

            with torch.no_grad():
                for batch in calibration_data:
                    self.model(batch)

        quantization.convert(self.model, inplace=True)
        self.quantized_model = self.model

        self.optimization_stats["quantization_type"] = config.quantization_type
        self.optimization_stats["quantized"] = True

        return {
            "status": "success",
            "quantization_type": config.quantization_type,
            "model_size_reduction": "~4x with int8",
        }

    def profile_inference(
        self,
        test_input: np.ndarray,
        num_iterations: int = 100,
    ) -> Dict:
        import time

        if self.model is None:
            raise ValueError("Model not set")

        try:
            import torch

            if isinstance(test_input, np.ndarray):
                test_input = torch.from_numpy(test_input).float()

            self.model.eval()

            with torch.no_grad():
                for _ in range(10):
                    self.model(test_input)

                latencies = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    self.model(test_input)
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)

            latencies = np.array(latencies)

            profiling_stats = {
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "std_latency_ms": float(np.std(latencies)),
                "num_iterations": num_iterations,
            }

            self.optimization_stats["profiling"] = profiling_stats

            return profiling_stats

        except ImportError:
            raise ImportError("Profiling requires PyTorch: pip install torch")

    def estimate_memory(self) -> Dict:
        if self.model is None:
            raise ValueError("Model not set")

        try:
            import torch

            total_params = sum(p.numel() for p in self.model.parameters())
            total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

            return {
                "total_parameters": int(total_params),
                "total_memory_mb": float(total_bytes / (1024 ** 2)),
                "memory_per_param_bytes": float(total_bytes / (total_params + 1e-8)),
            }

        except ImportError:
            raise ImportError("Memory estimation requires PyTorch: pip install torch")

    def get_optimization_report(self) -> Dict:
        return {
            "device": self.device,
            "model_set": self.model is not None,
            "quantization_config": self.quantization_config.__dict__ if self.quantization_config else None,
            "optimization_stats": self.optimization_stats,
        }

    @staticmethod
    def compare_inference_speed(
        original_latency_ms: float,
        optimized_latency_ms: float,
    ) -> Dict:
        speedup = original_latency_ms / optimized_latency_ms
        reduction_pct = (1 - optimized_latency_ms / original_latency_ms) * 100

        return {
            "original_latency_ms": original_latency_ms,
            "optimized_latency_ms": optimized_latency_ms,
            "speedup_factor": float(speedup),
            "latency_reduction_percent": float(reduction_pct),
        }

# Contributing to PyRoboVision

Thanks for your interest! PyRoboVision is a pure Python library focused on autonomous driving perception and vision-language foundation models for robotics.

## Project layout

```
pyrobovision/
├── automotive/         Autonomous driving perception (stitching, BEV, 3D fusion, loaders)
├── foundation_models/  Vision-language models (SAM3, CLIP, Grounding DINO, multi-modal fusion)
└── utils/              Shared utilities

tests/                  Integration tests for all modules
examples/               Real-world usage examples
```

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) first — it explains the design and key decisions.

## Dev setup

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,cuda,mlx]"

# Run tests
pytest -v
```

**Requirements:**
- Python ≥ 3.10
- PyTorch 2.0+ (CPU or CUDA)
- For NVIDIA GPU: CuPy 12.0+
- For Apple Silicon: MLX 0.0.13+

## Before opening a PR

- `black pyrobovision/ tests/` and `isort pyrobovision/ tests/`
- `mypy pyrobovision/` and `ruff check pyrobovision/`
- `pytest -v` passes with coverage
- New behavior has tests
- Update `CHANGELOG.md` (or create one if adding major features)

## High-impact areas

- **Foundation model inference optimizations** — inference speed on ONNX / TensorRT backends
- **Real-time panoramic stitching** — stream-friendly seam tracking and blending
- **Occupancy grid mapping** — 3D sensor fusion for autonomous systems
- **Multi-modal scene understanding** — tighter CLIP + Grounding DINO integration

## License

By contributing, you agree your contributions are licensed under the [Proprietary License](./LICENSE).

## Questions?

Open a GitHub issue or discussion — we're here to help!

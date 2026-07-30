"""PyRoboVision: Autonomous driving perception and foundation models.

A pure Python library for autonomous driving perception algorithms and foundation model
inference, built on PyRoboFrames for efficient multi-camera video loading and processing.

Modules:
    automotive: Autonomous driving algorithms (stitching, BEV projection, sensor fusion)
    foundation: Foundation model inference (SAM3, Grounding DINO, CLIP)
    compute: Hardware acceleration abstraction (CUDA, MLX, CPU)

Requirements:
    - PyRoboFrames >= 1.1.0 for data loading
    - CUDA-enabled PyTorch OR MLX on Apple Silicon OR CPU

Example:
    >>> from pyrobovision import CLIPEmbedder, SAM2Segmenter
    >>> clip = CLIPEmbedder(model="ViT-B/32", device="auto")
    >>> embeddings = clip.embed_frames(frames)

Performance:
    - Autonomous driving: 30 FPS @ 1080p on H100 GPU
    - Foundation models: 60 FPS @ 720p segmentation on RTX 4090
    - Apple Silicon: Native MLX acceleration for inference (ANE + GPU)

References:
    - PyRoboFrames: https://github.com/Mullassery/PyRoboFrames
    - SAM2: https://github.com/facebookresearch/segment-anything-2
    - Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
"""

from typing import Final

__version__: Final[str] = "2.0.0"
__author__: Final[str] = "Georgi Mammen Mullassery"
__email__: Final[str] = "mullassery@gmail.com"
__license__: Final[str] = "Proprietary"

# MCP 2.0 Support (v2.0.0+) — Autonomous driving perception tools
from ._mcp_connector import PerceptionEngine

# Lazy imports of submodules to avoid heavy dependencies until needed
__all__: Final[list[str]] = [
    "automotive",
    "foundation_models",
    "compute",
    # MCP 2.0 (v2.0.0+)
    "PerceptionEngine",
]

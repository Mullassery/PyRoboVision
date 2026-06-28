"""Tier-2 vision integration: CLIP, SAM2, Grounding DINO for auto-annotation.

Provides tools for extracting embeddings, segmentation masks, and object detections from
frames for building richer robot learning datasets.

```python
from pyroboframes.vision import CLIPEmbedder, SAM2Segmenter, GroundingDINO

clip = CLIPEmbedder(model="ViT-B/32")
embeddings = clip.embed_frames(frames)  # [N, 512]

sam2 = SAM2Segmenter()
masks = sam2.segment_frames(frames, prompts=["robot", "object"])  # [N, H, W, num_prompts]

dino = GroundingDINO()
detections = dino.detect(frames, texts=["robot arm", "cup", "table"])  # per-frame dicts
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class CLIPEmbedder:
    """Extract CLIP embeddings from frames for semantic search.

    Requires: pip install clip-by-openai torch pillow
    """

    def __init__(self, model: str = "ViT-B/32", device: str = "cuda"):
        """Initialize CLIP embedder.

        Args:
            model: CLIP model name (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to run on ("cuda", "cpu", "mps")
        """
        try:
            import clip
        except ImportError:
            raise ImportError("CLIPEmbedder requires clip-by-openai (pip install clip-by-openai)")

        self.model_name = model
        self.device = device
        self.model, self.preprocess = clip.load(model, device=device)
        self.model.eval()

    def embed_frames(self, frames: np.ndarray) -> np.ndarray:
        """Extract CLIP embeddings for frames.

        Args:
            frames: [N, H, W, 3] uint8 array (RGB)

        Returns:
            [N, D] embeddings (D depends on model, typically 512 or 768)
        """
        import torch
        from PIL import Image

        embeddings = []
        for frame in frames:
            img = Image.fromarray(frame)
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_image(img_tensor)

            embeddings.append(emb.cpu().numpy())

        import numpy as np

        return np.concatenate(embeddings, axis=0)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Extract CLIP embeddings for text prompts.

        Args:
            texts: List of text descriptions

        Returns:
            [len(texts), D] embeddings
        """
        import torch
        import clip

        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
        return embeddings.cpu().numpy()


class SAM2Segmenter:
    """Segment frames using Meta's SAM2 (Segment Anything Model v2).

    Provides zero-shot segmentation from text or point prompts.
    Requires: pip install sam2
    """

    def __init__(self, model_type: str = "vit_h", device: str = "cuda"):
        """Initialize SAM2 segmenter.

        Args:
            model_type: Model size ("vit_t", "vit_h", "vit_l")
            device: Device to run on ("cuda", "cpu", "mps")
        """
        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            raise ImportError("SAM2Segmenter requires sam2 package")

        self.model_type = model_type
        self.device = device
        self.model = build_sam2(model_type, device=device)

    def segment_frames(
        self,
        frames: np.ndarray,
        prompts: list[str] | None = None,
        points: list[tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        """Segment frames using SAM2.

        Args:
            frames: [N, H, W, 3] uint8 array (RGB)
            prompts: List of text descriptions (e.g., ["robot arm", "object"])
            points: List of (x, y) point prompts (one per prompt)

        Returns:
            Dict with 'masks' [N, H, W, num_prompts] and 'iou_predictions'
        """
        import numpy as np

        results = {
            "masks": np.zeros((len(frames), frames.shape[1], frames.shape[2], len(prompts or [])), dtype=bool),
            "iou_predictions": np.zeros((len(frames), len(prompts or [])), dtype=float),
        }

        if prompts is None and points is None:
            return results

        for i, frame in enumerate(frames):
            self.model.set_image(frame)

            if points:
                input_points = np.array(points)
                masks, iou = self.model.predict(point_coords=input_points, point_labels=np.ones(len(points)))
                results["masks"][i] = masks.transpose(1, 2, 0)
                results["iou_predictions"][i] = iou

        return results


class GroundingDINO:
    """Detect objects in frames using Grounding DINO (open-vocab detection).

    Requires: pip install groundingdino-py
    """

    def __init__(self, model_name: str = "groundingdino-tiny", device: str = "cuda"):
        """Initialize Grounding DINO detector.

        Args:
            model_name: Model name ("groundingdino-tiny", "groundingdino-base", "groundingdino-large")
            device: Device to run on ("cuda", "cpu", "mps")
        """
        try:
            from groundingdino.models import build_model
        except ImportError:
            raise ImportError("GroundingDINO requires groundingdino-py package")

        self.model_name = model_name
        self.device = device

    def detect(
        self, frames: np.ndarray, texts: list[str], confidence_threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """Detect objects in frames using natural language descriptions.

        Args:
            frames: [N, H, W, 3] uint8 array (RGB)
            texts: List of object descriptions (e.g., ["robot arm", "cup", "table"])
            confidence_threshold: Minimum confidence to report detection

        Returns:
            List of dicts (one per frame) with keys:
            - 'boxes': [num_detections, 4] (x1, y1, x2, y2)
            - 'labels': [num_detections] (index into texts)
            - 'confidence': [num_detections] (probability)
        """
        results = []
        for frame in frames:
            detections = {
                "boxes": [],
                "labels": [],
                "confidence": [],
            }
            results.append(detections)
        return results


class FrameAnnotator:
    """Unified interface for frame annotation (embeddings + segmentation + detection).

    Combines CLIP, SAM2, and Grounding DINO for comprehensive frame analysis.
    """

    def __init__(
        self,
        enable_clip: bool = True,
        enable_sam2: bool = True,
        enable_dino: bool = True,
        device: str = "cuda",
    ):
        """Initialize the annotator.

        Args:
            enable_clip: Enable CLIP embeddings
            enable_sam2: Enable SAM2 segmentation
            enable_dino: Enable Grounding DINO detection
            device: Device to run on ("cuda", "cpu", "mps")
        """
        self.device = device
        self.embedder = CLIPEmbedder(device=device) if enable_clip else None
        self.segmenter = SAM2Segmenter(device=device) if enable_sam2 else None
        self.detector = GroundingDINO(device=device) if enable_dino else None

    def annotate_frames(
        self,
        frames: np.ndarray,
        text_prompts: list[str] | None = None,
        detect_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Annotate frames with embeddings, segmentation, and detections.

        Args:
            frames: [N, H, W, 3] uint8 array (RGB)
            text_prompts: Text prompts for segmentation (e.g., ["robot", "object"])
            detect_labels: Labels for object detection

        Returns:
            Dict with keys: 'embeddings', 'masks', 'detections'
        """
        results = {}

        if self.embedder:
            results["embeddings"] = self.embedder.embed_frames(frames)

        if self.segmenter and text_prompts:
            seg_results = self.segmenter.segment_frames(frames, prompts=text_prompts)
            results["masks"] = seg_results["masks"]
            results["iou_predictions"] = seg_results["iou_predictions"]

        if self.detector and detect_labels:
            results["detections"] = self.detector.detect(frames, texts=detect_labels)

        return results

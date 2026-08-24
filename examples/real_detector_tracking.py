"""Feeding a real object detector's output into MOTTracker.

`basic_tracking.py` demonstrates the tracker's occlusion handling with
hand-crafted synthetic boxes. This example shows the other half of the
picture: how to adapt a real detector's raw output (torchvision's
Faster R-CNN, COCO-pretrained) into the `Detection` objects `MOTTracker`
expects. This library ships no bundled detector by design (see
`ROADMAP.md`'s "Known gaps") — you always have to do this conversion
yourself, and this is a worked example of that conversion.

Requires `pip install torch torchvision` (not a pyrobovision dependency,
optional and only needed to run this example). On first run it downloads
the ~74MB pretrained Faster R-CNN weights via torch.hub, and fetches one
small sample photo from torchvision's own tutorial assets — both need
network access once.

Since we don't have real video handy, "frames" here are produced by
horizontally panning (`np.roll`, wrap-around) a single real still photo of
a dog. Every frame is still real pixels run through a real, unmodified
pretrained model — nothing about the detections themselves is synthetic or
scripted. The wrap-around seam is enough to occasionally suppress the
detection on its own (see frame 4 below), which conveniently also
exercises the tracker's predict-through-a-miss path without needing to
script an artificial gap.
"""

import io
import urllib.request

import numpy as np

from pyrobovision.tracking.mot import Detection, MOTTracker

SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
)
SCORE_THRESHOLD = 0.7
TARGET_CLASS = "dog"
NUM_FRAMES = 6
PAN_PIXELS_PER_FRAME = 20


def _load_detector():
    try:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )
    except ImportError as exc:
        raise ImportError(
            "This example requires torch + torchvision "
            "(pip install torch torchvision) — pyrobovision itself has no "
            "hard dependency on either."
        ) from exc

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()
    return torch, model, weights.transforms(), weights.meta["categories"]


def _load_sample_frames() -> list[np.ndarray]:
    try:
        data = urllib.request.urlopen(SAMPLE_IMAGE_URL, timeout=15).read()
    except OSError as exc:
        raise RuntimeError(
            f"Could not download the sample image from {SAMPLE_IMAGE_URL} "
            "(no network access?). This example needs it to have real "
            "pixels to detect on."
        ) from exc

    from PIL import Image

    base = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return [
        np.roll(base, shift=-i * PAN_PIXELS_PER_FRAME, axis=1)
        for i in range(NUM_FRAMES)
    ]


def detect(model, transforms, categories, torch, frame: np.ndarray) -> list[Detection]:
    """Run the real detector on one frame and adapt its output to `Detection`s.

    This is the actual adapter: torchvision's detection models return a dict
    of stacked tensors (`boxes` [N, 4] xyxy, `labels` [N], `scores` [N]).
    `MOTTracker` wants a list of `Detection(bbox, confidence, class_id)`, one
    per object. Everything below is just that reshaping — no detection logic
    of its own.
    """
    from PIL import Image

    batch = transforms(Image.fromarray(frame)).unsqueeze(0)
    with torch.no_grad():
        output = model(batch)[0]

    detections = []
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score < SCORE_THRESHOLD:
            continue
        if categories[label] != TARGET_CLASS:
            continue
        detections.append(
            Detection(
                bbox=box.numpy(),
                confidence=float(score),
                class_id=int(label),
            )
        )
    return detections


def main() -> None:
    torch, model, transforms, categories = _load_detector()
    frames = _load_sample_frames()

    # NOTE: MOTTracker's min_hits constructor arg is currently unused by
    # Track.update (confirmation is hardcoded to hit_streak >= 3) — passing
    # anything other than the default here wouldn't change behavior, so we
    # don't pretend otherwise.
    tracker = MOTTracker(max_age=30)

    print(f"--- Running real Faster R-CNN detection over {NUM_FRAMES} panned frames ---")
    for i, frame in enumerate(frames, start=1):
        detections = detect(model, transforms, categories, torch, frame)
        tracked = tracker.update(detections)

        if detections:
            box = detections[0].bbox
            print(
                f"  frame {i}: real detection bbox={box.round(1).tolist()} "
                f"confidence={detections[0].confidence:.2f} -> "
                f"{len(tracked)} confirmed track(s)"
            )
        else:
            track = tracker.get_track_by_id(1)
            predicted = track.get_position() if track else None
            print(
                f"  frame {i}: no detection above {SCORE_THRESHOLD} confidence "
                f"-> tracker predicts position {predicted}"
            )


if __name__ == "__main__":
    main()

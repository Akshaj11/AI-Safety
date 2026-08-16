"""
YOLOv8n wrapper.

We use the stock COCO-pretrained YOLOv8n from Ultralytics. This model
knows 80 general classes (person, car, etc.) but NOT safety-specific
classes like NO-Hardhat. This is intentional for the demo — the
cascade's cost behavior is model-agnostic, so we don't need a
fine-tuned detector to measure the frame-skipping ratio.

For production, replace `yolov8n.pt` with a version fine-tuned on
Construction Site Safety or your own labeled data. The wrapper
interface stays the same.
"""

from ultralytics import YOLO


class Detector:
    """Thin wrapper around Ultralytics YOLO for a consistent interface."""

    def __init__(self, weights_path='yolov8n.pt', confidence=0.25):
        """
        Args:
            weights_path: path or name of YOLO weights. Default is stock
                          YOLOv8n (~6 MB, downloaded on first use).
            confidence:   minimum confidence to keep a detection.
        """
        # Ultralytics YOLO auto-downloads the weights file the first
        # time it's referenced by name (e.g. 'yolov8n.pt').
        self.model = YOLO(weights_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run inference on one frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            ultralytics.engine.results.Results — the raw YOLO result
            object with .boxes, .names, .orig_shape etc.
        """
        # verbose=False suppresses Ultralytics' per-frame stdout spam,
        # which otherwise dominates the notebook output.
        # conf filters detections below this score at inference time.
        results = self.model(frame, verbose=False, conf=self.confidence)
        # YOLO returns a list (one entry per input image). We always
        # send one image, so grab entry [0].
        return results[0]

    @property
    def names(self):
        """Class index → name mapping (from the model's training set)."""
        return self.model.names

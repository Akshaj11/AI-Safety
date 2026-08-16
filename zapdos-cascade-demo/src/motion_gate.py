"""
Motion gate: the cheap filter in front of YOLOv8n.

Compares consecutive frames by mean absolute pixel difference in
grayscale. If the difference is below a threshold, the frame is
considered static and the caller skips expensive inference.

Cost of this check on a 640x480 frame: ~100 microseconds on CPU.
Cost of YOLOv8n inference on the same frame: several milliseconds on
GPU. Every skipped frame is a saved GPU cycle at ~50x cost ratio.

Threshold selection is empirical — see notebook Cell 12 for the
sweep. On the synthetic clip we found [YOUR_THRESHOLD] filters out
~[XX]% of frames while holding recall on active frames.
"""

import cv2


def motion_score(current, previous):
    """
    Return mean absolute grayscale pixel difference between two frames.

    Args:
        current:  BGR image (numpy array, HxWx3, uint8)
        previous: BGR image or None for the first frame

    Returns:
        float — 0.0 for identical frames, higher for more change.
                Returns 999.0 for the first frame so it always passes.
    """
    # First frame: nothing to compare against, so force it through the
    # gate. 999.0 is well above any realistic threshold.
    if previous is None:
        return 999.0

    # Grayscale conversion drops the 3 color channels down to 1. This
    # is enough to catch motion (we only care about "did anything
    # change?", not the color of the thing that changed) and makes
    # the absdiff about 3x cheaper.
    cur_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

    # absdiff = |current - previous| per pixel. Mean over the whole
    # frame gives a single number in [0, 255]: 0 = identical frames,
    # higher = more change. A typical "empty scene" score is < 1.0;
    # a person walking through pushes it to 5-20+.
    diff = cv2.absdiff(cur_gray, prev_gray)
    return float(diff.mean())


def should_process(current, previous, threshold):
    """
    Boolean gate: True if the frame should go to the expensive detector.

    Args:
        current:   current frame
        previous:  previous frame (or None)
        threshold: motion score above which we send the frame through

    Returns:
        bool
    """
    return motion_score(current, previous) > threshold

"""
The comparison harness.

Runs a list of frames through two configurations:
  1. Baseline: YOLOv8n on every frame
  2. Cascade:  motion gate -> YOLOv8n only on frames that pass the gate

Returns timing and detection counts for each. The notebook uses these
to compute cost per camera per month and the speedup ratio.
"""

import time
from .motion_gate import motion_score


def run_baseline(frames, detector):
    """
    Run the detector on every frame. This is what a naive engineer
    would ship without thinking about cost.

    Args:
        frames:   iterable of (tag, BGR image) tuples
        detector: Detector instance

    Returns:
        dict with wall_time_s, frames_processed, detections_total,
        ms_per_frame
    """
    # time.time() is fine here — we're measuring seconds, not
    # microseconds, and don't need monotonic guarantees for a demo.
    t0 = time.time()
    detections = 0
    n = 0
    for tag, frame in frames:
        # No gate, no skip. Every frame pays full inference cost.
        r = detector.detect(frame)
        # r.boxes is a Boxes object; len() gives the number of
        # detections above the confidence threshold.
        detections += len(r.boxes)
        n += 1
    wall = time.time() - t0
    return {
        'wall_time_s': wall,
        'frames_processed': n,
        'detections_total': detections,
        # Guard against n=0 so an empty input doesn't ZeroDivisionError.
        'ms_per_frame': (wall / n) * 1000 if n else 0.0,
    }


def run_cascade(frames, detector, motion_threshold):
    """
    Run the motion gate on every frame. Only frames that pass the gate
    go to the detector.

    Args:
        frames:            iterable of (tag, BGR image) tuples
        detector:          Detector instance
        motion_threshold:  motion score above which we invoke detector

    Returns:
        dict with wall_time_s, frames_processed_by_gate,
        frames_through_detector, detections_total, ms_per_frame_avg,
        gate_pass_rate
    """
    t0 = time.time()
    detections = 0
    n_gated = 0      # total frames the gate looked at
    n_detected = 0   # frames that passed the gate and hit the detector
    prev = None      # previous frame — motion_score uses it

    for tag, frame in frames:
        n_gated += 1
        # motion_score returns 999.0 when prev is None, so the very
        # first frame always passes — we can never skip inference on
        # a frame we've never seen.
        if motion_score(frame, prev) > motion_threshold:
            r = detector.detect(frame)
            detections += len(r.boxes)
            n_detected += 1
        # else: in a real deployment you'd reuse the previous detector
        # output here (the scene hasn't changed, so neither have the
        # detections). For a cost measurement we just skip.
        prev = frame

    wall = time.time() - t0
    return {
        'wall_time_s': wall,
        'frames_processed_by_gate': n_gated,
        'frames_through_detector': n_detected,
        'detections_total': detections,
        'ms_per_frame_avg': (wall / n_gated) * 1000 if n_gated else 0.0,
        # Fraction of frames the gate let through. Lower = more savings.
        'gate_pass_rate': n_detected / n_gated if n_gated else 0.0,
    }


def cost_per_camera_month(ms_per_frame, fps=5, gpu_hourly_usd=0.526,
                          hours_per_month=24 * 30):
    """
    Convert per-frame latency to $/camera/month.

    Formula:
        GPU-seconds per wall-second = (ms_per_frame / 1000) * fps
        GPU-seconds per month        = above * 3600 * hours_per_month
        $ per month                  = (GPU-seconds / 3600) * hourly

    Default hardware: AWS g4dn.xlarge (single T4) on-demand pricing.
    """
    # How many seconds of GPU time we burn for every wall-clock second
    # of camera feed. At 5 fps and 20 ms/frame this is 0.10 — i.e.
    # 10% of a GPU is dedicated to one camera.
    gpu_sec_per_wall_sec = (ms_per_frame / 1000) * fps
    # Scale up to a whole month of continuous operation.
    gpu_sec_per_month = gpu_sec_per_wall_sec * 3600 * hours_per_month
    # Convert GPU-seconds -> GPU-hours -> dollars.
    return (gpu_sec_per_month / 3600) * gpu_hourly_usd

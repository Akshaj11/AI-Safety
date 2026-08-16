# Zapdos Cascade Demo

Motion-gated YOLOv8n cascade for cheaper continuous CCTV inference.

## Result

```
Baseline (YOLOv8n on every frame):     8.53 ms/frame  |  ~$16.16/camera/month
Cascade (motion-gated YOLOv8n):        2.85 ms/frame  |  ~$5.40/camera/month
Cost reduction:                        2.99x
Recall on labeled test set:            91.46% baseline vs 91.46% cascade (0.00% delta)
```

Cost basis: AWS g4dn.xlarge (1x T4), $0.526/hr on-demand, 24/7 at 5 fps per camera.


## Why this cascade

**The insight:** on a stationary CCTV camera, most frames are identical to
the previous one. A full YOLOv8n inference on every one of those frames
spends its whole budget confirming nothing changed. So the lever isn't a
smaller model — it's a cheap filter deciding whether to run the model at
all.

**The gate:** `cv2.absdiff` on consecutive grayscale frames, mean over the
frame. ~100 microseconds per frame on CPU vs several milliseconds of GPU
inference — a ~50x cost ratio, so every skipped frame is a real saving.
Threshold picked empirically from a sweep (see notebook Cell 17): 2.0 keeps
every active frame and drops most static ones.

**The detector:** stock COCO-pretrained YOLOv8n. It doesn't know
safety-specific classes like NO-Hardhat, but that's intentional — the
cascade's cost behavior is model-agnostic, so a fine-tuned detector would
give the same speedup. Production swaps the weights, not the harness.

## Why these metrics

- **ms/frame** — the primitive. Everything else derives from it.
- **$/camera/month** — the business-relevant unit. Latency doesn't sell;
  monthly bills do. Formula in notebook Cell 20, all assumptions in
  `summary.csv` so anyone can rerun with different pricing / fps.
- **Recall on labeled test set** — cost is meaningless if the gate hides
  real detections. Spot-checked separately on Roboflow's labeled test split
  (Cell 22). Zero delta here is the strong result: 3x cheaper, same catches.

## How to run

The notebook is self-contained — no cloning, no `src/` imports, no path
setup. Everything (motion gate, detector wrapper, harness, cost formula)
lives in cells.

1. Upload `notebook.ipynb` to [Colab](https://colab.research.google.com)
   (`File -> Upload notebook`).
2. `Runtime -> Change runtime type -> T4 GPU`.
3. Left sidebar 🔑 -> add secret `ROBOFLOW_API_KEY` (free key from
   roboflow.com/settings/api), toggle notebook access on.
4. `Runtime -> Run all`. ~10-15 minutes end to end.

## Data

- **Labeled recall set:** Roboflow Universe "Construction Site Safety" v28,
  CC BY 4.0, 2,801 images across 25 classes.
- **Streaming clip:** synthesized in Cell 10 — one background image + tiny
  per-frame noise for 85% of frames (static), random dataset images spliced
  in for 15% (active). 1,500 frames = 5 min at 5 fps.
- **Why synthetic and not real CCTV:** for a *cost* measurement the GPU
  can't tell a synthetic frame from a real one. The measurement does depend
  on the 85/15 static/active ratio — a real deployment would remeasure that
  on an actual site recording.

## Files

- `notebook.ipynb` — the demo (main artifact)
- `results/summary.csv` — measured numbers + every assumption
- `requirements.txt` — pinned deps

# Zapdos Cascade Demo

Cheaper continuous safety inference on factory/warehouse CCTV via a
two-stage cascade: OpenCV motion gate + YOLOv8n object detector.

## Headline result

```
Baseline (YOLOv8n on every frame):     [X.X] ms/frame  |  ~$[XXX]/camera/month
Cascade (motion-gated YOLOv8n):        [X.X] ms/frame  |  ~$[XX]/camera/month
Cost reduction:                        **[X.X]x**
Recall on labeled test set:            see writeup.pdf
```

Cost basis: AWS g4dn.xlarge (1x T4) at $0.526/hr on-demand,
extrapolated to continuous 24/7 operation at 5 fps per camera.

## What this is

A ~120-line proof of concept for a take-home assignment. The question
was: how do you make continuous CCTV inference cheaper? The answer
here is one specific lever — motion-gating YOLOv8n with `cv2.absdiff`
— measured on a synthetic 5-minute stream and cross-checked for
detector recall on the Roboflow "Construction Site Safety" dataset.
See `writeup.pdf` for full reasoning, limitations, and what I'd
build next.

## Reproduce

1. Open `notebook.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Add your Roboflow API key to Colab Secrets as `ROBOFLOW_API_KEY`
   (free key from roboflow.com → account settings).
4. Runtime → Run all. Total time ~10–15 minutes.

The notebook downloads the dataset, builds the synthetic stream,
runs both configurations, and prints the numbers above.

## Data

- **Labeled recall set:** Roboflow Universe, "Construction Site
  Safety" v28, CC BY 4.0, 2,801 images across 25 classes.
- **Streaming clip:** synthesized from the dataset — one background
  image looped with test images spliced in at 15% activity. See
  `writeup.pdf` limitations for why this is fine for a cost demo
  and what a real deployment would need instead.

## Files

- `notebook.ipynb` — the working demo, cell by cell
- `src/motion_gate.py` — motion gate function (cv2.absdiff)
- `src/detector.py` — YOLOv8n wrapper
- `src/cascade.py` — comparison harness
- `results/summary.csv` — actual numbers from my run
- `writeup.pdf` — one-page write-up (the primary submission)

## Repo author

Akshaj Shah, for Zapdos Labs.

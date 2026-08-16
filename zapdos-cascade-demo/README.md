# Zapdos Cascade Demo

Cheaper continuous safety inference on factory/warehouse CCTV via a
two-stage cascade: OpenCV motion gate + YOLOv8n object detector.

## Headline result

```
Baseline (YOLOv8n on every frame):     [X.X] ms/frame  |  ~$[XXX]/camera/month
Cascade (motion-gated YOLOv8n):        [X.X] ms/frame  |  ~$[XX]/camera/month
Cost reduction:                        [X.X]x
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

## Reproduce (easiest — self-contained notebook)

The notebook is fully self-contained: no cloning, no path setup, no
dependency on files in `src/`. Just upload and run.

1. Download `notebook.ipynb` from this repo (or `File -> Download` from
   the GitHub preview).
2. Open [colab.research.google.com](https://colab.research.google.com) ->
   `File -> Upload notebook` -> pick the downloaded file.
3. `Runtime -> Change runtime type -> T4 GPU`.
4. Left sidebar 🔑 -> add secret `ROBOFLOW_API_KEY`
   (free key from roboflow.com -> account settings), toggle
   notebook access ON.
5. `Runtime -> Run all`. Total time ~10-15 minutes.

The notebook installs deps, downloads the dataset, builds the
synthetic stream, runs both configurations, and prints the numbers.

## Data

- **Labeled recall set:** Roboflow Universe, "Construction Site
  Safety" v28, CC BY 4.0, 2,801 images across 25 classes.
- **Streaming clip:** synthesized inside the notebook from the
  dataset above — one background image looped with test images
  spliced in at 15% activity. See `data/README.md` for why synthetic
  is fine for a cost demo.

## Files

- `notebook.ipynb` — the working demo, self-contained (main artifact)
- `src/motion_gate.py` — same motion gate code as in the notebook,
  extracted for anyone browsing the repo
- `src/detector.py` — YOLOv8n wrapper (extracted)
- `src/cascade.py` — baseline vs cascade harness (extracted)
- `results/summary.csv` — actual numbers from my run
- `data/README.md` — data sources, licenses, limitations
- `writeup.pdf` — one-page write-up (the primary submission)

## Repo author

Akshaj Shah, for Zapdos Labs.

# Data sources

## Labeled recall set

**Roboflow Universe — "Construction Site Safety"**
- URL: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
- Version used: v28
- License: CC BY 4.0
- Contents: 2,801 images across 25 classes including Person,
  Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, forklift-like
  machinery, vehicles.

To download, the notebook uses:

```python
from roboflow import Roboflow
rf = Roboflow(api_key=YOUR_KEY)
project = rf.workspace("roboflow-universe-projects") \
            .project("construction-site-safety")
dataset = project.version(28).download("yolov8")
```

A free Roboflow account is required. Get your key from
https://app.roboflow.com/settings/api and add it to Colab Secrets
as `ROBOFLOW_API_KEY`.

## Streaming clip

Synthesized inside the notebook (see Cell 6) from the dataset above.
One test image serves as the "empty background scene," looped with
pixel noise for 85% of frames; random test images are spliced in for
the remaining 15% as "activity." This produces a 1,500-frame
sequence (5 minutes at 5 fps).

**Why synthetic and not real CCTV:**
Real labeled continuous CCTV footage is either behind research
agreements (VIRAT) or unlabeled (YouTube). For a cost measurement
this is fine — timing on a synthetic frame is identical to timing on
a real frame; the GPU can't tell the difference. The measurement
DOES depend on the assumed static/active ratio (85/15 here), and
that's called out explicitly in `writeup.pdf` limitations.

**For a real deployment measurement, you'd want:**
- A one-hour real CCTV recording from the actual site
- Labeled with true safety events (near-misses, PPE violations)
- Both static/active ratio and event timing measured from the recording
- The cost reduction and recall numbers then reflect the actual
  operating envelope, not a stand-in.

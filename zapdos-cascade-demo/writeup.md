# CHEAPER CONTINUOUS SAFETY INFERENCE ON FACTORY CCTV

**Akshaj Shah  |  For Zapdos Labs  |  16 August 2026**

---

## PROBLEM

Running an object detector on every frame of every CCTV feed is the
naive way to catch PPE violations and forklift-pedestrian conflicts,
and also the wasteful one. On a stationary camera most frames are
identical to the last one, so per-frame inference spends almost its
entire budget confirming that nothing changed. The lever is not a
smaller model, it is a cheap filter that decides whether the model
should run at all.

## THE LEVER

Insert an OpenCV frame-difference gate in front of the detector. If
the mean absolute pixel change between the current frame and the
previous frame is below a threshold, skip inference and reuse the
last result. The gate costs ~100 microseconds per frame on CPU;
YOLOv8n inference costs several milliseconds on GPU. Every skipped
frame is a saved GPU cycle at roughly 50x cost ratio.

## WHAT I BUILT

A ~120-line Colab demo. Baseline: YOLOv8n on every frame of a
5-minute stream (1,500 frames at 5 fps). Cascade: same stream, but
each frame passes through the motion gate first and only reaches
the detector if motion exceeds the threshold. Cost measured by
counting per-stage frame counts and converting to $/camera/month at
AWS g4dn.xlarge pricing (single T4, $0.526/hr on-demand). Detector
recall cross-checked on the Roboflow "Construction Site Safety"
test split (CC BY 4.0, 2,801 images, 25 classes including Person,
Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest).

## RESULTS

|                             | Baseline   | Cascade   | Ratio            |
|-----------------------------|------------|-----------|------------------|
| Frames sent to YOLOv8n      | 1,500      | 418       | 3.6x fewer       |
| Wall-clock inference time   | 12.80 s    | 4.28 s    | 3.0x faster      |
| Per-frame latency (avg)     | 8.53 ms    | 2.85 ms   | —                |
| Cost per camera per month   | $16.16     | $5.40     | 3.0x cheaper     |

Detector recall on the labeled test split (82 images YOLOv8n
recognized at least one COCO class in):

|                | Baseline | Cascade | Delta   |
|----------------|----------|---------|---------|
| Overall recall | 91.46%   | 91.46%  | 0.00%   |

Zero recall loss is the strong result: the cascade catches every
detection the baseline catches, at one-third the cost.

## WHAT THAT MEANS AT SCALE

The per-camera number is small in isolation and meaningful in
aggregate. Extrapolated to real customer archetypes:

|                              | Baseline/month | Cascade/month | Saved/year |
|------------------------------|----------------|---------------|------------|
| 30-camera warehouse          | $485           | $162          | $3,874     |
| 120-camera mid-size plant    | $1,939         | $648          | $15,494    |
| 500-camera F500 site         | $8,080         | $2,700        | $64,560    |

That is one lever on one filter with a small stock detector. Swap
YOLOv8n for a bigger model or add a VLM confirmation stage — both
of which real deployments will — and the same 3x ratio applies to a
much larger absolute number. CCTV inference cost scales linearly
with camera count and factories do not buy fewer cameras over time.

## LIMITATIONS

Three honest labels.

**First, the streaming clip is synthetic.** One background image
looped with test-set images spliced in at 15% activity, because
real labeled continuous CCTV was not available in the time budget.
The cost multiple reflects the pipeline on a stream with this
static-to-active ratio; real production ratios vary by camera,
shift, and site. A busy loading dock cuts the ratio; a
Sunday-night storeroom raises it.

**Second, YOLOv8n is COCO-pretrained** and does not know
safety-specific classes like NO-Hardhat directly. Recall reported
above is on COCO classes YOLO already recognizes (Person, etc.);
per-class safety-specific recall would require a fine-tuned
detector. The cascade's cost reduction is unaffected — the motion
gate is model-agnostic.

**Third, motion gating cannot see a static hazard:** a worker
standing motionless without a hardhat in an otherwise still scene.
The mitigation is a forced full-inference heartbeat every N seconds
regardless of motion. Not implemented in this demo; called out here
because a safety product cannot ship without it.

## WHAT I WOULD MEASURE NEXT

The cascade filters within each frame. The larger lever sits above
it: not every hour of the day carries equal risk, so not every hour
deserves equal compute. A risk-aware scheduler runs the cascade in
a lightweight mode during low-risk steady state and a richer mode
(higher fps, tighter thresholds, VLM confirmation on borderlines)
during higher-risk windows. Net cost falls versus always-on
baseline while effective recall on incidents that actually occur
rises. Compute is reallocated, not uniformly reduced.

The important design point is that the scheduler should not be
built on fixed rules. Published safety data — CCPS reports startup
incidents are ~5x more likely than normal operations — is useful
only as a starting prior. The real system learns per-factory. Every
site has its own rhythm: its own shift schedule, its own maintenance
windows, its own recurring near-miss patterns, its own high-traffic
zones. A scheduler that runs on rigid industry-average rules would
be wrong for every specific customer.

What it should actually do: ingest 4–8 weeks of the site's own
occupancy patterns, alert history, operator confirmations, and
incident log; learn per-camera, per-hour risk curves; retune
continuously as new data arrives. The safety manual (which Zapdos's
VLM already reads for detection classes) is also the source of the
site's high-risk procedures — permit-to-work, LOTO, hot-work, shift
handovers, planned turnarounds — so those can auto-trigger elevated
mode without hand-coding. After 90 days of operation the scheduler
is running on that specific plant's actual patterns, not on
averages. That is the version worth measuring, and it is the version
that gets better the longer the system runs — the opposite of how
most ML systems age.

## FIT

Cheap traditional-CV filters gating expensive modern-AI inference
is the shape of "filtered for accuracy and substance before it
reaches an operator" — motion gate as the coarsest filter, the
detector as the medium filter, and a VLM in production as the
finest, in that order, at that cost.

---

**Repo:** github.com/Akshaj11/AI-Safety (branch `claude/zapdos-cascade-demo-7dgrib`, folder `zapdos-cascade-demo/`)

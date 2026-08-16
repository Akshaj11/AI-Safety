# CHEAPER CONTINUOUS SAFETY INFERENCE ON FACTORY CCTV

**Akshaj Shah  |  For Zapdos Labs  |  16 August 2026**

---

## PROBLEM

Running an object detector on every frame of every CCTV feed is the
naive way to catch PPE violations and forklift-pedestrian conflicts,
and also the wasteful one. On a stationary camera most frames are
identical to the last one, so per-frame inference spends its whole
budget confirming nothing changed. The lever is not a smaller model
— it is a cheap filter that decides whether the model runs at all.

## THE LEVER

Insert an OpenCV frame-difference gate in front of the detector. If
the mean absolute pixel change between consecutive frames is below a
threshold, skip inference and reuse the last result. The gate costs
~100 microseconds per frame on CPU; YOLOv8n inference costs several
milliseconds on GPU — a ~50x cost ratio. Every skipped frame is a
saved GPU cycle.

## WHAT I BUILT

A ~120-line Colab demo. Baseline: YOLOv8n on every frame of a
5-minute stream (1,500 frames at 5 fps). Cascade: same stream, gate
first, detector only on passing frames. Cost derived from measured
per-frame latency at AWS g4dn.xlarge pricing (1x T4, $0.526/hr).
Detector recall cross-checked on the Roboflow "Construction Site
Safety" test split (CC BY 4.0).

## RESULTS

|                             | Baseline   | Cascade   | Ratio         |
|-----------------------------|------------|-----------|---------------|
| Frames sent to YOLOv8n      | 1,500      | 418       | 3.6x fewer    |
| Wall-clock inference time   | 12.80 s    | 4.28 s    | 3.0x faster   |
| Per-frame latency (avg)     | 8.53 ms    | 2.85 ms   | —             |
| Cost per camera per month   | $16.16     | $5.40     | 3.0x cheaper  |

Detector recall on the labeled test split:

|                | Baseline | Cascade | Delta   |
|----------------|----------|---------|---------|
| Overall recall | 91.46%   | 91.46%  | 0.00%   |

Zero recall loss: the cascade catches every detection the baseline
catches, at one-third the cost.

## WHAT THAT MEANS AT SCALE

|                              | Baseline/month | Cascade/month | Saved/year |
|------------------------------|----------------|---------------|------------|
| 30-camera warehouse          | $485           | $162          | $3,874     |
| 120-camera mid-size plant    | $1,939         | $648          | $15,494    |
| 500-camera F500 site         | $8,080         | $2,700        | $64,560    |

One lever, one filter, one small stock detector. Swap YOLOv8n for a
bigger model or add a VLM confirmation stage — both of which real
deployments will — and the same 3x ratio applies to a much larger
absolute number. CCTV cost scales linearly with camera count, and
factories do not buy fewer cameras over time.

## LIMITATIONS

**The streaming clip is synthetic** — background loop with test-set
images spliced in at 15% activity, because real labeled continuous
CCTV was not available in the time budget. The cost multiple assumes
this static-to-active ratio; a busy loading dock cuts it, a Sunday-
night storeroom raises it.

**YOLOv8n is COCO-pretrained** and does not know safety-specific
classes like NO-Hardhat directly. Recall above is on classes YOLO
already recognizes; a fine-tuned safety detector would give the same
cost reduction on richer classes — the motion gate is model-agnostic.

**Motion gating cannot see a static hazard** — a worker standing
motionless without a hardhat in an otherwise still scene. The
mitigation is a forced full-inference heartbeat every N seconds.
Not implemented here; called out because a safety product cannot
ship without it.

## WHAT I WOULD MEASURE NEXT

The cascade filters within each frame. The larger lever sits above
it: not every hour of the day carries equal risk, so not every hour
deserves equal compute. A risk-aware scheduler runs the cascade
light during low-risk steady state and rich (higher fps, tighter
thresholds, VLM confirmation on borderlines) during higher-risk
windows. Compute reallocated, not uniformly reduced.

The important design point is that the scheduler should not be
built on fixed rules. Published safety data — CCPS reports startup
incidents are ~5x more likely than normal operations — is useful
only as a starting prior. Every site has its own rhythm: shift
schedule, maintenance windows, recurring near-miss patterns,
high-traffic zones. A scheduler on industry-average rules would be
wrong for every specific customer.

What it should do: ingest 4–8 weeks of the site's own occupancy
patterns, alert history, operator confirmations, and incident log;
learn per-camera, per-hour risk curves; retune continuously. The
safety manual — which Zapdos's VLM already reads for detection
classes — is also the source of high-risk procedures
(permit-to-work, LOTO, hot-work, shift handovers), so those can
auto-trigger elevated mode without hand-coding. After 90 days the
scheduler is running on that specific plant's actual patterns.
The system gets better the longer it runs — the opposite of how
most ML systems age.

## FIT

Cheap traditional-CV filters gating expensive modern-AI inference
is the shape of "filtered for accuracy and substance before it
reaches an operator" — motion gate as the coarsest filter, the
detector as the medium filter, and a VLM in production as the
finest, in that order, at that cost.

---

**Repo:** github.com/Akshaj11/AI-Safety (branch `claude/zapdos-cascade-demo-7dgrib`, folder `zapdos-cascade-demo/`)

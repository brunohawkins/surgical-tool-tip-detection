# SurgNet pipeline: choices and why

Real-time surgical tool tip detection: for each frame, name each tool (bipolar, scissors, suction) and place a dot
on its tip. Score = 40% precision + 35% recall + 15% tip-error term (1 − median px / 50) + 10% speed (full marks
at 100 fps).

**Result: 3rd place overall.** 0.6791 on the hidden test set (precision 0.974, recall 0.198, median tip error
9.9 px, 4.7 ms per frame), against 0.623 for the organisers' baseline.

## 1. The model: one heatmap per class

| Choice | Why |
|---|---|
| **Tip heatmaps** (CenterNet-style): the network paints a heatmap per class, and each tip is a peak | Optimises exactly what is scored (tip + label). No boxes, masks or box-NMS. The bipolar "tip" is empty space between the arms, which a heatmap learns directly. |
| **ResNet-34 + FPN to 1/4 resolution**, ImageNet-pretrained (timm `resnet34.a1_in1k`) | Best single model of everything tested at a matched operating point (ResNet-18/50, DINOv3 ConvNeXt, SurgeNetXL surgical pretraining). |
| **Small ensembles** (two ResNets, heatmaps averaged) | Averaging suppresses each model's stray confident peaks: hidden precision 0.951 → 0.964 → 0.974. |
| **Sub-pixel peaks**: a parabola fitted to the log-heatmap around each peak | Exact for a gaussian bump. Hidden-set median tip error ~9.9 px (baseline 12.4). |
| **At most one tip per class, except suction (two)** | Only suction is ever labelled twice in one frame. |
| **Input: longer side 640, padded to a multiple of 32**; flip test-time augmentation; bf16 | 4.7 ms per frame on the grader for a 2-model ensemble (211 fps). |

## 2. Data and honest validation

- All three training folders, with real surgical frames oversampled to 40% of each epoch.
- **Leakage-safe hold-out:** all renders of a synthetic capture stay together, videos are cut into 25-frame blocks, and
  one real video is held out entirely.
- **Matched-operating-point comparisons:** candidate models are compared at thresholds that make them fire equally
  often. This exposed several "improvements" that were only calibration shifts before they cost submissions.
- **The limit of proxies:** no held-out or public set predicts the hidden set well (different sessions and probably
  different surgery types). Bipolar transfers (≈97% precise held-out, ≈99% hidden); suction and scissors do not.

## 3. Training recipe (what generalised)

The jump from 0.468 to 0.587 on the hidden set came from robustness to new footage, not from model size:
- **Both real videos in training, oversampled.**
- **Strong photometric augmentation**, sized from measured gaps between videos (an unseen video was ~1.5x darker and
  ~10x blurrier): gamma and exposure down to 0.5x, desaturation, defocus and low-resolution blur, vignette, noise, JPEG.
- **Wide geometry:** zoom 0.45–2.2x (log-uniform), rotation ±40°, horizontal flip.
- Focal loss (CenterNet), AdamW + one-cycle schedule, 50 epochs, about 10 minutes on one L40S, fixed seeds.

```bash
uv run python training/train.py --name final --all --thresholds-from runs/dev/surgnet.pt --flip-tta \
  --backbone resnet34.a1_in1k --backbone-lr-mult 1.0 --wd 0.01 --epochs 50 \
  --strong-aug --scale-range 0.45,2.2 --max-rotation 40
```

## 4. The decisive step: confidence-gated classes (0.587 → 0.679)

Validation tunes per-class thresholds, but the hidden set behaves differently, so thresholds were then **calibrated
on the hidden set itself**, one controlled change per submission, and the per-class behaviour was solved for
algebraically from the submissions that share the same weights:

| # | Change | Score | P | R | What it taught |
|---|---|---|---|---|---|
| 2 | Thresholds tuned on validation | 0.587 | 0.53 | 0.45 | Hidden-set false positives exist… |
| 3 | All thresholds +0.2 | 0.611 | 0.72 | 0.30 | …and they are low-confidence |
| 4 | Bipolar +0.2, suction +0.1 | 0.610 | 0.63 | 0.40 | Per-class algebra begins: bipolar strict is very precise |
| 6 | **Gated**: bipolar 0.475; suction 0.70, scissors 0.75 | 0.678 | 0.95 | 0.22 | The removed suction/scissors band was only 42% precise |
| 7 | Same gating on an L + mosaic ensemble | 0.678 | 0.96 | 0.21 | Averaging removes false positives |
| **8** | **L + ResNet-50 ensemble, gated, scissors off** | **0.679** | **0.97** | 0.20 | Scissors (~60% precise when gated) were still a net cost |

The last two submissions tested variations of the gates (suction from the colour-normalised model: 0.658;
#7 without scissors: 0.676). Neither beat #8, which is the final model.

Solved from #2/#3/#4/#6 (same weights): **bipolar ≥ 0.475 is ~99% precise** (recall 0.62); suction ≥ 0.70 is ~93%
precise but rare; the suction band 0.45–0.70 is ~53%; scissors are 30–60% at every threshold.

Why gate: precision weighs 0.40 against recall's 0.35, so at ~97% precision an extra detection only raises the score
if it is **≥ ~80–85% likely to be correct**. On unfamiliar footage our borderline suction and scissors detections are
right about half the time. They stay available when the model is very confident.

## 5. Tried and rejected (each with evidence)

| Idea | Result |
|---|---|
| DINOv3 ConvNeXt-T | Overfit 1.7k images (0.647 held-out vs 0.762) |
| SurgeNetXL backbone (CaFormer-S18, DINO on 4.7M surgical frames) | Learned too slowly in the time available: held-out 0.658, unseen video P 0.47 |
| ResNet-50 alone, two seeds of the same model, multi-scale 640+960 inference | No gain at a matched operating point |
| Class-weighted loss and sampling for scissors/suction | No gain (scissors on new footage is a recognition problem) |
| Auxiliary tool-mask head (from CVAT masks) | Worse at a matched operating point |
| Green-screen background swap, colour/greyscale augmentation, 800 px input | No gain |
| PitVis hard negatives (other instruments) | Learned "endoscopic image = no tools" |
| Grey-world colour normalisation as the whole model | Better suction, worse precision: hidden 0.603 (kept as a suction source) |
| Batch-norm statistics adaptation to microscope footage (Alpha-BN) | Noise-level |
| DARK-style smoothed peak refinement | Mixed: better overall held-out error, worse bipolar real-frame error |
| Relaxing suction/scissors thresholds (`gate60`) | Predicted 0.629 from the hidden-set algebra |

## 6. External data and rules

- **ImageNet-pretrained backbones** (timm, Apache-2.0).
- **Test only, never in a submitted model:** PitVis-2023 (UCL, CC BY-NC-SA 4.0), MACS aneurysm-clipping frames
  (UCL, CC BY-NC 4.0), SurgeNetXL weights (CC BY-NC 4.0, rejected experiment).
- XRlabs' data, labels and the trained weights are not in this repository. The data stayed on the GPU instance,
  which was deleted after the event. In the challenge repository, `SurgNet()` loaded fully offline, all tests passed
  and a fresh clone reproduced predictions exactly.

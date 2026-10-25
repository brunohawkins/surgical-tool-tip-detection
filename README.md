# Surgical tool tip detection in real time (SurgNet)

**3rd place overall** at the
[XRlabs Real-Time Vision Challenge](https://ukhackathons.com/events/real-time-vision-challenge-with-xrlabs-nvidia-x-playground-x-oxford-seed-fund)
with NVIDIA, Playground Global and Oxford Seed Fund, London, 24 October 2026. Solo entry, built in one day.

SurgNet looks at a frame of surgical video, names each instrument in view (bipolar forceps, suction or scissors)
and puts a dot on its tip, in under 5 ms per frame.

## Results on the hidden test set

| | |
|---|---|
| Leaderboard score | **0.6791** (organisers' baseline: 0.623) |
| Precision | 0.974 |
| Recall | 0.198 |
| Median tip error | 9.9 px |
| Speed | 4.74 ms per frame (211 fps) on an NVIDIA RTX PRO 6000 |

The score is 40% precision, 35% recall, 15% tip error and 10% speed (full marks at 100 fps).

## How it works

- **Data engineering.** Real surgical frames are only 28% of the 1,741 training frames, so they are oversampled to
  40% of each epoch. A whole real video is held out for validation, so near-identical frames never leak between
  training and testing. Augmentation is sized from measured differences between videos (the unseen one was about
  1.5× darker and 10× blurrier): exposure and gamma, defocus and low-resolution blur, desaturation, vignette, sensor
  noise, JPEG compression, zoom 0.45–2.2× and rotation ±40°.
- **Architecture.** CenterNet-style keypoint heat maps, one per tool, from an ImageNet-pretrained ResNet with a
  feature pyramid at 1/4 resolution, trained with focal loss. A parabola fit on the log heat map places each tip
  between grid cells. ResNet-34 beat ResNet-18, ResNet-50, DINOv3 ConvNeXt-T and SurgeNetXL at a matched
  operating point.
- **Tuning.** Precision outweighs recall, so an extra detection only helps if it is at least ~80% likely to be
  right. Each tool has its own confidence gate, calibrated by solving for per-tool precision across submissions of
  the same model: bipolar forceps report freely, suction only when very confident, scissors are off. The final
  model averages a ResNet-34 and a ResNet-50, each with flip test-time augmentation.

[PIPELINE.md](PIPELINE.md) has every choice, the evidence behind it, the training command and what didn't work.

## What's here

| Path | Contents |
|---|---|
| [`src/surgnet/inference/model.py`](src/surgnet/inference/model.py) | TipNet (backbone + feature pyramid → per-class heat maps), preprocessing, peak decoding with sub-pixel refinement |
| [`src/surgnet/inference/surgnet.py`](src/surgnet/inference/surgnet.py) | `SurgNet`: loads the ensemble, per-class routing and confidence gates; `predict(frame)` |
| [`training/train.py`](training/train.py) | Training: domain-balanced sampling, leakage-safe split, augmentation, focal loss, threshold tuning |
| [`training/match_rates.py`](training/match_rates.py) | Thresholds that make a candidate fire as often as a reference model, for fair comparisons |
| [`training/retune.py`](training/retune.py) | Sets per-class thresholds in a checkpoint |
| [`training/stress_test.py`](training/stress_test.py) | Robustness checks: frame sizes from 40×30 to 4K, blank and noise frames |
| [`training/eval_pitvis.py`](training/eval_pitvis.py), [`training/pitvis_negatives.py`](training/pitvis_negatives.py) | Tests on public PitVis footage, and a (rejected) hard-negative experiment |
| [`DATASETS.md`](DATASETS.md) | External data and licences |

## Not included

- **The challenge's interface and scoring package** (`surgnet.shared`, `surgnet.labelio`, `surgnet.scoring`,
  `surgnet.benchmark`), which belongs to the organisers. The code here imports it, so it runs inside the challenge
  repository rather than on its own.
- **XRlabs' training data and labels** (under NDA), and **the trained weights**, which were trained on that data.
- Dataset-specific video names in `training/train.py` are replaced with placeholders (`videoA`, `videoB`).

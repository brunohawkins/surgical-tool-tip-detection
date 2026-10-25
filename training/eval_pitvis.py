"""Presence check on open real surgery: PitVis-2023 (UCL, CC BY-NC-SA 4.0), endoscopic pituitary operations.

PitVis labels up to two visible instruments once per second. Three of its 18 instruments are ours (bipolar_forceps,
dural_scissors, suction); the other fifteen are distractors. There are no tip coordinates, so this scores presence
per frame: a class counts as predicted when SurgNet returns at least one tip of it. Test only, never trained on.

    uv run python training/eval_pitvis.py --videos ~/pitvis --weights runs/X/surgnet.pt [--dump DIR]

Data: https://doi.org/10.5522/04/26531686 (video_NN.mp4 + annotations_NN.csv).
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2

from surgnet.inference.surgnet import WEIGHTS_FILE, SurgNet
from surgnet.shared.detection import CLASSES

OURS = {1: 'bipolar', 4: 'scissors', 16: 'suction'}       # PitVis instrument id -> our class
OUT_OF_PATIENT, NONE_VISIBLE = -1, 0
COLOURS = {'bipolar': (255, 160, 0), 'scissors': (0, 220, 255), 'suction': (60, 255, 60)}


def sample_rows(csv_path: Path, n: int, rng: random.Random) -> list:
    """[(second, set of our classes, has distractor)]: half with our tools, half without, inside the patient only."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ids = {int(r['int_instrument1']), int(r['int_instrument2'])}
            if OUT_OF_PATIENT in ids:
                continue
            ours = {OURS[i] for i in ids if i in OURS}
            rows.append((int(r['int_time']), ours, any(i > 0 and i not in OURS for i in ids)))
    with_ours = [r for r in rows if r[1]]
    without = [r for r in rows if not r[1]]
    return sorted(rng.sample(with_ours, min(n // 2, len(with_ours))) + rng.sample(without, min(n // 2, len(without))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--videos', type=Path, required=True, help='folder with video_NN.mp4 and annotations_NN.csv')
    p.add_argument('--weights', type=Path, default=WEIGHTS_FILE)
    p.add_argument('--per-video', type=int, default=300)
    p.add_argument('--dump', type=Path, default=None, help='write overlays of a few frames here')
    p.add_argument('--dump-n', type=int, default=24)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    model = SurgNet(weights=args.weights)
    rng = random.Random(args.seed)
    tp, fp, fn = Counter(), Counter(), Counter()
    distractor_frames = distractor_alarms = empty_frames = empty_alarms = 0
    dumps = []
    for video in sorted(args.videos.expanduser().glob('video_*.mp4')):
        ann = video.with_name(video.name.replace('video_', 'annotations_').replace('.mp4', '.csv'))
        cap = cv2.VideoCapture(str(video))
        for second, truth, has_distractor in sample_rows(ann, args.per_video, rng):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            dets = model.predict(frame)
            pred = {d.class_name for d in dets}
            for c in CLASSES:
                tp[c] += c in pred and c in truth
                fp[c] += c in pred and c not in truth
                fn[c] += c not in pred and c in truth
            if not truth and has_distractor:
                distractor_frames += 1
                distractor_alarms += bool(pred)
            if not truth and not has_distractor:
                empty_frames += 1
                empty_alarms += bool(pred)
            if args.dump and rng.random() < 0.1 and len(dumps) < args.dump_n:
                dumps.append((f'{video.stem}_{second:05d}', frame, dets, truth))
        cap.release()

    report = {}
    for c in list(CLASSES) + ['overall']:
        t = sum(tp.values()) if c == 'overall' else tp[c]
        f_p = sum(fp.values()) if c == 'overall' else fp[c]
        f_n = sum(fn.values()) if c == 'overall' else fn[c]
        prec = t / (t + f_p) if t + f_p else 0.0
        rec = t / (t + f_n) if t + f_n else 0.0
        report[c] = {'precision': round(prec, 3), 'recall': round(rec, 3), 'tp': t, 'fp': f_p, 'fn': f_n}
        print(f'{c:9s} presence P {prec:.3f}  R {rec:.3f}  (tp {t} fp {f_p} fn {f_n})')
    report['false_alarm_on_other_instruments'] = distractor_alarms / max(1, distractor_frames)
    report['false_alarm_on_no_instrument'] = empty_alarms / max(1, empty_frames)
    print(f'false alarms: {distractor_alarms}/{distractor_frames} frames with only other instruments, '
          f'{empty_alarms}/{empty_frames} frames with no instrument')
    print(json.dumps(report))

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for name, frame, dets, truth in dumps:
            for d in dets:
                x, y = int(d.tip_xy[0]), int(d.tip_xy[1])
                cv2.circle(frame, (x, y), 9, COLOURS[d.class_name], 3)
                cv2.putText(frame, f'{d.class_name} {d.confidence:.2f}', (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            COLOURS[d.class_name], 2)
            cv2.putText(frame, 'truth: ' + (', '.join(sorted(truth)) or 'none of ours'), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(str(args.dump / f'{name}.jpg'), cv2.resize(frame, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    main()

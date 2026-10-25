"""Extract hard-negative frames from PitVis-2023 (UCL, CC BY-NC-SA 4.0): real surgery with none of our three tools.

Kept: seconds inside the patient where PitVis labels no instrument, or exactly one instrument that is not bipolar
forceps, dural scissors or suction. These teach the model what is not one of our tools (other instruments, tissue,
glare) on real footage. Use different videos from the ones eval_pitvis.py tests on.

    uv run python training/pitvis_negatives.py --videos ~/pitvis_train --out ~/pitvis_neg --per-video 150
"""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

import cv2

OURS = {1, 4, 16}                                  # bipolar_forceps, dural_scissors, suction
OUT_OF_PATIENT, NO_SECOND = -1, -2


def clean_seconds(csv_path: Path) -> list:
    out = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            a, b = int(r['int_instrument1']), int(r['int_instrument2'])
            if a == OUT_OF_PATIENT or a in OURS or b in OURS:
                continue
            if a == 0 or b == NO_SECOND:            # nothing visible, or a single other instrument
                out.append(int(r['int_time']))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='frames go to <out>/pitvis/')
    p.add_argument('--per-video', type=int, default=150)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    dest = args.out.expanduser() / 'pitvis'
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in sorted(args.videos.expanduser().glob('video_*.mp4')):
        seconds = clean_seconds(video.with_name(video.name.replace('video_', 'annotations_').replace('.mp4', '.csv')))
        seconds = sorted(rng.sample(seconds, min(args.per_video, len(seconds))))
        cap = cv2.VideoCapture(str(video))
        for s in seconds:
            cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(dest / f'{video.stem}_{s:05d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                total += 1
        cap.release()
        print(f'{video.name}: {len(seconds)} negative seconds sampled', flush=True)
    print(f'wrote {total} frames to {dest}')


if __name__ == '__main__':
    main()

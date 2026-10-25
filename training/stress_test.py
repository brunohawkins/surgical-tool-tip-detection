"""Robustness profile: official metrics on a run's held-out frames under perturbations the hidden set might contain.

    uv run python training/stress_test.py --weights runs/F_tta/surgnet.pt --val-run F_resnet34 [--no-second]

Each perturbation is applied to the frame (and the ground truth rescaled with it, so the match radius follows the
image diagonal exactly as the benchmark does). --no-second disables second tips of the same class.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from surgnet.inference.surgnet import SurgNet
from surgnet.labelio import load_cvat_xml
from surgnet.labelio.cvat import GroundTruthTip
from surgnet.scoring import compute_metrics, match_tips
from surgnet.shared.images import read_image

ERROR_CEILING_PX = 50.0


def perturb(img: np.ndarray, kind: str) -> tuple[np.ndarray, float, float]:
    """Returns (image, sx, sy): the scale applied to coordinates."""
    h, w = img.shape[:2]
    if kind.startswith('size'):                               # size480x270, size1344x1080: resize to that size
        tw, th = map(int, kind[4:].split('x'))
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA), tw / w, th / h
    if kind == 'blur':
        return cv2.GaussianBlur(img, (0, 0), 4.0), 1.0, 1.0
    if kind == 'dark':
        return cv2.convertScaleAbs(img, alpha=0.6, beta=-10), 1.0, 1.0
    if kind == 'jpeg30':
        return cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_COLOR), 1.0, 1.0
    if kind == 'center_crop':                                 # zoomed-in view: middle 60%
        x0, y0 = int(w * 0.2), int(h * 0.2)
        return img[y0:h - y0, x0:w - x0].copy(), 1.0, 1.0
    return img, 1.0, 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--val-run', required=True)
    p.add_argument('--data', type=Path, default=Path('~/data/train/images'))
    p.add_argument('--labels', type=Path, default=Path('~/data/train/annotations.xml'))
    p.add_argument('--kinds', default='none,size480x270,size1344x1080,blur,dark,jpeg30,center_crop')
    p.add_argument('--no-second', action='store_true')
    args = p.parse_args()

    model = SurgNet(weights=args.weights)
    if args.no_second:
        model.thresholds = [(t1, 1.01) for t1, _ in model.thresholds]
    gt = load_cvat_xml(args.labels.expanduser())
    names = [n for n in (Path('runs') / args.val_run / 'val_names.txt').read_text().split('\n') if n]
    images = {n: read_image(args.data.expanduser() / n) for n in names}
    for kind in args.kinds.split(','):
        per_image, n_pred = [], 0
        for n, img0 in images.items():
            img, sx, sy = perturb(img0, kind)
            h, w = img.shape[:2]
            if kind == 'center_crop':
                x0, y0 = int(img0.shape[1] * 0.2), int(img0.shape[0] * 0.2)
                tips = [GroundTruthTip(t.class_name, (t.tip_xy[0] - x0, t.tip_xy[1] - y0)) for t in gt[n].tips
                        if 0 <= t.tip_xy[0] - x0 < w and 0 <= t.tip_xy[1] - y0 < h]
            else:
                tips = [GroundTruthTip(t.class_name, (t.tip_xy[0] * sx, t.tip_xy[1] * sy)) for t in gt[n].tips]
            dets = model.predict(img)
            n_pred += len(dets)
            per_image.append(match_tips(dets, tips, 0.025 * math.hypot(w, h)))
        o = compute_metrics(per_image)['overall']
        err = o['median_error_px']
        score = 0.4 * o['precision'] + 0.35 * o['recall'] + 0.15 * (max(0.0, 1 - err / ERROR_CEILING_PX) if err else 0) + 0.1
        print(f'{kind:14s} score {score:.3f}  P {o["precision"]:.3f}  R {o["recall"]:.3f}  err {err and round(err, 1)}  '
              f'preds {n_pred}  tp {o["tp"]} fp {o["fp"]} fn {o["fn"]}', flush=True)


if __name__ == '__main__':
    main()

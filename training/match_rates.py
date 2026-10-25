"""Carry a threshold setting proven on the hidden set from one model to another (e.g. an ensemble).

Confidence scales differ between models, so the same number would mean a different operating point. Instead, per
class, the candidate's threshold is chosen so that it fires as often on held-out frames as the reference model does at
its (offset) thresholds. Writes a thresholds-only file for retune.py --thresholds-from.

    uv run python training/match_rates.py --ref runs/L_dev/surgnet.pt --offsets bipolar=0.2,scissors=0.1,suction=0.15 \
        --cand runs/L_dev/surgnet.pt runs/M_dev/surgnet.pt --val-run L_dev --out runs/thr_LM.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from surgnet.inference.model import TipNet
from surgnet.inference.surgnet import resolve_device
from surgnet.shared.detection import CLASSES
from surgnet.shared.images import read_image
from retune import members_of
from train import predict_peaks


def load(paths: list, device) -> list:
    nets = []
    for path in paths:
        for m in members_of(path):
            net = TipNet(m['config']['backbone'], width=int(m['config']['width']))
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in m['state_dict'].items()})
            nets.append(net.to(device).to(memory_format=torch.channels_last))
    return nets


def matched(ref_scores: np.ndarray, ref_t: float, cand_scores: np.ndarray) -> float:
    """Candidate threshold firing on the same number of frames as ref_scores >= ref_t."""
    k = int((ref_scores >= ref_t).sum())
    if k == 0:
        return 1.01
    return float(np.sort(cand_scores)[::-1][min(k, len(cand_scores)) - 1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ref', type=Path, required=True)
    p.add_argument('--offsets', default='bipolar=0.2,scissors=0.1,suction=0.15')
    p.add_argument('--cand', type=Path, nargs='+', required=True)
    p.add_argument('--val-run', required=True)
    p.add_argument('--data', type=Path, default=Path('~/data/train/images'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    device = resolve_device(None)
    names = [n for n in (Path('runs') / args.val_run / 'val_names.txt').read_text().split('\n') if n]
    images = {n: read_image(args.data.expanduser() / n) for n in names}
    ref_ck = torch.load(args.ref, map_location='cpu', weights_only=True)
    off = {k: float(v) for k, v in (kv.split('=') for kv in args.offsets.split(','))}
    norm = lambda path: bool(members_of(path)[0]['config'].get('normalise', False))
    ref = predict_peaks(load([args.ref], device), images, 640, device, True, norm(args.ref))
    cand = predict_peaks(load(args.cand, device), images, 640, device, True, norm(args.cand[0]))
    thresholds = {}
    for c, cls in enumerate(CLASSES):
        t1, t2 = ref_ck['thresholds'][cls]
        t1, t2 = min(0.95, t1 + off.get(cls, 0.0)), min(1.01, t2 + off.get(cls, 0.0))
        r1 = np.array([ref[n][0][c, 0] for n in names]); c1 = np.array([cand[n][0][c, 0] for n in names])
        r2 = np.array([ref[n][0][c, 1] for n in names]); c2 = np.array([cand[n][0][c, 1] for n in names])
        thresholds[cls] = [round(matched(r1, t1, c1), 4), 1.01 if t2 > 1 else round(matched(r2, t2, c2), 4)]
        print(f'{cls:9s} reference {t1:.3f}/{t2:.3f} fires on {(r1 >= t1).sum()} frames -> candidate {thresholds[cls]}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'thresholds': thresholds}, args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()

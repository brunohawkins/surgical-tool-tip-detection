"""Tune thresholds for one model or an ensemble on a run's held-out frames, or assemble final models into one file.

Tune (one checkpoint, or several for an ensemble; optional flip TTA):
    uv run python training/retune.py --ckpt runs/A/surgnet.pt runs/B/surgnet.pt --val-run A --flip-tta --out runs/AB/surgnet.pt

Assemble models trained on all data, with thresholds tuned on their development twins:
    uv run python training/retune.py --ckpt runs/A_all/surgnet.pt runs/B_all/surgnet.pt --thresholds-from runs/AB/surgnet.pt \
        --flip-tta --out runs/AB_all/surgnet.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

from surgnet.inference.model import MIN_DIST_FRAC, TipNet
from surgnet.inference.surgnet import resolve_device
from surgnet.labelio import load_cvat_xml
from surgnet.shared.detection import CLASSES
from surgnet.shared.images import read_image
from train import Tuner, predict_peaks


def members_of(path: Path) -> list:
    """[{'config', 'state_dict'}] for a single-model or an ensemble checkpoint."""
    ck = torch.load(path, map_location='cpu', weights_only=True)
    return ck.get('members') or [{'config': ck['config'], 'state_dict': ck['state_dict']}]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=Path, nargs='+', required=True)
    p.add_argument('--val-run', help='tune on runs/<val-run>/val_names.txt')
    p.add_argument('--thresholds-from', type=Path, help='copy thresholds instead of tuning')
    p.add_argument('--data', type=Path, default=Path('~/data/train/images'))
    p.add_argument('--labels', type=Path, default=Path('~/data/train/annotations.xml'))
    p.add_argument('--flip-tta', action='store_true')
    p.add_argument('--refine-sigma', type=float, default=0.0, help='DARK-style smoothing for sub-pixel refinement')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not (args.val_run or args.thresholds_from):
        p.error('give --val-run or --thresholds-from')

    members = [m for path in args.ckpt for m in members_of(path)]
    sizes = {int(m['config']['input_size']) for m in members}
    if len(sizes) != 1:
        p.error(f'members disagree on input size: {sizes}')
    config = {'input_size': sizes.pop(), 'flip_tta': args.flip_tta, 'min_dist_frac': MIN_DIST_FRAC,
              'backbone': members[0]['config']['backbone'], 'width': members[0]['config']['width'],
              'normalise': bool(members[0]['config'].get('normalise', False)), 'refine_sigma': args.refine_sigma,
              'members': [str(c) for c in args.ckpt]}
    val = {}
    if args.thresholds_from:
        src = torch.load(args.thresholds_from, map_location='cpu', weights_only=True)
        thresholds = [list(src['thresholds'][c]) for c in CLASSES]
    else:
        device = resolve_device(None)
        nets = []
        for m in members:
            net = TipNet(m['config']['backbone'], width=int(m['config']['width']))
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in m['state_dict'].items()})
            nets.append(net.to(device).to(memory_format=torch.channels_last))
        gt = load_cvat_xml(args.labels.expanduser())
        data = args.data.expanduser()
        names = [n for n in (Path('runs') / args.val_run / 'val_names.txt').read_text().split('\n') if n]
        tuner = Tuner(predict_peaks(nets, {n: read_image(data / n) for n in names}, config['input_size'], device,
                                    args.flip_tta, config['normalise']), gt)
        thresholds, val = tuner.tune()
        val['by_folder'] = tuner.by_folder(thresholds)
        print(f'proxy {val["proxy"]:.4f}  P {val["precision"]:.3f}  R {val["recall"]:.3f}  err {val["median_error_px"]}  '
              f'thresholds {dict(zip(CLASSES, thresholds))}')
        for f, s in val['by_folder'].items():
            print(f'  {f:16s} proxy {s["proxy"]:.4f}  P {s["precision"]:.3f}  R {s["recall"]:.3f}  err {s["median_error_px"]}')
    out = {'config': config, 'thresholds': dict(zip(CLASSES, thresholds)), 'val': val}
    if len(members) == 1:
        out['state_dict'] = members[0]['state_dict']
    else:
        out['members'] = members
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    if args.val_run:
        (args.out.parent / 'val_names.txt').write_text((Path('runs') / args.val_run / 'val_names.txt').read_text())
    print(f'saved {args.out} ({len(members)} model{"s" if len(members) > 1 else ""})')


if __name__ == '__main__':
    main()

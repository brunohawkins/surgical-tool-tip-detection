"""Train TipNet and write a checkpoint (weights + config + per-class thresholds) that SurgNet loads.

Development run (holds out data, tunes thresholds on it):
    uv run python training/train.py --name dev --data ~/data/train/images --labels ~/data/train/annotations.xml

Final run (trains on everything, reuses the development run's thresholds):
    uv run python training/train.py --name final --all --thresholds-from runs/dev/surgnet.pt --data ... --labels ...

The hold-out never splits near-identical frames: it is the real video(s) named by --holdout, plus --val-frac of the
remaining units of every folder, where a unit is every render of one synthetic capture (greenbackground/000030_*) or a
block of BLOCK consecutive frames of a video (phantom/frame_*, real_surgical/videoA_*).
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from surgnet.benchmark.runner import DEFAULT_RADIUS_FRAC, ERROR_CEILING_PX, LEADERBOARD_WEIGHTS
from surgnet.inference.model import (MIN_DIST_FRAC, STRIDE, TipNet, decode, ensemble_logits, fit, normalise, to_tensor)
from surgnet.inference.surgnet import resolve_device
from surgnet.labelio import load_cvat_xml
from surgnet.scoring import match_points
from surgnet.shared.detection import CLASSES
from surgnet.shared.images import find_images, read_image

SIGMA = 2.0                                     # heatmap gaussian, in cells
BLOCK = 25                                      # consecutive video frames that stay on one side of the split
SECOND_TIP_CLASSES = ('suction',)               # the only class ever labelled twice in one frame


def folder(name: str) -> str:
    return name.split('/')[0]


def capture_key(name: str) -> str:
    """'greenbackground/000030_L_12.jpg' -> 'greenbackground/000030' (one capture, many renders);
    'real_surgical/videoA_000120.jpg' -> 'real_surgical/videoA' and 'phantom/frame_020064.PNG' -> 'phantom/frame' (videos)."""
    return f'{folder(name)}/{name.rsplit("/", 1)[-1].split("_")[0]}'


def units(names: list) -> dict:
    """{name: split unit}. A synthetic capture is one unit; a video is cut into blocks of BLOCK consecutive frames."""
    out, videos = {}, defaultdict(list)
    for n in names:
        key = capture_key(n)
        if key.split('/')[1].isdigit():
            out[n] = key
        else:
            videos[key].append(n)
    for key, frames in videos.items():
        frames.sort(key=lambda n: int(re.findall(r'\d+', n.rsplit('/', 1)[-1])[-1]))
        for i, n in enumerate(frames):
            out[n] = f'{key}#{i // BLOCK}'
    return out


def split(names: list, holdout: list, val_frac: float, seed: int) -> tuple[list, list]:
    """Hold out whole videos named in `holdout`, plus val_frac of the remaining units of every folder."""
    unit = units(names)
    by_folder = defaultdict(set)
    for n in names:
        by_folder[folder(n)].add(unit[n])
    val, rng = {u for u in unit.values() if u.split('#')[0] in holdout}, random.Random(seed)
    for f in sorted(by_folder):
        keys = sorted(by_folder[f] - val)
        rng.shuffle(keys)
        val.update(keys[:round(len(keys) * val_frac)])
    return [n for n in names if unit[n] not in val], [n for n in names if unit[n] in val]


def load_cache(paths: list, max_side: int, workers: int) -> list:
    """[(BGR image downscaled so its longer side is at most max_side, (sx, sy) from original pixels)]."""
    def load(path):
        img = read_image(path)
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return img, (img.shape[1] / w, img.shape[0] / h)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(load, paths))


def decode_mask(m: ET.Element, height: int, width: int) -> np.ndarray:
    """CVAT 1.1 <mask>: run lengths alternating background/foreground, row-major inside its left/top/width/height box."""
    left, top, w, h = (int(float(m.get(k))) for k in ('left', 'top', 'width', 'height'))
    flat, pos, val = np.zeros(w * h, np.uint8), 0, 0
    for run in (int(c) for c in m.get('rle').split(',')):
        flat[pos:pos + run] = val
        pos, val = pos + run, val ^ 1
    full = np.zeros((height, width), np.uint8)
    full[top:top + h, left:left + w] = flat.reshape(h, w) * 255
    return full


def green_screen_assets(labels: Path, names: list, cache: list) -> tuple[list, list]:
    """For background swapping: per training frame, the dilated tool mask of green-screen frames (None otherwise), at
    cache resolution; plus the cached tool-free real/phantom frames (no masks, no points) to use as backgrounds."""
    index = {n: i for i, n in enumerate(names)}
    keep, backgrounds = [None] * len(names), []
    for im in ET.parse(str(labels)).getroot().iter('image'):
        i = index.get(im.get('name'))
        if i is None:
            continue
        masks, points = im.findall('mask'), im.findall('points')
        if folder(names[i]) == 'greenbackground':
            H, W = int(im.get('height')), int(im.get('width'))
            union = np.zeros((H, W), np.uint8)
            for m in masks:
                union |= decode_mask(m, H, W)
            union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
            ch, cw = cache[i][0].shape[:2]
            keep[i] = cv2.resize(union, (cw, ch), interpolation=cv2.INTER_NEAREST)
        elif not masks and not points:
            backgrounds.append(cache[i][0])
    return keep, backgrounds


def class_masks(labels: Path, names: list, cache: list) -> list:
    """Per training frame, a (h, w, 3) uint8 stack of tool masks by class at cache resolution (zeros if none)."""
    index = {n: i for i, n in enumerate(names)}
    out = [np.zeros(c[0].shape[:2] + (len(CLASSES),), np.uint8) for c in cache]
    for im in ET.parse(str(labels)).getroot().iter('image'):
        i = index.get(im.get('name'))
        if i is None or not im.findall('mask'):
            continue
        H, W = int(im.get('height')), int(im.get('width'))
        full = np.zeros((H, W, len(CLASSES)), np.uint8)
        for m in im.findall('mask'):
            c = CLASSES.index(m.get('label'))
            full[..., c] |= decode_mask(m, H, W)
        ch, cw = cache[i][0].shape[:2]
        out[i] = cv2.resize(full, (cw, ch), interpolation=cv2.INTER_NEAREST)
    return out


def swap_background(img: np.ndarray, keep: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Replace green pixels away from the tool with a random crop of a tool-free surgical frame. The tools reflect the
    green screen, so pixels inside the dilated tool mask are never replaced however green they are."""
    h, w = img.shape[:2]
    bh, bw = bg.shape[:2]
    cw = int(min(bw, bh * w / h) * random.uniform(0.5, 1.0))
    ch = int(cw * h / w)
    x0, y0 = random.randint(0, bw - cw), random.randint(0, bh - ch)
    patch = cv2.resize(bg[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        patch = patch[:, ::-1]
    green = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (35, 60, 0), (90, 255, 255))
    alpha = cv2.GaussianBlur(cv2.bitwise_and(green, cv2.bitwise_not(keep)), (5, 5), 0)[..., None] / 255.0
    return (img * (1 - alpha) + patch * alpha).astype(np.uint8)


class TipDataset(Dataset):
    """Optional green-screen swap, then one random affine (scale, rotation, shift, h-flip) straight into a square
    canvas, then photometric noise."""

    def __init__(self, cache: list, tips: list, size: int, keep: list = None, backgrounds: list = (), bg_swap: float = 0.0,
                 strong: bool = False, scale_range=(0.7, 1.35), max_rotation: float = 25.0, colour: bool = False,
                 norm: bool = False, masks: list = None, mosaic: float = 0.0):
        self.cache, self.tips, self.size = cache, tips, size
        self.keep, self.backgrounds, self.bg_swap = keep or [None] * len(cache), list(backgrounds), bg_swap
        self.strong, self.scale_range, self.max_rotation, self.colour = strong, scale_range, max_rotation, colour
        self.norm, self.masks, self.mosaic = norm, masks, mosaic
        hm = size // STRIDE
        self.gy, self.gx = np.mgrid[0:hm, 0:hm].astype(np.float32)

    def __len__(self):
        return len(self.cache)

    def _render(self, i: int, out: int):
        """Frame i through the random affine into an out x out canvas, then photometric noise.
        Returns (image, [(class, x, y)] of tips inside the canvas, affine matrix)."""
        img, (sx, sy) = self.cache[i]
        if self.keep[i] is not None and self.backgrounds and random.random() < self.bg_swap:
            img = swap_background(img, self.keep[i], random.choice(self.backgrounds))
        h, w = img.shape[:2]
        base = out / max(h, w)                                                            # where `fit` would put it
        zoom = math.exp(random.uniform(*np.log(self.scale_range)))                       # log-uniform zoom
        M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), random.uniform(-self.max_rotation, self.max_rotation),
                                    base * zoom)
        M[0, 2] += w * base / 2 + random.uniform(-0.2, 0.2) * out - w / 2                 # move the centre (pixel-centre
        M[1, 2] += h * base / 2 + random.uniform(-0.2, 0.2) * out - h / 2                 # coordinates throughout)
        if random.random() < 0.5:
            M[0] = -M[0]
            M[0, 2] += out - 1
        img = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        img = photometric(img, self.strong, self.colour)
        if self.norm:
            img = normalise(img)
        tips = []
        for c, x, y in self.tips[i]:
            x, y = M @ np.array([x * sx - 0.5, y * sy - 0.5, 1.0]) + 0.5
            if 0 <= x < out and 0 <= y < out:
                tips.append((c, x, y))
        return img, tips, M

    def __getitem__(self, i):
        size = self.size
        if self.mosaic and self.masks is None and random.random() < self.mosaic:
            # YOLO-style mosaic: four frames at half scale in a 2x2 grid. Tools appear small and in mixed contexts.
            half = size // 2
            img, tips = np.zeros((size, size, 3), np.uint8), []
            for q, j in enumerate([i] + random.choices(range(len(self)), k=3)):
                ox, oy = (q % 2) * half, (q // 2) * half
                tile, tile_tips, _ = self._render(j, half)
                img[oy:oy + half, ox:ox + half] = tile
                tips += [(c, x + ox, y + oy) for c, x, y in tile_tips]
        else:
            img, tips, M = self._render(i, size)
            if self.masks is not None:                                                    # same transform, 1/4 res
                mk = cv2.warpAffine(self.masks[i], M, (size, size), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
                mk = cv2.resize(mk, (size // STRIDE, size // STRIDE), interpolation=cv2.INTER_AREA)
                mask_target = torch.from_numpy(mk.transpose(2, 0, 1).astype(np.float32) / 255.0)

        target = np.zeros((len(CLASSES), size // STRIDE, size // STRIDE), np.float32)
        for c, x, y in tips:
            u, v = x / STRIDE - 0.5, y / STRIDE - 0.5
            np.maximum(target[c], np.exp(-((self.gx - u) ** 2 + (self.gy - v) ** 2) / (2 * SIGMA ** 2)), out=target[c])
            target[c, int(np.clip(round(v), 0, target.shape[1] - 1)), int(np.clip(round(u), 0, target.shape[2] - 1))] = 1.0
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        return (x, torch.from_numpy(target), mask_target) if self.masks is not None else (x, torch.from_numpy(target))


def photometric(img: np.ndarray, strong: bool = False, colour: bool = False) -> np.ndarray:
    """Colour, exposure, blur, sensor noise and compression: the ways real footage differs from renders.

    `strong` widens exposure, saturation and focus far enough to cover a dim, soft-focus microscope video: the unseen
    real video in our data is about 1.5x darker and 10x blurrier than the one we train on.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + (random.uniform(-25, 25) if colour else random.uniform(-8, 8))) % 180
    hsv[..., 1] *= random.uniform(0.5, 1.4) if strong else random.uniform(0.6, 1.4)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if colour and random.random() < 0.15:                                                 # tools come silver, blue,
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)     # purple or dark green
    if random.random() < (0.5 if strong else 0.3):                                        # gamma
        g = random.uniform(0.7, 2.0) if strong else random.uniform(0.7, 1.4)
        img = cv2.LUT(img, np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8))
    img = cv2.convertScaleAbs(img, alpha=random.uniform(0.5 if strong else 0.7, 1.3), beta=random.uniform(-30, 30))
    if strong and random.random() < 0.5:                                                  # defocus or low resolution
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(1.0, 4.0))
        else:
            h, w = img.shape[:2]
            f = random.uniform(2.0, 5.0)
            img = cv2.resize(cv2.resize(img, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA),
                             (w, h), interpolation=cv2.INTER_LINEAR)
    elif random.random() < 0.25:
        k = random.choice((3, 5, 7))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if strong and random.random() < 0.3:                                                  # microscope vignette
        h, w = img.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
        img = (img * np.clip(1.3 - random.uniform(0.3, 0.8) * r, 0, 1)[..., None]).astype(np.uint8)
    if random.random() < 0.2:
        img = np.clip(img + np.random.normal(0, random.uniform(3, 12), img.shape), 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 90)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor = None) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss, normalised by the number of tips in the batch. `class_weights` (C,)
    scales each class's heatmap, positives and negatives alike."""
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * (1 - pos)
    if class_weights is not None:
        pos_loss, neg_loss = pos_loss * class_weights.view(1, -1, 1, 1), neg_loss * class_weights.view(1, -1, 1, 1)
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp_min(1)


@torch.inference_mode()
def predict_peaks(net, images: dict, size: int, device, flip_tta: bool, norm: bool = False) -> dict:
    """{name: (scores (C, k), xy (C, k, 2) in original pixels, w, h)}, decoded exactly as SurgNet.predict does.
    `net` is a model or a list of models (an ensemble)."""
    nets = list(net) if isinstance(net, (list, tuple)) else [net]
    for n in nets:
        n.eval()
    out = {}
    for name, img in images.items():
        h, w = img.shape[:2]
        canvas, (sx, sy) = fit(img, size)
        if norm:
            canvas = normalise(canvas)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = ensemble_logits(nets, to_tensor(canvas, device), flip_tta)
        scores, xy = decode(logits)
        scores, xy = scores[0].cpu().numpy(), xy[0].cpu().numpy()
        xy[..., 0] = np.clip(xy[..., 0] / sx, 0, w - 1)
        xy[..., 1] = np.clip(xy[..., 1] / sy, 0, h - 1)
        out[name] = (scores, xy, w, h)
    return out


class Tuner:
    """Scores threshold settings fast. pick() keeps 0, 1 or 2 peaks per (image, class), so the official matching is
    run once for each of those three outcomes and every threshold setting just selects among them."""

    def __init__(self, peaks: dict, gt: dict, min_dist_frac: float = MIN_DIST_FRAC):
        self.names = list(peaks)
        n, C = len(self.names), len(CLASSES)
        self.s1, self.s2 = np.zeros((n, C)), np.full((n, C), -1.0)
        self.tp, self.fp, self.fn = (np.zeros((n, C, 3), int) for _ in range(3))
        self.err = np.full((n, C, 3, 2), np.nan)
        for i, name in enumerate(self.names):
            scores, xy, w, h = peaks[name]
            g = gt[name]
            radius = DEFAULT_RADIUS_FRAC * math.hypot(g.width, g.height)
            for c, cls in enumerate(CLASSES):
                gts = [t.tip_xy for t in g.tips if t.class_name == cls]
                second = next((j for j in range(1, scores.shape[1])
                               if math.hypot(*(xy[c, j] - xy[c, 0])) >= min_dist_frac * math.hypot(w, h)), None)
                self.s1[i, c] = scores[c, 0]
                options = [[], [xy[c, 0]], [xy[c, 0]]]
                if second is not None:
                    self.s2[i, c] = scores[c, second]
                    options[2] = [xy[c, 0], xy[c, second]]
                for state, preds in enumerate(options):
                    r = match_points([tuple(p) for p in preds], gts, radius)
                    self.tp[i, c, state], self.fp[i, c, state], self.fn[i, c, state] = \
                        len(r.matches), len(r.unmatched_preds), len(r.unmatched_gts)
                    for m, (_, _, d) in enumerate(r.matches):
                        self.err[i, c, state, m] = d

    def score(self, t1, t2, rows=None) -> dict:
        state = np.where(self.s1 >= np.asarray(t1), np.where(self.s2 >= np.asarray(t2), 2, 1), 0)
        rows = slice(None) if rows is None else rows
        take = lambda a: np.take_along_axis(a, state[..., None], axis=2)[..., 0][rows]
        tp, fp, fn = take(self.tp).sum(), take(self.fp).sum(), take(self.fn).sum()
        errs = np.take_along_axis(self.err, state[..., None, None], axis=2)[:, :, 0][rows].ravel()
        errs = errs[~np.isnan(errs)]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        median = float(np.median(errs)) if len(errs) else None
        err = min(1.0, max(0.0, 1 - median / ERROR_CEILING_PX)) if median is not None else 0.0
        w = LEADERBOARD_WEIGHTS
        return {'proxy': float(w['precision'] * precision + w['recall'] * recall + w['error'] * err),
                'precision': float(precision), 'recall': float(recall), 'median_error_px': median,
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}

    def tune(self) -> tuple[list, dict]:
        """Coordinate ascent: first-peak thresholds per class, then second-peak thresholds (1.01 = never)."""
        C = len(CLASSES)
        grid1 = np.round(np.arange(0.05, 0.951, 0.025), 3)
        grid2 = np.append(np.round(np.arange(0.2, 0.951, 0.05), 3), 1.01)
        t1, t2 = np.full(C, 0.3), np.full(C, 1.01)
        for _ in range(3):
            for t, grid in ((t1, grid1), (t2, grid2)):
                for c in range(C):
                    if t is t2 and CLASSES[c] not in SECOND_TIP_CLASSES:
                        continue
                    best, best_v = t[c], -1.0
                    for v in grid:
                        t[c] = v
                        s = self.score(t1, t2)['proxy']
                        if s > best_v + 1e-9:
                            best, best_v = v, s
                    t[c] = best
        return [[float(a), float(b)] for a, b in zip(t1, t2)], self.score(t1, t2)

    def by_folder(self, thresholds) -> dict:
        t1, t2 = [t[0] for t in thresholds], [t[1] for t in thresholds]
        folders = np.array([folder(n) for n in self.names])
        return {str(f): self.score(t1, t2, rows=folders == f) for f in sorted(set(folders))}


def save(path: Path, net, config: dict, thresholds: list, val: dict):
    state = {k: (v.detach().half() if v.is_floating_point() else v).cpu() for k, v in net.state_dict().items()
             if not k.startswith('mask_head.')}                                          # training-only head
    torch.save({'state_dict': state, 'config': config, 'thresholds': dict(zip(CLASSES, thresholds)), 'val': val}, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True, help='run directory: runs/<name>/')
    p.add_argument('--data', type=Path, default=Path('~/data/train/images'))
    p.add_argument('--labels', type=Path, default=Path('~/data/train/annotations.xml'))
    p.add_argument('--backbone', default='convnext_tiny.dinov3_lvd1689m')
    p.add_argument('--size', type=int, default=640, help='longer side of the network input')
    p.add_argument('--width', type=int, default=128, help='FPN channels')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3, help='head and FPN; the backbone gets lr * --backbone-lr-mult')
    p.add_argument('--backbone-lr-mult', type=float, default=0.2)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--domain-weights', default='real_surgical=0.4,phantom=0.3,greenbackground=0.3',
                   help='share of each epoch drawn from each folder')
    p.add_argument('--holdout', default='real_surgical/videoB', help='comma-separated capture keys held out whole')
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--all', action='store_true', help='train on everything (needs --thresholds-from)')
    p.add_argument('--thresholds-from', type=Path, default=None)
    p.add_argument('--flip-tta', action='store_true', help='average with a mirrored pass at inference')
    p.add_argument('--bg-swap', type=float, default=0.0, help='chance a green-screen frame gets a surgical background')
    p.add_argument('--strong-aug', action='store_true', help='heavy darkening, desaturation, defocus and vignette')
    p.add_argument('--scale-range', default='0.7,1.35', help='zoom range relative to the inference scale')
    p.add_argument('--max-rotation', type=float, default=25.0)
    p.add_argument('--colour-aug', action='store_true', help='wide hue shifts and occasional greyscale')
    p.add_argument('--pretrained-file', type=Path, default=None, help='backbone weights in timm layout (e.g. SurgeNetXL)')
    p.add_argument('--mosaic', type=float, default=0.0, help='chance a sample is a 2x2 mosaic of half-scale frames')
    p.add_argument('--mask-aux', type=float, default=0.0, help='weight of the auxiliary tool-mask loss (0 = off)')
    p.add_argument('--normalise', action='store_true', help='grey-world colour/contrast normalisation (train + inference)')
    p.add_argument('--class-boost', default='', help='e.g. scissors=2.5: sample frames holding that class more often')
    p.add_argument('--class-loss', default='', help='e.g. scissors=2: weight that class\'s heatmap loss')
    p.add_argument('--negatives', type=Path, default=None, help='folder of tool-free frames (give it a --domain-weights share)')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=0, help='hold-out split (keep 0 to compare runs on the same frames)')
    p.add_argument('--limit', type=int, default=None, help='use only the first N images (smoke tests)')
    args = p.parse_args()

    cv2.setNumThreads(1)                        # OpenCV's own pool deadlocks in forked DataLoader workers
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = True
    out_dir = Path('runs') / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    data = args.data.expanduser()
    gt = load_cvat_xml(args.labels.expanduser())
    names = [n for n in find_images(data) if n in gt][:args.limit]
    if args.all:
        train_names, val_names = names, []
    else:
        train_names, val_names = split(names, [k for k in args.holdout.split(',') if k], args.val_frac, args.split_seed)
    (out_dir / 'val_names.txt').write_text('\n'.join(val_names))
    paths = [data / n for n in train_names]
    if args.negatives:                          # tool-free frames from elsewhere: <dir>/<folder>/<image>
        neg_root = args.negatives.expanduser()
        negatives = find_images(neg_root)
        train_names, paths = train_names + negatives, paths + [neg_root / n for n in negatives]
    print(f'{len(train_names)} train {dict(Counter(map(folder, train_names)))}, '
          f'{len(val_names)} val {dict(Counter(map(folder, val_names)))}, device {device}', flush=True)

    t0 = time.time()
    cache = load_cache(paths, math.ceil(args.size * max(1.35, float(args.scale_range.split(',')[1]))), 8)
    val_images = dict(zip(val_names, (read_image(data / n) for n in val_names)))
    print(f'cached images in {time.time() - t0:.0f}s', flush=True)
    tips = [[(CLASSES.index(t.class_name), *t.tip_xy) for t in gt[n].tips] if n in gt else [] for n in train_names]
    share = dict(kv.split('=') for kv in args.domain_weights.split(','))
    counts = Counter(map(folder, train_names))
    weights = [float(share.get(folder(n), 0.0)) / counts[folder(n)] for n in train_names]
    boost = {k: float(v) for k, v in (kv.split('=') for kv in args.class_boost.split(',') if kv)}
    weights = [w * max([boost.get(CLASSES[c], 1.0) for c, _, _ in t] or [1.0]) for w, t in zip(weights, tips)]
    loss_w = {k: float(v) for k, v in (kv.split('=') for kv in args.class_loss.split(',') if kv)}
    class_weights = torch.tensor([loss_w.get(c, 1.0) for c in CLASSES], device=device)
    keep, backgrounds = green_screen_assets(args.labels.expanduser(), train_names, cache) if args.bg_swap else (None, ())
    if args.bg_swap:
        print(f'green-screen swap: {sum(k is not None for k in keep)} frames, {len(backgrounds)} backgrounds', flush=True)
    loader = DataLoader(TipDataset(cache, tips, args.size, keep, backgrounds, args.bg_swap, args.strong_aug,
                                   tuple(map(float, args.scale_range.split(','))), args.max_rotation, args.colour_aug, args.normalise,
                                   class_masks(args.labels.expanduser(), train_names, cache) if args.mask_aux else None,
                                   args.mosaic), batch_size=args.batch, drop_last=True,
                        sampler=WeightedRandomSampler(weights, len(train_names), generator=torch.Generator().manual_seed(args.seed)),
                        num_workers=args.workers, worker_init_fn=seed_worker, persistent_workers=args.workers > 0,
                        pin_memory=device.type == 'cuda')

    net = TipNet(args.backbone, pretrained=True, width=args.width, aux_masks=args.mask_aux > 0,
                 pretrained_file=str(args.pretrained_file.expanduser()) if args.pretrained_file else None)
    net = net.to(device).to(memory_format=torch.channels_last)
    backbone = list(net.backbone.parameters())
    ids = {id(q) for q in backbone}
    head = [q for q in net.parameters() if id(q) not in ids]
    opt = torch.optim.AdamW([{'params': backbone, 'lr': args.lr * args.backbone_lr_mult}, {'params': head, 'lr': args.lr}],
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * args.backbone_lr_mult, args.lr],
                                                total_steps=args.epochs * len(loader), pct_start=0.05)
    amp = device.type == 'cuda'
    config = {'backbone': args.backbone, 'input_size': args.size, 'width': args.width, 'flip_tta': args.flip_tta,
              'normalise': args.normalise,
              'min_dist_frac': MIN_DIST_FRAC, 'train_args': {k: str(v) for k, v in vars(args).items()}}

    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, total = time.time(), 0.0
        for batch in loader:
            x = batch[0].to(device, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
            y = batch[1].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = net(x, with_masks=args.mask_aux > 0)
            logits = out[0] if args.mask_aux > 0 else out
            loss = focal_loss(logits, y, class_weights)
            if args.mask_aux > 0:
                loss = loss + args.mask_aux * torch.nn.functional.binary_cross_entropy_with_logits(
                    out[1].float(), batch[2].to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            total += loss.item()
        print(f'epoch {epoch:3d}  loss {total / len(loader):.4f}  {time.time() - t0:.0f}s', flush=True)

        if val_names and (epoch % args.eval_every == 0 or epoch == args.epochs):
            tuner = Tuner(predict_peaks(net, val_images, args.size, device, args.flip_tta, args.normalise), gt)
            thresholds, val = tuner.tune()
            val['by_folder'] = tuner.by_folder(thresholds)
            val['epoch'] = epoch
            history.append({k: v for k, v in val.items()})
            print(f'  val proxy {val["proxy"]:.4f}  P {val["precision"]:.3f}  R {val["recall"]:.3f}  '
                  f'median err {val["median_error_px"]}  thresholds {thresholds}', flush=True)
            for f, s in val['by_folder'].items():
                print(f'    {f:16s} proxy {s["proxy"]:.4f}  P {s["precision"]:.3f}  R {s["recall"]:.3f}  '
                      f'err {s["median_error_px"]}  tp {s["tp"]} fp {s["fp"]} fn {s["fn"]}', flush=True)
            if best is None or val['proxy'] > best:
                best = val['proxy']
                save(out_dir / 'surgnet.pt', net, config, thresholds, val)
                print(f'  saved {out_dir / "surgnet.pt"}', flush=True)
    if args.all:
        src = torch.load(args.thresholds_from, map_location='cpu', weights_only=True)
        save(out_dir / 'surgnet.pt', net, config, [src['thresholds'][c] for c in CLASSES], {})
        print(f'saved {out_dir / "surgnet.pt"} with thresholds from {args.thresholds_from}', flush=True)
    (out_dir / 'history.json').write_text(json.dumps(history, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

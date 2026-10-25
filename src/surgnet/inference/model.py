"""TipNet: a pretrained timm backbone + FPN that paints one heatmap per class, peaking at each tool tip.

Shared by training and inference. A frame is scaled so its longer side is `size`, then padded right/bottom to a
multiple of PAD. The heatmap has STRIDE-pixel cells: a point at input pixel x sits at heatmap coordinate
u = x / STRIDE - 0.5, so cell j's centre is x = (j + 0.5) * STRIDE.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.detection import CLASSES

STRIDE = 4
PAD = 32
TOP_K = 4                                   # peaks kept per class before picking
MIN_DIST_FRAC = 0.03                        # a second tip of one class must be this far (x image diagonal) from the first
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TipNet(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = False, width: int = 128, aux_masks: bool = False,
                 pretrained_file: str = None):
        super().__init__()
        overlay = dict(file=pretrained_file) if pretrained_file else None                # e.g. surgical pretraining
        self.backbone = timm.create_model(backbone, pretrained=pretrained or bool(pretrained_file), features_only=True,
                                          pretrained_cfg_overlay=overlay)
        reductions = self.backbone.feature_info.reduction()
        channels = self.backbone.feature_info.channels()
        self.levels = [reductions.index(r) for r in (4, 8, 16, 32)]
        self.lateral = nn.ModuleList([nn.Conv2d(channels[i], width, 1) for i in self.levels])
        self.smooth = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(width, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, len(CLASSES), 1))
        nn.init.constant_(self.head[-1].bias, -2.19)                                     # start near p=0.1, as CenterNet
        if aux_masks:                                                                     # training only: tool outlines
            self.mask_head = nn.Sequential(nn.Conv2d(width, 64, 3, padding=1), nn.ReLU(inplace=True),
                                           nn.Conv2d(64, len(CLASSES), 1))
        cfg = getattr(self.backbone, 'pretrained_cfg', None) or {}
        mean, std = cfg.get('mean', IMAGENET_MEAN), cfg.get('std', IMAGENET_STD)
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1) * 255.0, persistent=False)

    def forward(self, x: torch.Tensor, with_masks: bool = False):
        """x: (B, 3, H, W) RGB in 0..255, H and W multiples of PAD. Returns heatmap logits (B, n_classes, H/4, W/4),
        plus tool-mask logits of the same shape when `with_masks` (training with the auxiliary head)."""
        feats = self.backbone((x - self.mean) / self.std)
        feats = [feats[i] for i in self.levels]
        p = self.lateral[3](feats[3])
        for i in (2, 1, 0):                                                                # top-down FPN to stride 4
            p = F.interpolate(p, size=feats[i].shape[-2:], mode='nearest') + self.lateral[i](feats[i])
        p = self.smooth(p)
        return (self.head(p), self.mask_head(p)) if with_masks else self.head(p)


def fit(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale so the longer side is `size`, pad right/bottom to a multiple of PAD. Returns (canvas, (sx, sy))."""
    h, w = image.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, 0, -(-nh // PAD) * PAD - nh, 0, -(-nw // PAD) * PAD - nw,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas, (nw / w, nh / h)


def normalise(canvas: np.ndarray, mean: float = 128.0, std: float = 48.0) -> np.ndarray:
    """Grey-world colour and contrast normalisation: each channel of the lit area to the same mean and spread.
    Sessions differ in light, white balance and exposure; this removes most of that before the network sees it.
    Near-black pixels (padding, rotation borders) are left out of the statistics."""
    small = canvas[::4, ::4]
    lit = (small.max(axis=2) > 12).astype(np.uint8)
    if int(lit.sum()) < 50:
        return canvas
    m, sd = cv2.meanStdDev(small, mask=lit)
    lut = np.clip((np.arange(256, dtype=np.float32)[:, None] - m.T) * (std / (sd.T + 1e-3)) + mean, 0, 255)
    return cv2.LUT(canvas, lut.astype(np.uint8).reshape(256, 1, 3))


def to_tensor(canvas: np.ndarray, device) -> torch.Tensor:
    """BGR uint8 (H, W, 3) -> RGB float (1, 3, H, W) on `device`, channels_last."""
    x = torch.from_numpy(canvas).to(device, non_blocking=True)
    return x.permute(2, 0, 1)[None].flip(1).float().contiguous(memory_format=torch.channels_last)


def forward_logits(net: nn.Module, x: torch.Tensor, flip_tta: bool = False) -> torch.Tensor:
    """Heatmap logits, optionally averaged with the horizontally flipped pass (one batch of two)."""
    if not flip_tta:
        return net(x)
    both = net(torch.cat([x, x.flip(3)]))
    return (both[:1] + both[1:].flip(3)) / 2


def ensemble_logits(nets: list, x: torch.Tensor, flip_tta: bool = False) -> torch.Tensor:
    """Mean heatmap logits over several models, each optionally with flip TTA."""
    out = forward_logits(nets[0], x, flip_tta)
    for net in nets[1:]:
        out = out + forward_logits(net, x, flip_tta)
    return out / len(nets)


def decode(logits: torch.Tensor, k: int = TOP_K, refine_sigma: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k local maxima per class. logits (B, C, h, w) -> scores (B, C, k), xy (B, C, k, 2) in input pixels.

    Each peak is refined to sub-cell precision by fitting a parabola to the log-heat through it and its neighbours,
    which is exact for a gaussian bump. With `refine_sigma` > 0 (cells) the heatmap is first smoothed with a gaussian
    for the fit only (DARK-style): network heatmaps are noisy and skewed, and smoothing removes most of that bias.
    Peak selection and scores always use the raw heatmap.
    """
    heat = logits.float().sigmoid()
    b, c, h, w = heat.shape
    peaks = heat * (heat == F.max_pool2d(heat, 3, 1, 1))
    scores, idx = peaks.view(b, c, -1).topk(min(k, h * w), dim=2)
    ys, xs = (idx // w).float(), (idx % w).float()
    fit_heat, limit = heat, 0.5
    if refine_sigma > 0:
        r = int(math.ceil(3 * refine_sigma))
        g = torch.exp(-torch.arange(-r, r + 1, device=heat.device, dtype=heat.dtype) ** 2 / (2 * refine_sigma ** 2))
        g = g / g.sum()
        fit_heat = F.conv2d(F.pad(heat.view(-1, 1, h, w), (r, r, 0, 0), mode='replicate'), g.view(1, 1, 1, -1))
        fit_heat = F.conv2d(F.pad(fit_heat, (0, 0, r, r), mode='replicate'), g.view(1, 1, -1, 1)).view(b, c, h, w)
        limit = 1.0
    lp = F.pad(fit_heat.clamp_min(1e-6).log(), (1, 1, 1, 1), mode='replicate')

    def at(dy, dx):
        return lp[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w].reshape(b, c, -1).gather(2, idx)

    l0 = at(0, 0)
    left, right, up, down = at(0, -1), at(0, 1), at(-1, 0), at(1, 0)
    dx = ((right - left) / (2 * (2 * l0 - right - left)).clamp_min(1e-6)).clamp(-limit, limit)
    dy = ((down - up) / (2 * (2 * l0 - down - up)).clamp_min(1e-6)).clamp(-limit, limit)
    xy = torch.stack([(xs + dx + 0.5) * STRIDE, (ys + dy + 0.5) * STRIDE], dim=-1)
    return scores, xy


def pick(scores: np.ndarray, xy: np.ndarray, thresholds, min_dist: float) -> list:
    """One image: scores (C, k), xy (C, k, 2). Returns [(class index, (x, y), score)].

    Per class: the best peak if it clears thresholds[c][0]; then the strongest other peak at least `min_dist` from
    it, if that clears thresholds[c][1]. Real frames can hold two tools of one class (two suctions), never more.
    """
    out = []
    for c in range(scores.shape[0]):
        if scores[c, 0] < thresholds[c][0]:
            continue
        out.append((c, (float(xy[c, 0, 0]), float(xy[c, 0, 1])), float(scores[c, 0])))
        for j in range(1, scores.shape[1]):
            if np.hypot(xy[c, j, 0] - xy[c, 0, 0], xy[c, j, 1] - xy[c, 0, 1]) >= min_dist:
                if scores[c, j] >= thresholds[c][1]:
                    out.append((c, (float(xy[c, j, 0]), float(xy[c, j, 1])), float(scores[c, j])))
                break
    return out

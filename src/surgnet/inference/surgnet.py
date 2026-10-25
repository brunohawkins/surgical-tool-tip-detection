"""SurgNet: the inference contract, backed by TipNet (see model.py).

    SurgNet()                      loads weights/surgnet.pt; device=None picks cuda, then mps, then cpu
    SurgNet.predict(image_bgr)     one BGR uint8 array of shape (H, W, 3) in -> list[ToolDetection] out

The checkpoint carries everything inference needs besides the code: backbone name, input size, per-class thresholds.
An ensemble checkpoint lists several models under 'members'; their heatmap logits are averaged. Each member gets its own
preprocessing (its config's 'normalise'), and an optional config 'routing' {class: [member indices]} takes each
class's heatmap from a subset of members (e.g. bipolar from one pair of models, suction from another).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch

from ..shared.detection import CLASSES, ToolDetection
from ..shared.images import read_image
from .model import MIN_DIST_FRAC, TipNet, decode, fit, forward_logits, normalise, pick, to_tensor

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / 'weights'      # ship your weights here
WEIGHTS_FILE = Path(os.environ.get('SURGNET_WEIGHTS', WEIGHTS_DIR / 'surgnet.pt'))   # override: benchmark other runs
WARMUP_SHAPES = ((1080, 1920, 3), (1080, 960, 3))                   # the usual frame sizes: tune kernels before timing


def resolve_device(device=None) -> torch.device:
    """None: best available. '0' or 0: cuda:0. Otherwise anything torch.device accepts."""
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return torch.device(f'cuda:{int(device)}')
    return torch.device(device)


class SurgNet:
    """Loads the model once; call `predict(image)` as often as needed."""

    def __init__(self, device=None, weights=WEIGHTS_FILE):
        self.device = resolve_device(device)
        ckpt = torch.load(weights, map_location='cpu', weights_only=True)
        cfg = ckpt['config']
        self.size = int(cfg['input_size'])
        self.flip_tta = bool(cfg.get('flip_tta', False))
        self.refine_sigma = float(cfg.get('refine_sigma', 0.0))
        self.min_dist_frac = float(cfg.get('min_dist_frac', MIN_DIST_FRAC))
        self.thresholds = [tuple(float(t) for t in ckpt['thresholds'][c]) for c in CLASSES]
        self.nets, self.member_norm = [], []
        for m in ckpt.get('members') or [{'config': cfg, 'state_dict': ckpt['state_dict']}]:
            self.member_norm.append(bool(m['config'].get('normalise', False)))
            net = TipNet(m['config']['backbone'], width=int(m['config']['width']))
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in m['state_dict'].items()})
            self.nets.append(net.to(self.device).eval().to(memory_format=torch.channels_last))
        routing = cfg.get('routing') or {}
        self.routing = [list(routing.get(c, range(len(self.nets)))) or [0] for c in CLASSES]
        self.amp = self.device.type == 'cuda'
        if self.amp:
            torch.backends.cudnn.benchmark = True
        for shape in WARMUP_SHAPES:
            self.predict(np.zeros(shape, np.uint8))

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> list[ToolDetection]:
        """Detect every tool in one BGR uint8 frame of shape (H, W, 3); tips in pixels of `image`, origin top-left."""
        h, w = image.shape[:2]
        canvas, (sx, sy) = fit(image, self.size)
        inputs = {False: None, True: None}
        member_logits = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            for net, norm in zip(self.nets, self.member_norm):
                if inputs[norm] is None:
                    inputs[norm] = to_tensor(normalise(canvas) if norm else canvas, self.device)
                member_logits.append(forward_logits(net, inputs[norm], self.flip_tta))
        stacked = torch.stack(member_logits)                                              # (M, 1, C, h, w)
        logits = torch.stack([stacked[r, :, c].mean(0) for c, r in enumerate(self.routing)], dim=1)
        scores, xy = decode(logits, refine_sigma=self.refine_sigma)
        scores, xy = scores[0].cpu().numpy(), xy[0].cpu().numpy()
        xy[..., 0] = np.clip(xy[..., 0] / sx, 0, w - 1)
        xy[..., 1] = np.clip(xy[..., 1] / sy, 0, h - 1)
        tips = pick(scores, xy, self.thresholds, self.min_dist_frac * float(np.hypot(w, h)))
        return [ToolDetection(CLASSES[c], p, s) for c, p, s in tips]

    def predict_path(self, image_path) -> list[ToolDetection]:
        """Convenience: read the image from disk, then `predict`. The benchmark does not use this."""
        return self.predict(read_image(image_path))

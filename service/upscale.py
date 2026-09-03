#!/usr/bin/env python3
"""Real-ESRGAN x2, in torch, on the card that is already there.

The packshot's detail ceiling is the generator: Qwen-Image-Edit normalises
every edit to about a megapixel and returns 768x1376 whatever goes in, and
asking it for more makes the picture worse -- at 704x1280 and 576x1024 the
print smeared. Measured. So more detail has to come from a second, separate
step, and that step is super-resolution.

Why the architecture is written out here rather than installed. Real-ESRGAN
ships as ONNX and as a pip package. The ONNX runs on this pod at 4.3 seconds
for a 256-pixel tile, which is seventy seconds for one packshot, because
onnxruntime here is the CPU build. Getting it onto the GPU means installing
onnxruntime-gpu, and the version that matches this CUDA is not the current one
-- the last attempt at that on this machine failed with "Require cuDNN 9.* and
CUDA 13.*". Replacing onnxruntime in the environment rembg depends on is a way
to break a working pod for a feature.

torch is already here and already on the GPU. RRDBNet is forty lines and its
weights load by name. The risk in writing an architecture out by hand is
getting it subtly wrong and never noticing, so it is checked against the ONNX
model on the same input -- see `verify` at the bottom, and TIMINGS.md for what
it reported.

BSD 3-Clause, which is what makes it usable at all: xinntao/Real-ESRGAN.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = os.environ.get("UPSCALE_WEIGHTS",
                         "/workspace/models/upscale/RealESRGAN_x2plus.pth")
# Tiles, because a 768x1376 image at once wants more VRAM than is spare on a
# card already holding two models. Overlapped and blended, or the seams show.
TILE = int(os.environ.get("UPSCALE_TILE", "512"))
OVERLAP = int(os.environ.get("UPSCALE_OVERLAP", "32"))


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """The x2 variant: the input is pixel-unshuffled by two first, which is why
    conv_first takes twelve channels rather than three."""

    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(12, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = F.pixel_unshuffle(x, downscale_factor=2)
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


_model = None


def model(device="cuda", dtype=torch.float32):
    global _model
    if _model is None:
        sd = torch.load(WEIGHTS, map_location="cpu")
        sd = sd.get("params_ema", sd.get("params", sd))
        net = RRDBNet()
        net.load_state_dict(sd, strict=True)
        net.eval().to(device=device, dtype=dtype)
        for p in net.parameters():
            p.requires_grad_(False)
        _model = net
    return _model


def _run(t, net):
    """One tensor through the network, in overlapping tiles."""
    _, _, h, w = t.shape
    out = torch.zeros((1, 3, h * 2, w * 2), device=t.device, dtype=t.dtype)
    weight = torch.zeros_like(out)
    step = TILE - OVERLAP
    for y in range(0, max(h - OVERLAP, 1), step):
        for x in range(0, max(w - OVERLAP, 1), step):
            y1, x1 = min(y + TILE, h), min(x + TILE, w)
            y0, x0 = max(y1 - TILE, 0), max(x1 - TILE, 0)
            # Padded to a multiple of two, because pixel_unshuffle needs it.
            tile = t[:, :, y0:y1, x0:x1]
            ph, pw = tile.shape[2] % 2, tile.shape[3] % 2
            if ph or pw:
                tile = F.pad(tile, (0, pw, 0, ph), mode="replicate")
            with torch.no_grad():
                got = net(tile)
            got = got[:, :, :(y1 - y0) * 2, :(x1 - x0) * 2]
            out[:, :, y0 * 2:y1 * 2, x0 * 2:x1 * 2] += got
            weight[:, :, y0 * 2:y1 * 2, x0 * 2:x1 * 2] += 1
    return out / weight.clamp(min=1)


def upscale(img, device="cuda"):
    """A PIL image in, twice the size out."""
    import numpy as np
    from PIL import Image

    net = model(device)
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
    out = _run(t, net).clamp(0, 1)
    a = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(a)


def verify(onnx_path="/workspace/models/upscale/real_esrgan_x2.onnx", size=128):
    """Does the hand-written architecture agree with the published model?

    The point of writing an architecture out is that it is fast and free; the
    danger is a transposition nobody notices until a customer sees a smeared
    photograph. So both are run on the same noise and the largest disagreement
    is printed. Anything past a pixel value or two means this file is wrong.
    """
    import numpy as np
    import onnxruntime as ort

    x = np.random.rand(1, 3, size, size).astype(np.float32)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    net = model("cpu")
    with torch.no_grad():
        mine = net(torch.from_numpy(x)).numpy()
    diff = np.abs(ref - mine)
    return {"max_diff": float(diff.max()), "mean_diff": float(diff.mean()),
            "ref_range": [float(ref.min()), float(ref.max())],
            "shape": list(mine.shape)}

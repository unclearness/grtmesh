# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Turn the clean renders into the synthetic dataset: depth in mm and color
with per-camera exposure and white-balance differences.

    python python/synthetic/make_dataset.py --render <dataset>/render \
        --out <dataset>/dance
    python python/synthetic/make_dataset.py --render <dataset>/render \
        --out <dataset>/dance_clean --no_perturbation

Output (read by `multiview_rgbd <out> ...`):

    <out>/cameras.json                     copied from the render
    <out>/perturbation.json                gains of every camera
    <out>/frames/NNNNNN/camKK_color.jpg    perturbed color (JPEG quality 95)
    <out>/frames/NNNNNN/camKK_depth.png    uint16 depth in mm (0 = invalid / background)
    <out>/clean/NNNNNN/camKK_color.png     unperturbed color (ground truth)

Perturbation. The renders are Lambertian, so a surface point has the same
color in every camera; the only difference between cameras is a fixed gain
that camera k applies to the linear-light color of every pixel:

    color_c *= exp(g_k,c(x, y)),
    g_k,c(x, y) = e_k + a_k x + b_k y          (exposure: offset + smooth gradient)
                + s_c t_k + m_k [c = G]         (white balance)

with normalized image coordinates x = 2u/(W-1) - 1, y = 2v/(H-1) - 1 in
[-1, 1] and s = (+1, 0, -1) for (R, G, B): t_k is a color-temperature shift
and m_k a green/magenta tint. This is the model the seam correction assumes:
a smooth multiplicative gain per camera (the luminance field corrects the
exposure part, the RGB field also the white balance). The exposure offsets
alternate between +EXPOSURE and -EXPOSURE around the circle (see EXPOSURE);
the other parameters are drawn from fixed uniform ranges with a fixed seed.
Everything is zero-mean over the even cameras and over the odd cameras
separately (the correction is relative, with a zero-mean gauge, so a gain
common to all cameras could not be recovered), and scaled down per camera
where needed so that |g| <= MAX_LOG_GAIN everywhere, inside the default
correction bound (0.15).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Half-widths of the uniform parameter ranges (log gain).
RANGES = {
    "a": 0.015,  # horizontal exposure gradient
    "b": 0.015,  # vertical exposure gradient
    "t": 0.03,   # color temperature (R up / B down)
    "m": 0.015,  # green tint
}
# Exposure offsets: within each group (even = reconstruction, odd = held-out),
# cameras alternate around the circle between +EXPOSURE and -EXPOSURE (the
# last one of an odd-sized group gets 0), so neighboring cameras, which share
# the most seams, differ by 2 * EXPOSURE in log exposure (about 22%).
EXPOSURE = 0.10
MAX_LOG_GAIN = 0.14   # strictly inside the default bound of the correction (0.15)
SEED = 20261020
JPEG_QUALITY = 95
MAX_DEPTH_M = 15.0    # the empty background renders at the camera clip end (20 m)


def log_gain(params: dict[str, float], width: int, height: int) -> np.ndarray:
    """H x W x 3 log gain in BGR order (OpenCV images)."""
    x = np.linspace(-1.0, 1.0, width)[None, :]
    y = np.linspace(-1.0, 1.0, height)[:, None]
    exposure = params["e"] + params["a"] * x + params["b"] * y
    r = exposure + params["t"]
    g = exposure + params["m"] + 0 * x * y
    b = exposure - params["t"]
    return np.stack([np.broadcast_to(b, g.shape), g, np.broadcast_to(r, g.shape)], axis=-1)


def peak(params: dict[str, float]) -> float:
    """max over the image and the channels of |g| (the gain is affine in x, y)."""
    corners = [(sx, sy) for sx in (-1, 1) for sy in (-1, 1)]
    values = []
    for sx, sy in corners:
        exposure = params["e"] + params["a"] * sx + params["b"] * sy
        values += [exposure + params["t"], exposure + params["m"], exposure - params["t"]]
    return max(abs(v) for v in values)


def draw_parameters(names: list[str]) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(SEED)
    values = {k: rng.uniform(-h, h, len(names)) for k, h in RANGES.items()}
    index = np.arange(len(names))
    values["e"] = np.zeros(len(names))
    for group in (index % 2 == 0, index % 2 == 1):
        members = index[group]
        for j, i in enumerate(members):
            values["e"][i] = 0.0 if (len(members) % 2 and j == len(members) - 1) else \
                (EXPOSURE if j % 2 == 0 else -EXPOSURE)
    for k in values:
        for group in (index % 2 == 0, index % 2 == 1):
            values[k][group] -= values[k][group].mean()
    keys = ["e", *RANGES]
    params = {name: {k: float(values[k][i]) for k in keys} for i, name in enumerate(names)}
    # Keep every camera inside the bound. Scaling one camera breaks the exact
    # zero mean only slightly; re-centering and re-checking converges quickly.
    for _ in range(20):
        for p in params.values():
            s = min(1.0, MAX_LOG_GAIN / max(peak(p), 1e-9))
            for k in p:
                p[k] *= s
        for group in (index % 2 == 0, index % 2 == 1):
            members = [names[i] for i in index[group]]
            for k in keys:
                mean = np.mean([params[n][k] for n in members])
                for n in members:
                    params[n][k] -= mean
        if max(peak(p) for p in params.values()) <= MAX_LOG_GAIN + 1e-6:
            break
    return params


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0), 1 / 2.4) - 0.055)


def perturb(bgr: np.ndarray, gain_log: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(bgr.astype(np.float64) / 255.0) * np.exp(gain_log)
    return np.clip(np.round(linear_to_srgb(np.clip(linear, 0, 1)) * 255.0), 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--render", type=Path, required=True, help="output dir of render_bl.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                   help="subset of the rendered frames (inclusive; default: all)")
    p.add_argument("--no_perturbation", action="store_true",
                   help="write the clean colors as input (reference upper bound), no clean/ dir")
    args = p.parse_args()

    meta = json.loads((args.render / "cameras.json").read_text())
    names = [c["name"] for c in meta["cameras"]]
    params = draw_parameters(names)
    if args.no_perturbation:
        params = {name: {k: 0.0 for k in ("e", *RANGES)} for name in names}
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.render / "cameras.json", args.out / "cameras.json")
    (args.out / "perturbation.json").write_text(json.dumps({
        "model": "g_c = e + a x + b y + s_c t + m [c=G], s = (+1, 0, -1) for (R, G, B); "
                 "color_linear *= exp(g_c); x = 2u/(W-1)-1, y = 2v/(H-1)-1",
        "seed": SEED, "exposure": EXPOSURE, "ranges": RANGES, "max_log_gain": MAX_LOG_GAIN,
        "zero_mean_groups": ["even cameras", "odd cameras"],
        "peak_log_gain": {n: peak(params[n]) for n in names},
        "cameras": params}, indent=2))
    gains = {c["name"]: log_gain(params[c["name"]], c["width"], c["height"]) for c in meta["cameras"]}
    first, last = args.frames if args.frames else meta["frames"]
    for frame in range(first, last + 1):
        frame_dir = args.out / "frames" / f"{frame:06d}"
        clean_dir = args.out / "clean" / f"{frame:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_perturbation:
            clean_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = args.render / name
            color = cv2.imread(str(src / f"color_{frame:04d}.png"), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(src / f"depth_{frame:04d}.exr"), cv2.IMREAD_UNCHANGED)
            if color is None or depth is None:
                raise FileNotFoundError(f"{src} frame {frame}")
            depth_m = depth[..., 0] if depth.ndim == 3 else depth
            depth_mm = np.where(np.isfinite(depth_m) & (depth_m > 0) & (depth_m < MAX_DEPTH_M),
                                np.round(depth_m * 1000.0), 0).astype(np.uint16)
            if not args.no_perturbation:
                shutil.copyfile(src / f"color_{frame:04d}.png", clean_dir / f"{name}_color.png")
            out = color if args.no_perturbation else perturb(color, gains[name])
            cv2.imwrite(str(frame_dir / f"{name}_color.jpg"), out,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(str(frame_dir / f"{name}_depth.png"), depth_mm)
        print(f"[dataset] frame {frame}", flush=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Pick held-out zoom boxes for the synthetic figure: boxes on the floor and on
the person where the uncorrected render shows a brightness step between two
source cameras and the correction removes it.

    python python/synthetic/find_seam_crops.py --dataset <dataset>/dance \
        --runs data_out/synthetic

For every held-out view the log ratio of the render's luminance to the ground
truth is box-filtered (texture detail and misregistration average out); a
seam shows up as a step of that ratio. The score of a box is the spread of the
ratio inside it without correction (95th - 5th percentile) minus the spread
with the luminance field, over fully covered boxes only. Prints the best
camera / frame / box per region as make_figure.py --crop arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dataset  # noqa: E402

HELD_OUT = ["cam01", "cam03", "cam05", "cam07", "cam09"]
LUMA_BGR = np.array([0.0722, 0.7152, 0.2126])


def luminance(bgr):
    return make_dataset.srgb_to_linear(bgr.astype(np.float64) / 255.0) @ LUMA_BGR


def above_floor(dataset: Path, cam: dict, frame: int) -> np.ndarray:
    d = cv2.imread(str(dataset / "frames" / f"{frame:06d}" / f"{cam['name']}_depth.png"), -1) / 1000.0
    h, w = d.shape
    k = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    t = np.array(cam["camera_to_world"])
    vv, uu = np.mgrid[0:h, 0:w]
    rays = np.linalg.inv(k) @ np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)])
    p = (t[:3, :3] @ (rays * d.ravel())).T + t[:3, 3]
    return ((p[:, 2] > 0.05) & (d.ravel() > 0)).reshape(h, w)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs=3, default=[5, 150, 5])
    p.add_argument("--floor_box", type=int, nargs=2, default=[200, 130])
    p.add_argument("--body_box", type=int, nargs=2, default=[110, 110])
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()
    cams = {c["name"]: c for c in json.loads((args.dataset / "cameras.json").read_text())["cameras"]}
    found = {"floor": [], "body": []}
    for frame in range(*args.frames):
        for name in HELD_OUT:
            gt = cv2.imread(str(args.dataset / "clean" / f"{frame:06d}" / f"{name}_color.png"))
            ygt = luminance(gt)
            renders = {v: cv2.imread(str(args.runs / v / "render" / name / f"{frame:06d}.png"))
                       for v in ("off", "lum")}
            covered = np.all([r.max(axis=2) > 12 for r in renders.values()], axis=0)
            ratio = {}
            for v, r in renders.items():
                num = cv2.blur(luminance(r) * covered, (9, 9))
                den = cv2.blur(ygt * covered, (9, 9))
                ratio[v] = np.log((num + 1e-4) / (den + 1e-4))
            body = above_floor(args.dataset, cams[name], frame)
            # Keep away from silhouettes: geometric error dominates there.
            inner = cv2.erode(covered.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            for kind, (bw, bh), region in (("floor", args.floor_box, inner & ~body),
                                           ("body", args.body_box, inner & body)):
                h, w = region.shape
                for y in range(0, h - bh, args.stride):
                    for x in range(0, w - bw, args.stride):
                        m = region[y:y + bh, x:x + bw]
                        if m.mean() < 0.97:
                            continue
                        spread = {v: np.subtract(*np.percentile(ratio[v][y:y + bh, x:x + bw][m], [95, 5]))
                                  for v in ratio}
                        found[kind].append((spread["off"] - spread["lum"], spread["off"], spread["lum"],
                                            name, frame, x, y, bw, bh))
        print(f"frame {frame} done", flush=True)
    for kind, rows in found.items():
        rows.sort(reverse=True)
        print(f"== {kind}: score, spread off -> lum (log), --crop argument")
        seen = set()
        for s, so, sl, name, frame, x, y, bw, bh in rows:
            key = (name, frame // 20)
            if key in seen:
                continue
            seen.add(key)
            print(f"  {s:.3f}  {so:.3f} -> {sl:.3f}  {name}:{frame}:{x},{y},{bw},{bh}")
            if len(seen) >= args.top:
                break


if __name__ == "__main__":
    main()

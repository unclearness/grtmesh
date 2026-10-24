# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Figure panels and gain-field recovery statistics for the synthetic sequence.

    python python/synthetic/make_figure.py --dataset <dataset>/dance \
        --runs data_out/synthetic --out doc/wacv-2027/figures/synth \
        --crop cam05:146:900,560,200,130 --crop2 cam07:140:760,160,110,110 --frame 140

Writes
  <out>/crop_<variant>.jpg, crop_gt.jpg     the --crop box (floor) of every variant and the
                                            ground truth, enlarged (boxes from find_seam_crops.py)
  <out>/crop2_<variant>.jpg, crop2_gt.jpg   the --crop2 box (person)
  <out>/crop_view.jpg, crop2_view.jpg       ground-truth views with the box
  <out>/fields.jpg                    per reconstruction camera (columns): input, applied
                                      exposure gain, gain recovered by the correction; grid
                                      cells in which the camera sees no surface are faded
  <out>/recovery.json                 recovered vs applied gain on the observed grid cells

The applied gain is known exactly (perturbation.json, an affine log gain per
camera and channel). The recovered gain is the negated field of the corrected
run (CC_DUMP_GAIN_DIR: log gains before the clamp, [cam][gy][gx][ch] with ch
in B, G, R order). The correction is relative with a zero-mean gauge over the
cameras and the injected gains are zero-mean over the reconstruction cameras,
so the two are compared directly. Only cells in which the camera sees a
surface (valid depth) are compared: seams exist only there, and elsewhere the
field stays near the identity by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gain_colormap import log_gain_rgb  # noqa: E402

RECON = ["cam02", "cam04", "cam06", "cam08", "cam10"]
LUMA_BGR = np.array([0.0722, 0.7152, 0.2126])


def load_gain(path: Path) -> np.ndarray:
    with path.open() as f:
        cams, gw, gh, nch, _ = f.readline().split()
        values = np.loadtxt(f, dtype=np.float32)
    return values.reshape(int(cams), int(gh), int(gw), int(nch))


def applied_grid(params: dict, width: int, height: int, gw: int, gh: int) -> np.ndarray:
    """Applied log gain at the grid cell centers, gh x gw x 3 (B, G, R)."""
    u = (np.arange(gw) + 0.5) * width / gw - 0.5
    v = (np.arange(gh) + 0.5) * height / gh - 0.5
    x = (2 * u / (width - 1) - 1)[None, :]
    y = (2 * v / (height - 1) - 1)[:, None]
    exposure = params["e"] + params["a"] * x + params["b"] * y
    return np.stack([exposure - params["t"], exposure + params["m"], exposure + params["t"]], axis=-1)


def recovered_grid(dump: np.ndarray, k: int) -> np.ndarray:
    """Gain the correction attributes to camera k (negated field), gh x gw x 3 (B, G, R)."""
    g = -dump[k]
    return np.repeat(g, 3, axis=2) if g.shape[2] == 1 else g


def observed_cells(dataset: Path, name: str, frame: int, gw: int, gh: int) -> np.ndarray:
    """Grid cells in which camera `name` sees a surface (>= 25% valid depth)."""
    depth = cv2.imread(str(dataset / "frames" / f"{frame:06d}" / f"{name}_depth.png"), cv2.IMREAD_UNCHANGED)
    return cv2.resize((depth > 0).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) >= 0.25


def field_image(log_gain: np.ndarray, observed: np.ndarray, size: tuple[int, int], bound: float) -> np.ndarray:
    """Scalar log-gain grid -> BGR image in the paper palette (bilinear, like the
    bake); cells without a visible surface are faded toward white."""
    up = cv2.resize(log_gain.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(log_gain_rgb(up, bound), cv2.COLOR_RGB2BGR).astype(np.float32)
    seen = cv2.resize(observed.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)[..., None]
    return (img * (0.3 + 0.7 * seen) + 255 * 0.7 * (1 - seen)).astype(np.uint8)


def recovery(dataset: Path, runs: Path, variant: str) -> dict:
    meta = json.loads((dataset / "cameras.json").read_text())
    cams = {c["name"]: c for c in meta["cameras"]}
    params = json.loads((dataset / "perturbation.json").read_text())["cameras"]
    first, last = meta["frames"]
    applied, recovered, seen = [], [], []
    for frame in range(first, last + 1):
        dump = load_gain(runs / variant / "gain" / f"gain_{frame - first:06d}.txt")
        gh, gw = dump.shape[1:3]
        for k, name in enumerate(RECON):
            c = cams[name]
            applied.append(applied_grid(params[name], c["width"], c["height"], gw, gh))
            recovered.append(recovered_grid(dump, k))
            seen.append(observed_cells(dataset, name, frame, gw, gh))
    a, r, m = np.stack(applied), np.stack(recovered), np.stack(seen)
    out = {"variant": variant, "frames": last - first + 1, "observed_cell_fraction": float(m.mean())}
    for label, fa, fr in (("luminance", a @ LUMA_BGR, r @ LUMA_BGR),
                          ("chroma_r_minus_b", a[..., 2] - a[..., 0], r[..., 2] - r[..., 0])):
        fa, fr = fa[m], fr[m]
        out[label] = {"pearson_r": float(np.corrcoef(fa, fr)[0, 1]) if fr.std() > 0 else 0.0,
                      "rmse": float(np.sqrt(np.mean((fr - fa) ** 2))),
                      "rmse_no_correction": float(np.sqrt(np.mean(fa ** 2)))}
    n = len(RECON)
    out["camera_mean_luminance_observed"] = {
        name: {"applied": float((a[k::n] @ LUMA_BGR)[m[k::n]].mean()),
               "recovered": float((r[k::n] @ LUMA_BGR)[m[k::n]].mean())}
        for k, name in enumerate(RECON)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--crop", required=True, help="camera:frame:x,y,w,h of the floor zoom")
    p.add_argument("--crop2", required=True, help="camera:frame:x,y,w,h of the person zoom")
    p.add_argument("--variants", default="off,lum,rgb")
    p.add_argument("--field_variant", default="lum")
    p.add_argument("--frame", type=int, default=140, help="frame of the fields panel")
    p.add_argument("--view_width", type=int, default=640)
    p.add_argument("--bound", type=float, default=0.15, help="color scale (log gain) of the gain panels")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"crops": {"crop": args.crop, "crop2": args.crop2}}

    for key, spec in (("crop", args.crop), ("crop2", args.crop2)):
        cam, frame, box = spec.split(":")
        x, y, w, h = (int(v) for v in box.split(","))
        frame = int(frame)
        views = {v: args.runs / v / "render" / cam / f"{frame:06d}.png" for v in args.variants.split(",")}
        views["gt"] = args.dataset / "clean" / f"{frame:06d}" / f"{cam}_color.png"
        scale = max(1, round(360 / max(w, h)))
        for name, path in views.items():
            img = cv2.imread(str(path))
            crop = cv2.resize(img[y:y + h, x:x + w], (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(args.out / f"{key}_{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if name == "gt":
                view = img.copy()
                cv2.rectangle(view, (x, y), (x + w, y + h), (40, 40, 230), 5)
                s = args.view_width / view.shape[1]
                view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(args.out / f"{key}_view.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 92])

    meta = json.loads((args.dataset / "cameras.json").read_text())
    cams = {c["name"]: c for c in meta["cameras"]}
    params = json.loads((args.dataset / "perturbation.json").read_text())["cameras"]
    dump = load_gain(args.runs / args.field_variant / "gain" / f"gain_{args.frame - meta['frames'][0]:06d}.txt")
    gh, gw = dump.shape[1:3]
    tw, th = 320, 180
    columns = []
    for k, name in enumerate(RECON):
        c = cams[name]
        inp = cv2.imread(str(args.dataset / "frames" / f"{args.frame:06d}" / f"{name}_color.jpg"))
        seen = observed_cells(args.dataset, name, args.frame, gw, gh)
        applied = applied_grid(params[name], c["width"], c["height"], gw, gh) @ LUMA_BGR
        recovered = recovered_grid(dump, k) @ LUMA_BGR
        col = [cv2.resize(inp, (tw, th), interpolation=cv2.INTER_AREA),
               field_image(applied, seen, (tw, th), args.bound),
               field_image(recovered, seen, (tw, th), args.bound)]
        columns.append(np.vstack([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in col]))
    cv2.imwrite(str(args.out / "fields.jpg"), np.hstack(columns), [cv2.IMWRITE_JPEG_QUALITY, 92])

    report["recovery"] = {v: recovery(args.dataset, args.runs, v) for v in ("lum", "rgb")
                          if (args.runs / v / "gain").exists()}
    (args.out / "recovery.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

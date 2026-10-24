# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Prepare a downloaded CMU Panoptic (Kinoptic) sequence for `cmu_panoptic`.

Input: a sequence directory as written by `panoptic_download.py` (or the
official `scripts/getData_kinoptic.sh` of the panoptic-toolbox):

    <seq_dir>/kcalibration_<seq>.json, calibration_<seq>.json
    <seq_dir>/synctables_<seq>.json, ksynctables_<seq>.json
    <seq_dir>/kinectVideos/kinect_50_KK.mp4
    <seq_dir>/kinect_shared_depth/KINECTNODE<K>/depthdata.dat

Steps (all run by default, select with --steps):

  extract  kinectVideos/kinect_50_KK.mp4 -> kinectImgs/50_KK/50_KK_NNNNNNNN.jpg
           (ffmpeg, JPEG -q:v 2, 1-based numbering as in the toolbox).
  export   For every HD frame, the nearest Kinect color and depth frame in
           universal time, written as
             <out_dir>/frames/hd_NNNNNNNN/kKK_rgb.jpg    (8-bit color)
             <out_dir>/frames/hd_NNNNNNNN/kKK_depth.png  (uint16, mm)
             <out_dir>/frames/hd_NNNNNNNN/meta.json      (sync record)
           plus the calibration and sync tables in <out_dir>/calib/.
  gt       Undistorted color images of the held-out Kinects,
           kKK_rgb.jpg.undist.jpg, the ground truth of the held-out
           re-rendering evaluation (python/eval_metrics.py).

`cmu_panoptic <out_root> <seq> <out_root>/<seq>/calib ...` then reads
<out_root>/<seq>/frames. The frame numbers on its command line are the HD
frame numbers (hd_NNNNNNNN), so a partial export (--hd_start/--hd_end) works
as long as it covers the reconstructed range.

The outputs are byte-identical to the exports used for the paper
(produced with OpenCV 4.11 and an ffmpeg 2026 build; ffmpeg only writes its
version into the intermediate JPEG comment).

Example (the two paper sequences, frames 400-2399 cover every experiment):

    python python/panoptic_prepare.py --seq_dir <dataset>/panoptic-toolbox/170407_office2 \
        --out_dir <dataset>/panoptic_export/170407_office2 --hd_start 0 --hd_end 2400
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEPTH_WIDTH, DEPTH_HEIGHT = 512, 424
PANEL = 50  # Kinect panel index in the Panoptic file names


@dataclass(frozen=True)
class KinectPick:
    kinect_id: int
    ok: bool
    reason: str
    color_index_1based: int | None
    depth_index_1based: int | None
    color_univ_time: float | None
    depth_univ_time: float | None
    color_dist: float | None
    depth_dist: float | None
    rgbd_time_diff: float | None
    rgb_path: str | None
    depth_path: str | None


def parse_id_list(s: str) -> list[int]:
    """"1-10" -> [1..10], "1,3,5" -> [1, 3, 5]."""
    s = s.strip()
    if "-" in s and "," not in s:
        lo, hi = (int(x) for x in s.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def extract_kinect_images(seq_dir: Path, kinect_ids: list[int], overwrite: bool) -> None:
    for k in kinect_ids:
        mp4 = seq_dir / "kinectVideos" / f"kinect_{PANEL}_{k:02d}.mp4"
        out_dir = seq_dir / "kinectImgs" / f"{PANEL}_{k:02d}"
        if not mp4.exists():
            print(f"[WARN] missing {mp4}")
            continue
        if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
            print(f"[extract] {out_dir} exists, skipped (--overwrite to redo)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-i", str(mp4), "-vsync", "0", "-q:v", "2",
               "-threads", "0", str(out_dir / f"{PANEL}_{k:02d}_%08d.jpg")]
        print("[extract]", " ".join(cmd))
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def read_depth_frame(depth_dat: Path, index_1based: int) -> np.ndarray:
    """One 512x424 uint16 frame of depthdata.dat, flipped left-right."""
    count = DEPTH_WIDTH * DEPTH_HEIGHT
    with depth_dat.open("rb") as f:
        f.seek(2 * count * (index_1based - 1))
        data = np.fromfile(f, dtype=np.uint16, count=count)
    if data.size != count:
        raise ValueError(f"depth frame {index_1based} out of range in {depth_dat}")
    return data.reshape((DEPTH_HEIGHT, DEPTH_WIDTH))[:, ::-1]


def median_dt(times: np.ndarray) -> float:
    dt = np.diff(times)
    dt = dt[np.isfinite(dt)]
    return float(np.median(dt)) if dt.size > 0 else 0.0


def export_rgbd(seq_dir: Path, out_dir: Path, kinect_ids: list[int],
                hd_start: int, hd_end: int | None) -> None:
    """HD-time-aligned Kinect RGB-D frames (nearest neighbor in univ_time)."""
    seq = seq_dir.name
    calib_out = out_dir / "calib"
    frames_out = out_dir / "frames"
    calib_out.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    ksync_path = seq_dir / f"ksynctables_{seq}.json"
    psync_path = seq_dir / f"synctables_{seq}.json"
    for p in (ksync_path, psync_path):
        if not p.exists():
            raise FileNotFoundError(p)
    for name in (ksync_path.name, psync_path.name, f"kcalibration_{seq}.json",
                 f"calibration_{seq}.json"):
        if (seq_dir / name).exists():
            shutil.copyfile(seq_dir / name, calib_out / name)

    ksync = load_json(ksync_path)
    hd_times = np.asarray(load_json(psync_path)["hd"]["univ_time"], dtype=np.float64)
    if hd_end is None:
        hd_end = int(hd_times.shape[0])

    # Accept a Kinect frame up to 2.5 frame periods (of KINECTNODE1) away
    # from the HD time.
    dt_color = median_dt(np.asarray(ksync["kinect"]["color"]["KINECTNODE1"]["univ_time"], dtype=np.float64))
    dt_depth = median_dt(np.asarray(ksync["kinect"]["depth"]["KINECTNODE1"]["univ_time"], dtype=np.float64))
    max_color_dist = 2.5 * dt_color if dt_color > 0 else 1e9
    max_depth_dist = 2.5 * dt_depth if dt_depth > 0 else 1e9
    thresholds = {
        "color_time_offset": 0.0,
        "max_color_dist": float(max_color_dist),
        "max_depth_dist": float(max_depth_dist),
        "dt_kc_median": float(dt_color),
        "dt_kd_median": float(dt_depth),
        "gate_mode": "none",
        "robust_sample_count": 2000,
        "robust_rgbd_per_kinect": {},
    }

    for hd_idx in range(hd_start, hd_end):
        t = float(hd_times[hd_idx])
        frame_dir = frames_out / f"hd_{hd_idx:08d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        picks: list[KinectPick] = []

        for k in kinect_ids:
            node = f"KINECTNODE{k}"
            try:
                color_times = np.asarray(ksync["kinect"]["color"][node]["univ_time"], dtype=np.float64)
                depth_times = np.asarray(ksync["kinect"]["depth"][node]["univ_time"], dtype=np.float64)
            except KeyError:
                picks.append(KinectPick(k, False, "Missing sync entry", *([None] * 9)))
                continue

            c0 = int(np.argmin(np.abs(color_times - t)))
            d0 = int(np.argmin(np.abs(depth_times - t)))
            c_dist = float(abs(color_times[c0] - t))
            d_dist = float(abs(depth_times[d0] - t))
            rgbd_diff = float(abs(depth_times[d0] - color_times[c0]))
            c1, d1 = c0 + 1, d0 + 1
            rgb_src = seq_dir / "kinectImgs" / f"{PANEL}_{k:02d}" / f"{PANEL}_{k:02d}_{c1:08d}.jpg"
            depth_dat = seq_dir / "kinect_shared_depth" / node / "depthdata.dat"

            def pick(ok: bool, reason: str, rgb_path: str, depth_path: str) -> KinectPick:
                return KinectPick(k, ok, reason, c1, d1, float(color_times[c0]),
                                  float(depth_times[d0]), c_dist, d_dist, rgbd_diff,
                                  rgb_path, depth_path)

            def reject(reason: str) -> None:
                print(f"[DEBUG] Kinect {k}: {reason}")
                picks.append(pick(False, reason, str(rgb_src), str(depth_dat)))

            if c_dist > max_color_dist:
                reject(f"Color time too far: {c_dist:.3f}")
                continue
            if d_dist > max_depth_dist:
                reject(f"Depth time too far: {d_dist:.3f}")
                continue
            if not rgb_src.exists():
                reject("RGB jpg missing")
                continue
            if not depth_dat.exists():
                reject("Depth dat missing")
                continue
            rgb = cv2.imread(str(rgb_src), cv2.IMREAD_COLOR)
            if rgb is None:
                reject("RGB decode failed")
                continue
            try:
                depth = read_depth_frame(depth_dat, d1)
            except Exception as e:  # noqa: BLE001 - recorded in meta.json
                reject(f"Depth read failed: {e}")
                continue

            rgb_dst = frame_dir / f"k{k:02d}_rgb.jpg"
            depth_dst = frame_dir / f"k{k:02d}_depth.png"
            cv2.imwrite(str(rgb_dst), rgb)
            cv2.imwrite(str(depth_dst), depth)
            picks.append(pick(True, "OK", str(rgb_dst.relative_to(out_dir)),
                              str(depth_dst.relative_to(out_dir))))

        meta = {"seq_name": seq, "hd_index_0based": hd_idx, "hd_univ_time": t,
                "gate_mode": "none", "thresholds": thresholds,
                "picks": [asdict(p) for p in picks]}
        (frame_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        ok_count = sum(1 for p in picks if p.ok)
        print(f"[export] hd {hd_idx} (t={t:.3f}): {ok_count}/{len(picks)} Kinects")


# ---------------------------------------------------------------------------
# gt
# ---------------------------------------------------------------------------
def undistort_ground_truth(out_dir: Path, kinect_ids: list[int], hd_start: int,
                           hd_end: int | None, overwrite: bool) -> None:
    """kKK_rgb.jpg -> kKK_rgb.jpg.undist.jpg (same K, distortion removed)."""
    calib_path = next((out_dir / "calib").glob("kcalibration_*.json"))
    sensors = load_json(calib_path)["sensors"]
    frame_dirs = sorted(p for p in (out_dir / "frames").iterdir()
                        if p.is_dir() and p.name.startswith("hd_"))
    for frame_dir in frame_dirs:
        hd_idx = int(frame_dir.name[3:])
        if hd_idx < hd_start or (hd_end is not None and hd_idx >= hd_end):
            continue
        for k in kinect_ids:
            src = frame_dir / f"k{k:02d}_rgb.jpg"
            dst = frame_dir / f"k{k:02d}_rgb.jpg.undist.jpg"
            if not src.exists() or (dst.exists() and not overwrite):
                continue
            s = sensors[k - 1]
            K = np.array(s["K_color"], dtype=np.float64)
            D = np.array(s["distCoeffs_color"], dtype=np.float64).flatten()
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            cv2.imwrite(str(dst), cv2.undistort(img, K, D, None, newCameraMatrix=K))
        print(f"[gt] {frame_dir.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seq_dir", type=Path, required=True, help="downloaded sequence, e.g. .../170407_office2")
    p.add_argument("--out_dir", type=Path, required=True, help="export root of this sequence, e.g. .../panoptic_export/170407_office2")
    p.add_argument("--steps", default="extract,export,gt", help="comma list of extract, export, gt")
    p.add_argument("--kinect_ids", default="1-10", help='Kinects to extract / export, e.g. "1-10"')
    p.add_argument("--hd_start", type=int, default=0, help="first HD frame to export")
    p.add_argument("--hd_end", type=int, default=-1, help="exclusive end HD frame (-1 = all)")
    p.add_argument("--gt_kinect_ids", default="1,3,5,7,9",
                   help="held-out Kinects whose undistorted images are the evaluation ground truth")
    p.add_argument("--gt_hd_start", type=int, default=400)
    p.add_argument("--gt_hd_end", type=int, default=700)
    p.add_argument("--overwrite", action="store_true", help="redo extract / gt outputs that exist")
    args = p.parse_args()

    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = steps - {"extract", "export", "gt"}
    if unknown:
        p.error(f"unknown steps: {sorted(unknown)}")
    kinect_ids = parse_id_list(args.kinect_ids)
    hd_end = None if args.hd_end < 0 else args.hd_end

    if "extract" in steps:
        extract_kinect_images(args.seq_dir, kinect_ids, args.overwrite)
    if "export" in steps:
        export_rgbd(args.seq_dir, args.out_dir, kinect_ids, args.hd_start, hd_end)
    if "gt" in steps:
        undistort_ground_truth(args.out_dir, parse_id_list(args.gt_kinect_ids),
                               args.gt_hd_start, args.gt_hd_end, args.overwrite)


if __name__ == "__main__":
    main()

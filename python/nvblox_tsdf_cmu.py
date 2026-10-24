# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.sensor import Sensor


# ----------------------------
# Panoptic export IO
# ----------------------------

@dataclass(frozen=True)
class Intrinsic:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = None  # Optional distortion coefficients (not used in this example)

@dataclass(frozen=True)
class CamCalib:
    kinect_id: int
    color_intr: Intrinsic
    T_color2world: np.ndarray  # 4x4 float64
    depth_intr: Intrinsic
    T_depth2world: np.ndarray  # 4x4 float64


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def guess_frame_root(processed_root: Path) -> Path:
    """
    panoptic_export style:
      processed_root/frames/hd_00000000/...
    or processed_root itself is frames/
    """
    if processed_root.name == "frames" and processed_root.exists():
        return processed_root
    frames = processed_root / "frames"
    if frames.exists():
        return frames
    raise FileNotFoundError(f"Could not find frames/ under: {processed_root}")


def iter_frame_dirs(frames_root: Path) -> list[Path]:
    dirs = [p for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("hd_")]
    dirs.sort(key=lambda p: p.name)
    return dirs


def read_color_rgb_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Missing color: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb  # uint8 RGB


def read_depth_u16_mm(path: Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Missing depth: {path}")
    if d.dtype != np.uint16 or d.ndim != 2:
        raise ValueError(f"Depth must be uint16 single-channel png: {path} got {d.dtype} shape={d.shape}")
    return d  # uint16 mm, 0 invalid


def find_paths_in_frame(frame_dir: Path, kid: int) -> tuple[Path, Path]:
    """
    panoptic_export naming (your pipeline):
      k00_rgb.jpg, k00_depth.png
    Keep robust matching like official examples (glob) but do not change dataset layout.
    """
    kid2 = f"{kid:02d}"
    # Preferred exact names
    candidates = [
        (frame_dir / f"k{kid2}_rgb.jpg", frame_dir / f"k{kid2}_depth.png"),
        (frame_dir / f"k{kid2}_rgb.png", frame_dir / f"k{kid2}_depth.png"),
        (frame_dir / f"k{kid2}.rgb.jpg", frame_dir / f"k{kid2}.depth.png"),
    ]
    for c, d in candidates:
        if c.exists() and d.exists():
            return c, d

    # Fallback glob (robust)
    color_hits = sorted(
        list(frame_dir.glob(f"*k{kid2}*rgb*.jpg"))
        + list(frame_dir.glob(f"*k{kid2}*rgb*.png"))
        + list(frame_dir.glob(f"*k{kid2}*color*.jpg"))
        + list(frame_dir.glob(f"*k{kid2}*color*.png"))
    )
    depth_hits = sorted(
        list(frame_dir.glob(f"*k{kid2}*depth*.png"))
        + list(frame_dir.glob(f"*k{kid2}*depth*.tif"))
    )
    if not color_hits or not depth_hits:
        raise FileNotFoundError(
            f"Could not find rgb/depth for k{kid2} in {frame_dir}\n"
            f"  color hits: {color_hits}\n"
            f"  depth hits: {depth_hits}"
        )
    return color_hits[0], depth_hits[0]


def load_panoptic_kcalibration(calib_json_path: Path, kinects: list[int]) -> list[CamCalib]:
    """
    Parse CMU Panoptic calibration json like:
      kcalibration_170407_office2.json

    sensors[i] == kinect i
    M_world2sensor : world -> camera
    """
    obj = read_json(calib_json_path)

    if "sensors" not in obj:
        raise ValueError(f'Invalid panoptic calib json: missing "sensors": {calib_json_path}')

    sensors = obj["sensors"]
    out: list[CamCalib] = []

    for kid in kinects:
        if kid < 1 or kid > len(sensors):
            raise IndexError(f"Kinect id {kid} out of range (1..{len(sensors)})")

        s = sensors[kid - 1]  # sensors[0] is kinect 1, etc.

        # ---- color intrinsics ----
        Kc = np.array(s["K_color"], dtype=np.float64)
        cdist = s.get("distCoeffs_color", None)
        if cdist is not None:
            cdist = np.array(cdist, dtype=np.float64).flatten().tolist()
        cintr = Intrinsic(
            width=int(s["color_width"]),
            height=int(s["color_height"]),
            fx=float(Kc[0, 0]),
            fy=float(Kc[1, 1]),
            cx=float(Kc[0, 2]),
            cy=float(Kc[1, 2]),
            dist=cdist
        )

        # ---- extrinsics ----
        # M_world2sensor : world -> camera
        M_world2sensor = np.array(s["M_world2sensor"], dtype=np.float64).reshape(4, 4)

        sensor2color = np.array(s["M_color"], dtype=np.float64).reshape(4, 4)
        T_world2color = sensor2color @ M_world2sensor
        # We need camera -> world
        T_color2world = np.linalg.inv(T_world2color)

        Kd = np.array(s["K_depth"], dtype=np.float64)
        ddist = s.get("distCoeffs_depth", None)
        if ddist is not None:
            ddist = np.array(ddist, dtype=np.float64).flatten().tolist()[:5]
        # ---- depth intrinsics/extrinsics ----
        Kd = np.array(s["K_depth"], dtype=np.float64)
        dintr = Intrinsic(
            width=int(s["depth_width"]),
            height=int(s["depth_height"]),
            fx=float(Kd[0, 0]),
            fy=float(Kd[1, 1]),
            cx=float(Kd[0, 2]),
            cy=float(Kd[1, 2]),
            dist=ddist
        )

        sensor2depth = np.array(s["M_depth"], dtype=np.float64).reshape(4, 4)
        T_world2depth = sensor2depth @ M_world2sensor
        T_depth2world = np.linalg.inv(T_world2depth)


        out.append(
            CamCalib(
                kinect_id=kid,
                color_intr=cintr,
                T_color2world=T_color2world,
                depth_intr=dintr,
                T_depth2world=T_depth2world
            )
        )

        print(out[-1])

        print(f"[INFO] Loaded Panoptic calib k{kid:02d}")

    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    # Match BEHAVE script feel: processed_root + kinects + range + tsdf params
    ap.add_argument("--processed_root", type=Path, required=True, help="panoptic_export sequence dir (contains frames/ and calibs/).")
    ap.add_argument("--kinects", type=str, required=True, help="Comma-separated kinect ids, e.g. 0,1,2,3,4,5,6,7,8,9")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--max_views", type=int, default=0, help="Limit views per frame (0=all).")

    ap.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size (m).")
    ap.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale (mm => 1000).")
    ap.add_argument("--max_integration_distance", type=float, default=6.0, help="Max integration distance (m).")

    ap.add_argument("--out_dir", type=Path, default=Path("nvblox_torch_out_panoptic"))
    ap.add_argument("--write_every", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Allow overriding calib json location (optional)
    ap.add_argument("--calib_json", type=Path, default=None, help="Override kcalibration.json path.")

    args = ap.parse_args()

    if args.device != "cuda":
        raise RuntimeError("nvblox_torch Mapper expects CUDA tensors in typical builds.")

    kinects = [int(x) for x in args.kinects.split(",") if x.strip() != ""]

    processed_root = args.processed_root.resolve()
    frames_root = guess_frame_root(processed_root)
    frame_dirs = iter_frame_dirs(frames_root)
    if not frame_dirs:
        raise RuntimeError(f"No hd_ frame folders found under {frames_root}")

    start = max(0, args.start)
    end = min(len(frame_dirs), start + max(0, args.count))

    if start >= end:
        raise ValueError("Invalid --start/--count")

    calib_json = args.calib_json
    if calib_json is None:
        #calib_json = args.processed_root / "calibs" / "kcalibration.json"
        for json_path in args.processed_root.glob("calib/kcalibration_*.json"):
            if json_path is not None and json_path.exists():
                calib_json = json_path
                break
    if calib_json is None or not calib_json.exists():
        raise FileNotFoundError(f"Missing calibration json: {calib_json}")

    calibs = load_panoptic_kcalibration(calib_json, kinects)

    # Build one Sensor per camera (intrinsics)
    sensors: dict[int, Sensor] = {}
    depth_sensors: dict[int, Sensor] = {}
    for cam in calibs:
        intr = cam.color_intr
        sensors[cam.kinect_id] = Sensor.from_camera(
            fu=intr.fx, fv=intr.fy, cu=intr.cx, cv=intr.cy, width=intr.width, height=intr.height
        )
        dintr = cam.depth_intr
        depth_sensors[cam.kinect_id] = Sensor.from_camera(
            fu=dintr.fx, fv=dintr.fy, cu=dintr.cx, cv=dintr.cy, width=dintr.width, height=dintr.height
        )

    # Mapper params (as in sun3d.py / behave script)
    projective_params = ProjectiveIntegratorParams()
    projective_params.projective_integrator_max_integration_distance_m = float(args.max_integration_distance)
    projective_params.projective_integrator_truncation_distance_vox = 4

    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_params)
    mapper = Mapper(
        voxel_sizes_m=float(args.voxel_size),
        mapper_parameters=mapper_params,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    #out_dir = args.out_dir / str()

    print(f"[INFO] frames_root={frames_root} frames={len(frame_dirs)} using [{start},{end})")
    print(f"[INFO] kinects={kinects} voxel={args.voxel_size} max_integration_distance={args.max_integration_distance}")
    print(f"[INFO] calib_json={calib_json}")

    total_int = 0.0
    total_mesh = 0.0
    all_times = []
    numpy_to_torch_times: list[float] = []
    add_depth_times: list[float] = []
    add_color_times: list[float] = []
    update_color_mesh_times: list[float] = []
    get_color_mesh_times: list[float] = []
    misc_times: list[float] = []

    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    log_path = args.out_dir / f"timings_{timestamp_str}.csv"
    f = open(log_path, "w", encoding="utf-8")
    f.write("frame,all_ms,numpy_to_torch_ms,add_depth_ms,add_color_ms,update_color_mesh_ms,get_color_mesh_ms,misc_ms\n")

    for i in range(start, end):
        fdir = frame_dirs[i]
        print(f"[INFO] Frame {i}: {fdir.name}")

        # Preload to reduce per-view I/O overhead variance (mirrors your BEHAVE script style)
        colors: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        valid_calibs = []
        for cam in calibs:
            try:
                c_path, d_path = find_paths_in_frame(fdir, cam.kinect_id)
            except FileNotFoundError as e:
                print(e)
                continue  
            rgb_u8 = read_color_rgb_u8(c_path)
            depth_u16 = read_depth_u16_mm(d_path)
            depth_m = depth_u16.astype(np.float32) / float(args.depth_scale)
            depth_m[(depth_u16 == 0) | (depth_m <= 0.0)] = 0.0
            # cv2.imshow("color", rgb_u8)
            # cv2.imshow("depth", ((depth_m / args.max_integration_distance) * 255).astype(np.uint8))
            # cv2.waitKey(-1)
            if cam.color_intr.dist is not None:
                # Note: This is a simple undistort that may not be perfect for all distortion models.
                # For more complex distortion, consider using OpenCV's fisheye module or custom undistortion.
                K = np.array([[cam.color_intr.fx, 0, cam.color_intr.cx], [0, cam.color_intr.fy, cam.color_intr.cy], [0, 0, 1]])
                D = np.array(cam.color_intr.dist)
                # h, w = rgb_u8.shape[:2]
                # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
                rgb_u8 = cv2.undistort(rgb_u8, K, D, None, newCameraMatrix=K)
            if cam.depth_intr.dist is not None:
                K = np.array([[cam.depth_intr.fx, 0, cam.depth_intr.cx], [0, cam.depth_intr.fy, cam.depth_intr.cy], [0, 0, 1]])
                D = np.array(cam.depth_intr.dist)
                h, w = depth_m.shape[:2]

                # 歪み補正マップを作成
                map1, map2 = cv2.initUndistortRectifyMap(
                    cameraMatrix=K,
                    distCoeffs=D,
                    R=None,
                    newCameraMatrix=K,
                    size=(w, h),
                    m1type=cv2.CV_32FC1
                )

                # 最近傍補完で remap
                depth_undistorted = cv2.remap(
                    depth_m,
                    map1,
                    map2,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                depth_m = depth_undistorted
            colors.append(rgb_u8)
            depths.append(depth_m)
            valid_calibs.append(cam)
        t_start = time.perf_counter()

        mapper.clear()
        # mapper_params = MapperParams()
        # mapper_params.set_projective_integrator_params(projective_params)
        # mapper = Mapper(
        #     voxel_sizes_m=float(args.voxel_size),
        #     mapper_parameters=mapper_params,
        # )

        t0 = time.perf_counter()
        used = 0

        t_numpy_to_torch: list[float] = []
        t_add_depth: list[float] = []
        t_add_color: list[float] = []

        for cam, rgb_u8, depth_m in zip(valid_calibs, colors, depths):
            t00 = time.perf_counter()
            depth_t = torch.from_numpy(depth_m).to(device="cuda", dtype=torch.float32)
            rgb_t = torch.from_numpy(rgb_u8).to(device="cuda", dtype=torch.uint8)

            # pose must be CPU 4x4 (nvblox_torch docstring)
            t_w_c = torch.from_numpy(cam.T_color2world.astype(np.float32)).to(device="cpu", dtype=torch.float32)
            t_w_d = torch.from_numpy(cam.T_depth2world.astype(np.float32)).to(device="cpu", dtype=torch.float32)

            sensor = sensors[cam.kinect_id]
            depth_sensor = depth_sensors[cam.kinect_id]
            t01 = time.perf_counter()
            t_numpy_to_torch.append(t01 - t00)

            t02 = time.perf_counter()
            # print(depth_sensor)
            # print(depth_t.shape, depth_t.dtype)
            # print(t_w_d.shape, t_w_d.dtype)
            # print(sensor)
            # print(rgb_t.shape, rgb_t.dtype)
            # hoge
            mapper.add_depth_frame(depth_t, t_w_d, depth_sensor)
            torch.cuda.synchronize()
            t03 = time.perf_counter()
            t_add_depth.append(t03 - t02)

            t03b = time.perf_counter()
            mapper.add_color_frame(rgb_t, t_w_c, sensor)
            torch.cuda.synchronize()
            t04 = time.perf_counter()
            t_add_color.append(t04 - t03b)

            used += 1
            if args.max_views > 0 and used >= int(args.max_views):
                break

        t1 = time.perf_counter()

        t11 = time.perf_counter()
        mapper.update_color_mesh()
        torch.cuda.synchronize()
        t12 = time.perf_counter()
        mesh = mapper.get_color_mesh()
        torch.cuda.synchronize()
        t22 = time.perf_counter()
        t2 = time.perf_counter()

        t_end = time.perf_counter()

        int_s = t1 - t0
        mesh_s = t2 - t1

        print(f"[TIME] views={used} integrate={int_s*1000:.1f} ms")
        print(f"[TIME]  - numpy_to_torch={np.sum(t_numpy_to_torch)*1000:.1f} ms add_depth={np.sum(t_add_depth)*1000:.1f} ms add_color={np.sum(t_add_color)*1000:.1f} ms")
        print(f"[TIME] views={used} mesh_update={mesh_s*1000:.1f} ms")
        print(f"[TIME]  - update_color_mesh={((t12-t11)*1000):.1f} ms")
        print(f"[INFO]  - get_color_mesh={((t22-t12)*1000):.1f} ms")

        all_times.append((t_end - t_start)*1000)
        numpy_to_torch_times.append(np.sum(t_numpy_to_torch)*1000)
        add_depth_times.append(np.sum(t_add_depth)*1000)
        add_color_times.append(np.sum(t_add_color)*1000)
        update_color_mesh_times.append((t12 - t11)*1000)
        get_color_mesh_times.append((t22 - t12)*1000)
        all_time = all_times[-1]
        numpy_to_torch_time = numpy_to_torch_times[-1]
        add_depth_time = add_depth_times[-1]
        add_color_time = add_color_times[-1]
        update_color_mesh_time = update_color_mesh_times[-1]
        get_color_mesh_time = get_color_mesh_times[-1]
        misc_time = all_time - (numpy_to_torch_time + add_depth_time + add_color_time + update_color_mesh_time + get_color_mesh_time)
        misc_times.append(misc_time)
        f.write(f"{i},{all_time},{numpy_to_torch_time},{add_depth_time},{add_color_time},{update_color_mesh_time},{get_color_mesh_time},{misc_time},\n")
        f.flush()
        
        total_int += int_s
        total_mesh += mesh_s

        if args.write_every > 0 and ((i - start) % args.write_every == 0):
            out_ply = args.out_dir / f"{fdir.name}.ply"
            mesh.save(str(out_ply))
            print(f"[INFO] Wrote mesh: {out_ply}")

    n = end - start
    if n > 0:
        print(f"[SUMMARY] frames={n} avg_integrate={(total_int/n)*1000:.1f} ms avg_mesh={(total_mesh/n)*1000:.1f} ms")
        print(f"[SUMMARY] out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
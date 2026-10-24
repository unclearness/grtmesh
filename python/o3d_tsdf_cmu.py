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
import open3d as o3d


# ----------------------------
# Panoptic export IO (same shape as your nvblox script)
# ----------------------------

@dataclass(frozen=True)
class Intrinsic:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray | None = None  # (N,) OpenCV distortion coefficients


@dataclass(frozen=True)
class CamCalib:
    kinect_id: int
    color_intr: Intrinsic
    T_color2world: np.ndarray  # 4x4 float64
    depth_intr: Intrinsic
    T_depth2world: np.ndarray  # 4x4 float64
    T_world2color: np.ndarray  # 4x4 float64
    T_world2depth: np.ndarray  # 4x4 float64


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


def _k_mat(intr: Intrinsic) -> np.ndarray:
    return np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def undistort_image(img: np.ndarray, intr: Intrinsic, is_depth: bool) -> np.ndarray:
    """Undistort color/depth using OpenCV. If intr.dist is None, returns img unchanged."""
    if intr.dist is None:
        return img
    K = _k_mat(intr)
    D = intr.dist.astype(np.float64)
    return cv2.undistort(img, K, D, None, K)

def undistort_depth(depth_u16: np.ndarray, intr: Intrinsic) -> np.ndarray:
    """Undistort depth image (uint16 mm) using OpenCV. Returns uint16 mm."""
    if intr.dist is None:
        return depth_u16
    K = _k_mat(intr)
    D = intr.dist.astype(np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K,
        distCoeffs=D,
        R=None,
        newCameraMatrix=K,
        size=(intr.width, intr.height),
        m1type=cv2.CV_32FC1
    )
    depth_undistorted = cv2.remap(
        depth_u16,
        map1,
        map2,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return depth_undistorted

def fill_depth_holes_u16_mm(depth_u16: np.ndarray, max_iter: int = 6, ksize: int = 3) -> np.ndarray:
    """Simple hole filling for uint16 depth (0 = invalid). Conservative: iterative dilation."""
    if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
        raise ValueError("depth_u16 must be uint16 HxW")
    out = depth_u16.copy()
    kernel = np.ones((ksize, ksize), np.uint8)
    for _ in range(max_iter):
        holes = out == 0
        if not np.any(holes):
            break
        dil = cv2.dilate(out, kernel, iterations=1)
        out[holes] = dil[holes]
    return out


def map_depth_to_color_u16_mm(
    depth_u16_mm: np.ndarray,
    depth_intr: Intrinsic,
    T_depth2color: np.ndarray,
    color_intr: Intrinsic,
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Project depth image into the color camera (both assumed undistorted). Output is uint16 mm in color resolution."""
    if depth_u16_mm.dtype != np.uint16 or depth_u16_mm.ndim != 2:
        raise ValueError("depth_u16_mm must be uint16 HxW")
    Hc, Wc = out_hw

    z_m = depth_u16_mm.astype(np.float32) / 1000.0
    valid = z_m > 0
    if not np.any(valid):
        return np.zeros((Hc, Wc), dtype=np.uint16)

    ys, xs = np.nonzero(valid)
    z = z_m[ys, xs]
    x = (xs.astype(np.float32) - float(depth_intr.cx)) / float(depth_intr.fx) * z
    y = (ys.astype(np.float32) - float(depth_intr.cy)) / float(depth_intr.fy) * z
    ones = np.ones_like(z, dtype=np.float32)
    P = np.stack([x, y, z, ones], axis=0)  # 4xN

    Tc = T_depth2color.astype(np.float32)
    Pc = Tc @ P
    xc, yc, zc = Pc[0, :], Pc[1, :], Pc[2, :]
    in_front = zc > 0
    if not np.any(in_front):
        return np.zeros((Hc, Wc), dtype=np.uint16)
    xc = xc[in_front]
    yc = yc[in_front]
    zc = zc[in_front]

    u = (xc / zc) * float(color_intr.fx) + float(color_intr.cx)
    v = (yc / zc) * float(color_intr.fy) + float(color_intr.cy)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
    ui = ui[inside]
    vi = vi[inside]
    zc = zc[inside]
    if ui.size == 0:
        return np.zeros((Hc, Wc), dtype=np.uint16)

    out_z = np.full((Hc, Wc), np.inf, dtype=np.float32)
    np.minimum.at(out_z, (vi, ui), zc)
    out_mm = np.where(np.isfinite(out_z), np.clip(out_z * 1000.0, 0.0, 65535.0).astype(np.uint16), 0)
    return out_mm


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
    Keep robust matching but do not change dataset layout.
    """
    kid2 = f"{kid:02d}"
    candidates = [
        (frame_dir / f"k{kid2}_rgb.jpg", frame_dir / f"k{kid2}_depth.png"),
        (frame_dir / f"k{kid2}_rgb.png", frame_dir / f"k{kid2}_depth.png"),
        (frame_dir / f"k{kid2}.rgb.jpg", frame_dir / f"k{kid2}.depth.png"),
    ]
    for c, d in candidates:
        if c.exists() and d.exists():
            return c, d

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


def _get_dist_coeffs(sensor: dict[str, Any], key_candidates: list[str]) -> np.ndarray | None:
    for k in key_candidates:
        if k in sensor:
            arr = np.array(sensor[k], dtype=np.float64).reshape(-1)
            if arr.size > 0:
                return arr
    return None


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
        color_dist = _get_dist_coeffs(s, ["distCoeffs_color", "dist_color", "D_color", "color_distCoeffs"])
        cintr = Intrinsic(
            width=int(s["color_width"]),
            height=int(s["color_height"]),
            fx=float(Kc[0, 0]),
            fy=float(Kc[1, 1]),
            cx=float(Kc[0, 2]),
            cy=float(Kc[1, 2]),
            dist=color_dist,
        )

        # ---- extrinsics ----
        # M_world2sensor : world -> camera
        M_world2sensor = np.array(s["M_world2sensor"], dtype=np.float64).reshape(4, 4)

        sensor2color = np.array(s["M_color"], dtype=np.float64).reshape(4, 4)
        T_world2color = sensor2color @ M_world2sensor
        T_color2world = np.linalg.inv(T_world2color)

        # ---- depth intrinsics/extrinsics ----
        Kd = np.array(s["K_depth"], dtype=np.float64)
        depth_dist = _get_dist_coeffs(s, ["distCoeffs_depth", "dist_depth", "D_depth", "depth_distCoeffs"])
        depth_dist = depth_dist[:5]
        dintr = Intrinsic(
            width=int(s["depth_width"]),
            height=int(s["depth_height"]),
            fx=float(Kd[0, 0]),
            fy=float(Kd[1, 1]),
            cx=float(Kd[0, 2]),
            cy=float(Kd[1, 2]),
            dist=depth_dist,
        )

        sensor2depth = np.array(s["M_depth"], dtype=np.float64).reshape(4, 4)
        T_world2depth = sensor2depth @ M_world2sensor
        T_depth2world = np.linalg.inv(T_world2depth)

        out.append(
            CamCalib(
                kinect_id=kid,
                color_intr=cintr,
                T_world2color=T_world2color,
                T_color2world=T_color2world,
                depth_intr=dintr,
                T_world2depth=T_world2depth,
                T_depth2world=T_depth2world,
            )
        )
        print(f"[INFO] Loaded Panoptic calib k{kid:02d}")

    return out


def _make_pinhole_intr(intr: Intrinsic) -> o3d.camera.PinholeCameraIntrinsic:
    pin = o3d.camera.PinholeCameraIntrinsic()
    pin.set_intrinsics(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
    return pin


def main() -> None:
    ap = argparse.ArgumentParser()
    # Keep the interface aligned with nvblox_tsdf_cmu.py
    ap.add_argument("--processed_root", type=Path, required=True, help="panoptic_export sequence dir (contains frames/ and calibs/).")
    ap.add_argument("--kinects", type=str, required=True, help="Comma-separated kinect ids, e.g. 0,1,2,3,4,5,6,7,8,9")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--max_views", type=int, default=0, help="Limit views per frame (0=all).")

    ap.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size (m).")
    ap.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale (mm => 1000).")
    ap.add_argument("--max_integration_distance", type=float, default=6.0, help="Max integration distance (m).")

    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--write_every", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Allow overriding calib json location (optional)
    ap.add_argument("--calib_json", type=Path, default=None, help="Override kcalibration.json path.")

    ap.add_argument("--uniform", action="store_true", help="Use UniformTSDFVolume instead of ScalableTSDFVolume (hashing).")

    # Important: Panoptic has separate color/depth cameras. Open3D classic TSDF assumes aligned RGB-D.
    # Default is depth-only fusion (geometry). Enable --use_color only if your RGB is already registered to depth.
    ap.add_argument("--use_color", action="store_true", help="Integrate color (requires RGB aligned to depth camera).")
    ap.add_argument(
        "--align_depth_to_color",
        action="store_true",
        help=(
            "If enabled, preprocess per view: undistort color, undistort depth, then project depth into the (undistorted) color image coordinates. "
            "This creates a registered RGB-D for Open3D TSDF integration using the color camera intrinsics/extrinsics."
        ),
    )
    ap.add_argument("--hole_fill_iter", type=int, default=6, help="Hole filling iterations for aligned depth (uint16 mm).")

    args = ap.parse_args()

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
        for json_path in args.processed_root.glob("calib/kcalibration_*.json"):
            if json_path.exists():
                calib_json = json_path
                break
    if calib_json is None or not calib_json.exists():
        raise FileNotFoundError(f"Missing calibration json: {calib_json}")

    calibs = load_panoptic_kcalibration(calib_json, kinects)

    # Open3D volume (classic pipeline): CPU only in most builds.
    # We'll keep --device for interface compatibility, and warn if cuda is requested.
    if args.device == "cuda":
        print("[WARN] Open3D classic TSDF (pipelines.integration) is usually CPU-only. "
              "If you need CUDA, consider Open3D tensor pipeline or your nvblox version.")

    if args.out_dir is None:
        args.out_dir = Path(f"o3d_tsdf_output") / str(args.voxel_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] frames_root={frames_root} frames={len(frame_dirs)} using [{start},{end})")
    print(f"[INFO] kinects={kinects} voxel={args.voxel_size} max_integration_distance={args.max_integration_distance}")
    print(f"[INFO] calib_json={calib_json}")
    print(f"[INFO] use_color={args.use_color} (default depth-only is recommended for Panoptic)")
    print(f"[INFO] align_depth_to_color={args.align_depth_to_color} hole_fill_iter={args.hole_fill_iter}")
    if args.align_depth_to_color and not args.use_color:
        print("[INFO] --align_depth_to_color is ON: color will be integrated regardless of --use_color.")

    # CSV log (compatible columns + more Open3D-specific timings)
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    if args.out_dir is None:
        args.out_dir = Path(f"o3d_tsdf_output") 
    log_path = args.out_dir / f"timings_{timestamp_str}.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(
            "frame,all_ms,io_ms,rgbd_ms,integrate_ms,extract_mesh_ms,post_ms,write_ms,misc_ms\n"
        )

        total_integrate = 0.0
        total_extract = 0.0

        for i in range(start, end):
            fdir = frame_dirs[i]
            print(f"[INFO] Frame {i}: {fdir.name}")

            t_all0 = time.perf_counter()

            # 1) I/O (preload)
            t_io0 = time.perf_counter()
            colors_u8: list[np.ndarray] = []
            depths_u16: list[np.ndarray] = []
            valid_calibs = []
            for cam in calibs:
                try:
                    c_path, d_path = find_paths_in_frame(fdir, cam.kinect_id)
                except FileNotFoundError as e:
                    print(e)
                    continue  
                need_color = bool(args.use_color or args.align_depth_to_color)
                if need_color:
                    colors_u8.append(read_color_rgb_u8(c_path))
                else:
                    # Avoid reading color if not used to reduce variance / match compute focus
                    colors_u8.append(None)  # type: ignore[arg-type]
                depths_u16.append(read_depth_u16_mm(d_path))
                valid_calibs.append(cam)
            t_io1 = time.perf_counter()

            # 2) Create TSDF volume per frame (match mapper.clear() behavior)
            trunc = 4.0 * float(args.voxel_size)
            if args.uniform:
                resolution = int(6.0 / args.voxel_size)
                length = float(args.voxel_size * resolution)
                volume = o3d.pipelines.integration.UniformTSDFVolume(
                    length=length,
                    resolution=resolution,
                    sdf_trunc=float(trunc),
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                    if args.use_color or args.align_depth_to_color
                    else o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
                    origin=np.array([ -2.8, -1.2, -0.2], dtype=np.float64)
                )
            else:
                volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=float(args.voxel_size),
                sdf_trunc=float(trunc),
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                if args.use_color or args.align_depth_to_color
                else o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
                )

            # 3) RGBD creation + integration (per view)
            t_rgbd0 = time.perf_counter()
            integrate_s = 0.0
            rgbd_s = 0.0

            used = 0
            for cam, rgb_u8, depth_u16 in zip(valid_calibs, colors_u8, depths_u16):
                # --- Optional preprocess: undistort + map depth -> (undistorted) color coordinates ---
                if args.align_depth_to_color:
                    if rgb_u8 is None:
                        raise RuntimeError("--align_depth_to_color requires RGB input per view")

                    rgb_ud = undistort_image(rgb_u8, cam.color_intr, is_depth=False)
                    depth_ud = undistort_depth(depth_u16, cam.depth_intr)

                    T_world2color = np.linalg.inv(cam.T_color2world)
                    T_depth2color = T_world2color @ cam.T_depth2world
                    aligned_depth_mm = map_depth_to_color_u16_mm(
                        depth_ud,
                        cam.depth_intr,
                        T_depth2color,
                        cam.color_intr,
                        out_hw=(cam.color_intr.height, cam.color_intr.width),
                    )
                    aligned_depth_mm = fill_depth_holes_u16_mm(aligned_depth_mm, max_iter=int(args.hole_fill_iter))

                    color_o3d = o3d.geometry.Image(rgb_ud)
                    depth_o3d = o3d.geometry.Image(aligned_depth_mm)

                    t_rgbd00 = time.perf_counter()
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_o3d,
                        depth_o3d,
                        depth_scale=1000.0,
                        depth_trunc=float(args.max_integration_distance),
                        convert_rgb_to_intensity=False,
                    )
                    t_rgbd01 = time.perf_counter()
                    rgbd_s += (t_rgbd01 - t_rgbd00)

                    intrinsic = _make_pinhole_intr(cam.color_intr)
                    extrinsic = cam.T_world2color.astype(np.float64)

                else:
                    depth_o3d = o3d.geometry.Image(depth_u16)

                    if args.use_color:
                        if rgb_u8 is None:
                            raise RuntimeError("Internal error: rgb_u8 is None while --use_color is set.")
                        if rgb_u8.shape[0] != depth_u16.shape[0] or rgb_u8.shape[1] != depth_u16.shape[1]:
                            rgb_u8_rs = cv2.resize(rgb_u8, (depth_u16.shape[1], depth_u16.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            rgb_u8_rs = rgb_u8
                        color_o3d = o3d.geometry.Image(rgb_u8_rs)
                    else:
                        color_o3d = o3d.geometry.Image(np.zeros((depth_u16.shape[0], depth_u16.shape[1], 3), dtype=np.uint8))

                    t_rgbd00 = time.perf_counter()
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_o3d,
                        depth_o3d,
                        depth_scale=float(args.depth_scale),
                        depth_trunc=float(args.max_integration_distance),
                        convert_rgb_to_intensity=False,
                    )
                    t_rgbd01 = time.perf_counter()
                    rgbd_s += (t_rgbd01 - t_rgbd00)

                    intrinsic = _make_pinhole_intr(cam.depth_intr)
                    extrinsic = cam.T_world2depth.astype(np.float64)

                t_int0 = time.perf_counter()
                volume.integrate(rgbd, intrinsic, extrinsic)
                t_int1 = time.perf_counter()
                integrate_s += (t_int1 - t_int0)

                used += 1
                if args.max_views > 0 and used >= int(args.max_views):
                    break

            t_rgbd1 = time.perf_counter()

            # 4) Extract mesh (Marching Cubes inside Open3D)
            t_mesh0 = time.perf_counter()
            mesh = volume.extract_triangle_mesh()
            t_mesh1 = time.perf_counter()

            # 5) Postprocess (normals etc.)
            t_post0 = time.perf_counter()
            mesh.compute_vertex_normals()
            t_post1 = time.perf_counter()

            # 6) Write
            t_wr0 = time.perf_counter()
            wrote = False
            if args.write_every > 0 and ((i - start) % args.write_every == 0):
                out_ply = args.out_dir / f"{fdir.name}.ply"
                o3d.io.write_triangle_mesh(str(out_ply), mesh, write_ascii=False, compressed=False)
                wrote = True
                print(f"[INFO] Wrote mesh: {out_ply}")
            t_wr1 = time.perf_counter()

            t_all1 = time.perf_counter()

            # Timing summary
            all_ms = (t_all1 - t_all0) * 1000.0
            io_ms = (t_io1 - t_io0) * 1000.0
            rgbd_ms = rgbd_s * 1000.0
            integrate_ms = integrate_s * 1000.0
            extract_ms = (t_mesh1 - t_mesh0) * 1000.0
            post_ms = (t_post1 - t_post0) * 1000.0
            write_ms = (t_wr1 - t_wr0) * 1000.0
            misc_ms = all_ms - (io_ms + rgbd_ms + integrate_ms + extract_ms + post_ms + write_ms)

            all_ms = integrate_ms + extract_ms + post_ms  # Recompute total for logging consistency

            print(f"[TIME] views={used} io={io_ms:.1f} ms rgbd={rgbd_ms:.1f} ms integrate={integrate_ms:.1f} ms")
            print(f"[TIME] mesh_extract={extract_ms:.1f} ms post={post_ms:.1f} ms write={write_ms:.1f} ms (wrote={wrote})")

            f.write(f"{i},{all_ms},{io_ms},{rgbd_ms},{integrate_ms},{extract_ms},{post_ms},{write_ms},{misc_ms}\n")

            total_integrate += integrate_s
            total_extract += (t_mesh1 - t_mesh0)

        n = end - start
        if n > 0:
            print(f"[SUMMARY] frames={n} avg_integrate={(total_integrate/n)*1000.0:.1f} ms avg_extract={(total_extract/n)*1000.0:.1f} ms")
            print(f"[SUMMARY] out_dir={args.out_dir}")
            print(f"[SUMMARY] timings_csv={log_path}")


if __name__ == "__main__":
    main()

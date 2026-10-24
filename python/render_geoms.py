# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BLENDER_PATH = "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"  # Update this path as needed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=Path, default=Path(BLENDER_PATH), help="Path to blender executable")
    parser.add_argument("--blender_script", type=Path, default=Path(__file__).parent / "render_geoms_bl.py")
    parser.add_argument("--calib_json", type=Path, default=None, help="Panoptic kcalibration_*.json")
    parser.add_argument("--cameras_json", type=Path, default=None,
                        help="cameras.json of the generic RGB-D layout (instead of --calib_json)")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE_NEXT", choices=["BLENDER_EEVEE_NEXT", "CYCLES"])
    parser.add_argument("--samples", type=int, default=1, help="Cycles samples (ignored for Eevee)")
    parser.add_argument("--transparent_bg", action="store_true", help="Render with transparent background")
    parser.add_argument("--camera_ids", type=str, default="",
                        help="comma-separated 1-based Kinect ids or camera names (default: all)")
    parser.add_argument("--frames", type=int, nargs=2, default=[400, 2400], metavar=("START", "END"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.blender),
        "-b",
        "-P",
        str(args.blender_script),
        "--",
        "--input_dir",
        str(args.input_dir.resolve()),  # Blender resolves relative paths against its own cwd
        "--output_dir",
        str(args.output_dir.resolve()),
        "--engine",
        args.engine,
        "--samples",
        str(args.samples),
    ]
    if args.calib_json is not None:
        cmd += ["--calib_json", str(args.calib_json.resolve())]
    if args.cameras_json is not None:
        cmd += ["--cameras_json", str(args.cameras_json.resolve())]
    cmd += ["--frames", str(args.frames[0]), str(args.frames[1])]
    if args.transparent_bg:
        cmd.append("--transparent_bg")
    if args.camera_ids:
        cmd += ["--camera_ids", args.camera_ids]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Orbit free-view render of a per-frame mesh sequence with Blender.

Wraps render_movie_bl.py with the camera path used for the CMU Panoptic
free-view videos (10 turns over frames 400-2400 around the dome center).

    python python/render_movie.py --input-dir <out_root>/<data_name>mesh \
        --output-dir renders [--mode obj|ply] [--blender <path to blender>]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="directory with NNNNNN.obj (or hd_NNNNNNNN.ply) meshes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("obj", "ply"), default="obj")
    parser.add_argument("--blender", default=shutil.which("blender") or "blender",
                        help="Blender executable (default: blender on PATH)")
    parser.add_argument("--frame-start", type=int, default=400)
    parser.add_argument("--frame-end", type=int, default=2400)
    args = parser.parse_args()

    blender_script = Path(__file__).parent / "render_movie_bl.py"
    cmd = [
        str(args.blender),
        "--background",
        "--factory-startup",
        "--python",
        str(blender_script),
        "--",
        "--obj-dir", str(args.input_dir),
        "--ply-dir", str(args.input_dir),
        "--output-dir", str(args.output_dir),
        "--obj-pattern", "{:06d}.obj",
        "--ply-pattern", "hd_{:08d}.ply",
        "--output-file-pattern", "render_{:08d}.png",
        "--scene-frame-start", str(args.frame_start),
        "--scene-frame-end", str(args.frame_end),
        "--num-turns", "10.0",
        "--target-x", "0.0",
        "--target-y", "-2.5",
        "--target-z", "0.0",
        "--camera-radius", "2.0",
        "--camera-height", "2.0",
        "--camera-lens-mm", "30.0",
        "--camera-clip-start", "0.01",
        "--camera-clip-end", "1000.0",
        "--ply-rot-x-deg", "90.0",
        "--ply-rot-y-deg", "0.0",
        "--ply-rot-z-deg", "0.0",
        "--camera-name", "OrbitCamera",
        "--pivot-name", "CameraPivot",
        "--collection-name", "PLYTemp",
        "--do-orphan-purge",
        "--mode", args.mode,
    ]
    print("Running command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

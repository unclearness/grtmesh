# SPDX-License-Identifier: GPL-3.0-or-later
"""Render clean color and depth of every camera of the synthetic scene.

    blender -b <dataset>/scene.blend -P python/synthetic/render_bl.py -- \
        --out <dataset>/render [--frames 1-150] [--cameras cam01,cam02]

Writes
    <out>/cameras.json                   intrinsics / poses (OpenCV axes), bounds
    <out>/<camera>/color_NNNN.png        8-bit sRGB, "Standard" view transform
    <out>/<camera>/depth_NNNN.exr        float32 z depth in meters (camera axis)

python/synthetic/make_dataset.py turns these into the dataset read by the
multiview_rgbd app (depth PNG in mm, exposure / white-balance perturbation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

# Blender camera (looks along -Z, Y up) -> OpenCV camera (Z forward, Y down).
BLENDER_TO_OPENCV = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", default="", help='e.g. "1-150" (default: the scene range)')
    p.add_argument("--cameras", default="", help="comma list (default: every camera)")
    p.add_argument("--samples", type=int, default=64, help="EEVEE TAA samples")
    return p.parse_args(argv)


def camera_record(cam: bpy.types.Object, scene: bpy.types.Scene) -> dict:
    """Pinhole intrinsics in OpenCV pixel coordinates (pixel centers at
    integers) and the camera-to-world matrix in OpenCV axes."""
    width = scene.render.resolution_x * scene.render.resolution_percentage // 100
    height = scene.render.resolution_y * scene.render.resolution_percentage // 100
    data = cam.data
    if data.sensor_fit not in ("HORIZONTAL", "AUTO") or width < height:
        raise ValueError("expected a horizontal sensor fit and a landscape image")
    if data.shift_x != 0.0 or data.shift_y != 0.0:
        raise ValueError("lens shift is not supported")
    fx = data.lens / data.sensor_width * width
    c2w = cam.matrix_world @ BLENDER_TO_OPENCV
    return {
        "name": cam.name, "width": width, "height": height,
        "fx": fx, "fy": fx, "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0,
        "camera_to_world": [list(row) for row in c2w],
    }


def setup_compositor(scene: bpy.types.Scene):
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.format.file_format = "PNG"
    out.format.color_mode = "RGB"
    out.format.color_depth = "8"
    out.file_slots.clear()
    out.file_slots.new("color_")
    out.file_slots.new("depth_")
    depth_slot = out.file_slots["depth_"]
    depth_slot.use_node_format = False
    depth_slot.format.file_format = "OPEN_EXR"
    depth_slot.format.color_mode = "RGB"  # OpenCV reads R/G/B, not a single V channel
    depth_slot.format.color_depth = "32"
    depth_slot.format.exr_codec = "ZIP"
    tree.links.new(layers.outputs["Image"], out.inputs["color_"])
    tree.links.new(layers.outputs["Depth"], out.inputs["depth_"])
    return out


def main():
    args = parse_args()
    scene = bpy.context.scene
    scene.eevee.taa_render_samples = args.samples
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    cameras = sorted((o for o in scene.objects if o.type == "CAMERA"), key=lambda o: o.name)
    if args.cameras:
        wanted = args.cameras.split(",")
        cameras = [c for c in cameras if c.name in wanted]
    if args.frames:
        lo, hi = (int(x) for x in args.frames.split("-"))
    else:
        lo, hi = scene.frame_start, scene.frame_end

    args.out.mkdir(parents=True, exist_ok=True)
    all_cameras = sorted((o for o in scene.objects if o.type == "CAMERA"), key=lambda o: o.name)
    meta = {
        "depth_scale": 1000.0,
        "bounds": json.loads(scene["grtmesh_bounds"]),
        "fps": scene.render.fps,
        "frames": [scene.frame_start, scene.frame_end],
        "cameras": [camera_record(c, scene) for c in all_cameras],
        "motion": json.loads(scene.get("grtmesh_motion", "{}")),
    }
    (args.out / "cameras.json").write_text(json.dumps(meta, indent=2))

    out = setup_compositor(scene)
    # The render result itself is not needed; the File Output node writes.
    scene.render.filepath = str(args.out / "_unused" / "")
    for cam in cameras:
        scene.camera = cam
        out.base_path = str(args.out / cam.name)
        for frame in range(lo, hi + 1):
            if (args.out / cam.name / f"depth_{frame:04d}.exr").exists():
                continue
            scene.frame_set(frame)
            bpy.ops.render.render(write_still=False)
            print(f"[render] {cam.name} frame {frame}", flush=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Vector


@dataclass
class Config:
    obj_dir: Path
    ply_dir: Path
    output_dir: Path
    obj_pattern: str
    ply_pattern: str
    output_file_pattern: str
    scene_frame_start: int
    scene_frame_end: int
    num_turns: float
    target_location: Vector
    camera_radius: float
    camera_height: float
    camera_lens_mm: float
    camera_clip_start: float
    camera_clip_end: float
    ply_rotation_euler_deg: tuple[float, float, float]
    camera_name: str
    pivot_name: str
    collection_name: str
    do_orphan_purge: bool
    mode: str


def parse_blender_cli_args() -> Config:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()

    parser.add_argument("--obj-dir", type=Path, required=True)
    parser.add_argument("--ply-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obj-pattern", type=str, default="{:06d}.obj")
    parser.add_argument("--ply-pattern", type=str, default="{:06d}.ply")
    parser.add_argument("--output-file-pattern", type=str, default="render_{:08d}.png")

    parser.add_argument("--scene-frame-start", type=int, default=400)
    parser.add_argument("--scene-frame-end", type=int, default=1400)
    parser.add_argument("--num-turns", type=float, default=10.0)

    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=-2.7)
    parser.add_argument("--target-z", type=float, default=0.0)

    parser.add_argument("--camera-radius", type=float, default=2.0)
    parser.add_argument("--camera-height", type=float, default=2.0)
    parser.add_argument("--camera-lens-mm", type=float, default=30.0)
    parser.add_argument("--camera-clip-start", type=float, default=0.01)
    parser.add_argument("--camera-clip-end", type=float, default=100.0)

    parser.add_argument("--ply-rot-x-deg", type=float, default=90.0)
    parser.add_argument("--ply-rot-y-deg", type=float, default=0.0)
    parser.add_argument("--ply-rot-z-deg", type=float, default=0.0)

    parser.add_argument("--camera-name", type=str, default="OrbitCamera")
    parser.add_argument("--pivot-name", type=str, default="CameraPivot")
    parser.add_argument("--collection-name", type=str, default="PLYTemp")

    parser.add_argument("--do-orphan-purge", action="store_true")
    parser.add_argument("--mode", type=str, choices=["obj", "ply"], default="obj")

    args = parser.parse_args(argv)

    return Config(
        obj_dir=args.obj_dir,
        ply_dir=args.ply_dir,
        output_dir=args.output_dir,
        obj_pattern=args.obj_pattern,
        ply_pattern=args.ply_pattern,
        output_file_pattern=args.output_file_pattern,
        scene_frame_start=args.scene_frame_start,
        scene_frame_end=args.scene_frame_end,
        num_turns=args.num_turns,
        target_location=Vector((args.target_x, args.target_y, args.target_z)),
        camera_radius=args.camera_radius,
        camera_height=args.camera_height,
        camera_lens_mm=args.camera_lens_mm,
        camera_clip_start=args.camera_clip_start,
        camera_clip_end=args.camera_clip_end,
        ply_rotation_euler_deg=(
            args.ply_rot_x_deg,
            args.ply_rot_y_deg,
            args.ply_rot_z_deg,
        ),
        camera_name=args.camera_name,
        pivot_name=args.pivot_name,
        collection_name=args.collection_name,
        do_orphan_purge=args.do_orphan_purge,
        mode=args.mode,
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def ensure_pivot(name: str, location: Vector) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=location)
        obj = bpy.context.active_object
        obj.name = name
    else:
        obj.location = location
    return obj


def ensure_camera(name: str, pivot: bpy.types.Object, cfg: Config) -> bpy.types.Object:
    camera = bpy.data.objects.get(name)
    if camera is None or camera.type != "CAMERA":
        bpy.ops.object.camera_add(
            location=(pivot.location.x, pivot.location.y - cfg.camera_radius, pivot.location.z + cfg.camera_height)
        )
        camera = bpy.context.active_object
        camera.name = name

    camera.location = (
        pivot.location.x,
        pivot.location.y - cfg.camera_radius,
        pivot.location.z + cfg.camera_height,
    )
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.lens = cfg.camera_lens_mm
    camera.data.clip_start = cfg.camera_clip_start
    camera.data.clip_end = cfg.camera_clip_end

    for constraint in list(camera.constraints):
        if constraint.type == "TRACK_TO":
            camera.constraints.remove(constraint)

    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = pivot
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = camera
    return camera


def set_camera_for_frame(
    camera: bpy.types.Object,
    pivot: bpy.types.Object,
    frame: int,
    cfg: Config,
) -> None:
    if cfg.scene_frame_end <= cfg.scene_frame_start:
        t = 0.0
    else:
        t = (frame - cfg.scene_frame_start) / float(cfg.scene_frame_end - cfg.scene_frame_start)

    angle = 2.0 * math.pi * cfg.num_turns * t
    x = pivot.location.x + cfg.camera_radius * math.cos(angle)
    y = pivot.location.y + cfg.camera_radius * math.sin(angle)
    z = pivot.location.z + cfg.camera_height
    camera.location = (x, y, z)


def remove_object_and_related_data(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return

    mesh = obj.data if obj.type == "MESH" else None
    materials: list[bpy.types.Material] = []

    if obj.type == "MESH" and obj.data is not None:
        materials = [mat for mat in obj.data.materials if mat is not None]

    bpy.data.objects.remove(obj, do_unlink=True)

    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)

    for material in materials:
        if material.users == 0:
            bpy.data.materials.remove(material)


def cleanup_temp_collection(collection: bpy.types.Collection, do_orphan_purge: bool) -> None:
    for obj in list(collection.objects):
        remove_object_and_related_data(obj)

    if do_orphan_purge:
        try:
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True,
                do_linked_ids=True,
                do_recursive=True,
            )
        except Exception:
            pass


def setup_render(scene: bpy.types.Scene) -> None:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = 1
    scene.eevee.use_shadows = False


def reconnect_all_image_texture_materials() -> None:
    for material in bpy.data.materials:
        if material.node_tree is None:
            continue
        if "Image Texture" not in material.node_tree.nodes:
            continue
        if "Material Output" not in material.node_tree.nodes:
            continue

        links = material.node_tree.links
        for link in list(links):
            links.remove(link)

        output = material.node_tree.nodes["Material Output"]
        texture = material.node_tree.nodes["Image Texture"]
        links.new(texture.outputs[0], output.inputs[0])


def import_obj(filepath: Path, target_collection: bpy.types.Collection) -> bpy.types.Object:
    before_names = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=str(filepath))
    after_names = set(bpy.data.objects.keys())

    new_names = list(after_names - before_names)
    if not new_names:
        raise RuntimeError(f"OBJ import did not create an object: {filepath}")

    obj = bpy.context.active_object
    if obj is None or obj.name not in new_names:
        obj = bpy.data.objects[new_names[0]]

    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)

    target_collection.objects.link(obj)
    return obj


def apply_transform(obj: bpy.types.Object, cfg: Config) -> None:
    obj.rotation_euler = tuple(math.radians(v) for v in cfg.ply_rotation_euler_deg)


def remove_old_camera(camera_name: str) -> None:
    old_camera = bpy.data.objects.get(camera_name)
    if old_camera is not None:
        bpy.data.objects.remove(old_camera, do_unlink=True)



# def frame_to_ply_path(frame: int) -> Path:
#     return ply_dir / file_pattern.format(frame)

def import_ply(filepath: Path, target_collection: bpy.types.Collection) -> bpy.types.Object:
    before_names = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=str(filepath))
    after_names = set(bpy.data.objects.keys())
    new_names = list(after_names - before_names)
    if not new_names:
        raise RuntimeError(f"PLY import did not create an object: {filepath}")
    obj = bpy.context.active_object
    if obj is None or obj.name not in new_names:
        obj = bpy.data.objects[new_names[0]]
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    target_collection.objects.link(obj)
    return obj

def create_object_local_material(obj: bpy.types.Object, frame: int) -> bpy.types.Material:
    material = bpy.data.materials.new(name=f"PLYMat_{frame:08d}")
    material.use_nodes = True
    nt = material.node_tree
    if nt is None:
        raise RuntimeError("Material node tree is None.")
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    color_attr = nodes.new(type="ShaderNodeVertexColor")
    color_attr.layer_name = "Col"
    color_attr.location = (0, 0)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (260, 0)
    # User-requested direct connection:
    # Color Attribute("Col") -> Material Output.Surface
    links.new(color_attr.outputs["Color"], output.inputs["Surface"])
    if len(obj.data.materials) == 0:
        obj.data.materials.append(material)
    else:
        obj.data.materials.clear()
        obj.data.materials.append(material)
    return material

# def apply_transform(obj: bpy.types.Object) -> None:
#     ply_rotation_euler_deg = (90.0, 0.0, 0.0)
#     obj.rotation_euler = tuple(math.radians(v) for v in ply_rotation_euler_deg)


def main() -> None:
    cfg = parse_blender_cli_args()
    clear_all_objects()

    scene = bpy.context.scene
    scene.frame_start = cfg.scene_frame_start
    scene.frame_end = cfg.scene_frame_end

    ensure_output_dir(cfg.output_dir)
    setup_render(scene)

    temp_collection = ensure_collection(cfg.collection_name)
    pivot = ensure_pivot(cfg.pivot_name, cfg.target_location)

    remove_old_camera(cfg.camera_name)
    camera = ensure_camera(cfg.camera_name, pivot, cfg)

    reconnect_all_image_texture_materials()
    cleanup_temp_collection(temp_collection, cfg.do_orphan_purge)

    for frame in range(cfg.scene_frame_start, cfg.scene_frame_end + 1):
        print(f"[INFO] Processing frame {frame}")

        if cfg.mode == "obj":
            obj_path = cfg.obj_dir / cfg.obj_pattern.format(frame)
            if not obj_path.exists():
                print(f"[WARN] File not found: {obj_path}")
                continue
        else:
            ply_path = cfg.ply_dir / cfg.ply_pattern.format(frame)
            if not ply_path.exists():
                print(f"[WARN] File not found: {ply_path}")
                continue

        scene.frame_set(frame)
        set_camera_for_frame(camera=camera, pivot=pivot, frame=frame, cfg=cfg)

        print(
            f"[INFO] frame={frame} lens={camera.data.lens} "
            f"location={tuple(camera.location)} "
            f"radius={cfg.camera_radius} height={cfg.camera_height}"
        )

        if cfg.mode == "obj":
            obj = import_obj(obj_path, temp_collection)
        else:
            obj = import_ply(ply_path, temp_collection)
            create_object_local_material(obj, frame)

        apply_transform(obj, cfg)
        reconnect_all_image_texture_materials()

        scene.render.filepath = str(cfg.output_dir / cfg.output_file_pattern.format(frame))
        bpy.ops.render.render(write_still=True)

        remove_object_and_related_data(obj)

        if cfg.do_orphan_purge:
            try:
                bpy.ops.outliner.orphans_purge(
                    do_local_ids=True,
                    do_linked_ids=True,
                    do_recursive=True,
                )
            except Exception:
                pass

    print("[INFO] Done.")


def clear_all_objects() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)

    for camera in list(bpy.data.cameras):
        if camera.users == 0:
            bpy.data.cameras.remove(camera)

    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)

    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True,
        )
    except Exception:
        pass

if __name__ == "__main__":
    main()
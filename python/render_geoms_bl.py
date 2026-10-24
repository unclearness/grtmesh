# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector, Euler
import mathutils
import numpy as np

# def _argv_after_double_dash() -> list[str]:
#     # Blender passes its own args; user args come after "--"
#     argv = []
#     if "--" in bpy.app.argv:
#         argv = bpy.app.argv[bpy.app.argv.index("--") + 1 :]
#     return argv


import sys

def set_background_rgba(bg_rgba: tuple[float, float, float, float]) -> None:
    r, g, b, a = bg_rgba
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    assert nt is not None

    # Find or create Background + World Output
    bg = None
    out = None
    for n in nt.nodes:
        if n.type == "BACKGROUND":
            bg = n
        elif n.type == "OUTPUT_WORLD":
            out = n
    if bg is None:
        bg = nt.nodes.new("ShaderNodeBackground")
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputWorld")

    # Ensure link
    linked = False
    for l in nt.links:
        if l.from_node == bg and l.to_node == out and l.to_socket.name == "Surface":
            linked = True
            break
    if not linked:
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    bg.inputs["Color"].default_value = (r, g, b, a)
    bg.inputs["Strength"].default_value = 1.0

def _argv_after_double_dash() -> list[str]:
    """
    Return arguments passed after '--' when running Blender with:
      blender -b -P script.py -- <args>
    """
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]

def _as_mat44(m: Any) -> mathutils.Matrix:
    # m: nested list 4x4
    print(m)
    if len(m) != 4 or any(len(r) != 4 for r in m):
        raise ValueError("Expected 4x4 matrix")
    return mathutils.Matrix(m)


def _as_mat33(m: Any) -> list[list[float]]:
    if len(m) != 3 or any(len(r) != 3 for r in m):
        raise ValueError("Expected 3x3 matrix")
    return [[float(m[r][c]) for c in range(3)] for r in range(3)]


def load_cmu_panoptic_kinect_color(calib_json: Path) -> list[dict[str, Any]]:
    data = json.loads(calib_json.read_text(encoding="utf-8"))
    sensors = data["sensors"]
    cams: list[dict[str, Any]] = []
    # print(sensors)
    for i, s in enumerate(sensors):
        K = _as_mat33(s["K_color"])
        w = int(s["color_width"])
        h = int(s["color_height"])
        M_world2sensor = _as_mat44(s["M_world2sensor"])
        M_color = _as_mat44(s["M_color"])
        world2color = M_color @ M_world2sensor  # world -> color camera
        # print(M_color)
        # print(M_world2sensor)
        # print(f"Camera {i+1}: K={K}, world2color=\n{world2color}") 
        # hoge
        cams.append(
            {
                "kinect_id_1start": i + 1,
                "out_name": str(i + 1),
                "width": w,
                "height": h,
                "fx": float(K[0][0]),
                "fy": float(K[1][1]),
                "cx": float(K[0][2]),
                "cy": float(K[1][2]),
                "world2color": world2color,
            }
        )
    return cams


def load_rgbd_sequence_cameras(cameras_json: Path) -> list[dict[str, Any]]:
    """Cameras of the generic RGB-D layout (cameras.json of
    app/rgbd_sequence_dataset.h: OpenCV axes, camera_to_world)."""
    data = json.loads(cameras_json.read_text(encoding="utf-8"))
    cams: list[dict[str, Any]] = []
    for c in data["cameras"]:
        c2w = np.array(c["camera_to_world"], dtype=np.float64)
        cams.append(
            {
                "out_name": c["name"],
                "width": int(c["width"]),
                "height": int(c["height"]),
                "fx": float(c["fx"]),
                "fy": float(c["fy"]),
                "cx": float(c["cx"]),
                "cy": float(c["cy"]),
                "world2color": np.linalg.inv(c2w),
            }
        )
    return cams


def list_mesh_files(input_dir: Path) -> list[Path]:
    exts = {".obj", ".ply"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .obj/.ply found in: {input_dir}")
    # enforce same extension (optional). If mixed, we still render in name order.
    return files


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)

    # purge orphan data (safe-ish)
    for _ in range(2):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def import_mesh(mesh_path: Path) -> bpy.types.Object:
    ext = mesh_path.suffix.lower()

    if ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(mesh_path))

    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis="Y", up_axis="Z")

        # # "No axis conversion": interpret OBJ coordinates as Blender coordinates.
        # # Blender internal axes are Forward=-Y, Up=Z. So set importer axes to these.
        # if hasattr(bpy.ops.wm, "obj_import"):
        #     # Blender 4.x+
        #     op = bpy.ops.wm.obj_import
        #     kwargs = {"filepath": str(mesh_path)}

        #     # Different builds may expose different property names; set if present.
        #     props = op.get_rna_type().properties.keys()
        #     if "forward_axis" in props:
        #         kwargs["forward_axis"] = "NEGATIVE_Y"
        #     if "up_axis" in props:
        #         kwargs["up_axis"] = "Z"
        #     if "global_scale" in props:
        #         kwargs["global_scale"] = 1.0

        #     op(**kwargs)
        # else:
        #     # Legacy importer
        #     op = bpy.ops.import_scene.obj
        #     kwargs = {
        #         "filepath": str(mesh_path),
        #         "axis_forward": "-Y",
        #         "axis_up": "Z",
        #         "global_scale": 1.0,
        #     }
        #     op(**kwargs)

    else:
        raise ValueError(f"Unsupported: {mesh_path}")

    obj = bpy.context.view_layer.objects.active
    if obj is None or obj.type != "MESH":
        for o in bpy.context.scene.objects:
            if o.type == "MESH":
                obj = o
                break
    if obj is None:
        raise RuntimeError(f"Failed to import mesh: {mesh_path}")

    return obj


def make_unlit_material_for_obj() -> bpy.types.Material:
    mat = bpy.data.materials.new(name="Unlit_OBJ")
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None

    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0

    # Try to find image texture node via principled? Instead, create texture node and let Blender auto-assign image?
    # Better: keep it simple: if the imported OBJ created materials, we will override them per-object later.
    # This material is a fallback pure white emission.
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_unlit_material_for_ply_vertex_color(mesh: bpy.types.Mesh) -> bpy.types.Material:
    mat = bpy.data.materials.new(name="Unlit_PLY_VCol")
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0

    # Blender's color attributes: mesh.color_attributes (new) or mesh.vertex_colors (old)
    attr_name: str | None = None
    if hasattr(mesh, "color_attributes") and len(mesh.color_attributes) > 0:
        attr_name = mesh.color_attributes[0].name
    elif hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
        attr_name = mesh.vertex_colors[0].name

    if attr_name is None:
        em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    else:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = attr_name
        nt.links.new(attr.outputs["Color"], em.inputs["Color"])

    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def force_unlit_for_obj_object(obj: bpy.types.Object) -> None:
    """
    Imported OBJ often brings materials + texture nodes.
    We rewrite each slot to an Emission shader and reuse the image texture if present.
    """
    if obj.type != "MESH":
        return

    if not obj.data.materials:
        obj.data.materials.append(make_unlit_material_for_obj())

    for i, mat in enumerate(obj.data.materials):
        if mat is None:
            mat = make_unlit_material_for_obj()
            obj.data.materials[i] = mat

        mat.use_nodes = True
        nt = mat.node_tree
        assert nt is not None

        # Find an existing Image Texture node if present
        img_node = None
        for n in nt.nodes:
            if n.type == "TEX_IMAGE":
                img_node = n
                break

        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0

        if img_node is not None and getattr(img_node, "image", None) is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img_node.image
            # Important: set colorspace to sRGB for albedo
            if tex.image is not None and hasattr(tex.image, "colorspace_settings"):
                tex.image.colorspace_settings.name = "sRGB"
            nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        else:
            em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])


def set_render_settings(engine: str, transparent_bg: bool, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine

    # Color management to reduce "looks different" surprises
    view = scene.view_settings
    view.view_transform = "Standard"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent_bg else "RGB"
    scene.render.film_transparent = bool(transparent_bg)

    if engine == "CYCLES":
        scene.cycles.samples = max(1, int(samples))
        scene.cycles.use_denoising = False

    set_background_rgba((0.0, 0.0, 0.0, 1.0))  

    # scene.render.film_transparent = True
    # scene.render.image_settings.color_mode = "RGBA"

# def setup_camera_from_intrinsics(
#     cam_obj: bpy.types.Object,
#     width: int,
#     height: int,
#     fx: float,
#     fy: float,
#     cx: float,
#     cy: float,
# ) -> None:
#     """
#     Convert pinhole intrinsics (pixels) to Blender camera params.

#     We use:
#     - sensor_width fixed (36mm)
#     - lens_mm = fx * sensor_width / width
#     - pixel aspect to account for fx != fy
#         pixel_aspect_y = fx / fy  (assuming pixel_aspect_x = 1)
#     - principal point -> shift_x, shift_y
#         shift_x = -(cx - width/2)/width
#         shift_y =  (cy - height/2)/height   (note sign; depends on convention)
#     """
#     scene = bpy.context.scene
#     scene.render.resolution_x = int(width)
#     scene.render.resolution_y = int(height)
#     scene.render.resolution_percentage = 100

#     cam = cam_obj.data
#     assert isinstance(cam, bpy.types.Camera)

#     cam.type = "PERSP"
#     cam.sensor_fit = "HORIZONTAL"
#     cam.sensor_width = 36.0  # mm (any constant works if consistent)
#     cam.lens = fx * cam.sensor_width / float(width)

#     # Handle non-square pixels so that fy matches too
#     scene.render.pixel_aspect_x = 1.0
#     scene.render.pixel_aspect_y = float(fx / fy) if fy != 0 else 1.0

#     # Principal point shift
#     cam.shift_x = -((cx - (width * 0.5)) / float(width))
#     cam.shift_y = +((cy - (height * 0.5)) / float(height))
#     # cam.shift_x = (cx - (width * 0.5)) / float(width)
#     # cam.shift_y = -(cy - (height * 0.5)) / float(height)


def get_sensor_size(cam: bpy.types.Camera) -> float:
    """ Returns the sensor size in millimeters based on the configured sensor_fit.

    :param cam: The camera object.
    :return: The sensor size in millimeters.
    """
    if cam.sensor_fit == 'VERTICAL':
        sensor_size_in_mm = cam.sensor_height
    else:
        sensor_size_in_mm = cam.sensor_width
    return sensor_size_in_mm


def get_view_fac_in_px(cam: bpy.types.Camera, pixel_aspect_x: float, pixel_aspect_y: float,
                       resolution_x_in_px: int, resolution_y_in_px: int) -> int:
    """ Returns the camera view in pixels.

    :param cam: The camera object.
    :param pixel_aspect_x: The pixel aspect ratio along x.
    :param pixel_aspect_y: The pixel aspect ratio along y.
    :param resolution_x_in_px: The image width in pixels.
    :param resolution_y_in_px: The image height in pixels.
    :return: The camera view in pixels.
    """
    # Determine the sensor fit mode to use
    if cam.sensor_fit == 'AUTO':
        if pixel_aspect_x * resolution_x_in_px >= pixel_aspect_y * resolution_y_in_px:
            sensor_fit = 'HORIZONTAL'
        else:
            sensor_fit = 'VERTICAL'
    else:
        sensor_fit = cam.sensor_fit

    # Based on the sensor fit mode, determine the view in pixels
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

    return view_fac_in_px


def set_intrinsics_from_blender_params(lens: float = None, image_width: int = None, image_height: int = None,
                                       clip_start: float = None, clip_end: float = None,
                                       pixel_aspect_x: float = None, pixel_aspect_y: float = None, shift_x: int = None,
                                       shift_y: int = None, lens_unit: str = None):
    """ Sets the camera intrinsics using blenders represenation.

    :param lens: Either the focal length in millimeters or the FOV in radians, depending on the given lens_unit.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param clip_start: Clipping start.
    :param clip_end: Clipping end.
    :param pixel_aspect_x: The pixel aspect ratio along x.
    :param pixel_aspect_y: The pixel aspect ratio along y.
    :param shift_x: The shift in x direction.
    :param shift_y: The shift in y direction.
    :param lens_unit: Either FOV or MILLIMETERS depending on whether the lens is defined as focal length in
                      millimeters or as FOV in radians.
    """

    cam_ob = bpy.context.scene.camera
    cam = cam_ob.data

    if lens_unit is not None:
        cam.lens_unit = lens_unit

    if lens is not None:
        # Set focal length
        if cam.lens_unit == 'MILLIMETERS':
            if lens < 1:
                raise Exception("The focal length is smaller than 1mm which is not allowed in blender: " + str(lens))
            cam.lens = lens
        elif cam.lens_unit == "FOV":
            cam.angle = lens
        else:
            raise Exception("No such lens unit: " + lens_unit)

    # Set resolution
    if image_width is not None:
        bpy.context.scene.render.resolution_x = image_width
    if image_height is not None:
        bpy.context.scene.render.resolution_y = image_height

    # Set clipping
    if clip_start is not None:
        cam.clip_start = clip_start
    if clip_end is not None:
        cam.clip_end = clip_end

    # Set aspect ratio
    if pixel_aspect_x is not None:
        bpy.context.scene.render.pixel_aspect_x = pixel_aspect_x
    if pixel_aspect_y is not None:
        bpy.context.scene.render.pixel_aspect_y = pixel_aspect_y

    # Set shift
    if shift_x is not None:
        cam.shift_x = shift_x
    if shift_y is not None:
        cam.shift_y = shift_y


def set_intrinsics_from_K_matrix(K: Union[np.ndarray, Matrix], image_width: int, image_height: int,
                                 clip_start: float = None, clip_end: float = None):
    """ Set the camera intrinsics via a K matrix.

    The K matrix should have the format:
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0,  1]]

    This method is based on https://blender.stackexchange.com/a/120063.

    :param K: The 3x3 K matrix.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param clip_start: Clipping start.
    :param clip_end: Clipping end.
    """

    K = Matrix(K)

    cam = bpy.context.scene.camera.data

    if abs(K[0][1]) > 1e-7:
        raise ValueError(f"Skew is not supported by blender and therefore "
                         f"not by BlenderProc, set this to zero: {K[0][1]} and recalibrate")

    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]

    # If fx!=fy change pixel aspect ratio
    pixel_aspect_x = pixel_aspect_y = 1
    if fx > fy:
        pixel_aspect_y = fx / fy
    elif fx < fy:
        pixel_aspect_x = fy / fx

    # Compute sensor size in mm and view in px
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    view_fac_in_px = get_view_fac_in_px(cam, pixel_aspect_x, pixel_aspect_y, image_width, image_height)
    sensor_size_in_mm = get_sensor_size(cam)

    # Convert focal length in px to focal length in mm
    f_in_mm = fx * sensor_size_in_mm / view_fac_in_px

    # Convert principal point in px to blenders internal format
    shift_x = (cx - (image_width - 1) / 2) / -view_fac_in_px
    shift_y = (cy - (image_height - 1) / 2) / view_fac_in_px * pixel_aspect_ratio

    # Finally set all intrinsics
    set_intrinsics_from_blender_params(f_in_mm, image_width, image_height, clip_start, clip_end, pixel_aspect_x,
                                       pixel_aspect_y, shift_x, shift_y, "MILLIMETERS")




def setup_camera_from_intrinsics(
    cam_obj: bpy.types.Object,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float
):
    scene = bpy.context.scene
    cam = cam_obj.data

    # 1. 解像度の設定
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100

    # 2. センサーサイズの設定
    cam.sensor_fit = "HORIZONTAL"
    sensor_width = cam.sensor_width  # デフォルトは36mm等

    # 3. 焦点距離 (Focal Length) の計算
    cam.lens = fx * sensor_width / float(width)

    # 4. ピクセルアスペクト比
    # $f_x \neq f_y$ の場合の違いを吸収します
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy

    # 5. プリンシパルポイント (Principal Point / Shift) の計算
    # 基準は HORIZONTAL のため、XもYも width で割ります
    cam.shift_x = (cx - width * 0.5) / float(width)
    
    # OpenCV(左上原点)の座標系からBlender(左下原点)に合わせるため符号を反転させます
    cam.shift_y = -(cy - height * 0.5) / float(width)


# def setup_camera_from_intrinsics( cam_obj: bpy.types.Object,
#     width: int,
#     height: int,
#     fx: float,
#     fy: float,
#     cx: float,
#     cy: float):
#     scene = bpy.context.scene
#     cam = cam_obj.data

#     cam.sensor_fit = "HORIZONTAL"
#     sensor_width = cam.sensor_width  # 例: 36mm

#     # lens from fx
#     cam.lens = fx * sensor_width / float(width)

#     # pixel aspect to absorb fx != fy
#     scene.render.pixel_aspect_x = 1.0
#     scene.render.pixel_aspect_y = fx / fy

#     # principal point
#     cam.shift_x = (cx - width * 0.5) / width
#     cam.shift_y = (cy - height * 0.5) / height

#     # resolution
#     scene.render.resolution_x = int(width)
#     scene.render.resolution_y = int(height)
#     scene.render.resolution_percentage = 100

# def set_camera_pose_from_world2color(cam_obj: bpy.types.Object, world2color: mathutils.Matrix) -> None:
#     """
#     world2color is world -> camera (OpenCV-ish camera coords: x right, y down, z forward).
#     Blender camera coords: x right, y up, -z forward (camera looks along -Z).

#     Convert world->cv to world->blender_cam by:
#       cv_to_bcam = diag(1, -1, -1)
#       world2bcam = cv_to_bcam @ world2cv
#       cam_world = inv(world2bcam)
#     """
#     cv_to_bcam = mathutils.Matrix(
#         (
#             (1.0, 0.0, 0.0, 0.0),
#             (0.0, -1.0, 0.0, 0.0),
#             (0.0, 0.0, -1.0, 0.0),
#             (0.0, 0.0, 0.0, 1.0),
#         )
#     )

#     world2bcam = cv_to_bcam @ world2color
#     cam_obj.matrix_world = world2bcam.inverted()

def set_camera_pose_from_world2color(cam_obj, world2color_4x4):
    # OpenCV cam -> Blender cam
    cv_to_bcam = np.array(
        [[1, 0, 0, 0],
         [0,-1, 0, 0],
         [0, 0,-1, 0],
         [0, 0, 0, 1]], dtype=np.float64
    )

    # world -> blenderCam
    world2bcam = cv_to_bcam @ world2color_4x4

    # Blender wants camera-to-world in matrix_world
    cam_world = np.linalg.inv(world2bcam)
    cam_obj.matrix_world = mathutils.Matrix(cam_world.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_json", type=Path, default=None, help="Panoptic kcalibration_*.json")
    parser.add_argument("--cameras_json", type=Path, default=None,
                        help="cameras.json of the generic RGB-D layout (instead of --calib_json)")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE_NEXT", choices=["BLENDER_EEVEE_NEXT", "CYCLES"])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--transparent_bg", action="store_true")
    parser.add_argument("--camera_ids", type=str, default="",
                        help="comma-separated 1-based Kinect ids (or camera names with "
                             "--cameras_json) to render (default: all)")
    parser.add_argument("--frames", type=int, nargs=2, default=[400, 2400], metavar=("START", "END"),
                        help="frame range [START, END) of the meshes NNNNNN.obj")
    args = parser.parse_args(_argv_after_double_dash())

    if (args.calib_json is None) == (args.cameras_json is None):
        parser.error("give exactly one of --calib_json and --cameras_json")
    if args.calib_json is not None:
        cams = load_cmu_panoptic_kinect_color(args.calib_json)
    else:
        cams = load_rgbd_sequence_cameras(args.cameras_json)
    if args.camera_ids:
        keep = {c.strip() for c in args.camera_ids.split(",") if c.strip()}
        cams = [c for c in cams if c["out_name"] in keep]
    mesh_files = list_mesh_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    set_render_settings(args.engine, args.transparent_bg, args.samples)

    # Create a camera object once
    bpy.ops.object.camera_add()
    cam_obj = bpy.context.view_layer.objects.active
    assert cam_obj is not None and cam_obj.type == "CAMERA"
    bpy.context.scene.camera = cam_obj

    # Optional: add a dummy light (not used for emission/unlit, but harmless)
    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 0.0))
    light_obj = bpy.context.view_layer.objects.active
    if light_obj is not None and light_obj.type == "LIGHT":
        light_obj.data.energy = 0.0

    # for caminfo in cams:
    #     kid = int(caminfo["kinect_id_1start"])
    #     out_cam_dir = args.output_dir / f"{kid}"
    #     out_cam_dir.mkdir(parents=True, exist_ok=True)

    #     setup_camera_from_intrinsics(
    #         cam_obj,
    #         width=int(caminfo["width"]),
    #         height=int(caminfo["height"]),
    #         fx=float(caminfo["fx"]),
    #         fy=float(caminfo["fy"]),
    #         cx=float(caminfo["cx"]),
    #         cy=float(caminfo["cy"]),
    #     )
    #     set_camera_pose_from_world2color(cam_obj, caminfo["world2color"])

    #     for fi, mesh_path in enumerate(mesh_files, start=1):
    #         # Remove previous meshes only (keep camera/light)
    #         for o in list(bpy.context.scene.objects):
    #             if o.type == "MESH":
    #                 bpy.data.objects.remove(o, do_unlink=True)

    #         obj = import_mesh(mesh_path)

    #         if mesh_path.suffix.lower() == ".ply":
    #             mat = make_unlit_material_for_ply_vertex_color(obj.data)
    #             obj.data.materials.clear()
    #             obj.data.materials.append(mat)
    #         else:
    #             force_unlit_for_obj_object(obj)

    #         out_path = out_cam_dir / f"{fi:06d}.png"
    #         bpy.context.scene.render.filepath = str(out_path)

    #         cam = cam_obj.data
    #         loc = cam_obj.matrix_world.to_translation()
    #         rot = cam_obj.matrix_world.to_euler()
    #         print("lens_mm", cam.lens, "shift", cam.shift_x, cam.shift_y,
    #             "loc", tuple(loc), "rot", tuple(rot))
    #         bpy.ops.render.render(write_still=True)
    #         print(f"[Kinect {kid:02d}] {mesh_path.name} -> {out_path}")

    # print("Done.")
    # Pre-create output dirs for each Kinect
    for caminfo in cams:
        (args.output_dir / caminfo["out_name"]).mkdir(parents=True, exist_ok=True)
    start, end = args.frames
#    for fi, mesh_path in enumerate(mesh_files, start=1):
    for fi, mesh_idx in enumerate(range(start, end), start=start):
        all_rendered = True
        for caminfo in cams:
            out_cam_dir = args.output_dir / caminfo["out_name"]
            out_path = out_cam_dir / f"{fi:06d}.png"
            if not out_path.exists():
                all_rendered = False
                break

        if all_rendered:
            print(f"All outputs already exist for frame {fi:06d}, skipping.")
            continue

        ply_names = [f"{mesh_idx:06d}.ply", f"hd_{mesh_idx:08d}.ply"]
        obj_name = f"{mesh_idx:06d}.obj"
        mesh_path = None
        for ply_name in ply_names:
            if (args.input_dir / ply_name).exists():
                mesh_path = args.input_dir / ply_name
                break
        if mesh_path is None:
            if (args.input_dir / obj_name).exists():
                mesh_path = args.input_dir / obj_name
            else:
                print(f"Warning: No mesh found for frame {fi:06d}, skipping.")
                continue
        # Remove previous meshes only (keep camera/light)
        for o in list(bpy.context.scene.objects):
            if o.type == "MESH":
                bpy.data.objects.remove(o, do_unlink=True)

        # Import this frame's mesh once
        obj = import_mesh(mesh_path)

        # Assign unlit material once
        if mesh_path.suffix.lower() == ".ply":
            mat = make_unlit_material_for_ply_vertex_color(obj.data)
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        else:
            force_unlit_for_obj_object(obj)

        # Render from every Kinect camera
        for caminfo in cams:

            cam_name = caminfo["out_name"]
            out_cam_dir = args.output_dir / cam_name
            out_path = out_cam_dir / f"{fi:06d}.png"
            if out_path.exists():
                print(f"Output already exists, skipping: {out_path}")
                continue

            # setup_camera_from_intrinsics(
            #     cam_obj,
            #     width=int(caminfo["width"]),
            #     height=int(caminfo["height"]),
            #     fx=float(caminfo["fx"]),
            #     fy=float(caminfo["fy"]),
            #     cx=float(caminfo["cx"]),
            #     cy=float(caminfo["cy"]),
            # )

            set_intrinsics_from_K_matrix(
                K=np.array([[caminfo["fx"], 0, caminfo["cx"]],
                            [0, caminfo["fy"], caminfo["cy"]],
                            [0, 0, 1]]),
                image_width=int(caminfo["width"]),
                image_height=int(caminfo["height"]),
            )
            set_camera_pose_from_world2color(cam_obj, caminfo["world2color"])

            # Ensure updates are reflected before render (important in headless)
            bpy.context.scene.camera = cam_obj
            bpy.context.view_layer.update()


            bpy.context.scene.render.filepath = str(out_path)

            cam = cam_obj.data
            loc = cam_obj.matrix_world.to_translation()
            rot = cam_obj.matrix_world.to_euler()
            print(
                f"[Frame {fi:06d} | camera {cam_name}]",
                "lens_mm", cam.lens,
                "shift", cam.shift_x, cam.shift_y,
                "loc", tuple(loc),
                "rot", tuple(rot),
            )

            bpy.ops.render.render(write_still=True)
            print(f"[camera {cam_name}] {mesh_path.name} -> {out_path}")
            # bpy.ops.wm.save_as_mainfile(filepath=str(Path(out_cam_dir) / "scene.blend"))
            # hoge

    print("Done.")

if __name__ == "__main__":
    main()
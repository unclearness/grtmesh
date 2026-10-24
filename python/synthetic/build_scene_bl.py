# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the synthetic multi-camera scene (.blend) used to test the seam
correction: an animated person on a floor, seen by ten RGB-D cameras.

Run inside Blender 4.5 with the MPFB 2 extension enabled and the CC0
"MakeHuman system assets" pack installed (see python/synthetic/README.md):

    blender -b -P python/synthetic/build_scene_bl.py -- \
        --out <dataset>/scene.blend \
        --mocap_dir <dataset>/downloads/cmu_mocap

Contents (all generated here, nothing is loaded from other .blend files):
  * a 3 x 3 m wooden floor and nothing else (the background is empty, so it
    has no depth and only the person and the floor are reconstructed);
  * human: MPFB 2 character (CC0 assets) with the "cmu_mb" rig, animated by
    CMU motion capture trial 05_02 (dance, from 2.5 s) resampled to 30 fps;
  * cameras cam01..cam10 on a circle around the center, looking down,
    1280 x 720, no lens distortion; color and depth come from the same camera.

Every material is made purely diffuse (no specular, coat, sheen or
subsurface scattering) and EEVEE's screen-space effects are disabled, so a
surface point has the same color in every camera: the only differences
between cameras are the exposure / white-balance gains added afterwards by
make_dataset.py.

The scene is deterministic: the same arguments give the same .blend.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import addon_utils
import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmu_mocap  # noqa: E402

FPS = 30
RES_X, RES_Y = 1280, 720
FLOOR_HALF = 1.5     # 3 x 3 m floor
LIGHT_HEIGHT = 2.8
CAMERA_HFOV_DEG = 70.0
NUM_CAMERAS = 10

# CMU ASF bone -> MPFB "cmu_mb" bone.
ASF_TO_RIG = {
    "root": "Hips", "lowerback": "LowerBack", "upperback": "Spine", "thorax": "Spine1",
    "lowerneck": "Neck", "upperneck": "Neck1", "head": "Head",
    "lhipjoint": "LHipJoint", "lfemur": "LeftUpLeg", "ltibia": "LeftLeg",
    "lfoot": "LeftFoot", "ltoes": "LeftToeBase",
    "rhipjoint": "RHipJoint", "rfemur": "RightUpLeg", "rtibia": "RightLeg",
    "rfoot": "RightFoot", "rtoes": "RightToeBase",
    "lclavicle": "LeftShoulder", "lhumerus": "LeftArm", "lradius": "LeftForeArm",
    "lwrist": "LeftHand", "lhand": "LeftFingerBase", "lfingers": "LeftHandFinger1",
    "lthumb": "LThumb",
    "rclavicle": "RightShoulder", "rhumerus": "RightArm", "rradius": "RightForeArm",
    "rwrist": "RightHand", "rhand": "RightFingerBase", "rfingers": "RightHandFinger1",
    "rthumb": "RThumb",
}
# ASF (Y up, subject faces +Z) -> Blender (Z up, character faces -Y).
ASF_TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mocap_dir", type=Path, required=True)
    p.add_argument("--subject", type=int, default=5)
    p.add_argument("--trial", type=int, default=2)
    p.add_argument("--mocap_start", type=int, default=300, help="first AMC frame (120 fps)")
    p.add_argument("--frames", type=int, default=150, help="frames at 30 fps")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Materials (procedural, view-independent: rough dielectrics)
# ---------------------------------------------------------------------------
def principled(name: str, roughness: float = 0.8):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    return mat, nodes, links, bsdf


def mat_wood_floor():
    """Light, low-contrast planks: thin dark joints and a faint grain, so a
    brightness step between two cameras is easy to see (like a wall)."""
    mat, nodes, links, bsdf = principled("floor_wood", 0.7)
    coord = nodes.new("ShaderNodeTexCoord")
    brick = nodes.new("ShaderNodeTexBrick")
    brick.inputs["Scale"].default_value = 1.2
    brick.inputs["Mortar Size"].default_value = 0.004
    brick.inputs["Brick Width"].default_value = 1.6
    brick.inputs["Row Height"].default_value = 0.18
    brick.inputs["Color1"].default_value = (0.22, 0.17, 0.12, 1)
    brick.inputs["Color2"].default_value = (0.20, 0.155, 0.11, 1)
    brick.inputs["Mortar"].default_value = (0.08, 0.06, 0.045, 1)
    wave = nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 6.0
    wave.inputs["Distortion"].default_value = 8.0
    wave.inputs["Detail"].default_value = 3.0
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.08
    links.new(coord.outputs["Object"], brick.inputs["Vector"])
    links.new(coord.outputs["Object"], wave.inputs["Vector"])
    links.new(brick.outputs["Color"], mix.inputs["A"])
    links.new(wave.outputs["Color"], mix.inputs["B"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


def add_plane(name, center, size, rotation, material):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=center, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0], size[1], 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def build_floor():
    add_plane("floor", (0, 0, 0), (2 * FLOOR_HALF, 2 * FLOOR_HALF), (0, 0, 0), mat_wood_floor())


def make_diffuse():
    """Lambertian appearance: zero every view-dependent lobe of every
    Principled BSDF (also inside node groups) and drop subsurface scattering."""
    def trees():
        for mat in bpy.data.materials:
            if mat.use_nodes and mat.node_tree is not None:
                yield mat.node_tree
        for group in bpy.data.node_groups:
            yield group

    settings = {"Metallic": 0.0, "Roughness": 1.0, "Specular IOR Level": 0.0,
                "Coat Weight": 0.0, "Sheen Weight": 0.0, "Subsurface Weight": 0.0,
                "Transmission Weight": 0.0}
    count = 0
    for tree in trees():
        for node in tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            for name, value in settings.items():
                socket = node.inputs.get(name)
                if socket is None:
                    continue
                for link in list(socket.links):
                    tree.links.remove(link)
                socket.default_value = value
            count += 1
    return count


def build_lights():
    for i, (x, y) in enumerate(((-1.2, -1.2), (1.2, -1.2), (-1.2, 1.2), (1.2, 1.2), (0, 0))):
        bpy.ops.object.light_add(type="AREA", location=(x, y, LIGHT_HEIGHT))
        light = bpy.context.object
        light.name = f"area_light_{i}"
        light.data.energy = 60.0
        light.data.size = 1.5
        light.data.color = (1.0, 1.0, 1.0)
    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35


def build_cameras():
    """cam01..cam10 on a 2.6 m circle looking down at (0, 0, 0.9), mounted at
    2.2 m or 1.7 m in the pattern H H L L H H L L H H, so the even (recon)
    and the odd (held-out) cameras both mix heights. View selection rejects a
    camera whose optical axis is within 10 deg of a surface's plane, so every
    camera looks down by at least 15 deg to texture the floor (like the
    Panoptic Kinects). A 70 deg horizontal FOV frames the person and the floor."""
    target = Vector((0.0, 0.0, 0.9))
    for i in range(NUM_CAMERAS):
        angle = 2 * math.pi * i / NUM_CAMERAS + math.radians(9)
        height = 2.2 if (i // 2) % 2 == 0 else 1.7
        location = Vector((2.6 * math.cos(angle), 2.6 * math.sin(angle), height))
        cam_data = bpy.data.cameras.new(f"cam{i + 1:02d}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = math.radians(CAMERA_HFOV_DEG)  # horizontal FOV
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.clip_start = 0.05
        cam_data.clip_end = 20.0
        cam = bpy.data.objects.new(f"cam{i + 1:02d}", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        cam.location = location
        cam.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()


# ---------------------------------------------------------------------------
# Human + motion
# ---------------------------------------------------------------------------
def build_human():
    from bl_ext.blender_org.mpfb.services.humanservice import HumanService

    info = HumanService._create_default_human_info_dict()
    info["phenotype"].update({"gender": 1.0, "age": 0.5, "muscle": 0.6, "weight": 0.5,
                              "height": 0.5,
                              "race": {"asian": 0.0, "caucasian": 1.0, "african": 0.0}})
    info["rig"] = "cmu_mb"
    info["eyes"] = "low-poly.mhclo"
    info["eyebrows"] = "eyebrow001.mhclo"
    info["eyelashes"] = "eyelashes01.mhclo"
    info["hair"] = "short02.mhclo"
    info["clothes"] = ["male_casualsuit02.mhclo", "shoes01.mhclo"]
    info["skin_mhmat"] = "young_caucasian_male.mhmat"
    info["skin_material_type"] = "MAKESKIN"  # plain BSDF, made diffuse below
    info["eyes_material_type"] = "MAKESKIN"
    settings = HumanService.get_default_deserialization_settings()
    HumanService.deserialize_from_dict(info, settings)
    return next(o for o in bpy.data.objects if o.type == "ARMATURE")


def apply_t_pose(rig):
    """The MPFB cmu_mb T-pose, which matches the CMU ASF rest pose."""
    import bl_ext.blender_org.mpfb as mpfb_pkg
    pose_path = Path(mpfb_pkg.__file__).parent / "data/poses/cmu_mb_fk/t-pose.json"
    pose = json.loads(pose_path.read_text())
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    for name, euler in pose["bone_rotations"].items():
        pb = rig.pose.bones[name]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = euler
    bpy.context.view_layer.update()
    t_pose = {pb.name: pb.matrix.copy() for pb in rig.pose.bones}
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
    return t_pose


def rotation_between(a: Vector, b: Vector) -> Matrix:
    return a.normalized().rotation_difference(b.normalized()).to_matrix()


def animate(rig, args):
    asf, amc = cmu_mocap.download(args.subject, args.trial, args.mocap_dir)
    skeleton = cmu_mocap.read_asf(asf)
    amc_frames = cmu_mocap.read_amc(amc)
    step = 120 // FPS
    picks = list(range(args.mocap_start, len(amc_frames), step))[:args.frames]
    if len(picks) < args.frames:
        raise ValueError(f"trial has only {len(picks)} frames at {FPS} fps")

    t_pose = apply_t_pose(rig)
    rest = {b.name: b.matrix_local.copy() for b in rig.data.bones}
    order = []

    def visit(bone):
        order.append(bone.name)
        for child in bone.children:
            visit(child)

    for bone in rig.data.bones:
        if bone.parent is None:
            visit(bone)
    rig_to_asf = {v: k for k, v in ASF_TO_RIG.items()}

    P = ASF_TO_BLENDER
    poses = [cmu_mocap.pose(skeleton, amc_frames[i]) for i in picks]
    # Scale: leg length of the character / of the subject (rest pose).
    _, _, rest_end = cmu_mocap.pose(skeleton, {"root": [0.0] * 6})
    src_leg = -rest_end["ltoes"][1]
    dst_leg = t_pose["Hips"].translation.z - min(t_pose["LeftToeBase"].translation.z, 0.0)
    scale = dst_leg / src_leg
    roots = np.array([p[1]["root"] for p in poses])
    center = roots.mean(axis=0)

    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for f, (rot, start, end) in enumerate(poses):
        posed = {}
        for name in order:
            pb = rig.pose.bones[name]
            bone = rig.data.bones[name]
            if bone.parent is not None:
                parent = bone.parent.name
                unposed = posed[parent] @ rest[parent].inverted() @ rest[name]
            else:
                unposed = rest[name]
            asf_name = rig_to_asf.get(name)
            if asf_name is None:
                target = unposed.to_3x3() @ (t_pose[name].to_3x3() @ unposed.to_3x3().inverted())
            else:
                g = Matrix((P @ rot[asf_name] @ P.T).tolist())
                target = g @ t_pose[name].to_3x3()
                if asf_name != "root" and skeleton.bones[asf_name].length > 0:
                    d = Vector((P @ (end[asf_name] - start[asf_name])).tolist())
                    if d.length > 1e-6:
                        target = rotation_between(target.col[1], d) @ target
            if bone.parent is None:
                offset = P @ (start["root"] - center)
                location = t_pose[name].translation.copy()
                location.x += scale * offset[0]
                location.y += scale * offset[1]
                location.z = scale * (P @ start["root"])[2]
            else:
                location = unposed.translation
            desired = Matrix.Translation(location) @ target.to_4x4()
            basis = unposed.inverted() @ desired
            loc, quat, _ = basis.decompose()
            pb.rotation_quaternion = quat
            pb.keyframe_insert("rotation_quaternion", frame=f + 1)
            if bone.parent is None:
                pb.location = loc
                pb.keyframe_insert("location", frame=f + 1)
            posed[name] = unposed @ Matrix.Translation(loc if bone.parent is None else (0, 0, 0)) @ quat.to_matrix().to_4x4()

    # Put the feet on the floor: the lowest toe over the clip is at z = 0.
    scene = bpy.context.scene
    lowest = math.inf
    for f in range(1, args.frames + 1):
        scene.frame_set(f)
        for name in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot"):
            pb = rig.pose.bones[name]
            lowest = min(lowest, (rig.matrix_world @ pb.head).z, (rig.matrix_world @ pb.tail).z)
    rig.location.z -= lowest - 0.02  # ankles/toes sit ~2 cm above the sole
    return {"subject": args.subject, "trial": args.trial, "amc_frames": [picks[0], picks[-1]],
            "scale": scale}


def main():
    args = parse_args()
    addon_utils.enable("bl_ext.blender_org.mpfb", default_set=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.unit_settings.system = "METRIC"

    build_floor()
    build_lights()
    build_cameras()
    rig = build_human()
    motion = animate(rig, args)
    print("diffuse BSDFs:", make_diffuse())
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = False  # no screen-space (view-dependent) GI
    scene["grtmesh_motion"] = json.dumps(motion)
    # The floor and the space above it; the empty background has no depth.
    scene["grtmesh_bounds"] = json.dumps({"min": [-FLOOR_HALF - 0.05, -FLOOR_HALF - 0.05, -0.05],
                                          "max": [FLOOR_HALF + 0.05, FLOOR_HALF + 0.05, 2.25]})
    scene.camera = bpy.data.objects["cam01"]
    scene.frame_set(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out), compress=True)
    print("saved", args.out, motion)


if __name__ == "__main__":
    main()

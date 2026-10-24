# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Minimal ASF/AMC reader for the CMU Graphics Lab Motion Capture Database.

The CMU database (http://mocap.cs.cmu.edu) places no restrictions on the use
of the data. This module needs only numpy, so it runs inside Blender's Python.

Conventions follow the ASF/AMC specification as used by the CMU data: each
bone has a local frame C (``axis``), a rest direction and a length; a frame of
motion gives Euler angles (degrees) for the bone's DOFs, and the bone's global
rotation is

    G_bone = G_parent @ C @ R(dofs) @ C^-1,      G_root = C_root @ R(root dofs) @ C_root^-1

so every bone has the identity global rotation in the rest pose. Positions are
in ASF length units (CMU: 1 unit = 1 / 0.45 inch) and the axes are Y-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import urllib.request

import numpy as np

CMU_URL = "http://mocap.cs.cmu.edu/subjects/{subject:02d}/{name}"
METERS_PER_UNIT = 0.0254 / 0.45  # CMU ASF length unit


def euler_matrix(angles_deg, order: str = "XYZ") -> np.ndarray:
    """Rotation applying the axes in `order` (first letter first)."""
    m = np.eye(3)
    for axis, angle in zip(order.upper(), np.deg2rad(angles_deg)):
        c, s = np.cos(angle), np.sin(angle)
        r = {"X": [[1, 0, 0], [0, c, -s], [0, s, c]],
             "Y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             "Z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        m = np.asarray(r) @ m
    return m


@dataclass
class Bone:
    name: str
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    length: float = 0.0
    axis: np.ndarray = field(default_factory=lambda: np.eye(3))  # C
    dofs: list[str] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)


@dataclass
class Skeleton:
    bones: dict[str, Bone]
    root_order: list[str]
    root_axis: np.ndarray

    def topological_order(self) -> list[str]:
        order, stack = [], ["root"]
        while stack:
            name = stack.pop()
            order.append(name)
            stack.extend(reversed(self.bones[name].children))
        return order


def read_asf(path: Path) -> Skeleton:
    lines = [ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    bones = {"root": Bone("root")}
    root_order, root_axis = ["TX", "TY", "TZ", "RX", "RY", "RZ"], np.eye(3)
    section, current = None, None
    for ln in lines:
        if ln.startswith(":"):
            section = ln.split()[0]
            continue
        tok = ln.split()
        if section == ":root":
            if tok[0] == "order":
                root_order = [t.upper() for t in tok[1:]]
            elif tok[0] == "orientation":
                root_axis = euler_matrix([float(t) for t in tok[1:4]], "XYZ")
        elif section == ":bonedata":
            if tok[0] == "begin":
                current = Bone("")
            elif tok[0] == "end":
                bones[current.name] = current
            elif tok[0] == "name":
                current.name = tok[1]
            elif tok[0] == "direction":
                current.direction = np.array([float(t) for t in tok[1:4]])
            elif tok[0] == "length":
                current.length = float(tok[1])
            elif tok[0] == "axis":
                current.axis = euler_matrix([float(t) for t in tok[1:4]], tok[4])
            elif tok[0] == "dof":
                current.dofs = [t.upper() for t in tok[1:]]
        elif section == ":hierarchy":
            if tok[0] in ("begin", "end"):
                continue
            parent = tok[0]
            for child in tok[1:]:
                bones[parent].children.append(child)
                bones[child].parent = parent
    return Skeleton(bones, root_order, root_axis)


def read_amc(path: Path) -> list[dict[str, list[float]]]:
    frames, current = [], None
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith(":"):
            continue
        tok = ln.split()
        if len(tok) == 1 and tok[0].isdigit():
            current = {}
            frames.append(current)
        elif current is not None:
            current[tok[0]] = [float(t) for t in tok[1:]]
    return frames


def pose(skeleton: Skeleton, frame: dict[str, list[float]]):
    """Global rotations (3x3), joint positions (bone start) and bone end
    positions of one AMC frame, in ASF units / axes."""
    rot, start, end = {}, {}, {}
    values = frame["root"]
    t = np.zeros(3)
    angles = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    order = []
    for name, v in zip(skeleton.root_order, values):
        if name.startswith("T"):
            t["XYZ".index(name[1])] = v
        else:
            angles[name[1]] = v
            order.append(name[1])
    c = skeleton.root_axis
    rot["root"] = c @ euler_matrix([angles[a] for a in order], "".join(order)) @ c.T
    start["root"] = end["root"] = t
    for name in skeleton.topological_order()[1:]:
        bone = skeleton.bones[name]
        values = frame.get(name, [0.0] * len(bone.dofs))
        angles = [0.0, 0.0, 0.0]
        for dof, v in zip(bone.dofs, values):
            angles["XYZ".index(dof[1])] = v
        # CMU bones list their DOFs as rx ry rz (subsets keep that order).
        rot[name] = rot[bone.parent] @ bone.axis @ euler_matrix(angles, "XYZ") @ bone.axis.T
        start[name] = end[bone.parent]
        end[name] = start[name] + bone.length * (rot[name] @ bone.direction)
    return rot, start, end


def download(subject: int, trial: int, out_dir: Path) -> tuple[Path, Path]:
    """Fetches <subject>.asf and <subject>_<trial>.amc into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in (f"{subject:02d}.asf", f"{subject:02d}_{trial:02d}.amc"):
        dst = out_dir / name
        if not dst.exists():
            urllib.request.urlretrieve(CMU_URL.format(subject=subject, name=name), dst)
        paths.append(dst)
    return paths[0], paths[1]

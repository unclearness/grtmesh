# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""How well the correction recovers the applied gains, measured at the seams.

    python python/synthetic/seam_recovery.py --dataset <dataset>/dance \
        --runs data_out/synthetic [--variant lum] [--json out.json]

The correction is relative: a seam constrains only the difference between the
two cameras that meet there. For every seam of the saved meshes (an edge whose
two faces are textured by different cameras; the atlas is the vertical stack
of the reconstruction cameras, so a face's camera and image position follow
from its UVs) this compares, on the solver grid,

  * the gain difference across the seam: applied (perturbation.json) vs
    recovered (negated field dump of the run), and
  * each camera's own gain at the seam: applied vs recovered.

The first is what removes the seams; the second also contains the level of
each camera, which the seams do not observe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figure as F  # noqa: E402


def load_obj(path: Path):
    vt, faces = [], []
    with path.open() as f:
        for line in f:
            if line.startswith("vt "):
                vt.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                parts = [p.split("/") for p in line.split()[1:4]]
                faces.append([(int(a) - 1, int(b) - 1) for a, b, *_ in parts])
    return np.array(vt), faces


def stats(applied: np.ndarray, recovered: np.ndarray) -> dict:
    return {"pearson_r": float(np.corrcoef(applied, recovered)[0, 1]),
            "slope_recovered_over_applied": float(np.sum(applied * recovered) / np.sum(applied ** 2)),
            "mean_abs_applied": float(np.abs(applied).mean()),
            "mean_abs_error": float(np.abs(applied - recovered).mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--variant", default="lum")
    p.add_argument("--frames", type=int, nargs=3, default=[5, 151, 10], metavar=("START", "END", "STRIDE"))
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    meta = json.loads((args.dataset / "cameras.json").read_text())
    cams = {c["name"]: c for c in meta["cameras"]}
    params = json.loads((args.dataset / "perturbation.json").read_text())["cameras"]
    n = len(F.RECON)
    width, height = cams[F.RECON[0]]["width"], cams[F.RECON[0]]["height"]
    rows = []
    for frame in range(*args.frames):
        vt, faces = load_obj(args.runs / args.variant / "mesh" / f"{frame:06d}.obj")
        dump = F.load_gain(args.runs / args.variant / "gain" / f"gain_{frame - meta['frames'][0]:06d}.txt")
        gh, gw = dump.shape[1:3]
        px = vt[:, 0] * width
        py = (1.0 - vt[:, 1]) * height * n
        face_cam = np.array([min(n - 1, int(py[f[0][1]] // height)) for f in faces])
        applied = [F.applied_grid(params[c], width, height, gw, gh) @ F.LUMA_BGR for c in F.RECON]
        recovered = [F.recovered_grid(dump, k) @ F.LUMA_BGR for k in range(n)]
        edges = {}
        for i, f in enumerate(faces):
            for (a, ta), (b, tb) in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                edges.setdefault((min(a, b), max(a, b)), []).append((i, ta, tb))
        for incident in edges.values():
            if len(incident) != 2:
                continue
            (fa, a0, a1), (fb, b0, b1) = incident
            ka, kb = face_cam[fa], face_cam[fb]
            if ka == kb:
                continue
            cells = []
            for k, t0, t1 in ((ka, a0, a1), (kb, b0, b1)):
                x = 0.5 * (px[t0] + px[t1])
                y = 0.5 * (py[t0] + py[t1]) - k * height
                cells.append((int(np.clip(y / height * gh, 0, gh - 1)), int(np.clip(x / width * gw, 0, gw - 1))))
            (ya, xa), (yb, xb) = cells
            rows.append((applied[ka][ya, xa] - applied[kb][yb, xb], recovered[ka][ya, xa] - recovered[kb][yb, xb],
                         applied[ka][ya, xa], recovered[ka][ya, xa]))
    r = np.array(rows)
    out = {"variant": args.variant, "frames": args.frames, "seam_edges": len(r),
           "difference_across_seam": stats(r[:, 0], r[:, 1]),
           "per_camera_gain_at_seam": stats(r[:, 2], r[:, 3])}
    print(json.dumps(out, indent=2))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

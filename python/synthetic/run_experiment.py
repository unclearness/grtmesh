# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Seam-correction experiment on the synthetic sequence: reconstruct from the
even cameras with several correction settings, re-render the meshes from the
odd (held-out) cameras and compare them with the unperturbed renders.

    python python/synthetic/run_experiment.py --dataset <dataset>/dance \
        --clean_dataset <dataset>/dance_clean \
        --out data_out/synthetic --exe bin/Release/multiview_rgbd.exe

Steps (each skipped when its output exists): recon -> render -> eval. The
metrics are PSNR / SSIM / LPIPS on the AND of every variant's rendered
coverage (python/eval_metrics.py), 5% trimmed means over all held-out images,
and the mean seam residual before / after correction reported by the
reconstruction (timing CSV), written to <out>/eval.json. The per-camera gain
fields of the corrected runs are dumped to <out>/<variant>/gain/ for figures.
Runs use the library defaults (24 x 16 grid, temporal EMA) with the overrides in CC.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_metrics as E  # noqa: E402

RECON_CAMERAS = "cam02,cam04,cam06,cam08,cam10"
HELD_OUT = ["cam01", "cam03", "cam05", "cam07", "cam09"]
# name -> multiview_rgbd flags. The correction uses the in-house setting of the
# paper (|g| <= 0.4, lambda_smooth = 5) and samples the seams on a 1/2 proxy
# instead of 1/8: the person covers only a small part of each image, and an
# 8 x 8 box at the seam would mix in background and other body parts.
CC = ["--color-correction", "field", "--color-correction-max-log-gain", "0.4",
      "--color-correction-lambda-smooth", "5", "--color-correction-proxy-downscale", "2"]
VARIANTS = {
    "off": ["--color-correction", "off"],
    "lum": CC,
    "rgb": [*CC, "--color-correction-space", "rgb"],
}


def tmean(values, trim=0.05):
    v = sorted(values)
    a = int(len(v) * trim)
    v = v[a:len(v) - a] if len(v) > 2 * a else v
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True, help="multiview_rgbd executable")
    p.add_argument("--clean_dataset", type=Path, default=None,
                   help="the same sequence without perturbation (make_dataset.py --no_perturbation); "
                        "adds the variant 'clean' (no correction) as an upper bound")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--voxel_size", default="0.01")
    p.add_argument("--frames", type=int, nargs=3, default=[1, 151, 1], metavar=("START", "END", "STRIDE"))
    p.add_argument("--device", default="cuda", help="LPIPS device")
    args = p.parse_args()

    out = args.out.resolve()
    names = args.variants.split(",")
    datasets = {k: args.dataset for k in names}
    if args.clean_dataset is not None:
        names.append("clean")
        datasets["clean"] = args.clean_dataset
        VARIANTS["clean"] = VARIANTS["off"]
    start, end, stride = args.frames
    for name in names:
        run_dir = out / name
        mesh_dir = run_dir / "mesh"
        if not (mesh_dir / f"{end - 1:06d}.obj").exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [str(args.exe), str(datasets[name]), str(run_dir), "--cameras", RECON_CAMERAS,
                   "--voxel-size", args.voxel_size, "--start-frame", str(start),
                   "--end-frame", str(end), *VARIANTS[name]]
            if VARIANTS[name] != VARIANTS["off"]:
                cmd += ["--color-correction-gain-dump-dir", str(run_dir / "gain")]
                (run_dir / "gain").mkdir(exist_ok=True)
            print("[recon]", " ".join(cmd), flush=True)
            with (run_dir / "stdout.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        render_dir = run_dir / "render"
        if not (render_dir / HELD_OUT[-1] / f"{end - 1:06d}.png").exists():
            cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "render_geoms.py"),
                   "--cameras_json", str(args.dataset / "cameras.json"),
                   "--input_dir", str(mesh_dir), "--output_dir", str(render_dir),
                   "--camera_ids", ",".join(HELD_OUT), "--frames", str(start), str(end)]
            print("[render]", " ".join(cmd), flush=True)
            with (run_dir / "render.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

    lp = E.LPIPSComputer(net="alex", device=args.device)
    acc = {k: {"psnr": [], "ssim": [], "lpips": []} for k in names}
    pairs = 0
    for frame in range(start, end, stride):
        for cam in HELD_OUT:
            gt_path = args.dataset / "clean" / f"{frame:06d}" / f"{cam}_color.png"
            preds = {k: out / k / "render" / cam / f"{frame:06d}.png" for k in names}
            if not gt_path.exists() or not all(v.exists() for v in preds.values()):
                continue
            gt = E.load_rgb_image(gt_path)
            mask = E.make_mask(list(preds.values()))
            if mask is None or mask.sum() == 0:
                continue
            pairs += 1
            for k, path in preds.items():
                pred = E.load_rgb_image(path)
                acc[k]["psnr"].append(E.compute_psnr(gt, pred, mask))
                acc[k]["ssim"].append(E.compute_ssim(gt, pred, mask))
                acc[k]["lpips"].append(lp.compute(gt, pred, mask))
    results = {k: {m: tmean(v) for m, v in a.items()} | {"n": len(a["psnr"])} for k, a in acc.items()}
    for k in names:
        rows = []
        for path in sorted((out / k).glob("timing_*.csv")):
            with path.open() as f:
                rows += [r for r in csv.DictReader(f)
                         if start <= int(r["frame"]) < end and float(r["cc_seam_before"]) > 0]
        if rows:
            results[k]["seam_before"] = float(np.mean([float(r["cc_seam_before"]) for r in rows]))
            results[k]["seam_after"] = float(np.mean([float(r["cc_seam_after"]) for r in rows]))
    print(f"{pairs} held-out images, shared mask over {names}")
    print(f"  {'variant':<10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    for k, r in results.items():
        seam = (f"  seam {r['seam_before']:.4f} -> {r['seam_after']:.4f}"
                if "seam_before" in r else "")
        print(f"  {k:<10} {r['psnr']:8.3f} {r['ssim']:8.4f} {r['lpips']:8.4f}{seam}")
    (out / "eval.json").write_text(json.dumps({
        "dataset": str(args.dataset), "recon_cameras": RECON_CAMERAS, "held_out": HELD_OUT,
        "frames": args.frames, "voxel_size": args.voxel_size,
        "variants": {k: VARIANTS[k] for k in names}, "pairs": pairs, "results": results}, indent=2))


if __name__ == "__main__":
    main()

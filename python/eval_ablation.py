# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Held-out re-rendering eval for camera-ready ablations (voxel size, occlusion).

Every variant is scored on the intersection of all variants' covered pixels;
the prediction dirs are given on the command line so one invocation
compares any set of same-build variants:

    PYTHONUTF8=1 ./.venv/Scripts/python.exe python/eval_ablation.py office2 \
        --gt_dir <export>/170407_office2/frames \
        --pred 2cm=data_out/cr/office2_2cm/render \
        --pred 1cm=data_out/cr/office2_1cm/render \
        --pred 5mm=data_out/cr/office2_5mm/render \
        --frames 400 700 3 --cams 1 3 5 7 9 --json out.json

All variants are scored on the AND of their non-black coverage, so the
numbers are comparable to each other but not to the published Table 2
(different build, different mask). Report deltas against the first --pred.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics as E  # noqa: E402

SCENES = ("office2", "band2")


def tmean(v, t=0.05):
    v = sorted(v)
    n = len(v)
    a = int(n * t)
    v = v[a:n - a] if n > 2 * a else v
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=SCENES)
    ap.add_argument("--gt_dir", required=True,
                    help="<export>/<seq>/frames with the kKK_rgb.jpg.undist.jpg ground truth")
    ap.add_argument("--pred", action="append", required=True,
                    help="name=render_dir (render_dir/<cam>/<frame:06d>.png); first is the reference")
    ap.add_argument("--frames", nargs=3, type=int, default=[400, 700, 3], metavar=("START", "END", "STRIDE"))
    ap.add_argument("--cams", nargs="+", type=int, default=[1, 3, 5, 7, 9])
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda", help="LPIPS device (use cpu while the GPU is timing runs)")
    args = ap.parse_args()

    preds = {}
    for p in args.pred:
        name, d = p.split("=", 1)
        preds[name] = Path(d)
    names = list(preds)
    gt = Path(args.gt_dir)
    lp = E.LPIPSComputer(net="alex", device=args.device)
    acc = {k: {"p": [], "s": [], "l": []} for k in names}
    used = 0
    for fr in range(*args.frames):
        for cam in args.cams:
            gp = E.get_gt_path(fr, cam, gt)
            pp = {k: E.get_pred_path(fr, cam, d) for k, d in preds.items()}
            if not gp.exists() or not all(p.exists() for p in pp.values()):
                continue
            g = E.load_rgb_image(gp)
            m = E.make_mask(list(pp.values()))
            if m is None or m.sum() == 0:
                continue
            used += 1
            for k in names:
                pr = E.load_rgb_image(pp[k])
                acc[k]["p"].append(E.compute_psnr(g, pr, m))
                acc[k]["s"].append(E.compute_ssim(g, pr, m))
                acc[k]["l"].append(lp.compute(g, pr, m))

    res = {k: {"psnr": tmean(a["p"]), "ssim": tmean(a["s"]), "lpips": tmean(a["l"]), "n": len(a["p"])}
           for k, a in acc.items()}
    ref = res[names[0]]
    print(f"[{args.scene}] {used} image pairs, shared mask over {names}")
    print(f"  {'variant':<10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} | {'dPSNR':>7} {'dSSIM':>8} {'dLPIPS':>8}")
    for k in names:
        r = res[k]
        print(f"  {k:<10} {r['psnr']:8.3f} {r['ssim']:8.4f} {r['lpips']:8.4f} | "
              f"{r['psnr']-ref['psnr']:+7.3f} {r['ssim']-ref['ssim']:+8.4f} {r['lpips']-ref['lpips']:+8.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"scene": args.scene, "frames": args.frames,
                                               "cams": args.cams, "pairs": used, "results": res}, indent=2))


if __name__ == "__main__":
    main()

# Reproducing the CMU Panoptic experiments

From the official download to the held-out re-rendering metrics of the paper.
Sequences: `170407_office2` and `160906_band2` (CMU Panoptic Studio, Kinoptic
part: ten Kinect v2 sensors). Replace `<dataset>` below with any directory for
the downloaded and exported data.

| Step | Tool | Disk per sequence |
|---|---|---|
| 1. Download | `python/panoptic_download.py` | ~28 GB (10 color videos + 10 depth streams) |
| 2. Prepare | `python/panoptic_prepare.py` (needs `ffmpeg` on `PATH`) | ~13 GB of extracted JPEGs, ~12 GB export for frames 0-2399 |
| 3. Build | CMake (see the README) | |
| 4. Reconstruct | `cmu_panoptic` | ~80 MB per frame (OBJ + atlas) |
| 5. Render held-out views | `python/render_geoms.py` (Blender 4.5) | |
| 6. Metrics | `python/eval_ablation.py` (`uv sync` for torch / lpips) | |

Python: `uv sync` installs every dependency declared in `pyproject.toml`
(Python 3.11.3). Blender: 4.5 LTS (tested with 4.5.5).

## 1. Download

```
python python/panoptic_download.py --out_root <dataset>/panoptic-toolbox 170407_office2 160906_band2
```

This fetches, from the official server, the files that the panoptic-toolbox
script `scripts/getData_kinoptic.sh <seq>` downloads (calibration, sync tables,
`kinectVideos/kinect_50_KK.mp4`, `kinect_shared_depth/KINECTNODE<K>/depthdata.dat`)
into the same layout; that script can be used instead. Interrupted downloads
resume. `--endpoint snu` selects the mirror.

## 2. Prepare

```
python python/panoptic_prepare.py --seq_dir <dataset>/panoptic-toolbox/170407_office2 ^
    --out_dir <dataset>/panoptic_export/170407_office2 --hd_start 0 --hd_end 2400
python python/panoptic_prepare.py --seq_dir <dataset>/panoptic-toolbox/160906_band2 ^
    --out_dir <dataset>/panoptic_export/160906_band2 --hd_start 0 --hd_end 2400
```

Three steps, each also available alone through `--steps`:

1. `extract`: `ffmpeg -vsync 0 -q:v 2` turns each Kinect color video into
   `kinectImgs/50_KK/50_KK_NNNNNNNN.jpg` (1-based, as in the toolbox).
2. `export`: for every HD frame, the Kinect color and depth frames nearest in
   universal time (sync tables), written as
   `frames/hd_NNNNNNNN/kKK_rgb.jpg` / `kKK_depth.png` (uint16 mm, flipped as in
   the toolbox) plus `meta.json` with the chosen indices and time offsets; the
   calibration and sync tables are copied to `calib/`.
3. `gt`: undistorted color of the held-out Kinects 1,3,5,7,9 for HD frames
   400-699 (`kKK_rgb.jpg.undist.jpg`), the ground truth of step 6.

The paper uses HD frames 400-699 (held-out evaluation) and 400-2399 (runtime,
free-view videos); `--hd_start 400 --hd_end 2400` is enough. Frame numbers on
the `cmu_panoptic` command line are HD frame numbers. The outputs are
byte-identical to the exports used for the paper (verified on frames
400-402 of both sequences with OpenCV 4.11).

## 3. Build

See [README.md](../README.md#configure-and-build-windows). The executable is
`bin/Release/cmu_panoptic.exe`.

## 4. Reconstruct (held-out protocol)

Reconstruct frames 400-699 from the even Kinects. Reconstruct every frame,
not every third: the seam correction smooths its gain field over time (EMA
0.15, the default), so the evaluated frames depend on the frames before them.
Every third frame (100 frames) is evaluated in step 6.

```
set "DS=<dataset>/panoptic_export"
set SEQ=170407_office2
bin\Release\cmu_panoptic.exe %DS% %SEQ% %DS%/%SEQ%/calib data_out/%SEQ%_off ^
    --kinect-ids 2,4,6,8,10 --start-frame 400 --end-frame 700
bin\Release\cmu_panoptic.exe %DS% %SEQ% %DS%/%SEQ%/calib data_out/%SEQ%_cc ^
    --kinect-ids 2,4,6,8,10 --start-frame 400 --end-frame 700 ^
    --color-correction field
```

The defaults are the paper's: 24x16 luminance gain field, `max_log_gain 0.15`,
`lambda_smooth 30`, 12 Jacobi iterations, EMA 0.15, a 0.10 m depth tolerance
for the visibility test (`--occlusion-threshold`), a four-voxel truncation
band and GPU input preprocessing. Add `--color-correction-space rgb` for the
per-channel variant. `VOXEL_SCALE=2` gives 1 cm voxels (5 mm, `=4`, needs the
`GRTMESH_LARGE_GRID` build); `NUM_KINECTS=N` reconstructs from Kinects 1..N
(sensor-count sweep).

Runtime: run frames 400-2399 with `--preload --serial --no-save-mesh` and
take the mean of the `e2e_ms` column of the timing CSV
`<out>/<seq>YYYYMMDD_HHMMSS.csv` over the fastest 95% of frames (input upload,
GPU preprocessing, reconstruction and texturing; file reading is excluded).
The paper's numbers come from a Visual Studio 2022 build (MSVC 14.44) with
CUDA 13.2; the default Visual Studio 2026 build produces the same geometry but
its runtimes may differ slightly. Free-view videos: reconstruct 400-2399 (the EMA is on by default).

## 5. Render the held-out views

```
python python/render_geoms.py --calib_json %DS%/%SEQ%/calib/kcalibration_%SEQ%.json ^
    --input_dir data_out/%SEQ%_off/%SEQ%mesh --output_dir data_out/%SEQ%_off/render ^
    --camera_ids 1,3,5,7,9 --frames 400 700
```

and the same for `_cc` (this renders all reconstructed frames; step 6 scores
every third). Blender 4.5 is expected at
`C:/Program Files/Blender Foundation/Blender 4.5/blender.exe` (`--blender` to
change). The meshes are rendered unlit (emission = atlas color) with EEVEE
into `render/<kinect>/<frame>.png`.

## 6. Metrics

```
python python/eval_ablation.py office2 --gt_dir %DS%/%SEQ%/frames ^
    --pred off=data_out/%SEQ%_off/render --pred cc=data_out/%SEQ%_cc/render ^
    --frames 400 700 3 --cams 1 3 5 7 9 --json data_out/%SEQ%_eval.json
```

PSNR / SSIM / LPIPS (AlexNet) against the undistorted held-out images, on the
AND of every variant's rendered (non-black) pixels, 5% trimmed mean over all
frame x camera pairs. The mask depends on which methods are compared (the
paper's Table 2 shares one mask among all baselines), and the absolute values
depend on the build and the renderer, so compare methods within one run; the
seam-correction effect is the difference `cc - off`.

# Synthetic multi-camera sequence with exposure / white-balance differences

A rendered stand-in for a multi-camera capture whose cameras differ in
exposure and white balance. Ten RGB-D cameras watch a person dancing on a
floor. The renders are Lambertian under fixed lighting, so a surface point has
the same color in every camera; each camera's color is then multiplied by its
own smooth gain (exposure and white balance), while the unperturbed renders
serve as ground truth. Reconstructing from the even cameras and re-rendering
into the odd ones measures how much of the inter-camera color difference the
seam correction removes. The differences follow the model of the correction
(multiplicative, low-frequency, within the default bound), so this is the
ideal case for the method.

| | |
|---|---|
| Scene | 3 x 3 m low-contrast plank floor and an MPFB 2 / MakeHuman character (CC0 assets) animated with CMU motion capture trial 05_02; empty background (no depth), all materials diffuse, EEVEE screen-space effects off |
| Cameras | `cam01`..`cam10` on a 2.6 m circle, 2.2 m / 1.7 m high, looking down at (0, 0, 0.9); 1280 x 720, 70 deg horizontal FOV, no lens distortion, depth registered to color |
| Frames | 150 at 30 fps (5 s) |
| Perturbation | per camera k and channel c, `color_c *= exp(e_k + a_k x + b_k y + s_c t_k + m_k [c = G])`: an exposure offset that alternates between +0.10 and -0.10 around the circle (neighbors differ by about 22%), a small linear gradient, a small color-temperature shift (`s = +1, 0, -1` for R, G, B) and green tint; at most 0.14 in log gain per channel, zero-mean over the even and over the odd cameras (`make_dataset.py`) |
| Split | reconstruct from `cam02,cam04,cam06,cam08,cam10`, evaluate on `cam01,cam03,cam05,cam07,cam09` |

## Dataset layout

```
dance/cameras.json                   intrinsics, camera_to_world (OpenCV axes), depth_scale, bounds
dance/perturbation.json              gain parameters of every camera
dance/frames/NNNNNN/camKK_color.jpg  perturbed input color (JPEG quality 95)
dance/frames/NNNNNN/camKK_depth.png  uint16 depth in mm (z along the optical axis, 0 = background)
dance/clean/NNNNNN/camKK_color.png   unperturbed color (ground truth)
```

This is the generic layout read by `multiview_rgbd` (documented in
`app/rgbd_sequence_dataset.h`); other calibrated RGB-D captures can be
converted to it.

## Reconstruct

```
bin/Release/multiview_rgbd.exe <dataset>/dance data_out/dance_lum ^
    --cameras cam02,cam04,cam06,cam08,cam10 --voxel-size 0.01 --color-correction field ^
    --color-correction-max-log-gain 0.4 --color-correction-lambda-smooth 5 ^
    --color-correction-proxy-downscale 2
```

The correction uses the paper's in-house setting (`|g| <= 0.4`,
`lambda_smooth = 5`) and samples the seams on a 1/2 proxy instead of the
default 1/8: the person covers only a small part of each image, and an 8 x 8
box at a seam would mix in background and other body parts.

`python/synthetic/run_experiment.py` runs the full comparison (no correction,
luminance and RGB fields with the setting above, and the unperturbed input as
an upper bound), renders the held-out views with
`python/render_geoms.py --cameras_json` and writes PSNR / SSIM / LPIPS and the
seam residuals to `eval.json`; `find_seam_crops.py` picks zoom boxes where a
seam is visible without correction, and `make_figure.py` draws the
supplementary figure (seam zoom-ins, applied vs recovered gains):

```
python python/synthetic/run_experiment.py --dataset <dataset>/dance ^
    --clean_dataset <dataset>/dance_clean --out data_out/synthetic ^
    --exe bin/Release/multiview_rgbd.exe
python python/synthetic/find_seam_crops.py --dataset <dataset>/dance --runs data_out/synthetic
python python/synthetic/make_figure.py --dataset <dataset>/dance ^
    --runs data_out/synthetic --out data_out/synthetic/figure ^
    --crop cam09:135:190,570,200,130 --crop2 cam03:70:550,290,110,110 --frame 70
```

## Regenerate the data

Requirements: Blender 4.5 LTS (tested with 4.5.5), the MPFB 2 extension and
the CC0 "MakeHuman system assets" pack, and the Python environment of `uv sync`
(see the top-level README). Install MPFB into a
separate Blender user directory so your own Blender setup is untouched
(PowerShell):

```
$env:BLENDER_USER_RESOURCES = "<dataset>\blender_user"
blender --online-mode -c extension sync
blender --online-mode -c extension install --enable mpfb
# MakeHuman system assets (CC0, 270 MB), unpacked into MPFB's user data
curl -L -o makehuman_system_assets_cc0.zip https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip
Expand-Archive makehuman_system_assets_cc0.zip "$env:BLENDER_USER_RESOURCES\extensions\.user\blender_org\mpfb\data"
```

Then build the scene, render and assemble the datasets (`--mocap_dir`
receives `05.asf` / `05_02.amc` from mocap.cs.cmu.edu):

```
blender -b -P python/synthetic/build_scene_bl.py -- --out <dataset>/scene.blend --mocap_dir <dataset>/downloads/cmu_mocap
blender -b <dataset>/scene.blend -P python/synthetic/render_bl.py -- --out <dataset>/render
python python/synthetic/make_dataset.py --render <dataset>/render --out <dataset>/dance
python python/synthetic/make_dataset.py --render <dataset>/render --out <dataset>/dance_clean --no_perturbation
```

`scene.blend` has every texture packed, so `render_bl.py` runs without MPFB.
Rendering uses EEVEE (64 samples); about 2 s per image on an RTX PRO 6000
(10 cameras x 150 frames).

| Script | Runs in | Purpose |
|---|---|---|
| `build_scene_bl.py` | Blender + MPFB | floor, character, motion retargeting, cameras, lights, diffuse materials -> `scene.blend` |
| `cmu_mocap.py` | any Python | ASF/AMC reader (global bone rotations per frame) |
| `render_bl.py` | Blender | clean color PNG + float depth EXR per camera and frame, `cameras.json` |
| `make_dataset.py` | Python + OpenCV | depth -> uint16 mm, per-camera exposure / white-balance gains |
| `run_experiment.py` | Python (`uv sync`) | reconstruct, render held-out views, PSNR / SSIM / LPIPS, seam residuals |
| `find_seam_crops.py` | Python (`uv sync`) | zoom boxes on the floor / the person where the correction removes a visible seam |
| `make_figure.py` | Python (`uv sync`) | seam zoom-ins of every variant, applied vs recovered gains |
| `seam_recovery.py` | Python (`uv sync`) | applied vs recovered gain differences across every seam (the correction is relative) and each camera's own gain at the seams |

## Credits and licenses

- Character: [MPFB 2](https://static.makehumancommunity.org/mpfb.html) and the
  MakeHuman system assets (skin, hair, eyes, clothes), CC0.
- Motion: CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu
  (subject 5, trial 2). The database was created with funding from NSF
  EIA-0196217; CMU places no restrictions on its use.
- Floor, cameras and lights are generated by `build_scene_bl.py`.

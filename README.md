# GPU-Native Real-Time Textured Mesh Reconstruction with Seam-Aware Multi-View Radiometric Correction (WACV 2027)

**grtmesh** (**G**PU, **R**eal-time, **T**extured **MESH**) is the library that implements the method, with namespaces `grtmesh` and `grtmesh::tex`. Source-available
under a noncommercial license; see [License](#license).

grtmesh turns synchronized depth + color images from several calibrated RGB-D
sensors into a textured triangle mesh (OBJ + a single UV atlas) every frame,
with the whole pipeline resident on the GPU: dense **TSDF fusion** of all views
in one kernel, **marching cubes**, **connected-component noise removal** on the
extracted mesh, **per-face view selection with UV inpainting** for faces no
camera sees well, packing into a **single texture atlas** (per-camera bands),
and finally a **low-frequency radiometric seam correction** that estimates a
smooth per-camera gain field from the texture seams and bakes it into the
atlas. Geometry, adjacency, view selection and the seam solve all stay on the
device; the host only stages the input images and receives the finished mesh.

## Citation

```bibtex
@inproceedings{sato2027grtmesh,
  title     = {GPU-Native Real-Time Textured Mesh Reconstruction with Seam-Aware Multi-View Radiometric Correction},
  author    = {Sato, Hiroyuki},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Requirements

### Operating system and compiler

| Platform | Notes |
|---|---|
| Windows 10/11, Visual Studio 2026 (MSVC v145) | First-class platform and the default below. Visual Studio 2022 (MSVC v143) is also supported; the paper's numbers were measured with a Visual Studio 2022 build. |
| Ubuntu 24.04, GCC 12.4 | Validated with CUDA 12.0, OpenCV 4.6 and an RTX 6000 Ada (sm_89). Use a GCC version that your CUDA toolkit supports as the host compiler. |

### GPU and CUDA

| Component | Notes |
|---|---|
| NVIDIA GPU, CUDA Toolkit 12.x or 13.x | **Required** (there is no CPU-only pipeline). Tested with **CUDA 12.0** on Ubuntu and **CUDA 13.2** on Windows (Visual Studio 2026 and 2022). CUDA 13 dropped `sm_<75`; the default `CMAKE_CUDA_ARCHITECTURES` list is chosen per toolkit version. |

### Build system

| Component | Notes |
|---|---|
| CMake >= 3.25 | C++20 is required for host and CUDA sources, including library consumers. |

### C++ libraries

| Library | Notes |
|---|---|
| OpenCV (C++ build) | **Apps only** (image I/O, undistortion, JPEG encoding). Point `OpenCV_DIR` at the directory containing `OpenCVConfig.cmake`. |
| OpenMP | Ships with MSVC; host-side parallel loops (`GRTMESH_USE_OPENMP`, default ON). |
| Eigen 3.4.1 | Git submodule `third_party/eigen`; used by the library. |
| nlohmann/json 3.12.0 | Git submodule `third_party/json`; used by the apps to load camera calibrations. |

### Python and Blender (optional)

| Component | Notes |
|---|---|
| Python 3.11.3, [uv](https://docs.astral.sh/uv/) | Data preparation, evaluation and rendering scripts under `python/`. `uv sync` installs every dependency declared in `pyproject.toml`; see [Python scripts](#python-scripts-python). |
| Blender 4.5 LTS | Render scripts (`python/**/*_bl.py`) and the synthetic sequence. Tested with Blender 4.5.5 LTS. The synthetic scene builder also needs the MPFB 2 extension ([python/synthetic/README.md](python/synthetic/README.md)). |
| ffmpeg | On `PATH` for `python/panoptic_prepare.py` (extracts the Kinect color videos). |

The camera count is limited only by memory: the per-camera state lives in
device buffers owned by each `Mesher`.

nlohmann/json is used by the applications to load camera intrinsics, lens
distortion and poses from Panoptic `kcalibration_*.json` and from the
`cameras.json` of the generic multi-view RGB-D layout. The per-frame
`meta.json` written by the Panoptic exporter records synchronization and
selected image paths; the C++ loader reads the RGB/depth images directly. The
`grtmesh` library has no JSON dependency.

## Data

- **CMU Panoptic** (paper experiments): [doc/reproduce_cmu_panoptic.md](doc/reproduce_cmu_panoptic.md)
  goes from the official download (`python/panoptic_download.py`) through the
  export (`python/panoptic_prepare.py`) to the held-out metrics.
- **Synthetic sequence** with per-camera exposure / white-balance differences
  and ground truth (Blender scene, rendering scripts, `multiview_rgbd` app):
  [python/synthetic/README.md](python/synthetic/README.md).

## Clone

```
git clone https://github.com/unclearness/grtmesh.git
cd grtmesh
git submodule update --init --recursive
```

On Windows, the `third_party/json` submodule contains very long file names;
enable long paths first (`git config --global core.longpaths true`) or clone
into a short directory such as `C:\src`.

Direct submodules: `third_party/eigen` (3.4.1) and `third_party/json`.
Eigen 3.4.1 includes the MSVC/CUDA C++20 compatibility fix
([Eigen issue #2690](https://gitlab.com/libeigen/eigen/-/issues/2690)).

## Configure and build (Windows)

Visual Studio 2026 is the default. Configure once from the repository root
(PowerShell):

```powershell
cmake -S . -B win_build -G "Visual Studio 18 2026" -A x64 `
  -T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2" `
  -DCMAKE_CUDA_ARCHITECTURES=75 `
  -DOpenCV_DIR="<path to the OpenCV build dir that contains OpenCVConfig.cmake>"
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU (e.g. `86`, `89`, `120`), or omit it
to build for every architecture the toolkit supports (slower to compile).
`-T cuda=<toolkit dir>` selects the CUDA toolkit for the Visual Studio
generator (nvcc and the Visual Studio CUDA integration together). Without it,
the generator uses the newest toolkit whose Visual Studio integration is
installed, which can differ from the toolkit you meant to use. If OpenCV is a
CUDA-enabled build, its CMake config calls the legacy `FindCUDA` module; then
also pass `-DCUDA_TOOLKIT_ROOT_DIR` with the same toolkit path.

Build everything (Release) with CMake, or a single project with MSBuild. Run
MSBuild from **PowerShell**; Git Bash mangles the `/p:` switches.

```powershell
cmake --build win_build --config Release
# or, per project:
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" win_build\grtmesh.vcxproj /p:Configuration=Release /p:Platform=x64 /m
& "C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe" win_build\cmu_panoptic.vcxproj /p:Configuration=Release /p:Platform=x64 /m
```

**Visual Studio 2022** works the same way with `-G "Visual Studio 17 2022"`
(MSBuild under `C:\Program Files\Microsoft Visual Studio\2022\Community\`). The
paper's runtime and quality numbers were measured with a Visual Studio 2022
build (MSVC 14.44, toolset v143) and CUDA 13.2. A Visual Studio 2026 build
(MSVC 14.51, toolset v145) of this code gives identical vertices and faces on
the paper's data; the UVs and atlas differ only as much as two runs of the same
build do (GPU scheduling), so measured runtimes may differ slightly from the
paper.

The static library goes to `lib/Release/grtmesh.lib`, executables to
`bin/Release/*.exe`. After editing a `.cu` file rebuild `grtmesh` first, then
the apps that link it.

## Configure and build (Ubuntu 24.04)

With the submodules initialized, CUDA Toolkit, OpenCV development files, and a
CUDA-supported host compiler installed, the portable build uses the system
compiler and the toolkit-compatible CUDA architecture list selected by this
project:

```sh
cmake -S . -B build_linux -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGRTMESH_BUILD_TESTS=ON
cmake --build build_linux --parallel 4
ctest --test-dir build_linux --output-on-failure
```

If `nvcc` does not support the system compiler, configure a fresh build
directory with a supported installed version explicitly. For example:

```sh
cmake -S . -B build_linux_gcc12 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DGRTMESH_BUILD_TESTS=ON
```

To shorten compilation for a machine-specific build, add
`-DCMAKE_CUDA_ARCHITECTURES=<compute-capability>` (for example, `89`) when that
architecture is supported by the installed toolkit. Otherwise omit it: the
project selects a portable architecture list compatible with the installed
toolkit. The executables are written to `bin/` and the library to `lib/`.

### CMake options

| Option | Default | Effect |
|---|---|---|
| `GRTMESH_LARGE_GRID` | OFF | 64-bit marching-cubes edge keys / int64 face cap for very fine grids (> 1e8 cells, e.g. 5 mm voxels in a room). Costs memory and time; leave OFF for grids up to ~300^3. |
| `GRTMESH_OUTPUT_SUFFIX` | `""` | Suffix appended to `bin/` and `lib/` so a variant build can coexist with the default one (e.g. `-DGRTMESH_LARGE_GRID=ON -DGRTMESH_OUTPUT_SUFFIX=_lg -B win_build_lg` -> `bin_lg/Release/`). |
| `GRTMESH_USE_OPENMP` | ON | OpenMP for host-side loops. |
| `GRTMESH_BUILD_APPS` | ON (OFF when added via `add_subdirectory`) | Build the sample / evaluation applications; requires OpenCV and the `json` submodule. |
| `GRTMESH_EIGEN_DIR` | `third_party/eigen` | Use another Eigen 3.4 checkout. |

### Using the library from another CMake project

```cmake
add_subdirectory(path/to/grtmesh)        # builds target `grtmesh`; apps are skipped
target_link_libraries(my_app PRIVATE grtmesh)
```

Only Eigen and the CUDA runtime are propagated; OpenCV and json are not
needed unless you build the apps.

## Applications (`bin/Release/`)

| Executable | Purpose |
|---|---|
| `cmu_panoptic` | Reconstruction of CMU Panoptic (Kinoptic) sequences; the main evaluation / free-view app. Full CLI below. |
| `multiview_rgbd` | Any calibrated multi-view RGB-D sequence in the generic layout (`cameras.json` + `frames/NNNNNN/<camera>_{color,depth}`, documented in `app/rgbd_sequence_dataset.h`), e.g. the synthetic sequence of `python/synthetic/`: `multiview_rgbd <dataset_dir> <out_dir> [--cameras cam02,cam04,...] [--voxel-size 0.02] [--occlusion-threshold F] [--color-correction ...]`. Writes `<out_dir>/mesh/NNNNNN.{obj,mtl,jpg}` and `<out_dir>/timing_YYYYMMDD_HHMMSS.csv` (same columns as below). The volume is the `bounds` box of `cameras.json`. |

The offline SPSR + mvs-texturing baseline app (`cmu_panoptic_poisson`) used in
the paper depends on external libraries that were removed from this branch; it
is kept on branch `wacv2027_1st_submission`.

### `cmu_panoptic` command line

```
Usage: cmu_panoptic <image_root> <data_name> <calib_dir> <out_root> [flags]
  Meshes are written to <out_root>/<data_name>mesh/NNNNNN.obj

Flags:
  --kinect-ids <comma list>   Kinect ids used for reconstruction,
                              e.g. 2,4,6,8,10 (held-out eval:
                              reconstruct from even, evaluate on
                              odd). Default: all Kinects listed in
                              the calibration.
  --color-correction field|global|off
  --color-correction-space lum|rgb
  --color-correction-grid W H
  --color-correction-iters N
  --color-correction-lambda-smooth F
  --color-correction-lambda-anchor F
  --color-correction-max-log-gain F
  --color-correction-data-weight F
  --color-correction-inset F
  --color-correction-reference-camera I
  --color-correction-proxy-downscale I
  --color-correction-temporal A  EMA of the gain field across
                              frames, A in (0,1); smaller =
                              stronger (default 0.15; outside
                              (0,1) = off)
  --color-correction-gain-dump-dir DIR
  --color-correction-profile  print the H2D/D2H/compute split
  --color-correction-cpu      solve the gain field on the CPU
                              reference path
  --color-correction-cpu-bake bake the corrected atlas on the CPU
  --start-frame N
  --end-frame N
  --max-frames N
  --frame-stride N
  --no-save-mesh
  --plain                   geometry-only pipeline (no texture stages)
  --occlusion-threshold F   absolute depth tolerance in meters (default 0.10)
  --profile-cuda            bracket frame processing for Nsight capture
  --save-point-cloud          also write <data_name>pc.ply (the
                              back-projected input of each frame)
  --volume-preset cmu-room|wide (default cmu-room)
  --preprocess cpu|gpu        input undistortion + depth-to-color
                              projection on the CPU (OpenMP) or
                              the GPU, where the results stay for
                              reconstruction (default gpu)
  --preload                   read and decode all frames into host
                              memory before the timed loop
  --serial                    preprocess each frame right before
                              reconstructing it; e2e_ms is then
                              the per-frame latency
  -h, --help                  print this message

Environment variables:
  NUM_KINECTS=N          use the first N Kinects {1..N} (sensor-count
                         sweep; when set > 0 it overrides --kinect-ids)
  VOXEL_SCALE=F          multiply the per-axis voxel count by F
  OCCLUSION_TH_SCALE=F   legacy: occlusion threshold = F * voxel size
                         (overrides --occlusion-threshold)
  CC_TEMPORAL, CC_DUMP_GAIN_DIR, CC_TRANSFER_PROFILE
                         same as the --color-correction-temporal /
                         -gain-dump-dir / -profile flags
```

Example (paper setting: held-out even Kinects, 24x16 luminance gain field,
EMA 0.15, 0.10 m depth tolerance; all of these are the defaults):

```
bin\Release\cmu_panoptic.exe <dataset>\panoptic_export 170407_office2 ^
    <dataset>\panoptic_export\170407_office2\calib data_out\office2 ^
    --kinect-ids 2,4,6,8,10 --start-frame 400 --max-frames 300 --volume-preset cmu-room ^
    --color-correction field
```

Input layout expected under `<image_root>/<data_name>/`: `frames/hd_NNNNNNNN/`
directories with per-Kinect color/depth images and `calib/` with the Kinoptic
calibration JSON, as written by `python/panoptic_prepare.py`
([doc/reproduce_cmu_panoptic.md](doc/reproduce_cmu_panoptic.md)). Frame
numbers are HD frame numbers, so a partial export works as long as it covers
the requested range. Images are read from disk frame by frame (the next frame
is loaded and undistorted while the current one is reconstructed), or all at
once before the loop with `--preload`.

Outputs (note there is no separator between `<data_name>` and the suffix):

```
<out_root>/<data_name>mesh/NNNNNN.obj          geometry + UVs (mtllib -> NNNNNN.mtl)
<out_root>/<data_name>mesh/NNNNNN.mtl          map_Kd -> NNNNNN.jpg
<out_root>/<data_name>mesh/NNNNNN.jpg          single texture atlas (per-camera bands, stacked vertically)
<out_root>/<data_name>pc.ply                   back-projected input points of each frame (--save-point-cloud)
<out_root>/<data_name>YYYYMMDD_HHMMSS.csv      per-frame timing (see below)
```

Timing CSV columns (milliseconds unless noted) start with
`frame,all,host2device,tsdf,mc,denoise,texture,device2host,misc,texture_only,inpaint,cc_added_ms,cc_seam_before,cc_seam_after,gpu_mem_used_mb`
where `all` is `Mesher::Process` (reconstruction and texturing, including the
final downloads), `texture = texture_only + inpaint`, `device2host` covers the
geometry and UV downloads, `misc` is the remainder of `all`, `cc_*` are the
seam-correction stage time and the mean absolute log-luminance seam residual
before/after correction, and `gpu_mem_used_mb` is the device memory in use
after the frame. Further columns split these stages and the transfers;
`e2e_ms` adds the input upload and preprocessing to `all` (the per-frame
latency with `--serial`) and `decode_ms` is the untimed file read.

The paper's runtimes are the mean of `e2e_ms` over the fastest 95% of frames
400-2399, run with `--preload --serial --no-save-mesh` (GPU preprocessing is
the default), from a Visual Studio 2022 build with CUDA 13.2 on an RTX PRO 6000
Blackwell Max-Q.

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `CC_TEMPORAL` | both apps (`grtmesh::app::ApplyColorCorrectionEnvironment` sets `ColorCorrectionConfig::temporal_alpha`; both also take `--color-correction-temporal`) | EMA of the gain field across frames, `A` in `(0,1)`; smaller = stronger smoothing. Default `0.15` (the paper setting, used for the videos and the held-out evaluation); `A <= 0` or `A >= 1` turns it off. |
| `CC_TRANSFER_PROFILE` | both apps (`ColorCorrectionConfig::profile_transfers`; `--color-correction-profile`) | Print the H2D / compute / D2H split of the seam-correction stage. |
| `CC_DUMP_GAIN_DIR` | both apps (`ColorCorrectionConfig::gain_dump_dir`; `--color-correction-gain-dump-dir`) | Directory to dump the per-frame gain field (`gain_NNNNNN.txt`, layout `[cam][gy][gx][ch]`, log gains before the clamp). |
| `NUM_KINECTS` | `app/panoptic_dataset.cc` (`PanopticKinopticDataset::Load`) | Reconstruct from the first N Kinects `{1..N}` (sensor-count sweep). When set `> 0` it **overrides** `--kinect-ids` (the original sensor-sweep semantics); precedence is `NUM_KINECTS` > `--kinect-ids` > all Kinects in the calibration. The selection source is printed at startup (`[INFO] Kinect ids (...)`), so unset the variable before held-out runs. |
| `VOXEL_SCALE` | `app/cmu_panoptic.cc` | Multiply the per-axis voxel count of the selected volume preset by `F` (finer grid for `F > 1`). |
| `OCCLUSION_TH_SCALE` | `app/cmu_panoptic.cc` | Legacy override: threshold = `F * voxel size`; takes precedence over `--occlusion-threshold`. Unset it for the fixed 0.10 m paper setting. |

## Library usage

Everything is reachable through the umbrella header `grtmesh/grtmesh.h`
(namespaces `grtmesh` for core types and geometry, `grtmesh::tex` for texturing and
seam correction). One `grtmesh::RgbdFrame` per sensor; all frames must share the
same depth size and the same color size. `Mesher` takes undistorted images:
either let `grtmesh::InputPreprocessor` undistort and register the raw sensor
images on the GPU (below), or undistort them yourself (the apps' CPU
reference, `grtmesh::app::UndistortPreprocessor`, uses OpenCV and gives the
same images).

```cpp
#include "grtmesh/grtmesh.h"

// Per-sensor input. Depth is float in any consistent metric unit (0 = invalid);
// the volume, voxel size and occlusion threshold use the same unit.
std::vector<grtmesh::RgbdFramePtr> frames;
for (int i = 0; i < num_sensors; ++i) {
  auto f = std::make_shared<grtmesh::RgbdFrame>();
  f->depth = grtmesh::DepthImage::Create(depth_w, depth_h);
  f->depth_intrinsics = /* grtmesh::CameraIntrinsics{fx, fy, cx, cy} */;
  f->depth_to_world = /* Eigen::Affine3f camera-to-world */;
  f->color = grtmesh::ColorImage::Create(color_w, color_h);   // 8-bit BGR
  f->color_intrinsics = /* ... */;
  f->color_to_world = /* ... */;
  f->depth_in_color = grtmesh::DepthImage::Create(color_w, color_h);  // depth reprojected into the color image
  frames.push_back(f);
}

grtmesh::VolumeParams volume;
volume.min_corner = {-2.f, -1.5f, 0.f};
volume.max_corner = { 2.f,  1.5f, 4.f};
volume.voxel_count = {200u, 150u, 200u};

grtmesh::Mesher mesher;
mesher.Init(volume, frames);   // allocates the TSDF volume, mesh buffers and camera records once

grtmesh::MesherParams params;
params.noise_removal.min_component_faces = 500;   // drop components with < 500 faces
params.color_correction.mode = grtmesh::tex::ColorCorrectionMode::kLowFreqField;  // seam correction (off by default)
params.color_correction.field_grid_w = 24;
params.color_correction.field_grid_h = 16;

for (;;) {
  // ... fill frames[i]->depth, frames[i]->depth_in_color, frames[i]->color ...
  grtmesh::Mesh mesh;
  grtmesh::FrameReport report;
  mesher.Process(frames, params, mesh, &report);  // fusion, MC, denoise, view selection, atlas (+ seam correction)
  // mesh.vertices / mesh.faces / mesh.uvs / mesh.uv_faces / mesh.atlas
  // report.timings (per stage, ms), report.color_correction (seam diagnostics)
}
```

GPU input preprocessing (`grtmesh/input_preprocessor.h`) starts from the
decoded sensor images and the lens calibration and leaves the undistorted
images on the device, where `Process` reads them in place:

```cpp
// One SensorFrame per sensor: calibrations (grtmesh::CameraCalibration with
// the OpenCV k1 k2 p1 p2 k3 k4 k5 k6 distortion), poses, depth_to_color, and
// pointers to the host pixels (BGR8 color, float depth) of the current frame.
std::vector<grtmesh::SensorFrame> sensor(num_sensors);
grtmesh::InputPreprocessor preprocessor;
preprocessor.Init(sensor);  // builds the undistortion maps once (host)
grtmesh::InputPreprocessor::DeviceFrames device;
for (;;) {
  // ... point sensor[i].color / sensor[i].depth at the new images ...
  preprocessor.Run(sensor, frames, device);  // upload, undistort, register
  const grtmesh::DeviceRgbdBatch batch = device.batch();
  mesher.Process(frames, params, mesh, &report, &batch);
}
```

The maps equal `cv::initUndistortRectifyMap(..., CV_16SC2)` of OpenCV 4.x's
AVX2 path entry for entry, so the result matches the OpenCV CPU path.
`preprocessor.Run(sensor, frames)` downloads the images into `frames`
instead.

`mesh.atlas` owns its pixel storage (a pooled pinned CUDA host buffer). It
stays valid after later frames and after the `Mesher` is destroyed. Copies of
an `Image` share pixels; `Clone()` makes an independent copy. Reusing the same
output `Mesh` object from frame to frame reuses its atlas buffer; retaining
output meshes makes the next frame take another buffer from the pool.

The atlas is packed on a host thread while the GPU stages run.
`params.build_atlas = false` skips the atlas and UVs (geometry only);
`params.download_face_assignments` additionally copies the per-face camera
assignment into `mesh.face_camera_ids` / `face_image_coords`. Instances are
independent (their own buffers, camera records and temporal correction
history) and may be interleaved; do not call one instance concurrently or
switch its CUDA device. `Init` starts a new sequence and resets the temporal
correction history.

Regression checks (requires a CUDA GPU): configure with
`-DGRTMESH_BUILD_TESTS=ON`, build, then run
`ctest --test-dir win_build -C Release --output-on-failure`.

`app/mesh_io.cc` (`grtmesh::app::SaveAsObj`) shows how to write the result as OBJ +
MTL + JPG.

### Runtime parameters (`grtmesh::MesherParams`, `include/grtmesh/mesher.h`)

Fusion:

- `fusion.truncation_distance` — unset (or `<= 0`) derives it from the voxel
  size (`4 x resolution`, the paper setting); the apps set `4 x` the smallest
  voxel edge explicitly.
  Each voxel keeps the running weighted mean of its observations (every
  observation has weight 1; the volume is cleared before every frame).
- `view_selection.occlusion_threshold` — depth tolerance of the per-face
  visibility test during view selection, in depth units.
- `view_selection.max_view_angle_deg` — cameras more oblique than this to the
  face normal are not candidates (default 80 deg, the published setting).
- `view_selection.inpaint_iterations` — propagate UVs from textured neighbors
  into faces no camera covers (default 20 iterations; `0` disables it; the
  paper runs use 5).
- `smooth_face_normals` — smooth face normals with vertex normals before
  view selection.

Atlas:

- `build_atlas` / `atlas_layout` — pack the per-camera images into
  `Mesh::atlas` (vertical bands by default; `kHorizontal`, `kTile`) and
  generate the UVs; `false` yields geometry only.
- `download_face_assignments` — also copy `face_camera_ids` /
  `face_image_coords` to the host (one extra transfer per frame).

Connected-component noise removal (small floating fragments produced by
marching cubes are dropped by looking at edge-connected components; the GPU
implementation uses approximate iterated label propagation,
which converges to the exact result with enough iterations):

- `noise_removal.min_component_faces` — components with fewer faces than
  this are removed; `<= 0` disables the stage. `100`-`1000` is a sensible range.
- `noise_removal.label_iterations` — maximum label-propagation
  iterations (default 100); `<= 0` disables the stage. Fewer iterations are
  faster but may under-merge; with a small iteration count and a large
  threshold parts of the foreground can disappear. Around 100 iterations
  clusters a 256^3 mesh exactly.
- `noise_removal.convergence_check_interval` — GPU path
  only. `>= 1` checks for convergence every N iterations and stops early;
  `<= 0` does **not** disable the check — `src/grtmesh/mesh_topology.cu`
  maps it to `8`, which is also the struct default. Each check costs one 4-byte
  device->host copy plus a stream sync, so `5`-`10` is the useful range.
  Early exit cannot be turned off on the GPU path.
### Seam color correction defaults (`grtmesh::tex::ColorCorrectionConfig`, `include/grtmesh/tex/color_correction.h`)

| Field | Default | Meaning |
|---|---|---|
| `mode` | `kOff` | `kLowFreqField` is the paper method, `kGlobalOnly` a one-gain-per-camera baseline (the same solve on a 1x1 field). |
| `space` | `kLuminanceOnly` | Single log-luminance gain (chroma preserved); `kLinearRgb` solves three independent channels. |
| `field_grid_w x field_grid_h` | `24 x 16` | Per-camera gain grid (the paper setting; `--color-correction-grid W H` overrides it). |
| `field_iterations` / `field_omega` | `12` / `0.7` | Matrix-free weighted-Jacobi iterations and under-relaxation. |
| `lambda_smooth` / `lambda_anchor` | `30` / `0.02` | Per-camera Laplacian smoothness; identity tether. |
| `seam_data_weight` | `1.0` | Seam data-term weight. |
| `reference_camera` | `-1` | Gauge: `-1` = zero-mean log gain over cameras, otherwise pin that camera to gain 1. |
| `max_log_gain` | `0.15` | Clamp on the log gain (gain in `[0.86, 1.16]`). |
| `proxy_downscale` / `seam_samples` / `sample_inset_proxy_pixels` | `8` / `8` / `0.5` | Residuals are sampled from a 1/8 linear-RGB proxy, 8 samples per seam, inset 0.5 proxy pixels. |
| `seam_max_dihedral_cos` | `0.5` | Seams whose faces fold more sharply than this are ignored. |
| `cpu_solve` / `cpu_bake` | `false` / `false` | Solve / bake on the CPU reference path (`src/grtmesh/tex/color_correction_cpu.cc`, same seam rules as the GPU path: fold skip, degenerate-face skip, dark-sample rejection, median residual) instead of CUDA, for verification. |
| `temporal_alpha` | `0.15` | EMA of the field across frames, in `(0, 1)`; the paper setting for the videos and the held-out evaluation. Values outside `(0, 1)` disable it (`CC_TEMPORAL` / `--color-correction-temporal`). |
| `gain_dump_dir` / `profile_transfers` | empty / `false` | Diagnostics: per-frame gain-field dump directory; print the upload / compute / download split. |

The temporal smoothing carries state from one `Process` call to the next, so
results depend on the frames processed before: the paper reconstructs frames
400-699 consecutively and evaluates every third one.

## Python scripts (`python/`)

Set up with `uv sync` (Python 3.11.3; every dependency is declared in
`pyproject.toml`, and PyTorch comes from the `cu130` index declared there) and
run scripts as `uv run python python/<script>.py`. The Blender scripts
(`*_bl.py`) run inside Blender 4.5 LTS (tested with 4.5.5) and are started by
their wrappers (`render_geoms.py`, `render_movie.py`) or as `blender -b -P`.

| Script | Purpose |
|---|---|
| `eval_metrics.py` | PSNR/SSIM/LPIPS helpers and the shared coverage mask (library used by `eval_ablation.py` and `synthetic/run_experiment.py`). |
| `eval_ablation.py` | Same-build comparison of several variants on their shared mask (`--pred name=render_dir`, `--gt_dir`). |
| `render_geoms.py` + `render_geoms_bl.py` | Blender/EEVEE render of per-frame meshes from the Kinect cameras (`--calib_json`) or the cameras of a `cameras.json` (`--cameras_json`) -> `<out>/<cam>/<frame>.png` (held-out eval renders). |
| `render_movie.py` + `render_movie_bl.py` | Orbit free-view render (10 turns, frames 400-2400): `render_movie.py --input-dir <meshes> --output-dir <out> [--blender <exe>]`. Params are CMU-dome-shared across scenes. |
| `nvblox_tsdf_cmu.py`, `o3d_tsdf_cmu.py` | Baseline reconstructions (run under WSL). |
| `panoptic_download.py`, `panoptic_prepare.py` | CMU Panoptic download (Python port of the toolbox's `getData_kinoptic.sh`) and export / held-out ground truth; see [doc/reproduce_cmu_panoptic.md](doc/reproduce_cmu_panoptic.md). |
| `synthetic/` | Synthetic sequence: Blender scene builder, renderer, perturbation, experiment driver; see [python/synthetic/README.md](python/synthetic/README.md). |

Held-out evaluation recipe: reconstruct with `--kinect-ids 2,4,6,8,10` ->
`render_geoms.py --camera_ids 1,3,5,7,9` -> `eval_ablation.py` (step-by-step
commands in [doc/reproduce_cmu_panoptic.md](doc/reproduce_cmu_panoptic.md)).
`render_geoms.py --cameras_json` renders meshes of the generic RGB-D layout.
Runtime numbers: the per-stage timing CSV written by `cmu_panoptic` (see
[`cmu_panoptic` command line](#cmu_panoptic-command-line)).

## Layout

```
CMakeLists.txt                  library target `grtmesh` (+ apps when GRTMESH_BUILD_APPS)
include/grtmesh/                 public API, namespace grtmesh
  grtmesh.h                      umbrella header
  image.h, camera.h, frame.h    Image<T,C> / ColorImage / DepthImage, CameraIntrinsics, RgbdFrame
  mesh.h, mesher.h              Mesh, Mesher (GPU pipeline), VolumeParams / MesherParams / FrameReport
include/grtmesh/tex/                  public API, namespace grtmesh::tex
  color_correction.h            ColorCorrectionConfig / Mode / Space / Stats
  atlas.h                       AtlasLayout, PackAtlas, FillColorAreaOutsideMeshBlack
src/common/                     cuda_check.h (fail-fast macros), device_buffer.h (RAII device / pinned
                                buffers), device_camera.h (camera record for the kernels), stopwatch.h
src/grtmesh/                     TSDF volume + marching cubes (tsdf_volume.cu), mesh topology /
                                connected components (mesh_topology.cu), Mesher (mesher.cc)
src/grtmesh/tex/                      per-face view selection + propagation (view_selection.cu), atlas_tiling.h,
                                seam color correction (color_correction_cuda.cu, color_correction_cpu.cc),
                                atlas packing (atlas.cc)
app/                            sample / evaluation apps (namespace grtmesh::app): undistortion and shared
                                flags (util.cc), dataset loaders (panoptic_dataset.cc,
                                rgbd_sequence_dataset.cc), OBJ/PLY writers (mesh_io.cc), one main per app
python/                         data preparation, evaluation, rendering, baselines, synthetic sequence
doc/                            reproduce_cmu_panoptic.md (paper experiments, step by step)
third_party/                    eigen, json (git submodules)
```

## License

This is source-available software, not open source. It is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE) (SPDX: `PolyForm-Noncommercial-1.0.0`):
you may use, modify and share it for noncommercial purposes (personal use, and
use by noncommercial organizations such as universities and public research
institutes), subject to the notice requirements in the license.

**Additional permission.** Evaluation and comparison of this software for the
purpose of academic publication or presentation (for example, running it as a
baseline in a paper) is permitted for everyone, including commercial
organizations.

**Commercial use** (products, services, or internal business use beyond the
above) requires a separate commercial license; see
[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

**Exceptions.** The Blender scripts (`python/**/*_bl.py`, which import `bpy`)
are licensed under the GNU GPL-3.0-or-later
([LICENSES/GPL-3.0-or-later.txt](LICENSES/GPL-3.0-or-later.txt)), as Blender
add-ons and scripts using its Python API are. Third-party components keep their
own licenses (Eigen: MPL2; nlohmann/json: MIT); CUDA and OpenCV are not
bundled.

Every first-party source file carries an `SPDX-License-Identifier` line.
Copyright (c) 2026 Hiroyuki Sato.

## C++ style checks

First-party C++ and CUDA code follows the Google C++ style guide and is compiled
as C++20. Existing exception handling is retained; CUDA extensions and
OpenMP pragmas are necessary platform features. Third-party and generated code
are excluded. These checks cover selected rules, not full semantic compliance.

Install the pinned tools in a separate environment (activate it on Windows with
`.lint-venv\Scripts\Activate.ps1`, or on Unix with
`source .lint-venv/bin/activate`):

```sh
python -m venv .lint-venv
python -m pip install -r tools/requirements-lint.txt
python tools/lint_cpp.py
python tools/lint_cpp.py --fix --format-only
```

The runner checks tracked files under `include/`, `src/`, `app/`, and `tests/`,
including `.cu` and `.cuh`. Formatting uses clang-format 19.1.5; cpplint 2.0.2
checks include hygiene and other source rules. Header guards use
`GRTMESH_<FULL_REPOSITORY_PATH>_` and are checked separately so checkout directory
names do not affect them. Mutable output references remain supported.

For compiler-aware checks, install clang-tidy 19 and configure a separate Ninja
build with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` and the normal CUDA / OpenCV
toolchain options. The Visual Studio generator does not emit this database.
Then run:

```sh
python tools/lint_cpp.py --tidy --build-dir build_lint
```

The optional clang-tidy pass checks host `.cc` translation units and included
first-party headers; CUDA kernels are checked by clang-format, cpplint, and the
nvcc build. GitHub Actions runs formatting, guard, and cpplint checks without
requiring CUDA or third-party submodules. Semantic CUDA validation remains the
Release build and existing regression / output-equivalence checks.

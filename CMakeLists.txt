# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
cmake_minimum_required(VERSION 3.25)

project(grtmesh
  LANGUAGES C CXX
  VERSION 1.0.0
  DESCRIPTION "grtmesh: GPU-native real-time textured mesh reconstruction with seam-aware radiometric correction")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# GRTMESH_OUTPUT_SUFFIX lets a variant build (e.g. GRTMESH_LARGE_GRID) coexist with
# the default build instead of overwriting bin/ and lib/.
set(GRTMESH_OUTPUT_SUFFIX "" CACHE STRING "Suffix appended to bin/ and lib/ output dirs")
option(GRTMESH_LARGE_GRID "int64 face-cap / 64-bit marching-cubes edge keys for ~5 mm voxel grids (>1e8 cells)" OFF)
option(GRTMESH_USE_OPENMP "Use OpenMP for host-side parallel loops" ON)

get_directory_property(GRTMESH_HAS_PARENT PARENT_DIRECTORY)
if(GRTMESH_HAS_PARENT)
  option(GRTMESH_BUILD_APPS "Build the sample / evaluation applications" OFF)
else()
  option(GRTMESH_BUILD_APPS "Build the sample / evaluation applications" ON)
endif()

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib${GRTMESH_OUTPUT_SUFFIX})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib${GRTMESH_OUTPUT_SUFFIX})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin${GRTMESH_OUTPUT_SUFFIX})

# ---------------------------------------------------------------------------
# Host compiler settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # /EHsc: proper exception handling (avoids C4530); /W4 instead of the noisy
  # /Wall; /MP parallel compile; /bigobj for the large CUDA translation units.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
                      $<$<COMPILE_LANGUAGE:CXX>:/MP> $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  # CUDA 13.x CCCL requires the conforming preprocessor on MSVC.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  add_compile_definitions(NOMINMAX)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-O2>)
endif()

# ---------------------------------------------------------------------------
# CUDA (required)
# ---------------------------------------------------------------------------
# Decide whether the user picked architectures BEFORE enable_language: CMake
# caches a detected default during enable_language, so a NOT DEFINED check
# afterwards never fires.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(GRTMESH_SET_DEFAULT_CUDA_ARCHS TRUE)
else()
  set(GRTMESH_SET_DEFAULT_CUDA_ARCHS FALSE)
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "grtmesh requires CUDA (nvcc not found). Set CMAKE_CUDA_COMPILER.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(GRTMESH_SET_DEFAULT_CUDA_ARCHS)
  # Keep this list valid for both CUDA 12.x and 13.x:
  # - CUDA 13 removed Maxwell/Pascal/Volta (< sm_75)
  # - sm_100 / sm_120 (Blackwell) need >= 12.8
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(GRTMESH_DEFAULT_CUDA_ARCHS 75 80 86 87 89 90 100 120)
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(GRTMESH_DEFAULT_CUDA_ARCHS 50 52 60 61 70 72 75 80 86 87 89 90 100 120)
  else()
    set(GRTMESH_DEFAULT_CUDA_ARCHS 50 52 60 61 70 72 75 80 86 87 89 90)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${GRTMESH_DEFAULT_CUDA_ARCHS} CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "grtmesh: CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets --extended-lambda")
if(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
  # CUDA 12.x nvcc rejects host compilers newer than it knows (e.g. VS2026).
  if(MSVC_VERSION GREATER_EQUAL 1950 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  endif()
endif()

# ---------------------------------------------------------------------------
# Third party (header-only; direct submodules of this repository)
# ---------------------------------------------------------------------------
set(GRTMESH_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen" CACHE PATH "Eigen root (contains Eigen/)")
if(NOT EXISTS "${GRTMESH_EIGEN_DIR}/Eigen/Core")
  message(FATAL_ERROR "Eigen not found at ${GRTMESH_EIGEN_DIR}. Run: git submodule update --init --recursive")
endif()
add_library(grtmesh_eigen INTERFACE)
target_include_directories(grtmesh_eigen SYSTEM INTERFACE "${GRTMESH_EIGEN_DIR}")

set(GRTMESH_JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include")

if(GRTMESH_USE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ---------------------------------------------------------------------------
# Library: grtmesh (namespaces grtmesh + grtmesh::tex)
# ---------------------------------------------------------------------------
set(GRTMESH_PUBLIC_HEADERS
  include/grtmesh/grtmesh.h
  include/grtmesh/image.h
  include/grtmesh/camera.h
  include/grtmesh/frame.h
  include/grtmesh/input_preprocessor.h
  include/grtmesh/mesh.h
  include/grtmesh/mesher.h
  include/grtmesh/tex/color_correction.h
  include/grtmesh/tex/atlas.h
)

set(GRTMESH_SOURCES
  src/common/cuda_check.h
  src/common/device_buffer.h
  src/common/device_camera.h
  src/common/stopwatch.h
  src/grtmesh/tsdf_volume.h
  src/grtmesh/tsdf_volume.cu
  src/grtmesh/mesh_topology.cuh
  src/grtmesh/mesh_topology.cu
  src/grtmesh/mesher.cc
  src/grtmesh/input_preprocessor.cc
  src/grtmesh/input_preprocessor_kernels.cuh
  src/grtmesh/input_preprocessor_kernels.cu
  src/grtmesh/undistort_maps.cc
  src/grtmesh/tex/atlas_tiling.h
  src/grtmesh/tex/view_selection.cuh
  src/grtmesh/tex/view_selection.cu
  src/grtmesh/tex/color_correction_internal.h
  src/grtmesh/tex/color_correction_cuda.cu
  src/grtmesh/tex/color_correction_cpu.cc
  src/grtmesh/tex/atlas.cc
)

add_library(grtmesh STATIC ${GRTMESH_PUBLIC_HEADERS} ${GRTMESH_SOURCES})
add_library(grtmesh::grtmesh ALIAS grtmesh)
target_include_directories(grtmesh
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(grtmesh PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(grtmesh PUBLIC grtmesh_eigen CUDA::cudart)
if(GRTMESH_USE_OPENMP)
  target_link_libraries(grtmesh PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(grtmesh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# Warning hygiene: Eigen headers compiled through nvcc emit host/device
# annotation diagnostics (20011/20012/20014) and, on MSVC, unknown-pragma C4068;
# the CUDA headers trip the code-page warning C4819 on non-UTF-8 locales.
target_compile_options(grtmesh PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20011,20012,20014>)
if(MSVC)
  target_compile_options(grtmesh PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4068>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
    # nvcc-generated host stubs (extern->static redefinition, unused parameter)
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4100>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4819>)
endif()
# undistort_maps.cc reproduces OpenCV's arithmetic exactly; its scalar part
# must not be contracted into fused multiply-adds (MSVC does not contract).
if(NOT MSVC)
  set_source_files_properties(src/grtmesh/undistort_maps.cc PROPERTIES
    COMPILE_OPTIONS -ffp-contract=off)
endif()
set_target_properties(grtmesh PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(GRTMESH_LARGE_GRID)
  target_compile_definitions(grtmesh PUBLIC GRTMESH_LARGE_GRID)
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GRTMESH_PUBLIC_HEADERS} ${GRTMESH_SOURCES})

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
if(GRTMESH_BUILD_APPS)
  find_package(OpenCV REQUIRED)

  add_library(grtmesh_app_common STATIC
    app/util.h
    app/util.cc
    app/panoptic_dataset.h
    app/panoptic_dataset.cc
    app/rgbd_sequence_dataset.h
    app/rgbd_sequence_dataset.cc
    app/parallel_runtime.h
    app/mesh_io.h
    app/mesh_io.cc
  )
  target_include_directories(grtmesh_app_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${GRTMESH_JSON_INCLUDE_DIR}
           ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(grtmesh_app_common PUBLIC grtmesh ${OpenCV_LIBS})
  if(MSVC)
    target_compile_options(grtmesh_app_common PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/wd4819>)
  endif()
  if(GRTMESH_USE_OPENMP)
    target_link_libraries(grtmesh_app_common PUBLIC OpenMP::OpenMP_CXX)
  endif()

  function(grtmesh_add_app TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE grtmesh_app_common)
    set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER "apps")
  endfunction()

  # CMU Panoptic (paper experiments) and the generic multi-view RGB-D layout
  # (synthetic sequence of python/synthetic/).
  grtmesh_add_app(cmu_panoptic                            app/cmu_panoptic.cc)
  grtmesh_add_app(multiview_rgbd                          app/multiview_rgbd.cc)
endif()

option(GRTMESH_BUILD_TESTS "Build CUDA regression tests" OFF)
if(GRTMESH_BUILD_TESTS)
  enable_testing()
  add_executable(grtmesh_regression tests/regression.cc)
  target_include_directories(grtmesh_regression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(grtmesh_regression PRIVATE grtmesh)
  add_test(NAME grtmesh_regression COMMAND grtmesh_regression)
endif()

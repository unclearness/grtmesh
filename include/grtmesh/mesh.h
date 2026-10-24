// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_INCLUDE_GRTMESH_MESH_H_
#define GRTMESH_INCLUDE_GRTMESH_MESH_H_

#include <array>
#include <cstdint>
#include <vector>

#include "Eigen/Geometry"
#include "grtmesh/image.h"

namespace grtmesh {

// Face camera id meaning "no camera sees this face".
inline constexpr int32_t kNoCamera = -1;

struct Mesh {
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> faces;
  std::vector<Eigen::Vector3f> face_normals;

  // Texture coordinates: three UVs per face, indexed through uv_faces, into
  // `atlas`. The atlas owns its storage and stays valid after later frames
  // and after the Mesher is destroyed. Empty when MesherParams::build_atlas is
  // false.
  std::vector<Eigen::Vector2f> uvs;
  std::vector<Eigen::Vector3i> uv_faces;
  ColorImage atlas;

  // Per-face view assignment: the camera that textures each face (kNoCamera
  // when none does) and the face's vertex positions, in pixels, in that
  // camera's image. Filled when MesherParams::download_face_assignments is
  // set (the CPU seam-correction path fills them as well).
  std::vector<int32_t> face_camera_ids;
  std::vector<std::array<Eigen::Vector2f, 3>> face_image_coords;
};

}  // namespace grtmesh

#endif  // GRTMESH_INCLUDE_GRTMESH_MESH_H_

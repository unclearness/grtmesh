// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#ifndef GRTMESH_APP_MESH_IO_H_
#define GRTMESH_APP_MESH_IO_H_

#include <string>
#include <vector>

#include "grtmesh/grtmesh.h"

namespace grtmesh::app {

// Writes <dir>/<base>.obj, <dir>/<base>.mtl and the atlas <dir>/<base>.jpg
// (mesh.atlas). OBJ layout: `mtllib`, `v x y z 1.0`, `vt u v 0`,
// `usemtl default_mat`, `f v/vt v/vt v/vt` (1-based).
bool SaveAsObj(const std::string& path, const grtmesh::Mesh& mesh);

// Writes an ASCII PLY (positions only) of every valid depth pixel of `frames`,
// back-projected with depth_intrinsics and moved to world space by
// depth_to_world.
void SavePointCloud(const std::string& path,
                    const std::vector<grtmesh::RgbdFramePtr>& frames);

}  // namespace grtmesh::app

#endif  // GRTMESH_APP_MESH_IO_H_

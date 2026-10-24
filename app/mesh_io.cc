// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "app/mesh_io.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "app/util.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

namespace grtmesh::app {

namespace {

// The single material every OBJ gets (values follow bunny.mtl).
constexpr const char* kDefaultMaterialName = "default_mat";

bool WriteMtl(const std::string& mtl_path, const std::string& diffuse_texname) {
  std::ofstream ofs(mtl_path);
  if (ofs.fail()) {
    std::cerr << "couldn't open mtl path: " << mtl_path << std::endl;
    return false;
  }

  const float ambient[3] = {0.117647f, 0.117647f, 0.117647f};   // Ka
  const float diffuse[3] = {0.752941f, 0.752941f, 0.752941f};   // Kd
  const float specular[3] = {0.752941f, 0.752941f, 0.752941f};  // Ks
  const float shininess = 8.0f;                                 // Ns
  const float dissolve = 1.0f;  // 1 == opaque (written as Tr = 1 - d)
  const int illum = 1;

  ofs << "newmtl " << kDefaultMaterialName << '\n'
      << "Ka " << ambient[0] << " " << ambient[1] << " " << ambient[2] << '\n'
      << "Kd " << diffuse[0] << " " << diffuse[1] << " " << diffuse[2] << '\n'
      << "Ks " << specular[0] << " " << specular[1] << " " << specular[2]
      << '\n'
      << "Tr " << 1.0f - dissolve << '\n'
      << "illum " << illum << '\n'
      << "Ns " << shininess << '\n';
  if (!diffuse_texname.empty()) {
    ofs << "map_Kd " << diffuse_texname << "\n";
  }
  ofs.close();

  return true;
}

bool WriteObj(const std::string& obj_path, const std::string& mtl_name,
              const grtmesh::Mesh& mesh) {
  std::ofstream ofs(obj_path);
  if (ofs.fail()) {
    std::cerr << "couldn't open obj path: " << obj_path << std::endl;
    return false;
  }

  ofs << "mtllib " << mtl_name << "\n"
      << "\n";

  // vertices (no vertex colors; the trailing 1.0 is the homogeneous w)
  for (const auto& v : mesh.vertices) {
    ofs << "v " << v.x() << " " << v.y() << " " << v.z() << " 1.0"
        << "\n";
  }

  // uv
  for (const auto& vt : mesh.uvs) {
    ofs << "vt " << vt.x() << " " << vt.y() << " 0"
        << "\n";
  }

  // faces: one material group, original face order, 1-based indices
  const bool write_uv_indices = !mesh.uv_faces.empty();
  if (!mesh.faces.empty()) {
    ofs << "usemtl " << kDefaultMaterialName << "\n";
  }
  for (size_t i = 0; i < mesh.faces.size(); i++) {
    ofs << "f";
    for (int j = 0; j < 3; j++) {
      ofs << " " << std::to_string(mesh.faces[i][j] + 1);
      if (write_uv_indices) {
        ofs << "/" << std::to_string(mesh.uv_faces[i][j] + 1);
      }
    }
    ofs << "\n";
  }

  ofs.close();

  return true;
}

}  // namespace

bool SaveAsObj(const std::string& path, const grtmesh::Mesh& mesh) {
  const std::filesystem::path obj(path);
  const std::filesystem::path dir = obj.parent_path();
  const std::string basename = obj.stem().string();
  const std::string diffuse_texname = basename + ".jpg";
  const std::string mtl_name = basename + ".mtl";

  WriteObj((dir / (basename + ".obj")).string(), mtl_name, mesh);
  WriteMtl((dir / mtl_name).string(),
           mesh.atlas.empty() ? "" : diffuse_texname);

  if (!mesh.atlas.empty()) {
    cv::Mat3b cv_tex;
    CopyToMat(cv_tex, mesh.atlas);
    cv::imwrite((dir / diffuse_texname).string(), cv_tex);
  }

  return true;
}

void SavePointCloud(const std::string& path,
                    const std::vector<grtmesh::RgbdFramePtr>& frames) {
  std::vector<Eigen::Vector3f> points;

  for (const auto& frame : frames) {
    const grtmesh::DepthImage& depth = frame->depth;
    const grtmesh::CameraIntrinsics& intrin = frame->depth_intrinsics;
    const Eigen::Affine3f& c2w = frame->depth_to_world;

    for (uint32_t y = 0; y < depth.height; y++) {
      for (uint32_t x = 0; x < depth.width; x++) {
        const float d = depth.data()[y * depth.width + x];
        if (d <= 0.f) {
          continue;
        }
        const Eigen::Vector3f p_c(
            (static_cast<float>(x) - intrin.cx) * d / intrin.fx,
            (static_cast<float>(y) - intrin.cy) * d / intrin.fy, d);
        points.push_back(c2w * p_c);
      }
    }
  }

  std::ofstream ofs(path);
  if (ofs.fail()) {
    std::cerr << "couldn't open ply: " << path << std::endl;
    return;
  }

  ofs << "ply\n"
      << "format ascii 1.0\n"
      << "element vertex " << points.size() << "\n"
      << "property float x\n"
      << "property float y\n"
      << "property float z\n"
      << "element face 0\n"
      << "property list uchar int vertex_indices\n"
      << "end_header\n";
  for (const auto& p : points) {
    ofs << p.x() << " " << p.y() << " " << p.z() << "\n";
  }
  ofs.close();
}

}  // namespace grtmesh::app

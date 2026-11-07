// The UV grid rasterizer claims the pixels on the mesh boundary: a grid aligned mesh fills every
// row and column of its grid (the top-left rule left the first row and the last column empty).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SurfaceMeshing.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>

using namespace vc::meshing;

namespace {
// A lattice plane of rows x cols vertices, `step` voxels apart, UV = voxel offsets (as vc_tifxyz2obj
// writes them for a surface of scale 1/step), faces wound as meshGrid winds them.
UvMesh plane(int rows, int cols, float step = 4.f)
{
    UvMesh m;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            m.vertices.push_back({10000.f + step * c, 8000.f + step * r, 6000.f});  // flat: 3-D spacing == step
            m.uvs.push_back({step * c, step * r});
        }
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r + 1 < rows; ++r)
        for (int c = 0; c + 1 < cols; ++c) {
            m.faces.push_back({{id(r + 1, c), id(r, c), id(r, c + 1)}, {id(r + 1, c), id(r, c), id(r, c + 1)}});
            m.faces.push_back({{id(r + 1, c), id(r, c + 1), id(r + 1, c + 1)}, {id(r + 1, c), id(r, c + 1), id(r + 1, c + 1)}});
        }
    return m;
}

bool valid(const cv::Vec3f& p) { return p[0] != -1.f; }

int validCount(const cv::Mat_<cv::Vec3f>& g)
{
    int n = 0;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c) n += valid(g(r, c));
    return n;
}

// Median 3-D distance between horizontally adjacent valid grid points: voxels per cell.
double voxelsPerCell(const cv::Mat_<cv::Vec3f>& g)
{
    std::vector<double> d;
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c + 1 < g.cols; ++c)
            if (valid(g(r, c)) && valid(g(r, c + 1))) d.push_back(cv::norm(g(r, c + 1) - g(r, c)));
    REQUIRE(!d.empty());
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    return d[d.size() / 2];
}
}

TEST_CASE("UvGridRasterizer covers every boundary row and column of a grid aligned mesh")
{
    auto m = plane(31, 41);
    UvGridRasterizer r;
    r.setSrcScale({0.25f, 0.25f});
    r.setMesh(m.vertices, m.uvs, m.faces);
    r.determineGridDimensions(1.f);
    std::unique_ptr<QuadSurface> s(r.createQuadSurface(1.f));
    const auto& g = *s->rawPointsPtr();
    REQUIRE(g.rows == 31);
    REQUIRE(g.cols == 41);
    CHECK(validCount(g) == 31 * 41);  // as built: 1200, row 0 and column 40 (71 cells) invalid
    CHECK(valid(g(30, 40)));
    CHECK(valid(g(0, 40)));
    CHECK(valid(g(30, 0)));
}


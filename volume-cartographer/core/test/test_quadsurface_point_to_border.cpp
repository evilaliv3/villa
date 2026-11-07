// Regression tests for the QuadSurface audit findings Q01, Q02, Q03, Q05, Q06 (audit reader
// quadsurface, 2026-11-07). Style of core/test/test_quadsurface_pointto.cpp. Each case fails on
// upstream/main 7e6b82164 and passes with the matching patch in patches/.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include <vc_reader/tifxyz.hpp>

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float spacing, float z)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(spacing * c, spacing * r, z);
    return m;
}

std::filesystem::path scratchDir(const char* name)
{
    const char* base = std::getenv("TMPDIR");
    auto dir = std::filesystem::path(base ? base : "/tmp") / "test_quadsurface_audit" / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

} // namespace

TEST_CASE("Q01: pointTo reaches a target in the first and last half cell of the grid")
{
    // 20 voxel cells (scale 0.05): the target sits 4 voxels inside the sheet's edge.
    auto pts = makePlanarGrid(64, 64, 20.f, 500.f);
    QuadSurface qs(pts, cv::Vec2f(0.05f, 0.05f));
    for (float gx : {0.f, 0.2f, 62.8f, 63.f}) {  // the grid has columns 0..63
        cv::Vec3f ptr(0, 0, 0);
        const cv::Vec3f target(20.f * gx, 20.f * 30.f, 500.f);
        const float d = qs.pointTo(ptr, target, /*th=*/0.05f, /*max_iters=*/1000);
        CHECK(d >= 0.0f);
        CHECK(d < 0.1f);
        CHECK(cv::norm(qs.coord(ptr) - target) < 0.1);
    }
    // An interior target still converges.
    cv::Vec3f ptr(0, 0, 0);
    CHECK(qs.pointTo(ptr, {20.f * 10.3f, 20.f * 12.7f, 500.f}, 0.05f, 1000) < 0.1f);
}


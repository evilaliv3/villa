// Regression test for the GridStore audit finding G01 (audit reader quadsurface, 2026-11-07).
// Style of core/test/test_gridstore.cpp. Fails on upstream/main 7e6b82164, passes with
// patches/G01-gridstore-index-crossed-buckets.patch.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <vector>

using vc::core::util::GridStore;

namespace {

bool returned(const GridStore& store, const cv::Rect& query, const std::vector<cv::Point>& path)
{
    for (const auto& p : store.get(query))
        if (p && *p == path) return true;
    return false;
}

} // namespace

TEST_CASE("G01: a query over a bucket a segment crosses returns the path even without a vertex there")
{
    GridStore store(cv::Rect(0, 0, 256, 256), 64);
    // Vertices in buckets (0, 1) and (1, 0); the 14 px segment passes through (65, 65) in bucket (1, 1).
    const std::vector<cv::Point> corner{{60, 70}, {70, 60}};
    store.add(corner);
    CHECK(returned(store, cv::Rect(64, 64, 64, 64), corner));
    // The buckets of the vertices still return it, and an unrelated bucket does not.
    CHECK(returned(store, cv::Rect(0, 64, 64, 64), corner));
    CHECK(returned(store, cv::Rect(64, 0, 64, 64), corner));
    CHECK_FALSE(returned(store, cv::Rect(192, 192, 64, 64), corner));
}

TEST_CASE("G01: a segment crossing a whole bucket column is found from that bucket alone")
{
    GridStore store(cv::Rect(0, 0, 256, 256), 64);
    const std::vector<cv::Point> across{{130, 100}, {5, 100}};  // right to left, 125 px
    store.add(across);
    // Bucket (1, 1) holds no vertex; the query rect ends inside it so the next bucket is not consulted.
    CHECK(returned(store, cv::Rect(64, 64, 1, 1), across));
}

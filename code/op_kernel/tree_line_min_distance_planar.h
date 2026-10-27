#pragma once
#include "tree_line_min_distance_planar_prepare.h"

namespace TreeDistanceSpatial {

#include "tree_line_min_distance_planar_candidates.h"

template <typename T>
__aicore__ inline bool PlanarRun(const GlobalTensor<T>& points,
    const GlobalTensor<T>& distances, const GlobalTensor<int32_t>& indices,
    const LocalTensor<float>& coordinates, const TreeLineMinDistanceTilingData& t,
    uint32_t batch, uint32_t begin, uint32_t owned)
{
    uint32_t side = 2;
    while ((side + 1) * (side + 1) <= t.m) ++side;
    float low[3] = {kGuardDist, kGuardDist, kGuardDist};
    float high[3] = {-kGuardDist, -kGuardDist, -kGuardDist};
    float width[3] = {0.0f, 0.0f, 0.0f};
    LocalTensor<float> inverse(TPosition::VECCALC, 65536, 1024);
    PrepareGeometry(coordinates, inverse, t.m, low, high, width);
    uint32_t axes[3] = {0, 1, 2};
    float cost[3];
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const float extent = high[axis] - low[axis];
        cost[axis] = extent > 0.0f && extent < 3.0e30f ? width[axis] / extent : kGuardDist;
    }
    for (uint32_t a = 0; a < 2; ++a)
        for (uint32_t b = a + 1; b < 3; ++b)
            if (cost[axes[b]] < cost[axes[a]]) { const uint32_t swap = axes[a]; axes[a] = axes[b]; axes[b] = swap; }
    if (!(cost[axes[1]] * (float)(int32_t)side < 0.5f)) return false;
    float origin[2], scale[2], radius[2];
    for (uint32_t k = 0; k < 2; ++k) {
        const uint32_t axis = axes[k];
        origin[k] = low[axis]; scale[k] = (float)(int32_t)(side - 1) / (high[axis] - low[axis]);
        const float magnitude = high[axis] > -low[axis] ? high[axis] : -low[axis];
        radius[k] = width[axis] * 0.5f + 0.0001f * (1.0f + magnitude);
        if (!(scale[k] > 0.0f && scale[k] < 3.0e30f)) return false;
    }
    LocalTensor<int32_t> heads(TPosition::VECCALC, 49152, 1024);
    LocalTensor<int32_t> next(TPosition::VECCALC, 53248, 1024);
    LocalTensor<int32_t> sizes(TPosition::VECCALC, 57344, 1024);
    uint32_t depth = 1;
    const auto cells = PreparePlanarCells(coordinates, heads, sizes, axes, origin, scale, side, t.m);
    for (uint32_t j = 0; j < t.m; ++j) {
        const uint32_t c = (uint32_t)cells.GetValue(j);
        const uint32_t size = (uint32_t)sizes.GetValue(c) + 1;
        if (size > 4) return false;
        sizes.SetValue(c, (int32_t)size); depth = size > depth ? size : depth;
        next.SetValue(j, heads.GetValue(c)); heads.SetValue(c, (int32_t)j);
    }
    LocalTensor<float> directions(TPosition::VECCALC, sizeof(T) == 4 ? 24576 : 0, 3072);
    Hop<HardEvent::S_V>();
    for (uint32_t axis = 0; axis < 3; ++axis)
        Copy<float>(directions[axis * 1024], PreparedDirection(axis), (uint64_t)64,
                    (uint8_t)(AlignLanes(t.m) / 64), {1, 1, 8, 8});
    Hop<HardEvent::V_S>();
    LocalTensor<T> raw(TPosition::VECCALC, 69632, 1536);
    LocalTensor<float> converted(TPosition::VECCALC, 75776, 1536);
    LocalTensor<float> dist(TPosition::VECCALC, 81920, 512);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 83968, 512);
    LocalTensor<T> narrow(TPosition::VECCALC, 86016, 512);
    bool compact = true;
    for (uint32_t done = 0; done < owned; done += 512) {
        const uint32_t count = owned - done < 512 ? owned - done : 512;
        if (done != 0) Hop<HardEvent::MTE3_S>();
        DataCopyPad(raw, points[((uint64_t)batch * t.n + begin + done) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        auto source = raw.template ReinterpretCast<float>();
        Hop<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) == 2) { Cast(converted, raw, RoundMode::CAST_NONE, count * 3); source = converted; }
        Hop<HardEvent::S_V>();
        if constexpr (sizeof(T) == 2) {
            PlanarCandidates<true>(source, coordinates, directions, heads, next, inverse, axes[0], axes[1],
                             origin, scale, radius, side, depth, count, dist, idx);
        } else {
            if (done == 0) {
                compact = PlanarCandidates<false>(source, coordinates, directions, heads, next, inverse, axes[0], axes[1],
                             origin, scale, radius, side, depth, count < 16 ? count : 16, dist, idx);
                Hop<HardEvent::S_V>();
            }
            if (compact) {
                compact = PlanarCandidates<false>(source, coordinates, directions, heads, next, inverse, axes[0], axes[1],
                             origin, scale, radius, side, depth, count, dist, idx);
            }
            if (!compact) {
                Hop<HardEvent::S_V>();
                PlanarCandidates<true>(source, coordinates, directions, heads, next, inverse, axes[0], axes[1],
                             origin, scale, radius, side, depth, count, dist, idx);
            }
        }
        Hop<HardEvent::S_V>();
        Sqrt(dist, dist, count);
        const uint64_t output = (uint64_t)batch * t.n + begin + done;
        if constexpr (sizeof(T) == 2) {
            Cast(narrow, dist, RoundMode::CAST_RINT, count); Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[output], narrow, {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        } else {
            Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[output], dist.template ReinterpretCast<T>(), {1, count * 4, 0, 0, 0});
        }
        DataCopyPad(indices[output], idx, {1, count * 4, 0, 0, 0});
    }
    return true;
}
}

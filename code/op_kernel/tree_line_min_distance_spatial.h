#pragma once

namespace TreeDistanceSpatial {

__aicore__ inline bool PreferDenseSample(const LocalTensor<float>& coordinates,
                                        uint32_t segments)
{
    const uint32_t samples = segments < 16 ? segments : 16;
    if (samples < 2) return false;
    float low[3] = {kGuardDist, kGuardDist, kGuardDist};
    float high[3] = {-kGuardDist, -kGuardDist, -kGuardDist};
    float width[3] = {0.0f, 0.0f, 0.0f};
    for (uint32_t k = 0; k < samples; ++k) {
        const uint32_t offset = 6 * (k * (segments - 1) / (samples - 1));
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const float a = coordinates.GetValue(offset + axis);
            const float c = coordinates.GetValue(offset + axis + 3);
            const float middle = a * 0.5f + c * 0.5f;
            low[axis] = middle < low[axis] ? middle : low[axis];
            high[axis] = middle > high[axis] ? middle : high[axis];
            width[axis] += a < c ? c - a : a - c;
        }
    }
    uint32_t axis = 0;
    if (high[1] - low[1] > high[axis] - low[axis]) axis = 1;
    if (high[2] - low[2] > high[axis] - low[axis]) axis = 2;
    const float extent = high[axis] - low[axis];
    if (!(extent > 0.0f && extent < 3.0e30f)) return true;
    return width[axis] * (float)(int32_t)(segments - 1) >
           32.0f * (float)(int32_t)samples * extent;
}

__aicore__ inline uint32_t Cell(float x, float low, float scale, uint32_t cells)
{
    const float position = (x - low) * scale;
    if (!(position > 0.0f)) return 0;
    if (position >= (float)(int32_t)(cells - 1)) return cells - 1;
    return (uint32_t)(int32_t)position;
}

// Endpoint bounds certify all candidates omitted by these bins.
__aicore__ inline uint32_t RoundedSegmentCell(float x, float low, float scale, uint32_t cells)
{
    const float position = (x - low) * scale + 0.5f;
    if (!(position > 0.0f)) return 0;
    if (position >= (float)(int32_t)(cells - 1)) return cells - 1;
    return (uint32_t)(int32_t)position;
}

__aicore__ inline void Visit(uint32_t cell, const LocalTensor<int32_t>& heads,
    const LocalTensor<int32_t>& next, const LocalTensor<float>& coordinates,
    const LocalTensor<float>& inverse, float px, float py, float pz,
    float& best, int32_t& winner)
{
    for (int32_t j = heads.GetValue(cell); j >= 0; j = next.GetValue(j)) {
        const uint32_t offset = 6u * (uint32_t)j;
        const float ax = coordinates.GetValue(offset);
        const float ay = coordinates.GetValue(offset + 1);
        const float az = coordinates.GetValue(offset + 2);
        const float ux = coordinates.GetValue(offset + 3);
        const float uy = coordinates.GetValue(offset + 4);
        const float uz = coordinates.GetValue(offset + 5);
        float dx = px - ax, dy = py - ay, dz = pz - az;
        float ratio = (dx * ux + dy * uy + dz * uz) * inverse.GetValue(j);
        ratio = ratio > 0.0f ? ratio : 0.0f;
        ratio = ratio < 1.0f ? ratio : 1.0f;
        dx -= ratio * ux;
        dy -= ratio * uy;
        dz -= ratio * uz;
        const float value = dx * dx + dy * dy + dz * dz;
        if (value < best || (value == best && j < winner)) {
            best = value;
            winner = j;
        }
    }
}

#include "tree_line_min_distance_spatial_prepare.h"
#include "tree_line_min_distance_spatial_vector.h"
#include "tree_line_min_distance_spatial_bounds.h"

}
#include "tree_line_min_distance_planar.h"
namespace TreeDistanceSpatial {

template <typename T, bool VECTOR = false>
__aicore__ inline bool Run(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                          const TreeLineMinDistanceTilingData& t)
{
    if (t.b * t.chunks > GetBlockNum()) return false;
    const uint32_t task = GetBlockIdx();
    const uint32_t batch = task / t.chunks;
    const uint32_t begin = (task % t.chunks) * t.chunkPoints;
    if (begin >= t.n) return true;
    const uint32_t owned = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T*)p);
    segments.SetGlobalBuffer((__gm__ T*)s);
    distances.SetGlobalBuffer((__gm__ T*)d);
    indices.SetGlobalBuffer((__gm__ int32_t*)i);
    LocalTensor<T> raw(TPosition::VECCALC, 0, 6144);
    LocalTensor<float> coordinates(TPosition::VECCALC, 24576, 6144);
    LocalTensor<int32_t> heads(TPosition::VECCALC, 49152, 1024);
    LocalTensor<int32_t> next(TPosition::VECCALC, 53248, 1024);
    LocalTensor<float> prefixHigh(TPosition::VECCALC, 57344, 1024);
    LocalTensor<float> suffixLow(TPosition::VECCALC, 61440, 1024);
    LocalTensor<float> inverse(TPosition::VECCALC, 65536, 1024);
    LocalTensor<T> pointRaw(TPosition::VECCALC, 69632, 384);
    LocalTensor<float> pointFloat(TPosition::VECCALC, 71168, 384);
    LocalTensor<float> dist(TPosition::VECCALC, 72704, 128);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 73216, 128);
    LocalTensor<T> narrow(TPosition::VECCALC, 73728, 128);
    LocalTensor<float> nearestCell(TPosition::VECCALC, 74240, 1024);
    DataCopyPad(raw, segments[(uint64_t)batch * t.m * 6],
                {1, (uint32_t)(t.m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    if constexpr (sizeof(T) == 2) {
        Hop<HardEvent::MTE2_V>();
        Cast(coordinates, raw, RoundMode::CAST_NONE, t.m * 6);
        Hop<HardEvent::V_S>();
    } else {
        coordinates = raw.template ReinterpretCast<float>();
        Hop<HardEvent::MTE2_S>();
    }

    if constexpr (VECTOR) {
        if (PreferDenseSample(coordinates, t.m))
            return PlanarRun(points, distances, indices, coordinates, t, batch, begin, owned);
    }

    float low[3] = {kGuardDist, kGuardDist, kGuardDist};
    float high[3] = {-kGuardDist, -kGuardDist, -kGuardDist};
    if constexpr (VECTOR) {
        float widths[3];
        PrepareGeometry(coordinates, inverse, t.m, low, high, widths);
    } else {
    for (uint32_t j = 0; j < t.m; ++j) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const float middle = coordinates.GetValue(6 * j + axis) * 0.5f +
                                 coordinates.GetValue(6 * j + axis + 3) * 0.5f;
            low[axis] = middle < low[axis] ? middle : low[axis];
            high[axis] = middle > high[axis] ? middle : high[axis];
        }
    }
    }
    uint32_t axis = 0;
    if (high[1] - low[1] > high[axis] - low[axis]) axis = 1;
    if (high[2] - low[2] > high[axis] - low[axis]) axis = 2;
    const float extent = high[axis] - low[axis];
    if (!(extent > 0.0f && extent < 3.0e30f)) return false;
    const uint32_t cells = t.m;
    const float scale = (float)(int32_t)(cells - 1) / extent;
    if (!(scale > 0.0f && scale < 3.0e30f)) return false;
    for (uint32_t c = 0; c < cells; ++c) {
        heads.SetValue(c, -1);
        prefixHigh.SetValue(c, -kGuardDist);
        suffixLow.SetValue(c, kGuardDist);
    }
    float widths = 0.0f;
    uint32_t maxCandidates = 1;
    for (uint32_t j = 0; j < t.m; ++j) {
        const uint32_t offset = 6 * j;
        const float a = coordinates.GetValue(offset + axis);
        const float c = coordinates.GetValue(offset + axis + 3);
        const float left = a < c ? a : c, right = a > c ? a : c;
        const uint32_t cell = VECTOR ? RoundedSegmentCell(a * 0.5f + c * 0.5f, low[axis], scale, cells)
                                     : Cell(a * 0.5f + c * 0.5f, low[axis], scale, cells);
        const int32_t previousHead = heads.GetValue(cell);
        next.SetValue(j, previousHead);
        heads.SetValue(cell, (int32_t)j);
        if constexpr (VECTOR) {
            if (maxCandidates < 4 && previousHead >= 0) {
                uint32_t depth = 1;
                int32_t previous = previousHead;
                while (previous >= 0 && depth < 4) {
                    ++depth;
                    previous = next.GetValue(previous);
                }
                maxCandidates = depth > maxCandidates ? depth : maxCandidates;
            }
        }
        const float oldHigh = prefixHigh.GetValue(cell);
        const float oldLow = suffixLow.GetValue(cell);
        prefixHigh.SetValue(cell, right > oldHigh ? right : oldHigh);
        suffixLow.SetValue(cell, left < oldLow ? left : oldLow);
        widths += right - left;
        if constexpr (VECTOR) {
            coordinates.SetValue(offset + 3, PreparedDirection(0).GetValue(j));
            coordinates.SetValue(offset + 4, PreparedDirection(1).GetValue(j));
            coordinates.SetValue(offset + 5, PreparedDirection(2).GetValue(j));
        } else {
        const float ux = coordinates.GetValue(offset + 3) - coordinates.GetValue(offset);
        const float uy = coordinates.GetValue(offset + 4) - coordinates.GetValue(offset + 1);
        const float uz = coordinates.GetValue(offset + 5) - coordinates.GetValue(offset + 2);
        const float length = ux * ux + uy * uy + uz * uz;
        inverse.SetValue(j, length > 0.0f ? 1.0f / length : 0.0f);
        coordinates.SetValue(offset + 3, ux);
        coordinates.SetValue(offset + 4, uy);
        coordinates.SetValue(offset + 5, uz);
        }
    }
    if (widths * scale > 8.0f * (float)(int32_t)t.m) return false;
    bool ordered = false;
    if constexpr (VECTOR) {
        if (maxCandidates == 1) ordered = OrderedBounds(prefixHigh, suffixLow, cells);
    }
    if (!ordered) {
    float highest = -kGuardDist, lowest = kGuardDist;
    for (uint32_t c = 0; c < cells; ++c) {
        const float right = prefixHigh.GetValue(c);
        highest = right > highest ? right : highest;
        prefixHigh.SetValue(c, highest);
        const uint32_t reverse = cells - 1 - c;
        const float left = suffixLow.GetValue(reverse);
        lowest = left < lowest ? left : lowest;
        suffixLow.SetValue(reverse, lowest);
    }
    }

    if constexpr (VECTOR) {
        if (maxCandidates == 1) {
            Hop<HardEvent::S_V>();
            CreateVecIndex(nearestCell, 0.0f, t.m);
            Hop<HardEvent::V_S>();
        } else {
        int32_t nearest = -1;
        for (uint32_t c = 0; c < cells; ++c) {
            if (heads.GetValue(c) >= 0) nearest = (int32_t)c;
            nearestCell.SetValue(c, (float)nearest);
        }
        nearest = (int32_t)cells;
        for (int32_t c = (int32_t)cells - 1; c >= 0; --c) {
            if (heads.GetValue(c) >= 0) nearest = c;
            const int32_t previous = (int32_t)nearestCell.GetValue(c);
            if (previous < 0 || (nearest < (int32_t)cells && nearest - c < c - previous)) {
                nearestCell.SetValue(c, (float)nearest);
            }
        }
        }
    }

    for (uint32_t done = 0; done < owned; done += 128) {
        const uint32_t count = owned - done < 128 ? owned - done : 128;
        if (done != 0) Hop<HardEvent::MTE3_S>();
        DataCopyPad(pointRaw, points[((uint64_t)batch * t.n + begin + done) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        LocalTensor<float> pointSource = pointRaw.template ReinterpretCast<float>();
        if constexpr (sizeof(T) == 2) {
            Hop<HardEvent::MTE2_V>();
            Cast(pointFloat, pointRaw, RoundMode::CAST_NONE, count * 3);
            Hop<HardEvent::V_S>();
            pointSource = pointFloat;
        } else {
            Hop<HardEvent::MTE2_S>();
        }
        if constexpr (VECTOR) {
            Hop<HardEvent::S_V>();
            Hop<HardEvent::MTE2_V>();
            VectorCandidates(pointSource, coordinates, heads, next, inverse,
                prefixHigh, suffixLow, nearestCell, low[axis], scale, axis,
                cells, count, maxCandidates, dist, idx);
        } else {
        for (uint32_t q = 0; q < count; ++q) {
            const float px = pointSource.GetValue(3 * q);
            const float py = pointSource.GetValue(3 * q + 1);
            const float pz = pointSource.GetValue(3 * q + 2);
            const float coordinate = axis == 0 ? px : (axis == 1 ? py : pz);
            const uint32_t center = Cell(coordinate, low[axis], scale, cells);
            float best = kGuardDist;
            int32_t winner = 0;
            Visit(center, heads, next, coordinates, inverse, px, py, pz, best, winner);
            int32_t left = (int32_t)center - 1;
            uint32_t right = center + 1;
            while (left >= 0 || right < cells) {
                const float limit = best + 0.0001f * (1.0f + best);
                if (left >= 0) {
                    const float gap = coordinate - prefixHigh.GetValue(left);
                    if (gap > 0.0f && gap * gap > limit) left = -1;
                    else {
                        Visit((uint32_t)left, heads, next, coordinates, inverse, px, py, pz, best, winner);
                        --left;
                    }
                }
                if (right < cells) {
                    const float gap = suffixLow.GetValue(right) - coordinate;
                    if (gap > 0.0f && gap * gap > limit) right = cells;
                    else {
                        Visit(right, heads, next, coordinates, inverse, px, py, pz, best, winner);
                        ++right;
                    }
                }
            }
            dist.SetValue(q, best);
            idx.SetValue(q, winner);
        }
        }
        Hop<HardEvent::S_V>();
        Sqrt(dist, dist, count);
        const uint64_t output = (uint64_t)batch * t.n + begin + done;
        if constexpr (sizeof(T) == 2) {
            Cast(narrow, dist, RoundMode::CAST_RINT, count);
            Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[output], narrow, {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        } else {
            Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[output], dist.template ReinterpretCast<T>(),
                        {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        }
        DataCopyPad(indices[output], idx, {1, count * 4, 0, 0, 0});
    }
    return true;
}
}

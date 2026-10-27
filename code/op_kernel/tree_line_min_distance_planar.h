#pragma once

namespace TreeDistanceSpatial {

__aicore__ inline void PlanarCandidates(const LocalTensor<float>& source,
    const LocalTensor<float>& coordinates, const LocalTensor<int32_t>& heads,
    const LocalTensor<int32_t>& next, const LocalTensor<float>& inverse,
    uint32_t axis0, uint32_t axis1, const float* low, const float* scale,
    const float* radius, uint32_t side, uint32_t depth, uint32_t realCount,
    const LocalTensor<float>& dist, const LocalTensor<int32_t>& idx)
{
    LocalTensor<float> px(TPosition::VECCALC, 90112, 512);
    LocalTensor<float> py(TPosition::VECCALC, 92160, 512);
    LocalTensor<float> pz(TPosition::VECCALC, 94208, 512);
    LocalTensor<float> cx(TPosition::VECCALC, 96256, 512);
    LocalTensor<float> cy(TPosition::VECCALC, 98304, 512);
    LocalTensor<int32_t> offsets(TPosition::VECCALC, 100352, 512);
    LocalTensor<int32_t> current(TPosition::VECCALC, 102400, 512);
    LocalTensor<float> currentFloat(TPosition::VECCALC, 104448, 512);
    LocalTensor<int32_t> following(TPosition::VECCALC, 106496, 512);
    LocalTensor<float> ax(TPosition::VECCALC, 108544, 512);
    LocalTensor<float> ay(TPosition::VECCALC, 110592, 512);
    LocalTensor<float> az(TPosition::VECCALC, 112640, 512);
    LocalTensor<float> ux(TPosition::VECCALC, 114688, 512);
    LocalTensor<float> uy(TPosition::VECCALC, 116736, 512);
    LocalTensor<float> uz(TPosition::VECCALC, 118784, 512);
    LocalTensor<float> dx(TPosition::VECCALC, 120832, 512);
    LocalTensor<float> dy(TPosition::VECCALC, 122880, 512);
    LocalTensor<float> dz(TPosition::VECCALC, 124928, 512);
    LocalTensor<float> ratio(TPosition::VECCALC, 126976, 512);
    LocalTensor<float> temp(TPosition::VECCALC, 129024, 512);
    LocalTensor<float> winner(TPosition::VECCALC, 131072, 512);
    LocalTensor<uint8_t> valid(TPosition::VECCALC, 133120, 256);
    LocalTensor<uint8_t> keep(TPosition::VECCALC, 135168, 256);
    LocalTensor<float> cellX(TPosition::VECCALC, 137216, 512);
    LocalTensor<float> cellY(TPosition::VECCALC, 139264, 512);
    LocalTensor<float> bound(TPosition::VECCALC, 141312, 512);
    LocalTensor<float> gap(TPosition::VECCALC, 143360, 512);
    LocalTensor<float> limit(TPosition::VECCALC, 145408, 512);
    LocalTensor<float> certified(TPosition::VECCALC, 147456, 512);
    const uint32_t count = AlignLanes(realCount);
    Duplicate(px, 0.0f, count); Duplicate(py, 0.0f, count); Duplicate(pz, 0.0f, count);
    CreateVecIndex(temp, 0.0f, count);
    Muls(temp, temp, 12.0f, count);
    Cast(offsets, temp, RoundMode::CAST_RINT, count);
    const auto bytes = offsets.ReinterpretCast<uint32_t>();
    Gather(px, source, bytes, 0u, realCount);
    Gather(py, source, bytes, 4u, realCount);
    Gather(pz, source, bytes, 8u, realCount);
    LocalTensor<float> coordinate0 = axis0 == 0 ? px : (axis0 == 1 ? py : pz);
    LocalTensor<float> coordinate1 = axis1 == 0 ? px : (axis1 == 1 ? py : pz);
    Adds(cx, coordinate0, -low[0], count); Muls(cx, cx, scale[0], count);
    Adds(cy, coordinate1, -low[1], count); Muls(cy, cy, scale[1], count);
    Maxs(cx, cx, 0.0f, count); Mins(cx, cx, (float)(int32_t)(side - 1), count);
    Maxs(cy, cy, 0.0f, count); Mins(cy, cy, (float)(int32_t)(side - 1), count);
    Cast(offsets, cx, RoundMode::CAST_FLOOR, count); Cast(cx, offsets, RoundMode::CAST_NONE, count);
    Cast(offsets, cy, RoundMode::CAST_FLOOR, count); Cast(cy, offsets, RoundMode::CAST_NONE, count);
    Duplicate(dist, kGuardDist, count); Duplicate(winner, 0.0f, count);
    for (int32_t x = -1; x <= 1; ++x) {
        Adds(cellX, cx, (float)x, count);
        Maxs(cellX, cellX, 0.0f, count); Mins(cellX, cellX, (float)(int32_t)(side - 1), count);
        Muls(cellX, cellX, (float)(int32_t)side, count);
        for (int32_t y = -1; y <= 1; ++y) {
            Adds(cellY, cy, (float)y, count);
            Maxs(cellY, cellY, 0.0f, count); Mins(cellY, cellY, (float)(int32_t)(side - 1), count);
            Add(temp, cellX, cellY, count); Muls(temp, temp, 4.0f, count);
            Cast(offsets, temp, RoundMode::CAST_RINT, count);
            Gather(current, heads, bytes, 0u, count);
            for (uint32_t step = 0; step < depth; ++step) {
                Cast(currentFloat, current, RoundMode::CAST_NONE, count);
                CompareScalar(valid, currentFloat, 0.0f, CMPMODE::GE, count);
                Maxs(temp, currentFloat, 0.0f, count); Muls(temp, temp, 24.0f, count);
                Cast(offsets, temp, RoundMode::CAST_RINT, count);
                Gather(ax, coordinates, bytes, 0u, count); Gather(ay, coordinates, bytes, 4u, count);
                Gather(az, coordinates, bytes, 8u, count); Gather(ux, coordinates, bytes, 12u, count);
                Gather(uy, coordinates, bytes, 16u, count); Gather(uz, coordinates, bytes, 20u, count);
                Maxs(temp, currentFloat, 0.0f, count); Muls(temp, temp, 4.0f, count);
                Cast(offsets, temp, RoundMode::CAST_RINT, count);
                Gather(following, next, bytes, 0u, count); Gather(temp, inverse, bytes, 0u, count);
                Sub(dx, px, ax, count); Sub(dy, py, ay, count); Sub(dz, pz, az, count);
                Mul(ratio, dx, ux, count); MulAddDst(ratio, dy, uy, count); MulAddDst(ratio, dz, uz, count);
                Mul(ratio, ratio, temp, count); Maxs(ratio, ratio, 0.0f, count); Mins(ratio, ratio, 1.0f, count);
                Mul(temp, ratio, ux, count); Sub(dx, dx, temp, count);
                Mul(temp, ratio, uy, count); Sub(dy, dy, temp, count);
                Mul(temp, ratio, uz, count); Sub(dz, dz, temp, count);
                Mul(temp, dx, dx, count); MulAddDst(temp, dy, dy, count); MulAddDst(temp, dz, dz, count);
                Select(temp, valid, temp, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
                Compare(keep, temp, dist, CMPMODE::LT, count);
                Select(winner, keep, currentFloat, winner, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                Min(dist, dist, temp, count);
                Compare(keep, temp, dist, CMPMODE::EQ, count);
                Min(temp, winner, currentFloat, count);
                Select(winner, keep, temp, winner, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                Select(current.ReinterpretCast<float>(), valid, following.ReinterpretCast<float>(),
                       current.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            }
        }
    }
    Duplicate(bound, kGuardDist, count);
    for (uint32_t axis = 0; axis < 2; ++axis) {
        const auto center = axis == 0 ? cx : cy;
        const auto coordinate = axis == 0 ? coordinate0 : coordinate1;
        Adds(gap, center, -1.0f, count); Muls(gap, gap, 1.0f / scale[axis], count);
        Adds(gap, gap, low[axis] + radius[axis], count); Sub(gap, coordinate, gap, count);
        Maxs(gap, gap, 0.0f, count); Mins(gap, gap, 1.0e18f, count); Mul(gap, gap, gap, count);
        CompareScalar(keep, center, 1.0f, CMPMODE::GT, count);
        Select(gap, keep, gap, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        Min(bound, bound, gap, count);
        Adds(gap, center, 2.0f, count); Muls(gap, gap, 1.0f / scale[axis], count);
        Adds(gap, gap, low[axis] - radius[axis], count); Sub(gap, gap, coordinate, count);
        Maxs(gap, gap, 0.0f, count); Mins(gap, gap, 1.0e18f, count); Mul(gap, gap, gap, count);
        CompareScalar(keep, center, (float)(int32_t)(side - 2), CMPMODE::LT, count);
        Select(gap, keep, gap, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        Min(bound, bound, gap, count);
    }
    Muls(limit, dist, 1.0001f, count); Adds(limit, limit, 0.0001f, count);
    Compare(keep, bound, limit, CMPMODE::GT, count);
    Duplicate(certified, 1.0f, count);
    Select(certified, keep, certified, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    Cast(idx, winner, RoundMode::CAST_RINT, count);
    Hop<HardEvent::V_S>();
    for (uint32_t q = 0; q < realCount; ++q) {
        if (certified.GetValue(q) != 0.0f) continue;
        float best = kGuardDist; int32_t index = 0;
        const float x = px.GetValue(q), y = py.GetValue(q), z = pz.GetValue(q);
        for (uint32_t cell = 0; cell < side * side; ++cell)
            Visit(cell, heads, next, coordinates, inverse, x, y, z, best, index);
        dist.SetValue(q, best); idx.SetValue(q, index);
    }
}

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
    for (uint32_t c = 0; c < side * side; ++c) { heads.SetValue(c, -1); sizes.SetValue(c, 0); }
    for (uint32_t j = 0; j < t.m; ++j) {
        uint32_t cell[2];
        for (uint32_t k = 0; k < 2; ++k) {
            const uint32_t offset = 6 * j + axes[k];
            cell[k] = Cell(coordinates.GetValue(offset) * 0.5f + coordinates.GetValue(offset + 3) * 0.5f,
                           origin[k], scale[k], side);
        }
        const uint32_t c = cell[0] * side + cell[1];
        const uint32_t size = (uint32_t)sizes.GetValue(c) + 1;
        if (size > 4) return false;
        sizes.SetValue(c, (int32_t)size); depth = size > depth ? size : depth;
        next.SetValue(j, heads.GetValue(c)); heads.SetValue(c, (int32_t)j);
    }
    for (uint32_t j = 0; j < t.m; ++j) {
        const uint32_t o = 6 * j;
        coordinates.SetValue(o + 3, PreparedDirection(0).GetValue(j));
        coordinates.SetValue(o + 4, PreparedDirection(1).GetValue(j));
        coordinates.SetValue(o + 5, PreparedDirection(2).GetValue(j));
    }
    LocalTensor<T> raw(TPosition::VECCALC, 69632, 1536);
    LocalTensor<float> converted(TPosition::VECCALC, 75776, 1536);
    LocalTensor<float> dist(TPosition::VECCALC, 81920, 512);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 83968, 512);
    LocalTensor<T> narrow(TPosition::VECCALC, 86016, 512);
    for (uint32_t done = 0; done < owned; done += 512) {
        const uint32_t count = owned - done < 512 ? owned - done : 512;
        if (done != 0) Hop<HardEvent::MTE3_S>();
        DataCopyPad(raw, points[((uint64_t)batch * t.n + begin + done) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        auto source = raw.template ReinterpretCast<float>();
        Hop<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) == 2) { Cast(converted, raw, RoundMode::CAST_NONE, count * 3); source = converted; }
        Hop<HardEvent::S_V>();
        PlanarCandidates(source, coordinates, heads, next, inverse, axes[0], axes[1],
                         origin, scale, radius, side, depth, count, dist, idx);
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

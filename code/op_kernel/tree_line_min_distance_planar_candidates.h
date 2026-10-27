#pragma once

__aicore__ inline void VisitPlanarSoA(uint32_t cell, const LocalTensor<int32_t>& heads,
    const LocalTensor<int32_t>& next, const LocalTensor<float>& coordinates,
    const LocalTensor<float>& directions, const LocalTensor<float>& inverse,
    float px, float py, float pz, float& best, int32_t& winner)
{
    for (int32_t j = heads.GetValue(cell); j >= 0; j = next.GetValue(j)) {
        const uint32_t offset = 6u * (uint32_t)j;
        const float ax = coordinates.GetValue(offset), ay = coordinates.GetValue(offset + 1);
        const float az = coordinates.GetValue(offset + 2);
        const float ux = directions.GetValue(j), uy = directions.GetValue(1024 + j);
        const float uz = directions.GetValue(2048 + j);
        float dx = px - ax, dy = py - ay, dz = pz - az;
        float ratio = (dx * ux + dy * uy + dz * uz) * inverse.GetValue(j);
        ratio = ratio > 0.0f ? ratio : 0.0f;
        ratio = ratio < 1.0f ? ratio : 1.0f;
        dx -= ratio * ux; dy -= ratio * uy; dz -= ratio * uz;
        const float value = dx * dx + dy * dy + dz * dz;
        if (value < best || (value == best && j < winner)) { best = value; winner = j; }
    }
}

template <bool WIDE = false>
__aicore__ inline bool PlanarCandidates(const LocalTensor<float>& source,
    const LocalTensor<float>& coordinates, const LocalTensor<float>& directions,
    const LocalTensor<int32_t>& heads,
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
    for (int32_t x = WIDE ? -1 : 0; x <= 1; ++x) {
        Adds(cellX, cx, (float)x, count);
        Maxs(cellX, cellX, 0.0f, count); Mins(cellX, cellX, (float)(int32_t)(side - 1), count);
        Muls(cellX, cellX, (float)(int32_t)side, count);
        for (int32_t y = WIDE ? -1 : 0; y <= 1; ++y) {
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
                Gather(az, coordinates, bytes, 8u, count);
                Maxs(temp, currentFloat, 0.0f, count); Muls(temp, temp, 4.0f, count);
                Cast(offsets, temp, RoundMode::CAST_RINT, count);
                Gather(ux, directions, bytes, 0u, count);
                Gather(uy, directions[1024], bytes, 0u, count);
                Gather(uz, directions[2048], bytes, 0u, count);
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
        Adds(gap, center, WIDE ? -1.5f : -0.5f, count); Muls(gap, gap, 1.0f / scale[axis], count);
        Adds(gap, gap, low[axis] + radius[axis], count); Sub(gap, coordinate, gap, count);
        Maxs(gap, gap, 0.0f, count); Mins(gap, gap, 1.0e18f, count); Mul(gap, gap, gap, count);
        CompareScalar(keep, center, WIDE ? 1.0f : 0.0f, CMPMODE::GT, count);
        Select(gap, keep, gap, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        Min(bound, bound, gap, count);
        Adds(gap, center, 1.5f, count); Muls(gap, gap, 1.0f / scale[axis], count);
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
    if constexpr (!WIDE) {
        for (uint32_t q = 0; q < realCount; ++q) {
            if (certified.GetValue(q) == 0.0f) {
                return false;
            }
        }
        return true;
    } else {
    for (uint32_t q = 0; q < realCount; ++q) {
        if (certified.GetValue(q) != 0.0f) continue;
        float best = kGuardDist; int32_t index = 0;
        const float x = px.GetValue(q), y = py.GetValue(q), z = pz.GetValue(q);
        for (uint32_t cell = 0; cell < side * side; ++cell)
            VisitPlanarSoA(cell, heads, next, coordinates, directions, inverse, x, y, z, best, index);
        dist.SetValue(q, best); idx.SetValue(q, index);
    }
        return false;
    }
}


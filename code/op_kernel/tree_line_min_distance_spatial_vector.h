#pragma once

__aicore__ inline void VectorCandidates(const LocalTensor<float>& pointSource,
    const LocalTensor<float>& coordinates, const LocalTensor<int32_t>& heads,
    const LocalTensor<int32_t>& next, const LocalTensor<float>& inverse,
    const LocalTensor<float>& prefixHigh, const LocalTensor<float>& suffixLow,
    const LocalTensor<float>& nearestCell, float low, float scale, uint32_t axis,
    uint32_t cells, uint32_t count, uint32_t maxCandidates, const LocalTensor<float>& dist,
    const LocalTensor<int32_t>& idx)
{
    LocalTensor<float> px(TPosition::VECCALC, 80000, 128);
    LocalTensor<float> py(TPosition::VECCALC, 80512, 128);
    LocalTensor<float> pz(TPosition::VECCALC, 81024, 128);
    LocalTensor<float> centerFloat(TPosition::VECCALC, 81536, 128);
    LocalTensor<int32_t> centers(TPosition::VECCALC, 82048, 128);
    LocalTensor<int32_t> offsets(TPosition::VECCALC, 82560, 128);
    LocalTensor<int32_t> current(TPosition::VECCALC, 83072, 128);
    LocalTensor<float> currentFloat(TPosition::VECCALC, 83584, 128);
    LocalTensor<int32_t> following(TPosition::VECCALC, 84096, 128);
    LocalTensor<float> ax(TPosition::VECCALC, 84608, 128);
    LocalTensor<float> ay(TPosition::VECCALC, 85120, 128);
    LocalTensor<float> az(TPosition::VECCALC, 85632, 128);
    LocalTensor<float> ux(TPosition::VECCALC, 86144, 128);
    LocalTensor<float> uy(TPosition::VECCALC, 86656, 128);
    LocalTensor<float> uz(TPosition::VECCALC, 87168, 128);
    LocalTensor<float> dx(TPosition::VECCALC, 87680, 128);
    LocalTensor<float> dy(TPosition::VECCALC, 88192, 128);
    LocalTensor<float> dz(TPosition::VECCALC, 88704, 128);
    LocalTensor<float> ratio(TPosition::VECCALC, 89216, 128);
    LocalTensor<float> temp(TPosition::VECCALC, 89728, 128);
    LocalTensor<float> bestIndex(TPosition::VECCALC, 90240, 128);
    LocalTensor<uint8_t> valid(TPosition::VECCALC, 90752, 256);
    LocalTensor<uint8_t> keep(TPosition::VECCALC, 91008, 256);
    LocalTensor<float> leftOK(TPosition::VECCALC, 91264, 128);
    LocalTensor<float> rightOK(TPosition::VECCALC, 91776, 128);
    LocalTensor<float> leftBound(TPosition::VECCALC, 92288, 128);
    LocalTensor<float> rightBound(TPosition::VECCALC, 92800, 128);
    LocalTensor<float> limit(TPosition::VECCALC, 93312, 128);
    LocalTensor<float> certified(TPosition::VECCALC, 94848, 128);
    const uint32_t realCount = count;
    count = AlignLanes(count);
    Duplicate(px, 0.0f, count);
    Duplicate(py, 0.0f, count);
    Duplicate(pz, 0.0f, count);
    CreateVecIndex(ratio, 0.0f, count);
    Muls(temp, ratio, 12.0f, count);
    Cast(offsets, temp, RoundMode::CAST_RINT, count);
    const LocalTensor<uint32_t> bytes = offsets.ReinterpretCast<uint32_t>();
    Gather(px, pointSource, bytes, 0u, realCount);
    Gather(py, pointSource, bytes, 4u, realCount);
    Gather(pz, pointSource, bytes, 8u, realCount);
    LocalTensor<float> coordinate = axis == 0 ? px : (axis == 1 ? py : pz);
    Adds(centerFloat, coordinate, -low, count);
    Muls(centerFloat, centerFloat, scale, count);
    Maxs(centerFloat, centerFloat, 0.0f, count);
    Mins(centerFloat, centerFloat, (float)(int32_t)(cells - 1), count);
    Cast(centers, centerFloat, RoundMode::CAST_FLOOR, count);
    Cast(centerFloat, centers, RoundMode::CAST_NONE, count);
    Muls(temp, centerFloat, 4.0f, count);
    Cast(offsets, temp, RoundMode::CAST_RINT, count);
    Gather(centerFloat, nearestCell, bytes, 0u, count);
    Cast(centers, centerFloat, RoundMode::CAST_RINT, count);
    Muls(temp, centerFloat, 4.0f, count);
    Cast(offsets, temp, RoundMode::CAST_RINT, count);
    Duplicate(dist, kGuardDist, count);
    Duplicate(bestIndex, 0.0f, count);
    Duplicate(certified, 1.0f, count);
    #pragma unroll
    for (int32_t neighbor = -1; neighbor <= 1; ++neighbor) {
        Adds(temp, centerFloat, (float)neighbor, count);
        Maxs(temp, temp, 0.0f, count);
        Mins(temp, temp, (float)(int32_t)(cells - 1), count);
        Muls(temp, temp, 4.0f, count);
        Cast(offsets, temp, RoundMode::CAST_RINT, count);
        Gather(current, heads, bytes, 0u, count);
        for (uint32_t step = 0; step < maxCandidates; ++step) {
        Cast(currentFloat, current, RoundMode::CAST_NONE, count);
        CompareScalar(valid, currentFloat, 0.0f, CMPMODE::GE, count);
        Maxs(temp, currentFloat, 0.0f, count);
        Muls(temp, temp, 24.0f, count);
        Cast(offsets, temp, RoundMode::CAST_RINT, count);
        Gather(ax, coordinates, bytes, 0u, count);
        Gather(ay, coordinates, bytes, 4u, count);
        Gather(az, coordinates, bytes, 8u, count);
        Gather(ux, coordinates, bytes, 12u, count);
        Gather(uy, coordinates, bytes, 16u, count);
        Gather(uz, coordinates, bytes, 20u, count);
        Maxs(temp, currentFloat, 0.0f, count);
        Muls(temp, temp, 4.0f, count);
        Cast(offsets, temp, RoundMode::CAST_RINT, count);
        Gather(following, next, bytes, 0u, count);
        Gather(temp, inverse, bytes, 0u, count);
        Sub(dx, px, ax, count);
        Sub(dy, py, ay, count);
        Sub(dz, pz, az, count);
        Mul(ratio, dx, ux, count);
        MulAddDst(ratio, dy, uy, count);
        MulAddDst(ratio, dz, uz, count);
        Mul(ratio, ratio, temp, count);
        Maxs(ratio, ratio, 0.0f, count);
        Mins(ratio, ratio, 1.0f, count);
        Mul(temp, ratio, ux, count); Sub(dx, dx, temp, count);
        Mul(temp, ratio, uy, count); Sub(dy, dy, temp, count);
        Mul(temp, ratio, uz, count); Sub(dz, dz, temp, count);
        Mul(temp, dx, dx, count);
        MulAddDst(temp, dy, dy, count);
        MulAddDst(temp, dz, dz, count);
        Select(temp, valid, temp, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        Compare(keep, temp, dist, CMPMODE::LT, count);
        Select(bestIndex, keep, currentFloat, bestIndex, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Min(dist, dist, temp, count);
        Compare(keep, temp, dist, CMPMODE::EQ, count);
        Min(temp, bestIndex, currentFloat, count);
        Select(bestIndex, keep, temp, bestIndex, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Select(current.ReinterpretCast<float>(), valid, following.ReinterpretCast<float>(),
               current.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }
        Cast(currentFloat, current, RoundMode::CAST_NONE, count);
        CompareScalar(keep, currentFloat, 0.0f, CMPMODE::LT, count);
        Select(certified, keep, certified, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }

    Adds(temp, centerFloat, -2.0f, count);
    Maxs(temp, temp, 0.0f, count);
    Muls(temp, temp, 4.0f, count);
    Cast(offsets, temp, RoundMode::CAST_RINT, count);
    Gather(leftBound, prefixHigh, bytes, 0u, count);
    Adds(temp, centerFloat, 2.0f, count);
    Mins(temp, temp, (float)(int32_t)(cells - 1), count);
    Muls(temp, temp, 4.0f, count);
    Cast(offsets, temp, RoundMode::CAST_RINT, count);
    Gather(rightBound, suffixLow, bytes, 0u, count);
    Sub(leftBound, coordinate, leftBound, count);
    Sub(rightBound, rightBound, coordinate, count);
    Maxs(leftBound, leftBound, 0.0f, count);
    Maxs(rightBound, rightBound, 0.0f, count);
    Mins(leftBound, leftBound, 1.0e18f, count);
    Mins(rightBound, rightBound, 1.0e18f, count);
    Mul(leftBound, leftBound, leftBound, count);
    Mul(rightBound, rightBound, rightBound, count);
    Muls(limit, dist, 1.0001f, count);
    Adds(limit, limit, 0.0001f, count);
    Duplicate(temp, 1.0f, count);
    Duplicate(leftOK, 0.0f, count);
    Compare(keep, leftBound, limit, CMPMODE::GT, count);
    Select(leftOK, keep, temp, leftOK, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    CompareScalar(keep, centerFloat, 1.0f, CMPMODE::LE, count);
    Select(leftOK, keep, temp, leftOK, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    Duplicate(rightOK, 0.0f, count);
    Compare(keep, rightBound, limit, CMPMODE::GT, count);
    Select(rightOK, keep, temp, rightOK, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    CompareScalar(keep, centerFloat, (float)(int32_t)(cells - 2), CMPMODE::GE, count);
    Select(rightOK, keep, temp, rightOK, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    Min(leftOK, leftOK, rightOK, count);
    Mul(certified, certified, leftOK, count);
    Cast(idx, bestIndex, RoundMode::CAST_RINT, count);
    Hop<HardEvent::V_S>();
    for (uint32_t q = 0; q < realCount; ++q) {
        if (certified.GetValue(q) != 0.0f) continue;
        const float x = px.GetValue(q), y = py.GetValue(q), z = pz.GetValue(q);
        const float target = axis == 0 ? x : (axis == 1 ? y : z);
        const uint32_t center = (uint32_t)centers.GetValue(q);
        float best = kGuardDist;
        int32_t winner = 0;
        Visit(center, heads, next, coordinates, inverse, x, y, z, best, winner);
        int32_t left = (int32_t)center - 1;
        uint32_t right = center + 1;
        while (left >= 0 || right < cells) {
            const float cutoff = best + 0.0001f * (1.0f + best);
            if (left >= 0) {
                const float gap = target - prefixHigh.GetValue(left);
                if (gap > 0.0f && gap * gap > cutoff) left = -1;
                else { Visit((uint32_t)left, heads, next, coordinates, inverse, x, y, z, best, winner); --left; }
            }
            if (right < cells) {
                const float gap = suffixLow.GetValue(right) - target;
                if (gap > 0.0f && gap * gap > cutoff) right = cells;
                else { Visit(right, heads, next, coordinates, inverse, x, y, z, best, winner); ++right; }
            }
        }
        dist.SetValue(q, best);
        idx.SetValue(q, winner);
    }
}

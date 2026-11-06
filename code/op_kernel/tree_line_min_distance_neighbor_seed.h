#pragma once

    __aicore__ inline LocalTensor<float> NeighborF(uint32_t offset)
    {
        return arena[oDx + offset];
    }

    // Both members of a pair examine the same input sample before choosing work.
    __aicore__ inline bool NeighborPairProbe(const LocalTensor<T>& probe, uint32_t point)
    {
        auto raw = arena[oSegRaw].template ReinterpretCast<T>();
        const float a0 = (float)raw.GetValue(0), c0 = (float)raw.GetValue(3);
        const float a1 = (float)raw.GetValue(6 * (m - 1));
        const float c1 = (float)raw.GetValue(6 * (m - 1) + 3);
        const float first = a0 + 0.5f * (c0 - a0);
        const float span = a1 + 0.5f * (c1 - a1) - first;
        const float extent = span < 0.0f ? -span : span;
        float w0 = c0 - a0, w1 = c1 - a1;
        if (w0 < 0.0f) w0 = -w0;
        if (w1 < 0.0f) w1 = -w1;
        if (!(extent > 0.0f) || (w0 > w1 ? w0 : w1) *
            (float)(int32_t)(m - 1) > 32.0f * extent) return false;
        neighborSign = span > 0.0f ? 1.0f : -1.0f;
        neighborOrigin = first * neighborSign;
        neighborExtent = extent;
        neighborStep = extent / (float)(int32_t)(m - 1);
        if (!(neighborStep > 1.0e-10f && neighborStep < 1.0e10f)) return false;
        const float coordinate = (float)probe.GetValue(point * 3) * neighborSign;
        const float delta = coordinate - neighborOrigin;
        const float padding = 8.0f * neighborStep;
        return delta >= -padding && delta <= neighborExtent + padding;
    }

    // A global envelope bounds every segment outside the three candidates.
    __aicore__ inline bool TryNeighborSeed(uint32_t batch, uint32_t start, uint32_t count)
    {
        static_assert(!NEIGHBOR || oDx + 5760 <= oDy);
        if (m < 128 || m > 256 || count > 64) return false;
        Hop<HardEvent::V_S>();
        const float sign = neighborSign, origin = neighborOrigin;
        const float extent = neighborExtent, step = neighborStep;
        auto pair = NeighborF(1536), reduction = NeighborF(1600);
        float margin, lowDelta, highDelta;
        if constexpr (ANALYTIC_BOUND) {
            auto lower = NeighborF(0), upper = NeighborF(mAligned), anchor = NeighborF(512);
            auto a = NeighborF(1024), c = NeighborF(1280);
            Muls(a, arena[oAx], sign, m);
            Muls(c, arena[oUx], sign, m);
            Add(c, a, c, m);
            Duplicate(lower, kGuardDist, mAligned);
            Duplicate(upper, kGuardDist, mAligned);
            Min(lower, a, c, m); Max(upper, a, c, m);
            CreateVecIndex(anchor, 0.0f, mAligned);
            Muls(anchor, anchor, step, mAligned); Adds(anchor, anchor, origin, mAligned);
            Sub(lower, lower, anchor, m);
            Sub(upper, anchor, upper, m);
            const uint32_t repeats = mAligned / kLanes;
            WholeReduceMin(pair, lower, (int32_t)kLanes, (int32_t)(2 * repeats),
                1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            Hop<HardEvent::V_S>();
            float minDelta = pair.GetValue(0), negMaxDelta = pair.GetValue(repeats);
            for (uint32_t k = 1; k < repeats; ++k) {
                const float low = pair.GetValue(k), high = pair.GetValue(repeats + k);
                if (low < minDelta) minDelta = low;
                if (high < negMaxDelta) negMaxDelta = high;
            }
            const float maxDelta = -negMaxDelta;
            const float absOrigin = origin < 0.0f ? -origin : origin;
            const float absLow = minDelta < 0.0f ? -minDelta : minDelta;
            const float absHigh = maxDelta < 0.0f ? -maxDelta : maxDelta;
            const float deltaBound = absLow > absHigh ? absLow : absHigh;
            // Endpoint magnitude is bounded by anchor magnitude plus deviation.
            // Inflate for rounding in step, the anchor, and the subtraction.
            const float bound = (absOrigin + extent + deltaBound) * 1.00001f + 0.0001f;
            if (!(bound < 1.0e10f)) return false;
            margin = 0.00000762939453125f *
                (bound + absOrigin + extent) + 0.0001f;
            lowDelta = minDelta - margin;
            highDelta = maxDelta + margin;
        } else {
            auto lower = NeighborF(0), upper = NeighborF(256), anchor = NeighborF(512);
            auto magnitude = NeighborF(768), a = NeighborF(1024), c = NeighborF(1280);
            Muls(a, arena[oAx], sign, m);
            Muls(c, arena[oUx], sign, m);
            Add(c, a, c, m);
            Duplicate(lower, kGuardDist, mAligned);
            Duplicate(upper, -kGuardDist, mAligned);
            Min(lower, a, c, m); Max(upper, a, c, m);
            Abs(a, lower, m); Abs(c, upper, m);
            Duplicate(magnitude, 0.0f, mAligned);
            Max(magnitude, a, c, m);
            CreateVecIndex(anchor, 0.0f, mAligned);
            Muls(anchor, anchor, step, mAligned); Adds(anchor, anchor, origin, mAligned);
            Sub(lower, lower, anchor, m); Sub(upper, upper, anchor, m);
            ReduceMin(pair, lower, reduction, mAligned, false);
            ReduceMax(pair[8], upper, reduction, mAligned, false);
            ReduceMax(pair[16], magnitude, reduction, mAligned, false);
            Hop<HardEvent::V_S>();
            const float bound = pair.GetValue(16);
            if (!(bound < 1.0e10f)) return false;
            margin = 0.00000762939453125f *
                (bound + (origin < 0.0f ? -origin : origin) + extent) + 0.0001f;
            lowDelta = pair.GetValue(0) - margin;
            highDelta = pair.GetValue(8) + margin;
        }
        if (!(highDelta - lowDelta <= 6.0f * step)) return false;

        constexpr uint32_t lanes = 3 * kLanes;
        auto px = NeighborF(1856), py = NeighborF(2048), pz = NeighborF(2240);
        auto projection = NeighborF(2432), seed = NeighborF(2496);
        auto offsets = NeighborF(2560).template ReinterpretCast<int32_t>();
        auto cx = NeighborF(2752), cy = NeighborF(2944), cz = NeighborF(3136);
        auto ux = NeighborF(3328), uy = NeighborF(3520), uz = NeighborF(3712);
        auto kx = NeighborF(3904), ky = NeighborF(4096), kz = NeighborF(4288);
        auto ratio = NeighborF(4480), value = NeighborF(4672), best = NeighborF(4864);
        auto winner = NeighborF(4928), trial = NeighborF(4992);
        auto resultIndex = NeighborF(5184).template ReinterpretCast<int32_t>();
        auto keep = NeighborF(5248).template ReinterpretCast<uint8_t>();
        auto scratch = NeighborF(5312), radius = NeighborF(5504);
        auto delta = NeighborF(5568), left = NeighborF(5632), right = NeighborF(5696);
        auto raw = arena[oPtRaw].template ReinterpretCast<T>();
        auto source = arena[oPtRaw];
        if constexpr (sizeof(T) == 2) {
            source = arena[oPtFlt];
            Cast(source, raw, RoundMode::CAST_NONE, count * 3);
        }
        auto pointOffsets = arena[oPtOff].template ReinterpretCast<uint32_t>();
        Duplicate(px, 0.0f, kLanes); Duplicate(py, 0.0f, kLanes); Duplicate(pz, 0.0f, kLanes);
        Gather(px, source, pointOffsets, 0u, count);
        Gather(py, source, pointOffsets, 4u, count);
        Gather(pz, source, pointOffsets, 8u, count);
        Copy(px[kLanes], px, (uint64_t)kLanes, (uint8_t)2, {1, 1, 8, 0});
        Copy(py[kLanes], py, (uint64_t)kLanes, (uint8_t)2, {1, 1, 8, 0});
        Copy(pz[kLanes], pz, (uint64_t)kLanes, (uint8_t)2, {1, 1, 8, 0});
        Muls(projection, px, sign, kLanes);
        Adds(seed, projection, -origin - 0.5f * (lowDelta + highDelta), kLanes);
        Muls(seed, seed, 1.0f / step, kLanes);
        Maxs(seed, seed, 0.0f, kLanes); Mins(seed, seed, (float)(int32_t)(m - 1), kLanes);
        Cast(resultIndex, seed, RoundMode::CAST_RINT, kLanes);
        Cast(seed, resultIndex, RoundMode::CAST_NONE, kLanes);
        Adds(trial, seed, -1.0f, kLanes);
        Copy(trial[kLanes], seed, (uint64_t)kLanes, (uint8_t)1, {1, 1, 8, 8});
        Adds(trial[2 * kLanes], seed, 1.0f, kLanes);
        Maxs(trial, trial, 0.0f, lanes); Mins(trial, trial, (float)(int32_t)(m - 1), lanes);
        Muls(scratch, trial, 4.0f, lanes);
        Cast(offsets, scratch, RoundMode::CAST_RINT, lanes);
        auto off = offsets.template ReinterpretCast<uint32_t>();
        Gather(cx, arena[oAx], off, 0u, lanes);
        Gather(cy, arena[oAy], off, 0u, lanes);
        Gather(cz, arena[oAz], off, 0u, lanes);
        Gather(ux, arena[oUx], off, 0u, lanes);
        Gather(uy, arena[oUy], off, 0u, lanes);
        Gather(uz, arena[oUz], off, 0u, lanes);
        Sub(cx, cx, px, lanes); Sub(cy, cy, py, lanes); Sub(cz, cz, pz, lanes);
        if constexpr (ANALYTIC_BOUND) {
            Gather(kx, arena[oLen], off, 0u, lanes);
            Muls(kx, kx, -1.0f, lanes);
            Mul(ratio, cx, ux, lanes);
            MulAddDst(ratio, cy, uy, lanes);
            MulAddDst(ratio, cz, uz, lanes);
            Div(ratio, ratio, kx, lanes);
        } else {
            Gather(kx, arena[oKx], off, 0u, lanes);
            Gather(ky, arena[oKy], off, 0u, lanes);
            Gather(kz, arena[oKz], off, 0u, lanes);
            Mul(ratio, cx, kx, lanes);
            MulAddDst(ratio, cy, ky, lanes);
            MulAddDst(ratio, cz, kz, lanes);
        }
        Maxs(ratio, ratio, 0.0f, lanes); Mins(ratio, ratio, 1.0f, lanes);
        MulAddDst(cx, ratio, ux, lanes);
        MulAddDst(cy, ratio, uy, lanes); MulAddDst(cz, ratio, uz, lanes);
        Mul(value, cx, cx, lanes);
        if constexpr (SPLIT) {
            Mul(cy, cy, cy, lanes); Mul(cz, cz, cz, lanes);
            Add(value, value, cy, lanes); Add(value, value, cz, lanes);
        } else {
            MulAddDst(value, cy, cy, lanes); MulAddDst(value, cz, cz, lanes);
        }
        Copy(best, value, (uint64_t)kLanes, (uint8_t)1, {1, 1, 8, 8});
        Copy(winner, trial, (uint64_t)kLanes, (uint8_t)1, {1, 1, 8, 8});
        Compare(keep, value[kLanes], best, CMPMODE::LT, kLanes);
        Select(best, keep, value[kLanes], best, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes);
        Select(winner, keep, trial[kLanes], winner, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes);
        Compare(keep, value[2 * kLanes], best, CMPMODE::LT, kLanes);
        Select(best, keep, value[2 * kLanes], best, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes);
        Select(winner, keep, trial[2 * kLanes], winner, SELMODE::VSEL_TENSOR_TENSOR_MODE, kLanes);
        auto distance = arena[oDist];
        Sqrt(distance, best, count);
        Cast(resultIndex, winner, RoundMode::CAST_RINT, count);
        if constexpr (ANALYTIC_BOUND) {
            // A rounded seed is within half a step of the envelope center.
            // When the seed is clamped, there is no omitted segment on the
            // clamped side. The same bound applies to the other side.
            const float clearance = 1.5f * step - 0.5f * (highDelta - lowDelta) - 2.0f * margin;
            if (clearance > 0.0f) {
                ReduceMax(pair, distance, reduction, count, false);
                Hop<HardEvent::V_S>();
                if (pair.GetValue(0) < clearance) {
                    Emit(batch, start, count, distance, resultIndex);
                    return true;
                }
                Hop<HardEvent::S_V>();
            }
        }
        Adds(radius, distance, margin, count);
        Muls(delta, seed, step, count); Adds(delta, delta, origin, count);
        Sub(delta, projection, delta, count);
        Adds(left, delta, 2.0f * step - highDelta, count);
        Muls(right, delta, -1.0f, count); Adds(right, right, 2.0f * step + lowDelta, count);
        CompareScalar(keep, seed, 1.0f, CMPMODE::GT, kLanes);
        Select(left, keep, left, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        CompareScalar(keep, seed, (float)(int32_t)(m - 2), CMPMODE::LT, kLanes);
        Select(right, keep, right, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        Min(left, left, right, count); Sub(left, left, radius, count);
        ReduceMin(pair, left, reduction, count, false);
        Hop<HardEvent::V_S>();
        if (!(pair.GetValue(0) > 0.0f)) {
            Hop<HardEvent::S_V>();
            return false;
        }
        Emit(batch, start, count, distance, resultIndex);
        return true;
    }

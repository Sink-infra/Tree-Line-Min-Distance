#pragma once

namespace AffineSegmentSearch {
constexpr uint32_t kPointCap = 128;
__aicore__ inline LocalTensor<float> F(uint32_t byte, uint32_t count = kPointCap)
{
    return LocalTensor<float>(TPosition::VECCALC, byte, count);
}

// All point lanes retain their own original position; no scattered stores.
template <bool EXTRA = false>
__aicore__ inline void Evaluate(uint32_t count, uint32_t segments)
{
    auto j = F(111872), high = F(112384), best = F(113920), winner = F(114432);
    auto cx = F(114944), cy = F(115456), cz = F(115968);
    auto vx = F(116480), vy = F(116992), vz = F(117504);
    auto ix = F(118016), iy = F(118528), iz = F(119040);
    auto ratio = F(119552), value = F(120064), scratch = F(120576);
    LocalTensor<int32_t> offsets(TPosition::VECCALC, 113408, kPointCap);
    LocalTensor<uint8_t> keep(TPosition::VECCALC, 121600, 256);
    LocalTensor<uint8_t> valid(TPosition::VECCALC, 121856, 32);
    LocalTensor<uint8_t> equal(TPosition::VECCALC, 121888, 32);
    Compare(valid, j, high, CMPMODE::LE, count);
    if constexpr (EXTRA) {
        Compare(equal, j, F(112896), CMPMODE::GE, count);
        And(valid.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(),
            equal.ReinterpretCast<uint16_t>(), 16);
    }
    Maxs(scratch, j, 0.0f, count); Mins(scratch, scratch, (float)(int32_t)(segments - 1), count);
    Muls(scratch, scratch, 4.0f, count);
    Cast(offsets, scratch, RoundMode::CAST_RINT, count);
    auto off = offsets.ReinterpretCast<uint32_t>();
    Gather(cx, F(0, 1024), off, 0u, count);
    Gather(cy, F(4096, 1024), off, 0u, count);
    Gather(cz, F(8192, 1024), off, 0u, count);
    Gather(vx, F(12288, 1024), off, 0u, count);
    Gather(vy, F(16384, 1024), off, 0u, count);
    Gather(vz, F(20480, 1024), off, 0u, count);
    Gather(ix, F(24576, 1024), off, 0u, count);
    Gather(iy, F(28672, 1024), off, 0u, count);
    Gather(iz, F(32768, 1024), off, 0u, count);
    Sub(cx, cx, F(110336), count); Sub(cy, cy, F(110848), count); Sub(cz, cz, F(111360), count);
    Mul(ratio, cx, ix, count); MulAddDst(ratio, cy, iy, count); MulAddDst(ratio, cz, iz, count);
    Maxs(ratio, ratio, 0.0f, count); Mins(ratio, ratio, 1.0f, count);
    MulAddDst(cx, ratio, vx, count); MulAddDst(cy, ratio, vy, count); MulAddDst(cz, ratio, vz, count);
    Mul(value, cx, cx, count); MulAddDst(value, cy, cy, count); MulAddDst(value, cz, cz, count);
    Select(value, valid, value, kGuardDist, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    Compare(keep, value, best, CMPMODE::GE, count);
    if constexpr (EXTRA) {
        Compare(equal, value, best, CMPMODE::EQ, count);
        Min(F(121088), winner, j, count);
        Select(scratch, keep, winner, j, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Select(winner, equal, F(121088), scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    } else {
        Select(winner, keep, winner, j, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }
    Min(best, best, value, count);
}

template <typename T>
__aicore__ inline bool Run(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                          const TreeLineMinDistanceTilingData& t)
{
    if (t.b * t.chunks > GetBlockNum() || t.m < 128) return false;
    const uint32_t task = GetBlockIdx(), batch = task / t.chunks;
    const uint32_t begin = (task % t.chunks) * t.chunkPoints;
    if (begin >= t.n) return true;
    const uint32_t owned = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
    const uint32_t prefix = (t.m - 1) / 64 * 64, tail = t.m - prefix;
    const uint32_t wideM = AlignLanes(t.m);
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T*)p); segments.SetGlobalBuffer((__gm__ T*)s);
    distances.SetGlobalBuffer((__gm__ T*)d); indices.SetGlobalBuffer((__gm__ int32_t*)i);
    LocalTensor<T> raw(TPosition::VECCALC, 36864, 6144);
    auto converted = F(61440, 6144), temp = F(102400, 1024), pair = F(106496, 64);
    LocalTensor<int32_t> segmentOffsets(TPosition::VECCALC, 86016, 1024);
    auto low = F(90112, 1024), high = F(94208, 1024), anchor = F(98304, 1024);
    LocalTensor<uint8_t> keep(TPosition::VECCALC, 121600, 256);
    DataCopyPad(raw, segments[(uint64_t)batch * t.m * 6],
        {1, (uint32_t)(t.m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    CreateVecIndex(temp, 0.0f, wideM); Muls(temp, temp, 24.0f, wideM);
    Cast(segmentOffsets, temp, RoundMode::CAST_RINT, wideM);
    LocalTensor<float> src = raw.template ReinterpretCast<float>();
    if constexpr (sizeof(T) == 2) {
        Hop<HardEvent::MTE2_V>();
        src = converted; Cast(src, raw, RoundMode::CAST_NONE, t.m * 6);
        Hop<HardEvent::V_S>();
    } else { Hop<HardEvent::MTE2_S>(); }
    float sampledSpan = -1.0f, sampledWidth = 0.0f;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const float a = src.GetValue(axis), c = src.GetValue(axis + 3);
        const uint32_t last = 6 * (prefix - 1) + axis;
        const float e = src.GetValue(last), f = src.GetValue(last + 3);
        const float firstDirection = c - a, lastDirection = f - e;
        const float span = e + 0.5f * lastDirection - a - 0.5f * firstDirection;
        const float magnitude = span < 0.0f ? -span : span;
        if (magnitude > sampledSpan) {
            sampledSpan = magnitude;
            const float w0 = firstDirection < 0.0f ? -firstDirection : firstDirection;
            const float w1 = lastDirection < 0.0f ? -lastDirection : lastDirection;
            sampledWidth = w0 > w1 ? w0 : w1;
        }
    }
    // A certified model spans at most sixteen index steps per segment.
    // This loose sample rejection precedes all coefficient preparation.
    if (!(sampledSpan > 0.0f) ||
        sampledWidth * (float)(int32_t)(prefix - 1) > 32.0f * sampledSpan) return false;
    Hop<HardEvent::MTE2_V>();
    auto absolute = sizeof(T) == 2 ? F(36864, 6144) : converted;
    Duplicate(absolute, 0.0f, AlignLanes(t.m * 6));
    Abs(absolute, src, t.m * 6);
    ReduceMax(pair, absolute, temp, AlignLanes(t.m * 6), false);
    Hop<HardEvent::V_S>();
    const float coordinateBound = pair.GetValue(0);
    if (!(coordinateBound < 1.0e10f)) return false;
    const auto off = segmentOffsets.ReinterpretCast<uint32_t>();
    auto ax = F(0, 1024), ay = F(4096, 1024), az = F(8192, 1024);
    auto ux = F(12288, 1024), uy = F(16384, 1024), uz = F(20480, 1024);
    Gather(ax, src, off, 0u, t.m); Gather(ay, src, off, 4u, t.m); Gather(az, src, off, 8u, t.m);
    Gather(ux, src, off, 12u, t.m); Gather(uy, src, off, 16u, t.m); Gather(uz, src, off, 20u, t.m);
    Sub(ux, ux, ax, t.m); Sub(uy, uy, ay, t.m); Sub(uz, uz, az, t.m);
    Duplicate(temp, 1.0f, wideM);
    Mul(temp, ux, ux, t.m); MulAddDst(temp, uy, uy, t.m); MulAddDst(temp, uz, uz, t.m);
    CompareScalar(keep, temp, 0.0f, CMPMODE::NE, wideM);
    Select(temp, keep, temp, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, wideM);
    Duplicate(high, -1.0f, wideM); Div(low, high, temp, wideM);
    Mul(F(24576, 1024), ux, low, t.m);
    Mul(F(28672, 1024), uy, low, t.m);
    Mul(F(32768, 1024), uz, low, t.m);
    Hop<HardEvent::V_S>();
    float origin = ax.GetValue(0) + 0.5f * ux.GetValue(0);
    float delta = ax.GetValue(prefix - 1) + 0.5f * ux.GetValue(prefix - 1) - origin;
    uint32_t axis = 0;
    float magnitude = delta < 0.0f ? -delta : delta;
    for (uint32_t a = 1; a < 3; ++a) {
        auto base = F(a * 4096, 1024), direction = F(12288 + a * 4096, 1024);
        const float start = base.GetValue(0) + 0.5f * direction.GetValue(0);
        const float diff = base.GetValue(prefix - 1) + 0.5f * direction.GetValue(prefix - 1) - start;
        const float span = diff < 0.0f ? -diff : diff;
        if (span > magnitude) { axis = a; magnitude = span; origin = start; delta = diff; }
    }
    const float sign = delta > 0.0f ? 1.0f : -1.0f;
    origin *= sign;
    const float step = magnitude / (float)(int32_t)(prefix - 1);
    if (!(step > 1.0e-10f && step < 1.0e10f)) return false;
    const float inverseStep = 1.0f / step;
    const float margin = 0.00000762939453125f *
        (coordinateBound + (origin < 0.0f ? -origin : origin) + magnitude) + 0.0001f;
    auto base = F(axis * 4096, 1024), direction = F(12288 + axis * 4096, 1024);
    Duplicate(low, kGuardDist, wideM); Duplicate(high, -kGuardDist, wideM);
    if (sign < 0.0f) {
        Muls(anchor, base, -1.0f, t.m); Muls(temp, direction, -1.0f, t.m);
        Add(high, anchor, temp, t.m); Min(low, anchor, high, t.m); Max(high, anchor, high, t.m);
    } else {
        Add(high, base, direction, t.m); Min(low, base, high, t.m); Max(high, base, high, t.m);
    }
    ReduceMin(pair, low[prefix], temp, 64, false);
    ReduceMax(pair[8], high[prefix], temp, 64, false);
    CreateVecIndex(anchor, 0.0f, prefix); Muls(anchor, anchor, step, prefix); Adds(anchor, anchor, origin, prefix);
    Sub(low, low, anchor, prefix); Sub(high, high, anchor, prefix);
    ReduceMin(pair[16], low, temp, prefix, false);
    ReduceMax(pair[24], high, temp, prefix, false);
    Hop<HardEvent::V_S>();
    const float tailLow = pair.GetValue(0), tailHigh = pair.GetValue(8);
    const float lowerDelta = pair.GetValue(16) - margin, upperDelta = pair.GetValue(24) + margin;
    if (!(upperDelta - lowerDelta <= 16.0f * step)) return false;

    LocalTensor<T> pointRaw(TPosition::VECCALC, 106752, 384);
    auto pointFloat = F(108288, 384);
    LocalTensor<int32_t> pointOffsets(TPosition::VECCALC, 109824, kPointCap);
    auto px = F(110336), py = F(110848), pz = F(111360);
    auto j = F(111872), jHigh = F(112384), jLow = F(112896);
    LocalTensor<int32_t> integer(TPosition::VECCALC, 113408, kPointCap);
    auto best = F(113920), winner = F(114432), scratch = F(120576), value = F(120064);
    auto projection = F(123648), seed = F(124160), radius = F(124672);
    bool stored = false;
    for (uint32_t done = 0; done < owned; done += kPointCap) {
        if (stored) Hop<HardEvent::MTE3_V>();
        const uint32_t real = owned - done < kPointCap ? owned - done : kPointCap;
        const uint32_t count = AlignLanes(real), wide = count;
        CreateVecIndex(scratch, 0.0f, wide); Muls(scratch, scratch, 12.0f, wide);
        Cast(pointOffsets, scratch, RoundMode::CAST_RINT, wide);
        DataCopyPad(pointRaw, points[((uint64_t)batch * t.n + begin + done) * 3],
            {1, (uint32_t)(real * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        Hop<HardEvent::MTE2_V>();
        LocalTensor<float> pointSource = pointRaw.template ReinterpretCast<float>();
        if constexpr (sizeof(T) == 2) {
            pointSource = pointFloat; Cast(pointSource, pointRaw, RoundMode::CAST_NONE, real * 3);
            Hop<HardEvent::V_S>();
        } else { Hop<HardEvent::MTE2_S>(); }
        const float firstX = pointSource.GetValue(0), firstY = pointSource.GetValue(1), firstZ = pointSource.GetValue(2);
        Duplicate(px, firstX, wide); Duplicate(py, firstY, wide); Duplicate(pz, firstZ, wide);
        auto pointBytes = pointOffsets.ReinterpretCast<uint32_t>();
        Gather(px, pointSource, pointBytes, 0u, real); Gather(py, pointSource, pointBytes, 4u, real);
        Gather(pz, pointSource, pointBytes, 8u, real);
        Muls(projection, F(110336 + axis * 512), sign, count);
        Adds(seed, projection, -origin - 0.5f * (lowerDelta + upperDelta), count);
        Muls(seed, seed, inverseStep, count);
        Maxs(seed, seed, 0.0f, count); Mins(seed, seed, (float)(int32_t)(prefix - 1), count);
        Cast(integer, seed, RoundMode::CAST_RINT, count); Cast(seed, integer, RoundMode::CAST_NONE, count);
        Duplicate(best, kGuardDist, wide); Duplicate(winner, 0.0f, wide);
        Duplicate(jHigh, (float)(int32_t)(prefix - 1), wide);
        // Seed candidates form a contiguous interval and establish an upper bound.
        // Additional candidates retain the smallest index explicitly on equal distances.
        for (int32_t shift = -2; shift <= 2; ++shift) {
            Adds(j, seed, (float)shift, count); Maxs(j, j, 0.0f, count);
            Mins(j, j, (float)(int32_t)(prefix - 1), count);
            Evaluate(count, t.m);
        }
        if (tail <= 8) {
            Duplicate(jHigh, (float)(int32_t)(t.m - 1), wide);
            for (uint32_t r = 0; r < tail; ++r) {
                Duplicate(j, (float)(int32_t)(prefix + r), wide); Evaluate(count, t.m);
            }
        }
        Sqrt(radius, best, count); Adds(radius, radius, margin, count);
        if (tail > 8) {
            Duplicate(value, kGuardDist, wide);
            Muls(value, projection, -1.0f, count); Adds(value, value, tailLow, count);
            Adds(scratch, projection, -tailHigh, count); Max(value, value, scratch, count);
            Maxs(value, value, 0.0f, count); Sub(value, value, radius, count);
            ReduceMin(pair, value, temp, wide, false); Hop<HardEvent::V_S>();
            if (!(pair.GetValue(0) > margin)) return false;
        }
        Sub(jLow, projection, radius, count); Adds(jLow, jLow, -origin - upperDelta, count);
        Muls(jLow, jLow, inverseStep, count);
        Cast(integer, jLow, RoundMode::CAST_CEIL, count); Cast(jLow, integer, RoundMode::CAST_NONE, count);
        Adds(jLow, jLow, -1.0f, count); Maxs(jLow, jLow, 0.0f, count);
        Mins(jLow, jLow, (float)(int32_t)(prefix - 1), count);
        Add(jHigh, projection, radius, count); Adds(jHigh, jHigh, -origin - lowerDelta, count);
        Muls(jHigh, jHigh, inverseStep, count);
        Cast(integer, jHigh, RoundMode::CAST_FLOOR, count); Cast(jHigh, integer, RoundMode::CAST_NONE, count);
        Adds(jHigh, jHigh, 1.0f, count); Maxs(jHigh, jHigh, 0.0f, count);
        Mins(jHigh, jHigh, (float)(int32_t)(prefix - 1), count);
        Duplicate(value, 0.0f, wide); Sub(value, jHigh, jLow, count); Adds(value, value, 1.0f, count);
        ReduceMax(pair, value, temp, wide, false); Hop<HardEvent::V_S>();
        const float maximum = pair.GetValue(0);
        if (!(maximum > 0.0f && maximum <= 32.0f)) return false;
        Sub(value, seed, jLow, count); Adds(value, value, -2.0f, count);
        Maxs(value, value, 0.0f, count); ReduceMax(pair, value, temp, wide, false);
        Sub(value, jHigh, seed, count); Adds(value, value, -2.0f, count);
        Maxs(value, value, 0.0f, count); ReduceMax(pair[8], value, temp, wide, false);
        Hop<HardEvent::V_S>();
        const uint32_t left = (uint32_t)(int32_t)pair.GetValue(0);
        const uint32_t right = (uint32_t)(int32_t)pair.GetValue(8);
        for (uint32_t r = 0; r < left; ++r) {
            Adds(j, seed, -(float)(int32_t)(3 + r), count); Evaluate<true>(count, t.m);
        }
        for (uint32_t r = 0; r < right; ++r) {
            Adds(j, seed, (float)(int32_t)(3 + r), count); Evaluate<true>(count, t.m);
        }
        auto result = F(122112);
        LocalTensor<int32_t> resultIndex(TPosition::VECCALC, 122624, kPointCap);
        LocalTensor<T> narrow(TPosition::VECCALC, 123136, kPointCap);
        Sqrt(result, best, count); Cast(resultIndex, winner, RoundMode::CAST_RINT, count);
        const uint64_t output = (uint64_t)batch * t.n + begin + done;
        if constexpr (sizeof(T) == 2) {
            PipeBarrier<PIPE_V>();
            Cast(narrow, result, RoundMode::CAST_RINT, count); Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[output], narrow, {1, (uint32_t)(real * sizeof(T)), 0, 0, 0});
        } else {
            Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[output], result.template ReinterpretCast<T>(), {1, real * 4, 0, 0, 0});
        }
        DataCopyPad(indices[output], resultIndex, {1, real * 4, 0, 0, 0});
        stored = true;
    }
    return true;
}
} // namespace AffineSegmentSearch



template <typename T>
__aicore__ inline void AffineOrFallback(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                       const TreeLineMinDistanceTilingData& t)
{
    {
        TPipe modelPipe;
        TBuf<TPosition::VECCALC> modelStorage;
        modelPipe.InitBuffer(modelStorage, 125184);
        if (AffineSegmentSearch::Run<T>(p, s, d, i, t)) return;
    }
    PipeBarrier<PIPE_ALL>();
    TPipe pipe;
    if (t.chunkPoints < 384) {
        auto grouped = t;
        grouped.group = HoistedPointGroupLayout::kPoints;
        grouped.tilePoints = HoistedPointGroupLayout::kPoints;
        KernelPointGroups<T, 3, true, false, true, true, false, false, false, true> op;
        op.Init(p, s, d, i, grouped, &pipe);
        op.Process();
    } else {
        KernelPointGroups<T, 3, true, false, true, true, false, false, true> op;
        op.Init(p, s, d, i, t, &pipe);
        op.Process();
    }
}

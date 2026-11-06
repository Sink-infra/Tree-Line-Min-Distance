#pragma once

template <typename T>
__aicore__ inline float SingletonSegmentDistance(const LocalTensor<T>& src,
    uint32_t j, float px, float py, float pz)
{
    const uint32_t base = 6 * j;
    const float ax = (float)src.GetValue(base), ay = (float)src.GetValue(base + 1), az = (float)src.GetValue(base + 2);
    const float ux = (float)src.GetValue(base + 3) - ax;
    const float uy = (float)src.GetValue(base + 4) - ay;
    const float uz = (float)src.GetValue(base + 5) - az;
    float dx = px - ax, dy = py - ay, dz = pz - az;
    const float length = ux * ux + uy * uy + uz * uz;
    const float dot = dx * ux + dy * uy + dz * uz;
    if (length > 0.0f && dot > 0.0f) {
        const float t = dot < length ? dot / length : 1.0f;
        dx -= t * ux; dy -= t * uy; dz -= t * uz;
    }
    return dx * dx + dy * dy + dz * dz;
}

template <typename T>
__aicore__ inline bool SingletonSparseSeed(const LocalTensor<T>& raw, uint32_t m,
    float px, float py, float pz, float& best, int32_t& winner)
{
    const float a0 = (float)raw.GetValue(0), c0 = (float)raw.GetValue(3);
    const float a1 = (float)raw.GetValue(6 * (m - 1)), c1 = (float)raw.GetValue(6 * (m - 1) + 3);
    const float origin = a0 + 0.5f * (c0 - a0);
    const float span = a1 + 0.5f * (c1 - a1) - origin;
    const float magnitude = span < 0.0f ? -span : span;
    float w0 = c0 - a0, w1 = c1 - a1;
    if (w0 < 0.0f) w0 = -w0;
    if (w1 < 0.0f) w1 = -w1;
    const float width = w0 > w1 ? w0 : w1;
    if (!(magnitude > 0.0f) || width * (float)(int32_t)(m - 1) > 32.0f * magnitude) return false;
    float location = (px - origin) * (float)(int32_t)(m - 1) / span;
    if (location < 0.0f) location = 0.0f;
    if (location > (float)(int32_t)(m - 1)) location = (float)(int32_t)(m - 1);
    if (!(location >= 0.0f && location <= (float)(int32_t)(m - 1))) return false;
    winner = (int32_t)(location + 0.5f);
    best = SingletonSegmentDistance(raw, (uint32_t)winner, px, py, pz);
    if (!(best >= 0.0f && best < kGuardDist)) return false;
    const float position = (px - origin) * (span > 0.0f ? 1.0f : -1.0f);
    float outside = 0.0f;
    if (position < 0.0f) outside = -position;
    else if (position > magnitude) outside = position - magnitude;
    const float reach = outside + 8.0f * magnitude / (float)(int32_t)(m - 1) + width;
    return best <= reach * reach;
}

// The projected segment lies inside its endpoint interval. Roundoff slack
// expands that interval before it is used as a lower bound.
template <typename T>
__aicore__ inline bool ScreenSingleton(const LocalTensor<float>& src,
    const LocalTensor<int32_t>& offsets, const LocalTensor<float>& ax,
    const LocalTensor<float>& cx, const LocalTensor<float>& lo,
    const LocalTensor<float>& hi, const LocalTensor<float>& bound,
    const LocalTensor<float>& temp, const LocalTensor<float>& pair,
    const LocalTensor<uint8_t>& mask, uint32_t m, uint32_t width,
    float px, float py, float pz, float& best, int32_t& winner,
    uint32_t& compactCount)
{
    if (m < 128) return false;
    const float threshold = best * 1.0001f + 1.0e-10f;
    if constexpr (sizeof(T) == 2) {
        // Each input coordinate is exactly representable in float. Across
        // the finite half range, the two coordinate subtractions, projected
        // residual, and radius endpoint rounding cost less than 1/8.
        const float radius = sqrt(threshold) + 0.125f;
        const auto highMask = temp.ReinterpretCast<uint8_t>();
        const uint8_t repeats = (uint8_t)(width / kLanes);
        const BinaryRepeatParams binary(1, 1, 1, 8, 8, 8);
        const UnaryRepeatParams unary(1, 1, 8, 8);
        SetMaskNorm();
        SetVectorMask<float>((int32_t)kLanes);
        Min<float, false>(lo, ax, cx, (uint64_t)kLanes, repeats, binary);
        Max<float, false>(hi, ax, cx, (uint64_t)kLanes, repeats, binary);
        CompareScalar<float, uint8_t, false>(mask, lo, px + radius, CMPMODE::LE,
            (uint64_t)kLanes, repeats, unary);
        CompareScalar<float, uint8_t, false>(highMask, hi, px - radius, CMPMODE::GE,
            (uint64_t)kLanes, repeats, unary);
        ResetMask();
        And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
            highMask.ReinterpretCast<uint16_t>(), width / 16);
    } else {
        Duplicate(bound, kGuardDist, width);
        Min(lo, ax, cx, m); Max(hi, ax, cx, m);
        Adds(lo, lo, -px, m);
        Muls(hi, hi, -1.0f, m); Adds(hi, hi, px, m);
        Max(bound, lo, hi, m);
        Abs(lo, ax, m); Abs(hi, cx, m);
        Max(lo, lo, hi, m);
        Maxs(lo, lo, px < 0.0f ? -px : px, m);
        Muls(lo, lo, 1.0e-5f, m); Adds(lo, lo, 1.0e-5f, m);
        Sub(bound, bound, lo, m); Maxs(bound, bound, 0.0f, m);
        Mul(bound, bound, bound, m);
        CompareScalar(mask, bound, threshold, CMPMODE::LE, width);
    }
    uint64_t kept = 0;
    GatherMask(lo.ReinterpretCast<int32_t>(), offsets, mask.ReinterpretCast<uint32_t>(),
               true, m, {1, 1, 0, 0}, kept);
    Hop<HardEvent::V_S>();
    if (kept == 0 || kept > 8) {
        if (kept >= 9 && kept <= 64) compactCount = (uint32_t)kept;
        return false;
    }
    const int32_t seedIndex = winner;
    for (uint32_t q = 0; q < (uint32_t)kept; ++q) {
        const int32_t j = lo.ReinterpretCast<int32_t>().GetValue(q) / 24;
        if (j == seedIndex) continue;
        const float value = SingletonSegmentDistance(src, (uint32_t)j, px, py, pz);
        if (value < best || (value == best && j < winner)) { best = value; winner = j; }
    }
    return best * 1.0001f + 1.0e-10f <= threshold;
}

template <typename T>
__aicore__ inline void FusedPointSegments(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                         uint32_t m)
{
    ICachePreLoad(4);
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p, 3);
    segments.SetGlobalBuffer((__gm__ T *)s, m * 6);
    distances.SetGlobalBuffer((__gm__ T *)d, 1);
    indices.SetGlobalBuffer((__gm__ int32_t *)i, 1);
    LocalTensor<T> pt(TPosition::VECCALC, 0, 16);
    LocalTensor<T> raw(TPosition::VECCALC, 64, 6144);
    LocalTensor<float> converted(TPosition::VECCALC, 24640, 6144);
    LocalTensor<int32_t> off(TPosition::VECCALC, 49216, 1024);
    LocalTensor<float> ax(TPosition::VECCALC, 53312, 1024);
    LocalTensor<float> ay(TPosition::VECCALC, 57440, 1024);
    LocalTensor<float> az(TPosition::VECCALC, 61568, 1024);
    LocalTensor<float> ux(TPosition::VECCALC, 65696, 1024);
    LocalTensor<float> uy(TPosition::VECCALC, 69824, 1024);
    LocalTensor<float> uz(TPosition::VECCALC, 73952, 1024);
    LocalTensor<float> length(TPosition::VECCALC, 78080, 1024);
    LocalTensor<float> ratio(TPosition::VECCALC, 82208, 1024);
    LocalTensor<float> dsq(TPosition::VECCALC, 86336, 1024);
    LocalTensor<float> work(TPosition::VECCALC, 90464, 1024);
    LocalTensor<float> pair(TPosition::VECCALC, 94592, 64);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 94848, 8);
    LocalTensor<T> narrow(TPosition::VECCALC, 94880, 16);
    LocalTensor<uint8_t> keep(TPosition::VECCALC, 94944, 256);
    const uint32_t width = AlignLanes(m);
    DataCopyPad(pt, points, {1, (uint32_t)(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(raw, segments, {1, (uint32_t)(m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    CreateVecIndex(work, 0.0f, width);
    Muls(work, work, 24.0f, width);
    Cast(off, work, RoundMode::CAST_RINT, width);
    if (width != m) {
        Duplicate(ax, kGuardCoord, width); Duplicate(ay, kGuardCoord, width); Duplicate(az, kGuardCoord, width);
        Duplicate(ux, 0.0f, width); Duplicate(uy, 0.0f, width); Duplicate(uz, 0.0f, width);
    }
    Hop<HardEvent::MTE2_S>();
    const float px = (float)pt.GetValue(0), py = (float)pt.GetValue(1), pz = (float)pt.GetValue(2);
    float screenedBest = kGuardDist;
    int32_t screenedWinner = 0;
    const bool mayScreen = m >= 128 && SingletonSparseSeed(raw, m, px, py, pz, screenedBest, screenedWinner);
    Hop<HardEvent::MTE2_V>();
    LocalTensor<float> src = raw.template ReinterpretCast<float>();
    if constexpr (sizeof(T) == 2) {
        src = converted;
        Cast(src, raw, RoundMode::CAST_NONE, m * 6);
        PipeBarrier<PIPE_V>();
    }
    Gather(ax, src, off.template ReinterpretCast<uint32_t>(), 0u, m);
    Gather(ux, src, off.template ReinterpretCast<uint32_t>(), 12u, m);

    uint32_t compactCount = 0;
    const bool screened = mayScreen && ScreenSingleton<T>(src, off, ax, ux, ay, az, dsq, work,
                                         pair, keep, m, width, px, py, pz,
                                         screenedBest, screenedWinner, compactCount);
    if (mayScreen && !screened) Hop<HardEvent::S_V>();
    const bool compacted = !screened && compactCount >= 9 && compactCount <= 64;
    const uint32_t activeCount = compacted ? compactCount : m;
    const uint32_t activeWidth = compacted ? 64 : width;
    if (screened) {
        pair.SetValue(0, screenedBest);
        idx.SetValue(0, screenedWinner);
    } else {
    if (compacted) {
        Copy(off, ay.template ReinterpretCast<int32_t>(), (uint64_t)64, 1, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        if (activeCount != 64) {
            Duplicate(ax, kGuardCoord, 64); Duplicate(ay, kGuardCoord, 64); Duplicate(az, kGuardCoord, 64);
            Duplicate(ux, 0.0f, 64); Duplicate(uy, 0.0f, 64); Duplicate(uz, 0.0f, 64);
        }
        Gather(ax, src, off.template ReinterpretCast<uint32_t>(), 0u, activeCount);
        Gather(ux, src, off.template ReinterpretCast<uint32_t>(), 12u, activeCount);
    }
    Gather(ay, src, off.template ReinterpretCast<uint32_t>(), 4u, activeCount);
    Gather(az, src, off.template ReinterpretCast<uint32_t>(), 8u, activeCount);
    Gather(uy, src, off.template ReinterpretCast<uint32_t>(), 16u, activeCount);
    Gather(uz, src, off.template ReinterpretCast<uint32_t>(), 20u, activeCount);
    Sub(ux, ux, ax, activeCount); Sub(uy, uy, ay, activeCount); Sub(uz, uz, az, activeCount);
    Adds(ax, ax, -px, activeCount); Adds(ay, ay, -py, activeCount); Adds(az, az, -pz, activeCount);
    Mul(length, ux, ux, activeWidth);
    MulAddDst(length, uy, uy, activeWidth); MulAddDst(length, uz, uz, activeWidth);
    Mul(ratio, ax, ux, activeWidth);
    MulAddDst(ratio, ay, uy, activeWidth); MulAddDst(ratio, az, uz, activeWidth);
    Muls(length, length, -1.0f, activeWidth);
    CompareScalar(keep, length, 0.0f, CMPMODE::NE, activeWidth);
    Select(length, keep, length, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, activeWidth);
    Div(ratio, ratio, length, activeWidth);
    Maxs(ratio, ratio, 0.0f, activeWidth); Mins(ratio, ratio, 1.0f, activeWidth);
    MulAddDst(ax, ratio, ux, activeWidth); MulAddDst(ay, ratio, uy, activeWidth); MulAddDst(az, ratio, uz, activeWidth);
    Mul(dsq, ax, ax, activeWidth);
    MulAddDst(dsq, ay, ay, activeWidth); MulAddDst(dsq, az, az, activeWidth);
    ReduceMin(pair, dsq, work, activeWidth, true);
    }
    if (!screened) {
        Hop<HardEvent::V_S>();
        const int32_t localWinner = pair.template ReinterpretCast<int32_t>().GetValue(1);
        idx.SetValue(0, compacted ? off.GetValue((uint32_t)localWinner) / 24 : localWinner);
    }
    Hop<HardEvent::S_V>();
    Sqrt(pair, pair, 1);
    if constexpr (sizeof(T) == 2) {
        Cast(narrow, pair, RoundMode::CAST_RINT, 1);
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances, narrow, {1, (uint32_t)sizeof(T), 0, 0, 0});
    } else {
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances, pair.template ReinterpretCast<T>(), {1, (uint32_t)sizeof(T), 0, 0, 0});
    }
    DataCopyPad(indices, idx, {1, 4, 0, 0, 0});
}

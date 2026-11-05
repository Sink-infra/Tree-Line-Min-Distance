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
    return best >= 0.0f && best < kGuardDist;
}

// The projected segment lies inside its endpoint interval. Roundoff slack
// expands that interval before it is used as a lower bound.
__aicore__ inline bool ScreenSingleton(const LocalTensor<float>& src,
    const LocalTensor<int32_t>& offsets, const LocalTensor<float>& ax,
    const LocalTensor<float>& cx, const LocalTensor<float>& lo,
    const LocalTensor<float>& hi, const LocalTensor<float>& bound,
    const LocalTensor<float>& temp, const LocalTensor<float>& pair,
    const LocalTensor<uint8_t>& mask, uint32_t m, uint32_t width,
    float px, float py, float pz, float& best, int32_t& winner)
{
    if (m < 128) return false;
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
    const float threshold = best * 1.0001f + 1.0e-10f;
    CompareScalar(mask, bound, threshold, CMPMODE::LE, width);
    uint64_t kept = 0;
    GatherMask(lo.ReinterpretCast<int32_t>(), offsets, mask.ReinterpretCast<uint32_t>(),
               true, m, {1, 1, 0, 0}, kept);
    Hop<HardEvent::V_S>();
    if (kept == 0 || kept > 8) return false;
    const int32_t seedIndex = winner;
    for (uint32_t q = 0; q < (uint32_t)kept; ++q) {
        const int32_t j = lo.ReinterpretCast<int32_t>().GetValue(q) / 24;
        if (j == seedIndex) continue;
        const float value = SingletonSegmentDistance(src, (uint32_t)j, px, py, pz);
        if (value < best || (value == best && j < winner)) { best = value; winner = j; }
    }
    return best * 1.0001f + 1.0e-10f <= threshold;
}

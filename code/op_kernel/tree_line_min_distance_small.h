#pragma once
#include "tree_line_min_distance_tiny_geometry.h"

template <typename T, uint32_t SEG, bool LATE = false, bool DIRECT = false, bool ROOT = false, bool TWO_STEPS = false, uint32_t PRELOAD = 2>
__aicore__ inline void SinglePoint(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i)
{
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p, 3);
    segments.SetGlobalBuffer((__gm__ T *)s, SEG * 6);
    distances.SetGlobalBuffer((__gm__ T *)d, 1);
    indices.SetGlobalBuffer((__gm__ int32_t *)i, 1);
    LocalTensor<T> pt(TPosition::VECCALC, 0, 16);
    LocalTensor<T> seg(TPosition::VECCALC, 64, SEG <= 2 ? 16 : 64);
    LocalTensor<float> dist(TPosition::VECCALC, 128, 8);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 160, 8);
    LocalTensor<T> narrow(TPosition::VECCALC, 192, 16);
    if constexpr (ROOT) {
        GlobalTensor<uint64_t> outputD, outputI;
        outputD.SetGlobalBuffer((__gm__ uint64_t*)(d));
        outputI.SetGlobalBuffer((__gm__ uint64_t*)(i));
        DataCachePreload(outputD, (int64_t)0);
        DataCachePreload(outputI, (int64_t)0);
    }
    if constexpr (DIRECT) {
        RefreshSource(points, 0, 3);
        RefreshSource(segments, 0, SEG * 6);
    } else {
    DataCopyPad(pt, points, {1, (uint32_t)(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(seg, segments, {1, (uint32_t)(SEG * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    if constexpr (LATE) ICachePreLoad(PRELOAD);
    Hop<HardEvent::MTE2_S>();
    }
    const float px = (float)(DIRECT ? points.GetValue(0) : pt.GetValue(0));
    const float py = (float)(DIRECT ? points.GetValue(1) : pt.GetValue(1));
    const float pz = (float)(DIRECT ? points.GetValue(2) : pt.GetValue(2));
    float best = kGuardDist;
    int32_t winner = 0;
    for (uint32_t j = 0; j < SEG; ++j) {
        const float ax = (float)(DIRECT ? segments.GetValue(6 * j) : seg.GetValue(6 * j));
        const float ay = (float)(DIRECT ? segments.GetValue(6 * j + 1) : seg.GetValue(6 * j + 1));
        const float az = (float)(DIRECT ? segments.GetValue(6 * j + 2) : seg.GetValue(6 * j + 2));
        float value;
        if constexpr (ROOT && SEG >= 3) {
            const float bx=(float)(DIRECT ? segments.GetValue(6 * j + 3) : seg.GetValue(6 * j + 3)),by=(float)(DIRECT ? segments.GetValue(6 * j + 4) : seg.GetValue(6 * j + 4)),bz=(float)(DIRECT ? segments.GetValue(6 * j + 5) : seg.GetValue(6 * j + 5));
            value=TinyGeometryDsq(px,py,pz,ax,ay,az,bx,by,bz);
        } else {
        const float ux = (float)(DIRECT ? segments.GetValue(6 * j + 3) : seg.GetValue(6 * j + 3)) - ax;
        const float uy = (float)(DIRECT ? segments.GetValue(6 * j + 4) : seg.GetValue(6 * j + 4)) - ay;
        const float uz = (float)(DIRECT ? segments.GetValue(6 * j + 5) : seg.GetValue(6 * j + 5)) - az;
        float dx = px - ax, dy = py - ay, dz = pz - az;
        const float dot = dx * ux + dy * uy + dz * uz;
        const float len = ux * ux + uy * uy + uz * uz;
        if (len > 0.0f && dot > 0.0f) {
            const float ratio = dot < len ? dot / len : 1.0f;
            dx -= ratio * ux;
            dy -= ratio * uy;
            dz -= ratio * uz;
        }
        value = dx * dx + dy * dy + dz * dz;
        }
        if (j == 0 || value < best) { best = value; winner = (int32_t)j; }
    }
    if constexpr (ROOT) {
        distances.SetValue(0, (T)(TWO_STEPS ? TinyRootTwoSteps(best) : (SEG == 1 && sizeof(T)==2 ? TinyRootResidual(best) : TinyRoot(best))));
        indices.SetValue(0, winner);
        return;
    }
    dist.SetValue(0, best);
    idx.SetValue(0, winner);
    Hop<HardEvent::S_V>();
    Sqrt(dist, dist, 1);
    if constexpr (sizeof(T) == 2) {
        Cast(narrow, dist, RoundMode::CAST_RINT, 1);
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances, narrow, {1, (uint32_t)sizeof(T), 0, 0, 0});
    } else {
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances, dist.template ReinterpretCast<T>(), {1, (uint32_t)sizeof(T), 0, 0, 0});
    }
    DataCopyPad(indices, idx, {1, 4, 0, 0, 0});
}

template <typename T, uint32_t POINTS, uint32_t SEGS, bool LATE = false, bool DIRECT = false, bool ROOT = false>
__aicore__ inline void FixedSmallBatch(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i)
{
    if constexpr (!LATE && SEGS == 2) ICachePreLoad(2);
    const uint32_t batch = GetBlockIdx();
    constexpr uint32_t n = POINTS;
    constexpr uint32_t segCount = SEGS;
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    distances.SetGlobalBuffer((__gm__ T *)d);
    indices.SetGlobalBuffer((__gm__ int32_t *)i);
    LocalTensor<T> pt(TPosition::VECCALC, 0, 32);
    LocalTensor<T> seg(TPosition::VECCALC, 128, 64);
    LocalTensor<float> dist(TPosition::VECCALC, 384, 8);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 416, 8);
    LocalTensor<T> narrow(TPosition::VECCALC, 448, 16);
    if constexpr (ROOT) {
        GlobalTensor<uint64_t> outputD, outputI;
        outputD.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ T*)d + (uint64_t)batch * n));
        outputI.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ int32_t*)i + (uint64_t)batch * n));
        DataCachePreload(outputD, (int64_t)0);
        DataCachePreload(outputI, (int64_t)0);
    }
    if constexpr (DIRECT) {
        RefreshSource(points, (uint64_t)batch * n * 3, n * 3);
        RefreshSource(segments, (uint64_t)batch * segCount * 6, segCount * 6);
    } else {
    DataCopyPad(pt, points[(uint64_t)batch * n * 3],
                {1, (uint32_t)(n * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(seg, segments[(uint64_t)batch * segCount * 6],
                {1, (uint32_t)(segCount * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    if constexpr (LATE) ICachePreLoad(2);
    Hop<HardEvent::MTE2_S>();
    }
    #pragma unroll
    for (uint32_t point = 0; point < n; ++point) {
        const float px = (float)(DIRECT ? points.GetValue((uint64_t)batch * n * 3 + (3 * point)) : pt.GetValue(3 * point));
        const float py = (float)(DIRECT ? points.GetValue((uint64_t)batch * n * 3 + (3 * point + 1)) : pt.GetValue(3 * point + 1));
        const float pz = (float)(DIRECT ? points.GetValue((uint64_t)batch * n * 3 + (3 * point + 2)) : pt.GetValue(3 * point + 2));
        float best = kGuardDist;
        int32_t winner = 0;
        #pragma unroll
        for (uint32_t j = 0; j < segCount; ++j) {
            const float ax = (float)(DIRECT ? segments.GetValue((uint64_t)batch * segCount * 6 + (6 * j)) : seg.GetValue(6 * j));
            const float ay = (float)(DIRECT ? segments.GetValue((uint64_t)batch * segCount * 6 + (6 * j + 1)) : seg.GetValue(6 * j + 1));
            const float az = (float)(DIRECT ? segments.GetValue((uint64_t)batch * segCount * 6 + (6 * j + 2)) : seg.GetValue(6 * j + 2));
            const float ux = (float)(DIRECT ? segments.GetValue((uint64_t)batch * segCount * 6 + (6 * j + 3)) : seg.GetValue(6 * j + 3)) - ax;
            const float uy = (float)(DIRECT ? segments.GetValue((uint64_t)batch * segCount * 6 + (6 * j + 4)) : seg.GetValue(6 * j + 4)) - ay;
            const float uz = (float)(DIRECT ? segments.GetValue((uint64_t)batch * segCount * 6 + (6 * j + 5)) : seg.GetValue(6 * j + 5)) - az;
            float dx = px - ax, dy = py - ay, dz = pz - az;
            const float dot = dx * ux + dy * uy + dz * uz;
            const float len = ux * ux + uy * uy + uz * uz;
            if (len > 0.0f && dot > 0.0f) {
                const float ratio = dot < len ? dot / len : 1.0f;
                dx -= ratio * ux; dy -= ratio * uy; dz -= ratio * uz;
            }
            const float value = dx * dx + dy * dy + dz * dz;
            if (j == 0 || value < best) { best = value; winner = (int32_t)j; }
        }
        if constexpr (ROOT) {
            distances.SetValue((uint64_t)batch * n + point, (T)TinyRoot(best));
            indices.SetValue((uint64_t)batch * n + point, winner);
        } else {
            dist.SetValue(point, best);
            idx.SetValue(point, winner);
        }
    }
    if constexpr (ROOT) return;
    Hop<HardEvent::S_V>();
    Sqrt(dist, dist, n);
    if constexpr (sizeof(T) == 2) {
        Cast(narrow, dist, RoundMode::CAST_RINT, n);
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances[(uint64_t)batch * n], narrow,
                    {1, (uint32_t)(n * sizeof(T)), 0, 0, 0});
    } else {
        Hop<HardEvent::V_MTE3>();
        DataCopyPad(distances[(uint64_t)batch * n], dist.template ReinterpretCast<T>(),
                    {1, (uint32_t)(n * sizeof(T)), 0, 0, 0});
    }
    DataCopyPad(indices[(uint64_t)batch * n], idx, {1, n * 4, 0, 0, 0});
}



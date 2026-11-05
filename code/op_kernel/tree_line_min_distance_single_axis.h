#pragma once

#include "tree_line_min_distance_singleton_screen.h"

template <typename T>
__aicore__ inline void SingleSegmentTask(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                        const TreeLineMinDistanceTilingData &t)
{
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    distances.SetGlobalBuffer((__gm__ T *)d);
    indices.SetGlobalBuffer((__gm__ int32_t *)i);
    LocalTensor<T> seg(TPosition::VECCALC, 0, 16);
    LocalTensor<T> raw(TPosition::VECCALC, 64, 3072);
    LocalTensor<float> converted(TPosition::VECCALC, 12352, 3072);
    LocalTensor<int32_t> off(TPosition::VECCALC, 24640, 1024);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 28736, 1024);
    LocalTensor<float> px(TPosition::VECCALC, 32832, 1024);
    LocalTensor<float> py(TPosition::VECCALC, 36960, 1024);
    LocalTensor<float> pz(TPosition::VECCALC, 41088, 1024);
    LocalTensor<float> ratio(TPosition::VECCALC, 45216, 1024);
    LocalTensor<float> temp(TPosition::VECCALC, 49344, 1024);
    LocalTensor<float> dist(TPosition::VECCALC, 53472, 1024);
    LocalTensor<T> narrow(TPosition::VECCALC, 57600, 1024);
    const uint32_t width = AlignLanes(t.tilePoints);
    CreateVecIndex(temp, 0.0f, width);
    Muls(temp, temp, 12.0f, width);
    Cast(off, temp, RoundMode::CAST_RINT, width);
    Duplicate(idx, 0, width);
    uint32_t cached = 0xFFFFFFFFu;
    float ax = 0, ay = 0, az = 0, ux = 0, uy = 0, uz = 0;
    float kx = 0, ky = 0, kz = 0;
    bool stored = false;
    for (uint32_t task = GetBlockIdx(); task < t.b * t.chunks; task += GetBlockNum()) {
        const uint32_t batch = task / t.chunks;
        const uint32_t begin = (task % t.chunks) * t.chunkPoints;
        if (begin >= t.n) continue;
        const uint32_t count = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
        if (stored) Hop<HardEvent::MTE3_V>();
        const bool fresh = cached != batch;
        if (fresh) {
            DataCopyPad(seg, segments[(uint64_t)batch * 6],
                        {1, (uint32_t)(6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        }
        DataCopyPad(raw, points[((uint64_t)batch * t.n + begin) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        if (fresh) {
            Hop<HardEvent::MTE2_S>();
            ax = (float)seg.GetValue(0); ay = (float)seg.GetValue(1); az = (float)seg.GetValue(2);
            ux = (float)seg.GetValue(3) - ax;
            uy = (float)seg.GetValue(4) - ay;
            uz = (float)seg.GetValue(5) - az;
            const float length = ux * ux + uy * uy + uz * uz;
            const float inv = length > 0.0f ? 1.0f / length : 0.0f;
            kx = ux * inv; ky = uy * inv; kz = uz * inv;
            cached = batch;
        }
        Hop<HardEvent::MTE2_V>();
        LocalTensor<float> src = raw.template ReinterpretCast<float>();
        if constexpr (sizeof(T) == 2) {
            src = converted;
            Cast(src, raw, RoundMode::CAST_NONE, count * 3);
            PipeBarrier<PIPE_V>();
        }
        Gather(px, src, off.template ReinterpretCast<uint32_t>(), 0u, count);
        Gather(py, src, off.template ReinterpretCast<uint32_t>(), 4u, count);
        Gather(pz, src, off.template ReinterpretCast<uint32_t>(), 8u, count);
        Adds(px, px, -ax, count); Adds(py, py, -ay, count); Adds(pz, pz, -az, count);
        Muls(ratio, px, kx, count);
        Axpy(ratio, py, ky, count); Axpy(ratio, pz, kz, count);
        Maxs(ratio, ratio, 0.0f, count); Mins(ratio, ratio, 1.0f, count);
        Axpy(px, ratio, -ux, count); Axpy(py, ratio, -uy, count); Axpy(pz, ratio, -uz, count);
        Mul(dist, px, px, count);
        MulAddDst(dist, py, py, count); MulAddDst(dist, pz, pz, count);
        Sqrt(dist, dist, count);
        if constexpr (sizeof(T) == 2) {
            Cast(narrow, dist, RoundMode::CAST_RINT, count);
            Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[(uint64_t)batch * t.n + begin], narrow,
                        {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        } else {
            Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[(uint64_t)batch * t.n + begin], dist.template ReinterpretCast<T>(),
                        {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        }
        DataCopyPad(indices[(uint64_t)batch * t.n + begin], idx, {1, count * 4, 0, 0, 0});
        stored = true;
    }
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

    const bool screened = mayScreen && ScreenSingleton(src, off, ax, ux, ay, az, dsq, work,
                                         pair, keep, m, width, px, py, pz,
                                         screenedBest, screenedWinner);
    if (mayScreen && !screened) Hop<HardEvent::S_V>();
    if (screened) {
        pair.SetValue(0, screenedBest);
        idx.SetValue(0, screenedWinner);
    } else {
    Gather(ay, src, off.template ReinterpretCast<uint32_t>(), 4u, m);
    Gather(az, src, off.template ReinterpretCast<uint32_t>(), 8u, m);
    Gather(uy, src, off.template ReinterpretCast<uint32_t>(), 16u, m);
    Gather(uz, src, off.template ReinterpretCast<uint32_t>(), 20u, m);
    Sub(ux, ux, ax, m); Sub(uy, uy, ay, m); Sub(uz, uz, az, m);
    Adds(ax, ax, -px, m); Adds(ay, ay, -py, m); Adds(az, az, -pz, m);
    Mul(length, ux, ux, width);
    MulAddDst(length, uy, uy, width); MulAddDst(length, uz, uz, width);
    Mul(ratio, ax, ux, width);
    MulAddDst(ratio, ay, uy, width); MulAddDst(ratio, az, uz, width);
    Muls(length, length, -1.0f, width);
    CompareScalar(keep, length, 0.0f, CMPMODE::NE, width);
    Select(length, keep, length, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, width);
    Div(ratio, ratio, length, width);
    Maxs(ratio, ratio, 0.0f, width); Mins(ratio, ratio, 1.0f, width);
    MulAddDst(ax, ratio, ux, width); MulAddDst(ay, ratio, uy, width); MulAddDst(az, ratio, uz, width);
    Mul(dsq, ax, ax, width);
    MulAddDst(dsq, ay, ay, width); MulAddDst(dsq, az, az, width);
    ReduceMin(pair, dsq, work, width, true);
    }
    if (!screened) {
        Hop<HardEvent::V_S>();
        idx.SetValue(0, pair.template ReinterpretCast<int32_t>().GetValue(1));
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

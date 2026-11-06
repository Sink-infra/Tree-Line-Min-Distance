#pragma once

template <typename T, bool ONE_TILE = false>
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
    if constexpr (!ONE_TILE) {
        CreateVecIndex(temp, 0.0f, width);
        Muls(temp, temp, 12.0f, width);
        Cast(off, temp, RoundMode::CAST_RINT, width);
        Duplicate(idx, 0, width);
    }
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
        const bool fresh = ONE_TILE || cached != batch;
        if (fresh) {
            DataCopyPad(seg, segments[(uint64_t)batch * 6],
                        {1, (uint32_t)(6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            if constexpr (ONE_TILE) SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        DataCopyPad(raw, points[((uint64_t)batch * t.n + begin) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        if constexpr (ONE_TILE) {
            CreateVecIndex(temp, 0.0f, width);
            Muls(temp, temp, 12.0f, width);
            Cast(off, temp, RoundMode::CAST_RINT, width);
            Duplicate(idx, 0, width);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            DataCopyPad(indices[(uint64_t)batch * t.n + begin], idx,
                        {1, count * 4, 0, 0, 0});
        }
        if (fresh) {
            if constexpr (ONE_TILE) WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            else Hop<HardEvent::MTE2_S>();
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
        if constexpr (ONE_TILE) {
            break;
        } else {
            DataCopyPad(indices[(uint64_t)batch * t.n + begin], idx,
                        {1, count * 4, 0, 0, 0});
        }
        stored = true;
    }
}

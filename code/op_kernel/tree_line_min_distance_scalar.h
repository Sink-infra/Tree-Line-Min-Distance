#pragma once

template <typename T>
__aicore__ inline void RefreshSource(const GlobalTensor<T> &tensor,
                                       uint64_t begin, uint32_t count)
{
    const uint32_t misalignment = (uint32_t)((uint64_t)tensor.GetPhyAddr(begin) & 63u);
    DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE>(tensor[begin]);
    for (uint32_t j = (64u - misalignment) / sizeof(T); j < count; j += 64u / sizeof(T)) {
        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE>(tensor[begin + j]);
    }
}

template <typename T>
__aicore__ inline void ScalarTask(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                 const TreeLineMinDistanceTilingData &t)
{
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    distances.SetGlobalBuffer((__gm__ T *)d);
    indices.SetGlobalBuffer((__gm__ int32_t *)i);
    LocalTensor<float> dist(TPosition::VECCALC, 30720, 512);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 32768, 512);
    LocalTensor<T> narrow(TPosition::VECCALC, 34816, 512);
    uint32_t cached = 0xFFFFFFFFu;
    bool stored = false;
    for (uint32_t task = GetBlockIdx(); task < t.b * t.chunks; task += GetBlockNum()) {
        const uint32_t batch = task / t.chunks;
        const uint32_t begin = (task % t.chunks) * t.chunkPoints;
        if (begin >= t.n) continue;
        const uint32_t owned = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
        if (cached != batch) {
            RefreshSource(segments, (uint64_t)batch * t.m * 6, t.m * 6);
            cached = batch;
        }
        for (uint32_t done = 0; done < owned; done += t.tilePoints) {
            const uint32_t count = owned - done < t.tilePoints ? owned - done : t.tilePoints;
            if (stored) Hop<HardEvent::MTE3_S>();
            const uint64_t pointBase = ((uint64_t)batch * t.n + begin + done) * 3;
            RefreshSource(points, pointBase, count * 3);
            for (uint32_t q = 0; q < count; ++q) {
                const float px = (float)points.GetValue(pointBase + 3 * q);
                const float py = (float)points.GetValue(pointBase + 3 * q + 1);
                const float pz = (float)points.GetValue(pointBase + 3 * q + 2);
                float best = kGuardDist;
                int32_t winner = 0;
                for (uint32_t j = 0; j < t.m; ++j) {
                    const uint32_t pos = 6 * j;
                    const float ax = (float)segments.GetValue((uint64_t)batch * t.m * 6 + pos);
                    const float ay = (float)segments.GetValue((uint64_t)batch * t.m * 6 + pos + 1);
                    const float az = (float)segments.GetValue((uint64_t)batch * t.m * 6 + pos + 2);
                    const float ux = (float)segments.GetValue((uint64_t)batch * t.m * 6 + pos + 3) - ax;
                    const float uy = (float)segments.GetValue((uint64_t)batch * t.m * 6 + pos + 4) - ay;
                    const float uz = (float)segments.GetValue((uint64_t)batch * t.m * 6 + pos + 5) - az;
                    float dx = px - ax, dy = py - ay, dz = pz - az;
                    const float dot = dx * ux + dy * uy + dz * uz;
                    const float len = ux * ux + uy * uy + uz * uz;
                    if (len > 0.0f && dot > 0.0f) {
                        const float ratio = dot < len ? dot / len : 1.0f;
                        dx -= ratio * ux;
                        dy -= ratio * uy;
                        dz -= ratio * uz;
                    }
                    const float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < best) { best = d2; winner = (int32_t)j; }
                }
                dist.SetValue(q, best);
                idx.SetValue(q, winner);
            }
            Hop<HardEvent::S_V>();
            Sqrt(dist, dist, count);
            const uint64_t output = (uint64_t)batch * t.n + begin + done;
            if constexpr (sizeof(T) == 2) {
                Cast(narrow, dist, RoundMode::CAST_RINT, count);
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], narrow, {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            } else {
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], dist.template ReinterpretCast<T>(),
                            {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            }
            DataCopyPad(indices[output], idx, {1, count * 4, 0, 0, 0});
            stored = true;
        }
    }
}

template <typename T>
__aicore__ inline void LegacyScalarTask(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                 const TreeLineMinDistanceTilingData &t)
{
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    distances.SetGlobalBuffer((__gm__ T *)d);
    indices.SetGlobalBuffer((__gm__ int32_t *)i);
    LocalTensor<T> seg(TPosition::VECCALC, 0, 6144);
    LocalTensor<T> raw(TPosition::VECCALC, 24576, 1536);
    LocalTensor<float> dist(TPosition::VECCALC, 30720, 512);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 32768, 512);
    LocalTensor<T> narrow(TPosition::VECCALC, 34816, 512);
    uint32_t cached = 0xFFFFFFFFu;
    bool stored = false;
    for (uint32_t task = GetBlockIdx(); task < t.b * t.chunks; task += GetBlockNum()) {
        const uint32_t batch = task / t.chunks;
        const uint32_t begin = (task % t.chunks) * t.chunkPoints;
        if (begin >= t.n) continue;
        const uint32_t owned = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
        if (cached != batch) {
            DataCopyPad(seg, segments[(uint64_t)batch * t.m * 6],
                        {1, (uint32_t)(t.m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            cached = batch;
        }
        for (uint32_t done = 0; done < owned; done += t.tilePoints) {
            const uint32_t count = owned - done < t.tilePoints ? owned - done : t.tilePoints;
            if (stored) Hop<HardEvent::MTE3_S>();
            DataCopyPad(raw, points[((uint64_t)batch * t.n + begin + done) * 3],
                        {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            Hop<HardEvent::MTE2_S>();
            for (uint32_t q = 0; q < count; ++q) {
                const float px = (float)raw.GetValue(3 * q);
                const float py = (float)raw.GetValue(3 * q + 1);
                const float pz = (float)raw.GetValue(3 * q + 2);
                float best = kGuardDist;
                int32_t winner = 0;
                for (uint32_t j = 0; j < t.m; ++j) {
                    const uint32_t pos = 6 * j;
                    const float ax = (float)seg.GetValue(pos);
                    const float ay = (float)seg.GetValue(pos + 1);
                    const float az = (float)seg.GetValue(pos + 2);
                    const float ux = (float)seg.GetValue(pos + 3) - ax;
                    const float uy = (float)seg.GetValue(pos + 4) - ay;
                    const float uz = (float)seg.GetValue(pos + 5) - az;
                    float dx = px - ax, dy = py - ay, dz = pz - az;
                    const float dot = dx * ux + dy * uy + dz * uz;
                    const float len = ux * ux + uy * uy + uz * uz;
                    const float flat = dx * dx + dy * dy + dz * dz;
                    float d2;
                    if (dot <= 0.0f || len == 0.0f) d2 = flat;
                    else if (dot >= len) d2 = flat - dot - dot + len;
                    else d2 = flat - dot * dot / len;
                    if (d2 < best) { best = d2; winner = (int32_t)j; }
                }
                dist.SetValue(q, best);
                idx.SetValue(q, winner);
            }
            Hop<HardEvent::S_V>();
            Sqrt(dist, dist, count);
            const uint64_t output = (uint64_t)batch * t.n + begin + done;
            if constexpr (sizeof(T) == 2) {
                Cast(narrow, dist, RoundMode::CAST_RINT, count);
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], narrow, {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            } else {
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], dist.template ReinterpretCast<T>(),
                            {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            }
            DataCopyPad(indices[output], idx, {1, count * 4, 0, 0, 0});
            stored = true;
        }
    }
}

template <typename T, uint32_t SEG>
__aicore__ inline void SmallScalarTask(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                 const TreeLineMinDistanceTilingData &t)
{
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    distances.SetGlobalBuffer((__gm__ T *)d);
    indices.SetGlobalBuffer((__gm__ int32_t *)i);
    LocalTensor<T> seg(TPosition::VECCALC, 0, 6144);
    LocalTensor<T> raw(TPosition::VECCALC, 24576, 1536);
    LocalTensor<float> dist(TPosition::VECCALC, 30720, 512);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 32768, 512);
    LocalTensor<T> narrow(TPosition::VECCALC, 34816, 512);
    uint32_t cached = 0xFFFFFFFFu;
    bool stored = false;
    for (uint32_t task = GetBlockIdx(); task < t.b * t.chunks; task += GetBlockNum()) {
        const uint32_t batch = task / t.chunks;
        const uint32_t begin = (task % t.chunks) * t.chunkPoints;
        if (begin >= t.n) continue;
        const uint32_t owned = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
        if (cached != batch) {
            DataCopyPad(seg, segments[(uint64_t)batch * SEG * 6],
                        {1, (uint32_t)(SEG * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            cached = batch;
        }
        for (uint32_t done = 0; done < owned; done += t.tilePoints) {
            const uint32_t count = owned - done < t.tilePoints ? owned - done : t.tilePoints;
            if (stored) Hop<HardEvent::MTE3_S>();
            DataCopyPad(raw, points[((uint64_t)batch * t.n + begin + done) * 3],
                        {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            Hop<HardEvent::MTE2_S>();
            for (uint32_t q = 0; q < count; ++q) {
                const float px = (float)raw.GetValue(3 * q);
                const float py = (float)raw.GetValue(3 * q + 1);
                const float pz = (float)raw.GetValue(3 * q + 2);
                float best = kGuardDist;
                int32_t winner = 0;
                #pragma unroll
                for (uint32_t j = 0; j < SEG; ++j) {
                    const uint32_t pos = 6 * j;
                    const float ax = (float)seg.GetValue(pos);
                    const float ay = (float)seg.GetValue(pos + 1);
                    const float az = (float)seg.GetValue(pos + 2);
                    const float ux = (float)seg.GetValue(pos + 3) - ax;
                    const float uy = (float)seg.GetValue(pos + 4) - ay;
                    const float uz = (float)seg.GetValue(pos + 5) - az;
                    float dx = px - ax, dy = py - ay, dz = pz - az;
                    const float dot = dx * ux + dy * uy + dz * uz;
                    const float len = ux * ux + uy * uy + uz * uz;
                    const float flat = dx * dx + dy * dy + dz * dz;
                    float d2;
                    if (dot <= 0.0f || len == 0.0f) d2 = flat;
                    else if (dot >= len) d2 = flat - dot - dot + len;
                    else d2 = flat - dot * dot / len;
                    if (d2 < best) { best = d2; winner = (int32_t)j; }
                }
                dist.SetValue(q, best);
                idx.SetValue(q, winner);
            }
            Hop<HardEvent::S_V>();
            Sqrt(dist, dist, count);
            const uint64_t output = (uint64_t)batch * t.n + begin + done;
            if constexpr (sizeof(T) == 2) {
                Cast(narrow, dist, RoundMode::CAST_RINT, count);
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], narrow, {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            } else {
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], dist.template ReinterpretCast<T>(),
                            {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            }
            DataCopyPad(indices[output], idx, {1, count * 4, 0, 0, 0});
            stored = true;
        }
    }
}

template <typename T, uint32_t SEG>
__aicore__ inline void SmallScalarTwoPoints(GM_ADDR p, GM_ADDR s, GM_ADDR d,
    GM_ADDR i, uint32_t n)
{
    const uint32_t begin = 2 * GetBlockIdx();
    const uint32_t count = begin + 1 < n ? 2 : 1;
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    distances.SetGlobalBuffer((__gm__ T *)d);
    indices.SetGlobalBuffer((__gm__ int32_t *)i);
    LocalTensor<T> seg(TPosition::VECCALC, 0, 48);
    LocalTensor<T> raw(TPosition::VECCALC, 24576, 8);
    LocalTensor<float> dist(TPosition::VECCALC, 30720, 8);
    LocalTensor<int32_t> idx(TPosition::VECCALC, 32768, 8);
    DataCopyPad(seg, segments, {1, (uint32_t)(SEG * 6 * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(raw, points[begin * 3], {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    Hop<HardEvent::MTE2_S>();
    for (uint32_t q = 0; q < count; ++q) {
        const float px = (float)raw.GetValue(3 * q);
        const float py = (float)raw.GetValue(3 * q + 1);
        const float pz = (float)raw.GetValue(3 * q + 2);
        float best = kGuardDist;
        int32_t winner = 0;
        #pragma unroll
        for (uint32_t j = 0; j < SEG; ++j) {
            const uint32_t pos = 6 * j;
            const float ax = (float)seg.GetValue(pos);
            const float ay = (float)seg.GetValue(pos + 1);
            const float az = (float)seg.GetValue(pos + 2);
            const float ux = (float)seg.GetValue(pos + 3) - ax;
            const float uy = (float)seg.GetValue(pos + 4) - ay;
            const float uz = (float)seg.GetValue(pos + 5) - az;
            float dx = px - ax, dy = py - ay, dz = pz - az;
            const float dot = dx * ux + dy * uy + dz * uz;
            const float len = ux * ux + uy * uy + uz * uz;
            const float flat = dx * dx + dy * dy + dz * dz;
            float d2;
            if (dot <= 0.0f || len == 0.0f) d2 = flat;
            else if (dot >= len) d2 = flat - dot - dot + len;
            else d2 = flat - dot * dot / len;
            if (d2 < best) { best = d2; winner = (int32_t)j; }
        }
        dist.SetValue(q, best);
        idx.SetValue(q, winner);
    }
    Hop<HardEvent::S_V>();
    Sqrt(dist, dist, count);
    Hop<HardEvent::V_MTE3>();
    DataCopyPad(distances[begin], dist, {1, count * 4, 0, 0, 0});
    DataCopyPad(indices[begin], idx, {1, count * 4, 0, 0, 0});
}

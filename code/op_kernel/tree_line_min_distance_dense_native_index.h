#pragma once

template <typename T>
__aicore__ inline void DensePointNativeIndexTask(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                    const TreeLineMinDistanceTilingData& t)
{
    constexpr uint32_t tile = 3328;
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    points.SetGlobalBuffer((__gm__ T*)p);
    segments.SetGlobalBuffer((__gm__ T*)s);
    distances.SetGlobalBuffer((__gm__ T*)d);
    indices.SetGlobalBuffer((__gm__ int32_t*)i);
    LocalTensor<float> ax(TPosition::VECCALC, 0, 1024);
    LocalTensor<float> ay(TPosition::VECCALC, 4096, 1024);
    LocalTensor<float> az(TPosition::VECCALC, 8192, 1024);
    LocalTensor<float> ux(TPosition::VECCALC, 12288, 1024);
    LocalTensor<float> uy(TPosition::VECCALC, 16384, 1024);
    LocalTensor<float> uz(TPosition::VECCALC, 20480, 1024);
    LocalTensor<float> kx(TPosition::VECCALC, 24576, 1024);
    LocalTensor<float> ky(TPosition::VECCALC, 28672, 1024);
    LocalTensor<float> kz(TPosition::VECCALC, 32768, 1024);
    LocalTensor<float> dx(TPosition::VECCALC, 36864, tile);
    LocalTensor<float> dy(TPosition::VECCALC, 50176, tile);
    LocalTensor<float> dz(TPosition::VECCALC, 63488, tile);
    LocalTensor<float> ratio(TPosition::VECCALC, 77056, tile);
    LocalTensor<float> dsq(TPosition::VECCALC, 90368, tile);
    LocalTensor<float> best(TPosition::VECCALC, 103936, tile);
    LocalTensor<float> winner(TPosition::VECCALC, 117248, tile);
    LocalTensor<float> px(TPosition::VECCALC, 130816, tile);
    LocalTensor<float> py(TPosition::VECCALC, 144128, tile);
    LocalTensor<float> pz(TPosition::VECCALC, 157440, tile);
    LocalTensor<int32_t> pointOffsets(TPosition::VECCALC, 171008, tile);
    LocalTensor<int32_t> segmentOffsets(TPosition::VECCALC, 184320, 1024);
    LocalTensor<half> narrow(TPosition::VECCALC, 50176, tile);
    LocalTensor<uint8_t> keep(TPosition::VECCALC, 36864, 512);
    LocalTensor<T> raw(TPosition::VECCALC, 36864, tile * 3);
    LocalTensor<float> converted(TPosition::VECCALC, 77056, tile * 3);
    LocalTensor<float> indexPacket(TPosition::VECCALC,171008,4096);
    const uint32_t segmentsWide=AlignLanes(t.m);
    uint32_t cached = 0xFFFFFFFFu;
    bool stored = false;
    for (uint32_t task = GetBlockIdx(); task < t.b * t.chunks; task += GetBlockNum()) {
        const uint32_t batch = task / t.chunks;
        const uint32_t begin = (task % t.chunks) * t.chunkPoints;
        if (begin >= t.n) continue;
        const uint32_t owned = t.n - begin < t.chunkPoints ? t.n - begin : t.chunkPoints;
        if (stored) Hop<HardEvent::MTE3_V>();
        if (cached != batch) {
            Hop<HardEvent::V_S>();
            CreateVecIndex(dx,0.0f,segmentsWide);
            Muls(dx,dx,24.0f,segmentsWide);
            Cast(segmentOffsets,dx,RoundMode::CAST_RINT,segmentsWide);
            Hop<HardEvent::V_MTE2>();
            DataCopyPad(raw, segments[(uint64_t)batch * t.m * 6],
                        {1, (uint32_t)(t.m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            Hop<HardEvent::MTE2_V>();
            LocalTensor<float> src = raw.template ReinterpretCast<float>();
            if constexpr (sizeof(T) == 2) {
                src = converted;
                Cast(src, raw, RoundMode::CAST_NONE, t.m * 6);
            }
            const LocalTensor<uint32_t> off = segmentOffsets.ReinterpretCast<uint32_t>();
            Gather(ax, src, off, 0u, t.m);
            Gather(ay, src, off, 4u, t.m);
            Gather(az, src, off, 8u, t.m);
            Gather(ux, src, off, 12u, t.m);
            Gather(uy, src, off, 16u, t.m);
            Gather(uz, src, off, 20u, t.m);
            Sub(ux, ux, ax, t.m);
            Sub(uy, uy, ay, t.m);
            Sub(uz, uz, az, t.m);
            Duplicate(best, 1.0f, segmentsWide);
            Mul(best, ux, ux, t.m);
            MulAddDst(best, uy, uy, t.m);
            MulAddDst(best, uz, uz, t.m);
            CompareScalar(keep, best, 0.0f, CMPMODE::NE, segmentsWide);
            Select(best, keep, best, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, segmentsWide);
            Duplicate(dsq, -1.0f, segmentsWide);
            Div(winner, dsq, best, segmentsWide);
            Mul(kx, ux, winner, t.m);
            Mul(ky, uy, winner, t.m);
            Mul(kz, uz, winner, t.m);
            cached = batch;
        }

        for (uint32_t done = 0; done < owned; done += tile) {
            if (done != 0) Hop<HardEvent::MTE3_V>();
            const uint32_t count = owned - done < tile ? owned - done : tile;
            const uint32_t wide = AlignLanes(count);
            Hop<HardEvent::V_S>();
            CreateVecIndex(dx,0.0f,tile);
            Muls(dx,dx,12.0f,tile);
            Cast(pointOffsets,dx,RoundMode::CAST_RINT,tile);
            Hop<HardEvent::V_MTE2>();
            DataCopyPad(raw, points[((uint64_t)batch * t.n + begin + done) * 3],
                        {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            Hop<HardEvent::MTE2_V>();
            LocalTensor<float> src = raw.template ReinterpretCast<float>();
            if constexpr (sizeof(T) == 2) {
                src = converted;
                Cast(src, raw, RoundMode::CAST_NONE, count * 3);
            }
            if (wide != count) {
                Duplicate(px, 0.0f, wide);
                Duplicate(py, 0.0f, wide);
                Duplicate(pz, 0.0f, wide);
            }
            const LocalTensor<uint32_t> off = pointOffsets.ReinterpretCast<uint32_t>();
            Gather(px, src, off, 0u, count);
            Gather(py, src, off, 4u, count);
            Gather(pz, src, off, 8u, count);
            Muls(px, px, -1.0f, wide);
            Muls(py, py, -1.0f, wide);
            Muls(pz, pz, -1.0f, wide);
            Duplicate(winner, 0.0f, wide);
            Hop<HardEvent::V_S>();
            const uint32_t packetCount=segmentsWide*4;
            CreateVecIndex(indexPacket,0.0f,packetCount);
            Muls(indexPacket,indexPacket,0.25f,packetCount);
            auto packetInt=indexPacket.ReinterpretCast<int32_t>();
            Cast(packetInt,indexPacket,RoundMode::CAST_FLOOR,packetCount);
            Cast(indexPacket,packetInt,RoundMode::CAST_NONE,packetCount);
            Hop<HardEvent::V_S>();
            const uint8_t repeats = wide / 64;
            const UnaryRepeatParams unary(1, 1, 8, 8);
            const BinaryRepeatParams binary(1, 1, 1, 8, 8, 8);
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
            for (int32_t j = 0; j < (int32_t)t.m; ++j) {
                const float sax = ax.GetValue(j), say = ay.GetValue(j), saz = az.GetValue(j);
                const float sux = ux.GetValue(j), suy = uy.GetValue(j), suz = uz.GetValue(j);
                const float skx = kx.GetValue(j), sky = ky.GetValue(j), skz = kz.GetValue(j);
                Adds<float, false>(dx, px, sax, MASK_PLACEHOLDER, repeats, unary);
                Adds<float, false>(dy, py, say, MASK_PLACEHOLDER, repeats, unary);
                Adds<float, false>(dz, pz, saz, MASK_PLACEHOLDER, repeats, unary);
                Muls<float, false>(ratio, dx, skx, MASK_PLACEHOLDER, repeats, unary);
                Axpy<float, float, false>(ratio, dy, sky, MASK_PLACEHOLDER, repeats, unary);
                Axpy<float, float, false>(ratio, dz, skz, MASK_PLACEHOLDER, repeats, unary);
                Maxs<float, false>(ratio, ratio, 0.0f, MASK_PLACEHOLDER, repeats, unary);
                Mins<float, false>(ratio, ratio, 1.0f, MASK_PLACEHOLDER, repeats, unary);
                Axpy<float, float, false>(dx, ratio, sux, MASK_PLACEHOLDER, repeats, unary);
                Axpy<float, float, false>(dy, ratio, suy, MASK_PLACEHOLDER, repeats, unary);
                Axpy<float, float, false>(dz, ratio, suz, MASK_PLACEHOLDER, repeats, unary);
                if (j == 0) {
                    Mul<float, false>(best, dx, dx, MASK_PLACEHOLDER, repeats, binary);
                    MulAddDst<float, float, false>(best, dy, dy, MASK_PLACEHOLDER, repeats, binary);
                    MulAddDst<float, float, false>(best, dz, dz, MASK_PLACEHOLDER, repeats, binary);
                } else {
                    Mul<float, false>(dsq, dx, dx, MASK_PLACEHOLDER, repeats, binary);
                    MulAddDst<float, float, false>(dsq, dy, dy, MASK_PLACEHOLDER, repeats, binary);
                    MulAddDst<float, float, false>(dsq, dz, dz, MASK_PLACEHOLDER, repeats, binary);
                    Compare<float, uint8_t, false>(keep, dsq, best, CMPMODE::GE, MASK_PLACEHOLDER, repeats, binary);
                    Min<float, false>(best, best, dsq, MASK_PLACEHOLDER, repeats, binary);
                    SetCmpMask(indexPacket[(uint32_t)j*4]);
                    PipeBarrier<PIPE_V>();
                    Select<float,uint8_t>(winner,keep,winner,repeats,binary);
                }
            }
            ResetMask();
            Sqrt(best, best, count);
            LocalTensor<int32_t> outIndex = winner.ReinterpretCast<int32_t>();
            Cast(outIndex, winner, RoundMode::CAST_RINT, count);
            const uint64_t output = (uint64_t)batch * t.n + begin + done;
            if constexpr (sizeof(T) == 2) {
                PipeBarrier<PIPE_V>();
                Cast(narrow, best, RoundMode::CAST_RINT, count);
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], narrow, {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            } else {
                Hop<HardEvent::V_MTE3>();
                DataCopyPad(distances[output], best.template ReinterpretCast<T>(),
                            {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
            }
            DataCopyPad(indices[output], outIndex, {1, count * 4, 0, 0, 0});
            stored = true;
        }
    }
}

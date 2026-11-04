#pragma once
#include "tree_line_min_distance_hoist_layout.h"

template <typename T, uint32_t MODE, bool LARGE = false, bool SHORT = false, bool WIDE = false, bool SPLIT = false, bool BANK = false, bool STRIDED = false, bool AXIS = false, bool HOIST = false>
class KernelPointGroups {
    static constexpr uint32_t oDx = HOIST ? HoistedPointGroupLayout::oDx : (WIDE && BANK && !STRIDED) ? 9280 : STRIDED ? 9216 : WIDE ? 9224 : LARGE ? 9224 : ::oDx;
    static constexpr uint32_t oDy = HOIST ? HoistedPointGroupLayout::oDy : (WIDE && BANK && !STRIDED) ? 17472 : STRIDED ? 17472 : WIDE ? (BANK ? 17480 : 17424) : LARGE ? 15376 : ::oDy;
    static constexpr uint32_t oDz = HOIST ? HoistedPointGroupLayout::oDz : (WIDE && BANK && !STRIDED) ? 25664 : STRIDED ? 25664 : WIDE ? (BANK ? 25672 : 25624) : LARGE ? 21528 : ::oDz;
    static constexpr uint32_t oTt = HOIST ? HoistedPointGroupLayout::oTt : (WIDE && BANK && !STRIDED) ? 33952 : STRIDED ? 33824 : WIDE ? (BANK ? 33864 : 33824) : LARGE ? 27680 : ::oTt;
    static constexpr uint32_t oDsq = HOIST ? HoistedPointGroupLayout::oDsq : (WIDE && BANK && !STRIDED) ? 9280 : STRIDED ? 9216 : WIDE ? 9224 : LARGE ? 9224 : ::oDsq;
    static constexpr uint32_t oSegRaw = LARGE ? 9224 : ::oSegRaw;
    static constexpr uint32_t oSegFlt = LARGE ? 15376 : ::oSegFlt;
    static constexpr uint32_t oNegOne = LARGE ? 21528 : ::oNegOne;
    static constexpr uint32_t oLen = LARGE ? 22552 : ::oLen;
    static constexpr uint32_t oInvLen = LARGE ? 23576 : ::oInvLen;
    static constexpr uint32_t oSegOff = HOIST ? HoistedPointGroupLayout::oSegOff : (WIDE && BANK && !STRIDED) ? 42144 : STRIDED ? 42048 : WIDE ? (BANK ? 42056 : 42016) : LARGE ? 33824 : ::oSegOff;
    static constexpr uint32_t oPtRaw = HOIST ? HoistedPointGroupLayout::oPtRaw : (WIDE && BANK && !STRIDED) ? 43168 : STRIDED ? 43072 : WIDE ? (BANK ? 43080 : 43040) : LARGE ? 34848 : ::oPtRaw;
    static constexpr uint32_t oPtFlt = HOIST ? HoistedPointGroupLayout::oPtFlt : (WIDE && BANK && !STRIDED) ? 43552 : STRIDED ? 43456 : WIDE ? (BANK ? 43464 : 43424) : LARGE ? 36384 : ::oPtFlt;
    static constexpr uint32_t oPtOff = HOIST ? HoistedPointGroupLayout::oPtOff : (WIDE && BANK && !STRIDED) ? 43936 : STRIDED ? 43840 : WIDE ? (BANK ? 43848 : 43808) : LARGE ? 37920 : ::oPtOff;
    static constexpr uint32_t oDiv8 = HOIST ? HoistedPointGroupLayout::oDiv8 : (WIDE && BANK && !STRIDED) ? 44064 : STRIDED ? 43968 : WIDE ? (BANK ? 43976 : 43936) : LARGE ? 38432 : ::oDiv8;
    static constexpr uint32_t oPx = HOIST ? HoistedPointGroupLayout::oPx : (WIDE && BANK && !STRIDED) ? 44064 : STRIDED ? 43968 : WIDE ? (BANK ? 43976 : 43936) : LARGE ? 39200 : ::oPx;
    static constexpr uint32_t oPy = HOIST ? HoistedPointGroupLayout::oPy : (WIDE && BANK && !STRIDED) ? 44192 : STRIDED ? 44096 : WIDE ? (BANK ? 44104 : 44064) : LARGE ? 39328 : ::oPy;
    static constexpr uint32_t oPz = HOIST ? HoistedPointGroupLayout::oPz : (WIDE && BANK && !STRIDED) ? 44320 : STRIDED ? 44224 : WIDE ? (BANK ? 44232 : 44192) : LARGE ? 39456 : ::oPz;
    static constexpr uint32_t oPxv = HOIST ? HoistedPointGroupLayout::oPxv : (WIDE && BANK && !STRIDED) ? 44448 : STRIDED ? 44352 : WIDE ? (BANK ? 44360 : 44320) : LARGE ? 39584 : ::oPxv;
    static constexpr uint32_t oPyv = HOIST ? HoistedPointGroupLayout::oPyv : (WIDE && BANK && !STRIDED) ? 44448 : STRIDED ? 44864 : WIDE ? (BANK ? 44360 : 44320) : LARGE ? 40352 : ::oPyv;
    static constexpr uint32_t oPzv = HOIST ? HoistedPointGroupLayout::oPzv : (WIDE && BANK && !STRIDED) ? 44448 : STRIDED ? 45376 : WIDE ? (BANK ? 44360 : 44320) : LARGE ? 41120 : ::oPzv;
    static constexpr uint32_t oPair = HOIST ? HoistedPointGroupLayout::oPair : (WIDE && BANK && !STRIDED) ? 45472 : STRIDED ? 45888 : WIDE ? (BANK ? 45384 : 45344) : LARGE ? 41888 : ::oPair;
    static constexpr uint32_t oCmin = HOIST ? HoistedPointGroupLayout::oCmin : (WIDE && BANK && !STRIDED) ? 45728 : STRIDED ? 46144 : WIDE ? (BANK ? 45640 : 45600) : LARGE ? 42144 : ::oCmin;
    static constexpr uint32_t oCidx = HOIST ? HoistedPointGroupLayout::oCidx : (WIDE && BANK && !STRIDED) ? 45856 : STRIDED ? 46272 : WIDE ? (BANK ? 45768 : 45728) : LARGE ? 42272 : ::oCidx;
    static constexpr uint32_t oBest = HOIST ? HoistedPointGroupLayout::oBest : (WIDE && BANK && !STRIDED) ? 45984 : STRIDED ? 46400 : WIDE ? (BANK ? 45896 : 45856) : LARGE ? 42400 : ::oBest;
    static constexpr uint32_t oFidx = HOIST ? HoistedPointGroupLayout::oFidx : (WIDE && BANK && !STRIDED) ? 46112 : STRIDED ? 46528 : WIDE ? (BANK ? 46024 : 45984) : LARGE ? 42912 : ::oFidx;
    static constexpr uint32_t oRes = HOIST ? HoistedPointGroupLayout::oRes : (WIDE && BANK && !STRIDED) ? 46240 : STRIDED ? 46656 : WIDE ? (BANK ? 46152 : 46112) : LARGE ? 43424 : ::oRes;
    static constexpr uint32_t oRedW = HOIST ? HoistedPointGroupLayout::oRedW : (WIDE && BANK && !STRIDED) ? 46240 : STRIDED ? 46656 : WIDE ? (BANK ? 46152 : 46112) : LARGE ? 44448 : ::oRedW;
    static constexpr uint32_t oDist = HOIST ? HoistedPointGroupLayout::oDist : (WIDE && BANK && !STRIDED) ? 46240 : STRIDED ? 46656 : WIDE ? (BANK ? 46152 : 46112) : LARGE ? 44960 : ::oDist;
    static constexpr uint32_t oIdx = HOIST ? HoistedPointGroupLayout::oIdx : (WIDE && BANK && !STRIDED) ? 46368 : STRIDED ? 46784 : WIDE ? (BANK ? 46280 : 46240) : LARGE ? 45472 : ::oIdx;
    static constexpr uint32_t oHalf = HOIST ? HoistedPointGroupLayout::oHalf : (WIDE && BANK && !STRIDED) ? 46496 : STRIDED ? 46912 : WIDE ? (BANK ? 46408 : 46368) : LARGE ? 45984 : ::oHalf;
    static constexpr uint32_t oMask = HOIST ? HoistedPointGroupLayout::oMask : (WIDE && BANK && !STRIDED) ? 46560 : STRIDED ? 46976 : WIDE ? (BANK ? 46472 : 46432) : LARGE ? 46240 : ::oMask;
    static constexpr uint32_t oArena = HOIST ? HoistedPointGroupLayout::oArena : (WIDE && BANK && !STRIDED) ? 46624 : STRIDED ? 47040 : WIDE ? (BANK ? 46536 : 46496) : LARGE ? 46304 : ::oArena;
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    TBuf<TPosition::VECCALC> arenaBuf;
    LocalTensor<float> arena;
    static constexpr uint32_t mode = MODE;
    uint32_t b, n, m, chunks, chunkPoints, tilePoints, group;
    uint32_t mAligned, mRepeat;
    bool orderedAxis = false;
    float axisLow = 0.0f, axisHigh = 0.0f, axisZ = 0.0f;

public:
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                const TreeLineMinDistanceTilingData &t, TPipe *pipe)
    {
        b = t.b;
        n = t.n;
        m = t.m;
        chunks = t.chunks;
        chunkPoints = t.chunkPoints;
        tilePoints = t.tilePoints;
        group = t.group;
        mAligned = SHORT ? kLanes : AlignLanes(m);
        mRepeat = mAligned / kLanes;
        points.SetGlobalBuffer((__gm__ T *)p);
        segments.SetGlobalBuffer((__gm__ T *)s);
        distances.SetGlobalBuffer((__gm__ T *)d);
        indices.SetGlobalBuffer((__gm__ int32_t *)i);
        pipe->InitBuffer(arenaBuf, oArena * sizeof(float));
        arena = arenaBuf.Get<float>();
        if (mode == 2) {
            return;
        }
        // Lane-invariant address tables and constants, built once per kernel.
        LocalTensor<float> scratch = arena[oLen];
        BuildScale(scratch, arena[oSegOff], 24.0f, mAligned);
        if (mode != 0) {
            BuildScale(scratch, arena[oPtOff], 12.0f, AlignLanes(tilePoints));
        }
        Duplicate(arena[oNegOne], -1.0f, mAligned);
    }

    __aicore__ inline void ProcessOne()
    {
        LoadSegments(0);
        LoadPoints(0, 0, 1);
        Hop<HardEvent::MTE2_V>();
        DeriveSegments();
        Hop<HardEvent::MTE2_S>();
        SegmentLanes(0, 0, 1);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tasks = b * chunks;
        const uint32_t stride = GetBlockNum();
        uint32_t cached = 0xFFFFFFFFu;
        bool stored = false;
        for (uint32_t task = GetBlockIdx(); task < tasks; task += stride) {
            const uint32_t batch = task / chunks;
            const uint32_t begin = (task - batch * chunks) * chunkPoints;
            if (begin >= n) {
                continue;
            }
            uint32_t owned = n - begin;
            if (owned > chunkPoints) {
                owned = chunkPoints;
            }
            const bool fresh = cached != batch;
            // Both reads are issued before the wait so the two global memory
            // latencies overlap instead of stacking.
            if (fresh) {
                LoadSegments(batch);
            }
            const uint32_t head = owned < tilePoints ? owned : tilePoints;
            LoadPoints(batch, begin, head);
            if (fresh) {
                cached = batch;
                if constexpr (AXIS) {
                    Hop<HardEvent::MTE2_S>();
                    orderedAxis = DetectOrderedAxis();
                }
                if (mode != 2) {
                    if constexpr (AXIS) {
                        if (!orderedAxis) { Hop<HardEvent::MTE2_V>(); DeriveSegments(); }
                    } else { Hop<HardEvent::MTE2_V>(); DeriveSegments(); }
                }
            }
            if (mode == 1) {
                // Segment parameters are consumed as scalars here.
                if (fresh) {
                    Hop<HardEvent::V_S>();
                } else {
                    Hop<HardEvent::MTE2_V>();
                }
            } else if (mode == 3) {
                if (!fresh) {
                    Hop<HardEvent::MTE2_V>();
                }
            } else {
                Hop<HardEvent::MTE2_S>();
            }
            for (uint32_t done = 0; done < owned; done += tilePoints) {
                uint32_t count = owned - done;
                if (count > tilePoints) {
                    count = tilePoints;
                }
                // The previous tile's stores must retire before its result
                // buffers are rewritten.
                if (stored) {
                    Hop<HardEvent::MTE3_V>();
                }
                stored = true;
                if (done != 0) {
                    if (mode != 0) {
                        Hop<HardEvent::V_MTE2>();
                    }
                    LoadPoints(batch, begin + done, count);
                    if (mode == 0) {
                        Hop<HardEvent::MTE2_S>();
                    } else {
                        Hop<HardEvent::MTE2_V>();
                    }
                }
                if (mode == 3) {
                    if constexpr (AXIS) {
                        if (orderedAxis) OrderedAxisLanes(batch, begin + done, count);
                        else BlockedLanes(batch, begin + done, count);
                    } else BlockedLanes(batch, begin + done, count);
                } else if (mode == 2) {
                    ScalarLanes(batch, begin + done, count);
                } else if (mode == 1) {
                    PointLanes(batch, begin + done, count);
                } else {
                    SegmentLanes(batch, begin + done, count);
                }
            }
        }
    }

private:

    __aicore__ inline bool DetectOrderedAxis()
    {
        LocalTensor<T> raw = arena[oSegRaw].template ReinterpretCast<T>();
        const float x0 = (float)raw.GetValue(0), x1 = (float)raw.GetValue(3);
        const float z0 = (float)raw.GetValue(2);
        float previous = (float)raw.GetValue(1);
        for (uint32_t j = 0; j < m; ++j) {
            const uint32_t k = 6 * j;
            const float y = (float)raw.GetValue(k + 1);
            if ((float)raw.GetValue(k) != x0 || (float)raw.GetValue(k + 3) != x1 ||
                (float)raw.GetValue(k + 2) != z0 || (float)raw.GetValue(k + 5) != z0 ||
                (float)raw.GetValue(k + 4) != y || y < previous) return false;
            previous = y;
        }
        axisLow = x0 < x1 ? x0 : x1;
        axisHigh = x0 < x1 ? x1 : x0;
        axisZ = z0;
        LocalTensor<float> y = arena[oAy];
        for (uint32_t j = 0; j < m; ++j) y.SetValue(j, (float)raw.GetValue(6 * j + 1));
        return true;
    }

    __aicore__ inline void AxisGatherY(const LocalTensor<float> &dst,
        const LocalTensor<float> &index, uint32_t count)
    {
        LocalTensor<float> bytes = arena[oDx + 1792];
        LocalTensor<int32_t> offsets = arena[oDx + 1920].ReinterpretCast<int32_t>();
        Muls(bytes, index, 4.0f, count);
        Cast(offsets, bytes, RoundMode::CAST_RINT, count);
        Gather(dst, arena[oAy], offsets.ReinterpretCast<uint32_t>(), 0u, count);
    }

    __aicore__ inline void AxisLowerBound(const LocalTensor<float> &low,
        const LocalTensor<float> &target, uint32_t count)
    {
        LocalTensor<float> high = arena[oDx + 384], mid = arena[oDx + 512];
        LocalTensor<float> ym = arena[oDx + 640], plus = arena[oDx + 768];
        LocalTensor<int32_t> rounded = arena[oDx + 896].ReinterpretCast<int32_t>();
        LocalTensor<uint8_t> keep = arena[oMask].ReinterpretCast<uint8_t>();
        Duplicate(low, 0.0f, count);
        Duplicate(high, (float)(int32_t)m, count);
        uint32_t remaining = m + 1;
        while (remaining > 1) {
            Add(mid, low, high, count);
            Muls(mid, mid, 0.5f, count);
            Cast(rounded, mid, RoundMode::CAST_FLOOR, count);
            Cast(mid, rounded, RoundMode::CAST_NONE, count);
            Mins(mid, mid, (float)((int32_t)m - 1), count);
            AxisGatherY(ym, mid, count);
            Compare(keep, ym, target, CMPMODE::LT, count);
            Adds(plus, mid, 1.0f, count);
            Select(low, keep, plus, low, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            Select(high, keep, high, mid, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            remaining = (remaining + 1) / 2;
        }
    }

    __aicore__ inline void OrderedAxisLanes(uint32_t batch, uint32_t start, uint32_t count)
    {
        const uint32_t wide = AlignLanes(count);
        Hop<HardEvent::S_V>();
        Hop<HardEvent::MTE2_V>();
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        LocalTensor<float> src = arena[oPtRaw];
        if constexpr (sizeof(T) == 2) {
            src = arena[oPtFlt];
            Cast(src, raw, RoundMode::CAST_NONE, count * 3);
        }
        LocalTensor<float> px = arena[oDx], py = arena[oDx + 128], pz = arena[oDx + 256];
        LocalTensor<float> next = arena[oDx + 1024], left = arena[oDx + 1152];
        LocalTensor<float> right = arena[oDx + 1280], leftY = arena[oDx + 1408];
        LocalTensor<float> rightY = arena[oDx + 1536], nearestY = arena[oDx + 1664];
        LocalTensor<uint32_t> pointOffsets = arena[oPtOff].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> keep = arena[oMask].ReinterpretCast<uint8_t>();
        Duplicate(px, 0.0f, wide);
        Duplicate(py, 0.0f, wide);
        Duplicate(pz, 0.0f, wide);
        Gather(px, src, pointOffsets, 0u, count);
        Gather(py, src, pointOffsets, 4u, count);
        Gather(pz, src, pointOffsets, 8u, count);
        AxisLowerBound(next, py, wide);
        Adds(left, next, -1.0f, wide);
        Maxs(left, left, 0.0f, wide);
        Mins(right, next, (float)((int32_t)m - 1), wide);
        AxisGatherY(leftY, left, wide);
        AxisGatherY(rightY, right, wide);
        Sub(left, py, leftY, wide);
        Sub(right, rightY, py, wide);
        Compare(keep, left, right, CMPMODE::LE, wide);
        Select(nearestY, keep, leftY, rightY, SELMODE::VSEL_TENSOR_TENSOR_MODE, wide);
        AxisLowerBound(next, nearestY, wide);
        LocalTensor<int32_t> idx = arena[oIdx].ReinterpretCast<int32_t>();
        Cast(idx, next, RoundMode::CAST_RINT, wide);
        Maxs(left, px, axisLow, wide);
        Mins(left, left, axisHigh, wide);
        Sub(px, px, left, wide);
        Sub(py, py, nearestY, wide);
        Adds(pz, pz, -axisZ, wide);
        Mul(px, px, px, wide);
        Mul(py, py, py, wide);
        Mul(pz, pz, pz, wide);
        Add(px, px, py, wide);
        Add(px, px, pz, wide);
        LocalTensor<float> dist = arena[oDist];
        Sqrt(dist, px, count);
        Emit(batch, start, count, dist, idx);
    }

    // Fills an int32 address table with step * index.
    __aicore__ inline void BuildScale(const LocalTensor<float> &scratch,
                                      const LocalTensor<float> &table, float step, uint32_t span)
    {
        CreateVecIndex(scratch, 0.0f, span);
        Muls(scratch, scratch, step, span);
        Cast(table.ReinterpretCast<int32_t>(), scratch, RoundMode::CAST_RINT, span);
    }

    __aicore__ inline void LoadSegments(uint32_t batch)
    {
        LocalTensor<T> raw = arena[oSegRaw].template ReinterpretCast<T>();
        DataCopyPad(raw, segments[(uint64_t)batch * m * 6],
                    {1, (uint32_t)(m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    }

    __aicore__ inline void LoadPoints(uint32_t batch, uint32_t start, uint32_t count)
    {
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        DataCopyPad(raw, points[((uint64_t)batch * n + start) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    }

    // Splits the interleaved endpoints into lane arrays and derives the
    // direction, the guarded inverse length and the fused projection
    // coefficients once for the whole batch.
    __aicore__ inline void DeriveSegments()
    {
        LocalTensor<T> raw = arena[oSegRaw].template ReinterpretCast<T>();
        LocalTensor<float> ax = arena[oAx], ay = arena[oAy], az = arena[oAz];
        LocalTensor<float> ux = arena[oUx], uy = arena[oUy], uz = arena[oUz];
        LocalTensor<float> kx = arena[oKx], ky = arena[oKy], kz = arena[oKz];
        LocalTensor<float> len = arena[oLen], invLen = arena[oInvLen];
        LocalTensor<uint32_t> off = arena[oSegOff].ReinterpretCast<uint32_t>();

        if (mAligned != m) {
            // Written before the gather and never touched by the count-M
            // arithmetic below, so the tail keeps losing every reduction and
            // the segment loops can run on whole 256B repeats.
            Duplicate(ax, kGuardCoord, mAligned);
            Duplicate(ay, kGuardCoord, mAligned);
            Duplicate(az, kGuardCoord, mAligned);
            Duplicate(ux, 0.0f, mAligned);
            Duplicate(uy, 0.0f, mAligned);
            Duplicate(uz, 0.0f, mAligned);
            Duplicate(kx, 0.0f, mAligned);
            Duplicate(ky, 0.0f, mAligned);
            Duplicate(kz, 0.0f, mAligned);
        }
        LocalTensor<float> src = arena[oSegRaw];
        if constexpr (sizeof(T) == 2) {
            src = arena[oSegFlt];
            Cast(src, raw, RoundMode::CAST_NONE, m * 6);
            PipeBarrier<PIPE_V>();
        }
        Gather(ax, src, off, 0u, m);
        Gather(ay, src, off, 4u, m);
        Gather(az, src, off, 8u, m);
        Gather(ux, src, off, 12u, m);
        Gather(uy, src, off, 16u, m);
        Gather(uz, src, off, 20u, m);
        Sub(ux, ux, ax, m);
        Sub(uy, uy, ay, m);
        Sub(uz, uz, az, m);
        Mul(len, ux, ux, m);
        MulAddDst(len, uy, uy, m);
        MulAddDst(len, uz, uz, m);
        // Exact zero-length test on the quantised value; no division by L = 0.
        LocalTensor<uint8_t> keep = arena[oMask].ReinterpretCast<uint8_t>();
        CompareScalar(keep, len, 0.0f, CMPMODE::NE, mAligned);
        Select(len, keep, len, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mAligned);
        Div(invLen, arena[oNegOne], len, mAligned);
        // Degenerate segments keep u = 0 exactly, so k = 0 and t clamps to 0.
        Mul(kx, ux, invLen, m);
        Mul(ky, uy, invLen, m);
        Mul(kz, uz, invLen, m);
    }

    // Blocked lanes: one call spans a whole group of points against the same
    // 64-segment window, so the fixed cost of issuing a vector instruction is
    // paid once per group instead of once per point.
    __aicore__ inline void BlockedLanes(uint32_t batch, uint32_t start, uint32_t count)
    {
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        LocalTensor<float> src = arena[oPtRaw];
        if constexpr (sizeof(T) == 2) {
            src = arena[oPtFlt];
            Cast(src, raw, RoundMode::CAST_NONE, count * 3);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> ax = arena[oAx], ay = arena[oAy], az = arena[oAz];
        LocalTensor<float> ux = arena[oUx], uy = arena[oUy], uz = arena[oUz];
        LocalTensor<float> kx = arena[oKx], ky = arena[oKy], kz = arena[oKz];
        LocalTensor<float> dx = arena[oDx], dy = arena[oDy], dz = arena[oDz];
        LocalTensor<float> tt = arena[oTt], dsq = arena[oDsq];
        LocalTensor<float> px = arena[oPx], py = arena[oPy], pz = arena[oPz];
        LocalTensor<float> pxv = arena[oPxv], pyv = arena[oPyv], pzv = arena[oPzv];
        LocalTensor<float> pair = arena[oPair], cmin = arena[oCmin], cidx = arena[oCidx];
        LocalTensor<float> best = arena[oBest], bidx = arena[oFidx];
        LocalTensor<uint32_t> ptOff = arena[oPtOff].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> keep = arena[oMask].ReinterpretCast<uint8_t>();
        LocalTensor<float> dist = arena[oDist];
        LocalTensor<int32_t> out = arena[oIdx].ReinterpretCast<int32_t>();

        // One point per 32B block of the expanded coordinate, so the blocked
        // calls below read it with a zero block stride.
        const BinaryRepeatParams spread(1, 1, 0, STRIDED ? 16 : 8, 0, 1);
        const BinaryRepeatParams fold(1, 1, 1, STRIDED ? 16 : 8, STRIDED ? 16 : 8, 0);
        const BinaryRepeatParams plain(1, 1, 1, STRIDED ? 16 : 8, STRIDED ? 16 : 8, STRIDED ? 16 : 8);
        const UnaryRepeatParams clamp(1, 1, STRIDED ? 16 : 8, STRIDED ? 16 : 8);

        for (uint32_t done = 0; done < count; done += group) {
            uint32_t span = count - done;
            if (span > group) {
                span = group;
            }
            const uint8_t reps = (uint8_t)span;
            const uint32_t carry = LARGE ? AlignLanes(span) : kLanes;
            const uint16_t pairRepeats = LARGE ? (uint16_t)(AlignLanes(2 * span) / kLanes) : 2;
            const uint32_t base = 12u * done;
            Gather(px, src, ptOff, base, span);
            Gather(py, src, ptOff, base + 4u, span);
            Gather(pz, src, ptOff, base + 8u, span);
            if constexpr (!WIDE || STRIDED || HOIST) {
            Brcb(pxv, px, (uint8_t)((span + 7) / 8), {1, 8});
            Brcb(pyv, py, (uint8_t)((span + 7) / 8), {1, 8});
            Brcb(pzv, pz, (uint8_t)((span + 7) / 8), {1, 8});
            }
            if constexpr (!SHORT) {
                Duplicate(best, kGuardDist, carry);
                Duplicate(bidx, 0.0f, carry);
            }

            for (uint32_t r = 0; r < (SHORT ? 1u : mRepeat); ++r) {
                const uint32_t seg = r * kLanes;
                SetVectorMask<float>((int32_t)kLanes);
                if constexpr (WIDE && !STRIDED && !HOIST) {
                    PipeBarrier<PIPE_V>();
                    Brcb(pxv, px, (uint8_t)((span + 7) / 8), {1, 8});
                }
                Sub<float, false>(dx, ax[seg], pxv, MASK_PLACEHOLDER, reps, spread);
                if constexpr (WIDE && !STRIDED && !HOIST) {
                    PipeBarrier<PIPE_V>();
                    Brcb(pxv, py, (uint8_t)((span + 7) / 8), {1, 8});
                }
                Sub<float, false>(dy, ay[seg], pyv, MASK_PLACEHOLDER, reps, spread);
                if constexpr (WIDE && !STRIDED && !HOIST) {
                    PipeBarrier<PIPE_V>();
                    Brcb(pxv, pz, (uint8_t)((span + 7) / 8), {1, 8});
                }
                Sub<float, false>(dz, az[seg], pzv, MASK_PLACEHOLDER, reps, spread);
                Mul<float, false>(tt, dx, kx[seg], MASK_PLACEHOLDER, reps, fold);
                MulAddDst<float, float, false>(tt, dy, ky[seg], MASK_PLACEHOLDER, reps, fold);
                MulAddDst<float, float, false>(tt, dz, kz[seg], MASK_PLACEHOLDER, reps, fold);
                Maxs<float, false>(tt, tt, 0.0f, MASK_PLACEHOLDER, reps, clamp);
                Mins<float, false>(tt, tt, 1.0f, MASK_PLACEHOLDER, reps, clamp);
                MulAddDst<float, float, false>(dx, tt, ux[seg], MASK_PLACEHOLDER, reps, fold);
                MulAddDst<float, float, false>(dy, tt, uy[seg], MASK_PLACEHOLDER, reps, fold);
                MulAddDst<float, float, false>(dz, tt, uz[seg], MASK_PLACEHOLDER, reps, fold);
                Mul<float, false>(dsq, dx, dx, MASK_PLACEHOLDER, reps, plain);
                if constexpr (SPLIT) {
                    Mul<float, false>(dy, dy, dy, MASK_PLACEHOLDER, reps, plain);
                    Mul<float, false>(dz, dz, dz, MASK_PLACEHOLDER, reps, plain);
                    Add<float, false>(dsq, dsq, dy, MASK_PLACEHOLDER, reps, plain);
                    Add<float, false>(dsq, dsq, dz, MASK_PLACEHOLDER, reps, plain);
                } else {
                    MulAddDst<float, float, false>(dsq, dy, dy, MASK_PLACEHOLDER, reps, plain);
                    MulAddDst<float, float, false>(dsq, dz, dz, MASK_PLACEHOLDER, reps, plain);
                }
                ResetMask();
                // Each repeat holds one point against this segment window.
                WholeReduceMin(pair, dsq, (int32_t)kLanes, (int32_t)reps, 1, 1, STRIDED ? 16 : 8);
                uint64_t kept = 0;
                GatherMask(cmin, pair, (uint8_t)1, false, 0, {1, pairRepeats, 8, 8}, kept);
                GatherMask(cidx, pair, (uint8_t)2, false, 0, {1, pairRepeats, 8, 8}, kept);
                if constexpr (!SHORT) {
                Cast(pair, cidx.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, carry);
                Adds(pair, pair, (float)(int32_t)seg, carry);
                // GE keeps the incumbent on ties, and the window reduction
                // already returns its own first minimum, so the smallest
                // segment index always wins.
                Compare(keep, cmin, best, CMPMODE::GE, carry);
                Min(best, best, cmin, carry);
                Select(bidx, keep, bidx, pair, SELMODE::VSEL_TENSOR_TENSOR_MODE, carry);
                }
            }
            if constexpr (SHORT) {
                Sqrt(dist, cmin, span);
                Emit(batch, start + done, span, dist, cidx.ReinterpretCast<int32_t>());
            } else {
                Sqrt(dist, best, span);
                Cast(out, bidx, RoundMode::CAST_RINT, span);
                Emit(batch, start + done, span, dist, out);
            }
            if (done + group < count) {
                Hop<HardEvent::MTE3_V>();
            }
        }
    }

    // Scalar lanes: the pair count one core owns is small enough that the
    // fixed cost of a vector instruction outweighs the whole reduction.
    __aicore__ inline void ScalarLanes(uint32_t batch, uint32_t start, uint32_t count)
    {
        LocalTensor<T> seg = arena[oSegRaw].template ReinterpretCast<T>();
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        LocalTensor<float> dist = arena[oDist];
        LocalTensor<float> winners = arena[oBest];
        LocalTensor<int32_t> idx = arena[oIdx].ReinterpretCast<int32_t>();
        const int32_t total = (int32_t)m;
        for (uint32_t p = 0; p < count; ++p) {
            const float px = (float)raw.GetValue(3 * p);
            const float py = (float)raw.GetValue(3 * p + 1);
            const float pz = (float)raw.GetValue(3 * p + 2);
            float best = kGuardDist;
            int32_t winner = 0;
            for (int32_t j = 0; j < total; ++j) {
                const uint32_t pos = 6u * (uint32_t)j;
                const float ax = (float)seg.GetValue(pos);
                const float ay = (float)seg.GetValue(pos + 1);
                const float az = (float)seg.GetValue(pos + 2);
                const float ux = (float)seg.GetValue(pos + 3) - ax;
                const float uy = (float)seg.GetValue(pos + 4) - ay;
                const float uz = (float)seg.GetValue(pos + 5) - az;
                const float qx = px - ax;
                const float qy = py - ay;
                const float qz = pz - az;
                const float dot = qx * ux + qy * uy + qz * uz;
                const float len = ux * ux + uy * uy + uz * uz;
                const float flat = qx * qx + qy * qy + qz * qz;
                float d2;
                if (dot <= 0.0f || len == 0.0f) {
                    d2 = flat;                       // closest point is the start
                } else if (dot >= len) {
                    d2 = flat - dot - dot + len;     // closest point is the end
                } else {
                    d2 = flat - dot * dot / len;     // projection falls inside
                }
                if (d2 < best) {
                    best = d2;
                    winner = j;
                }
            }
            dist.SetValue(p, best);
            winners.SetValue(p, (float)winner);
        }
        Hop<HardEvent::S_V>();
        Sqrt(dist, dist, count);
        Cast(idx, winners, RoundMode::CAST_RINT, count);
        Emit(batch, start, count, dist, idx);
    }

    // Segment lanes: one point at a time, the whole segment set in vector
    // lanes, and ReduceMin supplying the tie-breaking argmin for free.
    __aicore__ inline void SegmentLanes(uint32_t batch, uint32_t start, uint32_t count)
    {
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        LocalTensor<float> ax = arena[oAx], ay = arena[oAy], az = arena[oAz];
        LocalTensor<float> ux = arena[oUx], uy = arena[oUy], uz = arena[oUz];
        LocalTensor<float> kx = arena[oKx], ky = arena[oKy], kz = arena[oKz];
        LocalTensor<float> dx = arena[oDx], dy = arena[oDy], dz = arena[oDz];
        LocalTensor<float> tt = arena[oTt], dsq = arena[oDsq];
        LocalTensor<float> res = arena[oRes], work = arena[oRedW];
        const uint32_t lanes = mAligned;

        for (uint32_t p = 0; p < count; ++p) {
            const float px = (float)raw.GetValue(3 * p);
            const float py = (float)raw.GetValue(3 * p + 1);
            const float pz = (float)raw.GetValue(3 * p + 2);
            Adds(dx, ax, -px, lanes);
            Adds(dy, ay, -py, lanes);
            Adds(dz, az, -pz, lanes);
            Mul(tt, dx, kx, lanes);
            MulAddDst(tt, dy, ky, lanes);
            MulAddDst(tt, dz, kz, lanes);
            Maxs(tt, tt, 0.0f, lanes);
            Mins(tt, tt, 1.0f, lanes);
            MulAddDst(dx, tt, ux, lanes);
            MulAddDst(dy, tt, uy, lanes);
            MulAddDst(dz, tt, uz, lanes);
            Mul(dsq, dx, dx, lanes);
            MulAddDst(dsq, dy, dy, lanes);
            MulAddDst(dsq, dz, dz, lanes);
            ReduceMin(res[2 * p], dsq, work, lanes, true);
        }

        LocalTensor<float> dist = arena[oDist];
        LocalTensor<float> idxFloat = arena[oIdx];
        LocalTensor<int32_t> idx = arena[oIdx].ReinterpretCast<int32_t>();
        const uint16_t repeats = (uint16_t)(AlignLanes(2 * count) / kLanes);
        uint64_t kept = 0;
        GatherMask(dist, res, (uint8_t)1, false, 0, {1, repeats, 8, 8}, kept);
        GatherMask(idxFloat, res, (uint8_t)2, false, 0, {1, repeats, 8, 8}, kept);
        Sqrt(dist, dist, count);
        Emit(batch, start, count, dist, idx);
    }

    // Point lanes: the point tile stays in vector lanes while segments are
    // consumed as scalars, so the instruction count follows M instead of N.
    __aicore__ inline void PointLanes(uint32_t batch, uint32_t start, uint32_t count)
    {
        const uint32_t wide = AlignLanes(count);
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        LocalTensor<float> ax = arena[oAx], ay = arena[oAy], az = arena[oAz];
        LocalTensor<float> ux = arena[oUx], uy = arena[oUy], uz = arena[oUz];
        LocalTensor<float> kx = arena[oKx], ky = arena[oKy], kz = arena[oKz];
        LocalTensor<float> px = arena[oPxv], py = arena[oPyv], pz = arena[oPzv];
        LocalTensor<float> dx = arena[oDx], dy = arena[oDy], dz = arena[oDz];
        LocalTensor<float> tt = arena[oTt], dsq = arena[oDsq];
        LocalTensor<float> best = arena[oBest], fidx = arena[oFidx];
        LocalTensor<uint32_t> off = arena[oPtOff].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> keep = arena[oMask].ReinterpretCast<uint8_t>();

        if (count != wide) {
            // Padding lanes hold finite coordinates so no tail lane makes NaN.
            Duplicate(px, 0.0f, wide);
            Duplicate(py, 0.0f, wide);
            Duplicate(pz, 0.0f, wide);
        }
        LocalTensor<float> src = arena[oPtRaw];
        if constexpr (sizeof(T) == 2) {
            src = arena[oPtFlt];
            Cast(src, raw, RoundMode::CAST_NONE, count * 3);
            PipeBarrier<PIPE_V>();
        }
        Gather(px, src, off, 0u, count);
        Gather(py, src, off, 4u, count);
        Gather(pz, src, off, 8u, count);
        // Keep the negated coordinate so each segment costs one Adds per axis.
        Muls(px, px, -1.0f, wide);
        Muls(py, py, -1.0f, wide);
        Muls(pz, pz, -1.0f, wide);
        Duplicate(fidx, 0.0f, wide);

        const int32_t total = (int32_t)m;
        for (int32_t j = 0; j < total; ++j) {
            const float sax = ax.GetValue(j);
            const float say = ay.GetValue(j);
            const float saz = az.GetValue(j);
            const float sux = ux.GetValue(j);
            const float suy = uy.GetValue(j);
            const float suz = uz.GetValue(j);
            const float skx = kx.GetValue(j);
            const float sky = ky.GetValue(j);
            const float skz = kz.GetValue(j);
            Adds(dx, px, sax, wide);
            Adds(dy, py, say, wide);
            Adds(dz, pz, saz, wide);
            Muls(tt, dx, skx, wide);
            Axpy(tt, dy, sky, wide);
            Axpy(tt, dz, skz, wide);
            Maxs(tt, tt, 0.0f, wide);
            Mins(tt, tt, 1.0f, wide);
            Axpy(dx, tt, sux, wide);
            Axpy(dy, tt, suy, wide);
            Axpy(dz, tt, suz, wide);
            if (j == 0) {
                Mul(best, dx, dx, wide);
                MulAddDst(best, dy, dy, wide);
                MulAddDst(best, dz, dz, wide);
            } else {
                Mul(dsq, dx, dx, wide);
                MulAddDst(dsq, dy, dy, wide);
                MulAddDst(dsq, dz, dz, wide);
                // GE keeps the incumbent on ties, so the smallest index wins.
                Compare(keep, dsq, best, CMPMODE::GE, wide);
                Min(best, best, dsq, wide);
                Select(fidx, keep, fidx, static_cast<float>(j), SELMODE::VSEL_TENSOR_SCALAR_MODE, wide);
            }
        }

        LocalTensor<float> dist = arena[oDist];
        LocalTensor<int32_t> idx = arena[oIdx].ReinterpretCast<int32_t>();
        Sqrt(dist, best, count);
        Cast(idx, fidx, RoundMode::CAST_RINT, count);
        Emit(batch, start, count, dist, idx);
    }

    __aicore__ inline void Emit(uint32_t batch, uint32_t start, uint32_t count,
                                const LocalTensor<float> &dist, const LocalTensor<int32_t> &idx)
    {
        if constexpr (sizeof(T) == 2) {
            LocalTensor<T> narrow = arena[oHalf].template ReinterpretCast<T>();
            Cast(narrow, dist, RoundMode::CAST_RINT, count);
            Hop<HardEvent::V_MTE3>();
            DataCopyPad(distances[(uint64_t)batch * n + start], narrow,
                        {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        } else {
            Hop<HardEvent::V_MTE3>();
            LocalTensor<T> plain = dist.template ReinterpretCast<T>();
            DataCopyPad(distances[(uint64_t)batch * n + start], plain,
                        {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        }
        DataCopyPad(indices[(uint64_t)batch * n + start], idx, {1, count * 4, 0, 0, 0});
    }
};

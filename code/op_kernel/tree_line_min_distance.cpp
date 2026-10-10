#include "kernel_operator.h"
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

using namespace AscendC;

namespace {
constexpr float kGuardDist = 3.0e38f;
// A guard start point this far out loses every reduction while its squared
// distance still fits comfortably inside fp32.
constexpr float kGuardCoord = 1.0e18f;
constexpr uint32_t kLanes = 64;      // fp32 elements in one 256B vector repeat
constexpr uint32_t kGroupCap = 64;   // points carried by one blocked call
constexpr uint32_t kBlockLanes = kGroupCap * kLanes;
constexpr uint32_t kTileCap = 512;   // points resident in one tile

// Single arena carved by hand. Segment lane parameters stay live across a
// whole batch; the derivation scratch is overlaid on the block work area,
// which is only written after the derivation finishes.
constexpr uint32_t oAx = 0, oAy = 1024, oAz = 2048;
constexpr uint32_t oUx = 3072, oUy = 4096, oUz = 5120;
constexpr uint32_t oKx = 6144, oKy = 7168, oKz = 8192;
constexpr uint32_t oDx = 9224, oDy = 13328, oDz = 17432, oTt = 21536, oDsq = 25640;
// derivation scratch, overlaid on the work area
constexpr uint32_t oSegRaw = 9216;    // 6144 T
constexpr uint32_t oSegFlt = 15360;   // 6144 float, fp16 input only
constexpr uint32_t oNegOne = 21504;   // 1024 float, constant -1
constexpr uint32_t oLen = 22528;      // 1024 float
constexpr uint32_t oInvLen = 23552;   // 1024 float
// persistent
constexpr uint32_t oSegOff = 29736;   // 1024 uint32, 24 * j
constexpr uint32_t oPtRaw = 30760;    // 1536 T
constexpr uint32_t oPtFlt = 32296;    // 1536 float
constexpr uint32_t oPtOff = 33832;    // 512 uint32, 12 * i
constexpr uint32_t oDiv8 = 34344;     // 512 uint32, 4 * (l / 8)
constexpr uint32_t oPx = 34856, oPy = 34920, oPz = 34984;
constexpr uint32_t oPxv = 35048, oPyv = 35560, oPzv = 36072;
constexpr uint32_t oPair = 36584;     // 128 float, value/index pairs
constexpr uint32_t oCmin = 36712, oCidx = 36776;
constexpr uint32_t oBest = 36840, oFidx = 37352;
constexpr uint32_t oRes = 37864;      // 1024 float
constexpr uint32_t oRedW = 38888;     // 512 float
constexpr uint32_t oDist = 39400;     // 512 float
constexpr uint32_t oIdx = 39912;      // 512 int32
constexpr uint32_t oHalf = 40424;     // 512 half
constexpr uint32_t oMask = 40680;     // 256B compare mask
constexpr uint32_t oArena = 40744;    // arena size in floats

__aicore__ inline uint32_t AlignLanes(uint32_t v)
{
    return (v + kLanes - 1) / kLanes * kLanes;
}

// One producer/consumer handshake. Draining a single queue costs far less
// than a full-pipeline barrier, which matters most on the shapes whose whole
// runtime is a handful of transfers and calls.
template <HardEvent E>
__aicore__ inline void Hop()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}
}  // namespace

template <typename T, uint32_t MODE, bool LARGE = false, bool SHORT = false, bool WIDE = false>
class KernelTreeLineMinDistance {
    static constexpr uint32_t oDx = WIDE ? 9224 : LARGE ? 9224 : ::oDx;
    static constexpr uint32_t oDy = WIDE ? 17424 : LARGE ? 15376 : ::oDy;
    static constexpr uint32_t oDz = WIDE ? 25624 : LARGE ? 21528 : ::oDz;
    static constexpr uint32_t oTt = WIDE ? 33824 : LARGE ? 27680 : ::oTt;
    static constexpr uint32_t oDsq = WIDE ? 9224 : LARGE ? 9224 : ::oDsq;
    static constexpr uint32_t oSegRaw = LARGE ? 9224 : ::oSegRaw;
    static constexpr uint32_t oSegFlt = LARGE ? 15376 : ::oSegFlt;
    static constexpr uint32_t oNegOne = LARGE ? 21528 : ::oNegOne;
    static constexpr uint32_t oLen = LARGE ? 22552 : ::oLen;
    static constexpr uint32_t oInvLen = LARGE ? 23576 : ::oInvLen;
    static constexpr uint32_t oSegOff = WIDE ? 42016 : LARGE ? 33824 : ::oSegOff;
    static constexpr uint32_t oPtRaw = WIDE ? 43040 : LARGE ? 34848 : ::oPtRaw;
    static constexpr uint32_t oPtFlt = WIDE ? 43424 : LARGE ? 36384 : ::oPtFlt;
    static constexpr uint32_t oPtOff = WIDE ? 43808 : LARGE ? 37920 : ::oPtOff;
    static constexpr uint32_t oDiv8 = WIDE ? 43936 : LARGE ? 38432 : ::oDiv8;
    static constexpr uint32_t oPx = WIDE ? 43936 : LARGE ? 39200 : ::oPx;
    static constexpr uint32_t oPy = WIDE ? 44064 : LARGE ? 39328 : ::oPy;
    static constexpr uint32_t oPz = WIDE ? 44192 : LARGE ? 39456 : ::oPz;
    static constexpr uint32_t oPxv = WIDE ? 44320 : LARGE ? 39584 : ::oPxv;
    static constexpr uint32_t oPyv = WIDE ? 44320 : LARGE ? 40352 : ::oPyv;
    static constexpr uint32_t oPzv = WIDE ? 44320 : LARGE ? 41120 : ::oPzv;
    static constexpr uint32_t oPair = WIDE ? 45344 : LARGE ? 41888 : ::oPair;
    static constexpr uint32_t oCmin = WIDE ? 45600 : LARGE ? 42144 : ::oCmin;
    static constexpr uint32_t oCidx = WIDE ? 45728 : LARGE ? 42272 : ::oCidx;
    static constexpr uint32_t oBest = WIDE ? 45856 : LARGE ? 42400 : ::oBest;
    static constexpr uint32_t oFidx = WIDE ? 45984 : LARGE ? 42912 : ::oFidx;
    static constexpr uint32_t oRes = WIDE ? 46112 : LARGE ? 43424 : ::oRes;
    static constexpr uint32_t oRedW = WIDE ? 46112 : LARGE ? 44448 : ::oRedW;
    static constexpr uint32_t oDist = WIDE ? 46112 : LARGE ? 44960 : ::oDist;
    static constexpr uint32_t oIdx = WIDE ? 46240 : LARGE ? 45472 : ::oIdx;
    static constexpr uint32_t oHalf = WIDE ? 46368 : LARGE ? 45984 : ::oHalf;
    static constexpr uint32_t oMask = WIDE ? 46432 : LARGE ? 46240 : ::oMask;
    static constexpr uint32_t oArena = WIDE ? 46496 : LARGE ? 46304 : ::oArena;
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    TBuf<TPosition::VECCALC> arenaBuf;
    LocalTensor<float> arena;
    static constexpr uint32_t mode = MODE;
    uint32_t b, n, m, chunks, chunkPoints, tilePoints, group;
    uint32_t mAligned, mRepeat;

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
                if (mode != 2) {
                    Hop<HardEvent::MTE2_V>();
                    DeriveSegments();
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
                    BlockedLanes(batch, begin + done, count);
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
        const BinaryRepeatParams spread(1, 1, 0, 8, 0, 1);
        const BinaryRepeatParams fold(1, 1, 1, 8, 8, 0);
        const BinaryRepeatParams plain(1, 1, 1, 8, 8, 8);
        const UnaryRepeatParams clamp(1, 1, 8, 8);

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
            if constexpr (!WIDE) {
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
                if constexpr (WIDE) {
                    PipeBarrier<PIPE_V>();
                    Brcb(pxv, px, (uint8_t)((span + 7) / 8), {1, 8});
                }
                Sub<float, false>(dx, ax[seg], pxv, MASK_PLACEHOLDER, reps, spread);
                if constexpr (WIDE) {
                    PipeBarrier<PIPE_V>();
                    Brcb(pxv, py, (uint8_t)((span + 7) / 8), {1, 8});
                }
                Sub<float, false>(dy, ay[seg], pyv, MASK_PLACEHOLDER, reps, spread);
                if constexpr (WIDE) {
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
                MulAddDst<float, float, false>(dsq, dy, dy, MASK_PLACEHOLDER, reps, plain);
                MulAddDst<float, float, false>(dsq, dz, dz, MASK_PLACEHOLDER, reps, plain);
                ResetMask();
                // Each repeat holds one point against this segment window.
                WholeReduceMin(pair, dsq, (int32_t)kLanes, (int32_t)reps, 1, 1, 8);
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

template <typename T>
__aicore__ inline void InvalidateInputs(const GlobalTensor<T> &tensor,
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
            InvalidateInputs(segments, (uint64_t)batch * t.m * 6, t.m * 6);
            cached = batch;
        }
        for (uint32_t done = 0; done < owned; done += t.tilePoints) {
            const uint32_t count = owned - done < t.tilePoints ? owned - done : t.tilePoints;
            if (stored) Hop<HardEvent::MTE3_S>();
            const uint64_t pointBase = ((uint64_t)batch * t.n + begin + done) * 3;
            InvalidateInputs(points, pointBase, count * 3);
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
    DataCopyPad(pt, points, {1, (uint32_t)(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(seg, segments, {1, (uint32_t)(SEG * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    Hop<HardEvent::MTE2_S>();
    const float px = (float)pt.GetValue(0);
    const float py = (float)pt.GetValue(1);
    const float pz = (float)pt.GetValue(2);
    float best = kGuardDist;
    int32_t winner = 0;
    for (uint32_t j = 0; j < SEG; ++j) {
        const float ax = (float)seg.GetValue(6 * j);
        const float ay = (float)seg.GetValue(6 * j + 1);
        const float az = (float)seg.GetValue(6 * j + 2);
        const float ux = (float)seg.GetValue(6 * j + 3) - ax;
        const float uy = (float)seg.GetValue(6 * j + 4) - ay;
        const float uz = (float)seg.GetValue(6 * j + 5) - az;
        float dx = px - ax, dy = py - ay, dz = pz - az;
        const float dot = dx * ux + dy * uy + dz * uz;
        const float len = ux * ux + uy * uy + uz * uz;
        if (len > 0.0f && dot > 0.0f) {
            const float ratio = dot < len ? dot / len : 1.0f;
            dx -= ratio * ux;
            dy -= ratio * uy;
            dz -= ratio * uz;
        }
        const float value = dx * dx + dy * dy + dz * dz;
        if (j == 0 || value < best) { best = value; winner = (int32_t)j; }
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

template <typename T, uint32_t POINTS, uint32_t SEGS>
__aicore__ inline void FixedSmallBatch(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i)
{
    if constexpr (SEGS == 2) ICachePreLoad(2);
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
    DataCopyPad(pt, points[(uint64_t)batch * n * 3],
                {1, (uint32_t)(n * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(seg, segments[(uint64_t)batch * segCount * 6],
                {1, (uint32_t)(segCount * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    Hop<HardEvent::MTE2_S>();
    #pragma unroll
    for (uint32_t point = 0; point < n; ++point) {
        const float px = (float)pt.GetValue(3 * point);
        const float py = (float)pt.GetValue(3 * point + 1);
        const float pz = (float)pt.GetValue(3 * point + 2);
        float best = kGuardDist;
        int32_t winner = 0;
        #pragma unroll
        for (uint32_t j = 0; j < segCount; ++j) {
            const float ax = (float)seg.GetValue(6 * j);
            const float ay = (float)seg.GetValue(6 * j + 1);
            const float az = (float)seg.GetValue(6 * j + 2);
            const float ux = (float)seg.GetValue(6 * j + 3) - ax;
            const float uy = (float)seg.GetValue(6 * j + 4) - ay;
            const float uz = (float)seg.GetValue(6 * j + 5) - az;
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
        dist.SetValue(point, best); idx.SetValue(point, winner);
    }
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

template <typename DT_TREE_POINTS, uint32_t PATH>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments,
                                                  GM_ADDR min_distance, GM_ADDR nearest_segment_idx,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    if constexpr (PATH >= 60 && PATH <= 73) {
        FixedSmallBatch<DT_TREE_POINTS, (PATH - 60) % 7 + 2, (PATH - 60) / 7 + 1>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 21 && PATH <= 26) {
        SinglePoint<DT_TREE_POINTS, PATH - 18>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH == 2) {
        SinglePoint<DT_TREE_POINTS, 1>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH == 3) {
        SinglePoint<DT_TREE_POINTS, 2>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, t, tiling);
        if constexpr (PATH >= 90 && PATH <= 95) {
            SmallScalarTask<DT_TREE_POINTS, PATH - 87>(tree_points, line_segments, min_distance, nearest_segment_idx, t);
        } else if constexpr (PATH == 1) {
            LegacyScalarTask<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx, t);
        } else if constexpr (PATH == 4) {
            ScalarTask<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx, t);
        } else {
            if constexpr (PATH == 10) ICachePreLoad(3);
            TPipe pipe;
            KernelTreeLineMinDistance<DT_TREE_POINTS, PATH == 5 ? 1 : (PATH == 6 || PATH == 50 || PATH == 51 || PATH == 80 ? 3 : 0), (PATH == 50 || PATH == 51), PATH == 80, PATH == 51> op;
            op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, t, &pipe);
            if constexpr (PATH == 10) op.ProcessOne();
            else op.Process();
        }
    }
}

#include "kernel_operator.h"
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

using namespace AscendC;

namespace {
constexpr float kGuardDist = 3.0e38f;
// A guard start point this far out loses every reduction while its squared
// distance still fits comfortably inside fp32.
constexpr float kGuardCoord = 1.0e18f;
constexpr uint32_t kLanes = 64;   // fp32 elements in one 256B vector repeat

// Single arena carved by hand. Segment lane parameters stay live across a
// whole batch, and the segment staging area is disjoint from the point tile
// so both input transfers can be in flight at the same time.
constexpr uint32_t oAx = 0, oAy = 1024, oAz = 2048;
constexpr uint32_t oUx = 3072, oUy = 4096, oUz = 5120;
constexpr uint32_t oKx = 6144, oKy = 7168, oKz = 8192;
constexpr uint32_t oSegRaw = 9216;    // 6144 T
constexpr uint32_t oSegFlt = 15360;   // 6144 float, fp16 input only
constexpr uint32_t oSegOff = 21504;   // 1024 uint32
constexpr uint32_t oNegOne = 22528;   // 1024 float, constant -1
constexpr uint32_t oLen = 23552;      // 1024 float
constexpr uint32_t oInvLen = 24576;   // 1024 float
constexpr uint32_t oPtRaw = 25600;    // 1536 T
constexpr uint32_t oPtOff = 27136;    // 512 uint32
constexpr uint32_t oPx = 27648, oPy = 28160, oPz = 28672;
constexpr uint32_t oDx = 29184, oDy = 30208, oDz = 31232;
constexpr uint32_t oTt = 32256, oDsq = 33280;
constexpr uint32_t oBest = 34304, oFidx = 34816;
constexpr uint32_t oRes = 35328;      // 1024 float
constexpr uint32_t oRedW = 36352;     // 512 float
constexpr uint32_t oDist = 36864;     // 512 float
constexpr uint32_t oIdx = 37376;      // 512 int32
constexpr uint32_t oHalf = 37888;     // 512 half
constexpr uint32_t oMask = 38144;     // 256B compare mask
constexpr uint32_t oArena = 38400;    // arena size in floats

__aicore__ inline uint32_t AlignLanes(uint32_t v)
{
    return (v + kLanes - 1) / kLanes * kLanes;
}
}  // namespace

template <typename T>
class KernelTreeLineMinDistance {
    GlobalTensor<T> points, segments, distances;
    GlobalTensor<int32_t> indices;
    TBuf<TPosition::VECCALC> arenaBuf;
    LocalTensor<float> arena;
    uint32_t b, n, m, mode, chunks, chunkPoints, tilePoints, mAligned;

public:
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                                const TreeLineMinDistanceTilingData &t, TPipe *pipe)
    {
        b = t.b;
        n = t.n;
        m = t.m;
        mode = t.mode;
        chunks = t.chunks;
        chunkPoints = t.chunkPoints;
        tilePoints = t.tilePoints;
        mAligned = AlignLanes(m);
        points.SetGlobalBuffer((__gm__ T *)p);
        segments.SetGlobalBuffer((__gm__ T *)s);
        distances.SetGlobalBuffer((__gm__ T *)d);
        indices.SetGlobalBuffer((__gm__ int32_t *)i);
        pipe->InitBuffer(arenaBuf, oArena * sizeof(float));
        arena = arenaBuf.Get<float>();
        if (mode == 2) {
            return;
        }
        // Lane-invariant tables and constants, built once per kernel.
        LocalTensor<float> scratch = arena[oLen];
        CreateVecIndex(scratch, 0.0f, mAligned);
        Muls(scratch, scratch, 24.0f, mAligned);
        Cast(arena[oSegOff].ReinterpretCast<int32_t>(), scratch, RoundMode::CAST_RINT, mAligned);
        if (mode == 1) {
            const uint32_t widest = AlignLanes(tilePoints);
            CreateVecIndex(scratch, 0.0f, widest);
            Muls(scratch, scratch, 12.0f, widest);
            Cast(arena[oPtOff].ReinterpretCast<int32_t>(), scratch, RoundMode::CAST_RINT, widest);
        }
        Duplicate(arena[oNegOne], -1.0f, mAligned);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tasks = b * chunks;
        const uint32_t stride = GetBlockNum();
        uint32_t cached = 0xFFFFFFFFu;
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
            // Both reads are issued before the barrier so the two global
            // memory latencies overlap instead of stacking.
            if (fresh) {
                LoadSegments(batch);
            }
            const uint32_t head = owned < tilePoints ? owned : tilePoints;
            LoadPoints(batch, begin, head);
            PipeBarrier<PIPE_ALL>();
            if (fresh) {
                cached = batch;
                if (mode != 2) {
                    DeriveSegments();
                    if (mode == 1) {
                        PipeBarrier<PIPE_ALL>();
                    }
                }
            }
            for (uint32_t done = 0; done < owned; done += tilePoints) {
                uint32_t count = owned - done;
                if (count > tilePoints) {
                    count = tilePoints;
                }
                if (done != 0) {
                    LoadPoints(batch, begin + done, count);
                    PipeBarrier<PIPE_ALL>();
                }
                if (mode == 2) {
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

    // Scalar lanes: the pair count one core owns is small enough that the
    // fixed cost of a vector instruction outweighs the whole reduction.
    __aicore__ inline void ScalarLanes(uint32_t batch, uint32_t start, uint32_t count)
    {
        LocalTensor<T> seg = arena[oSegRaw].template ReinterpretCast<T>();
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        LocalTensor<float> dist = arena[oDist];
        LocalTensor<int32_t> idx = arena[oIdx].ReinterpretCast<int32_t>();
        const int32_t total = (int32_t)m;
        for (uint32_t p = 0; p < count; ++p) {
            const float px = (float)raw.GetValue(3 * p);
            const float py = (float)raw.GetValue(3 * p + 1);
            const float pz = (float)raw.GetValue(3 * p + 2);
            float best = kGuardDist;
            int32_t winner = 0;
            for (int32_t j = 0; j < total; ++j) {
                const uint32_t base = 6u * (uint32_t)j;
                const float ax = (float)seg.GetValue(base);
                const float ay = (float)seg.GetValue(base + 1);
                const float az = (float)seg.GetValue(base + 2);
                const float ux = (float)seg.GetValue(base + 3) - ax;
                const float uy = (float)seg.GetValue(base + 4) - ay;
                const float uz = (float)seg.GetValue(base + 5) - az;
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
            idx.SetValue(p, winner);
        }
        PipeBarrier<PIPE_ALL>();
        Sqrt(dist, dist, count);
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
        LocalTensor<float> px = arena[oPx], py = arena[oPy], pz = arena[oPz];
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
            src = arena[oDx];
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
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(distances[(uint64_t)batch * n + start], narrow,
                        {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        } else {
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T> plain = dist.template ReinterpretCast<T>();
            DataCopyPad(distances[(uint64_t)batch * n + start], plain,
                        {1, (uint32_t)(count * sizeof(T)), 0, 0, 0});
        }
        DataCopyPad(indices[(uint64_t)batch * n + start], idx, {1, count * 4, 0, 0, 0});
    }
};

template <typename DT_TREE_POINTS>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments,
                                                  GM_ADDR min_distance, GM_ADDR nearest_segment_idx,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, t, tiling);
    TPipe pipe;
    KernelTreeLineMinDistance<DT_TREE_POINTS> op;
    op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, t, &pipe);
    op.Process();
}

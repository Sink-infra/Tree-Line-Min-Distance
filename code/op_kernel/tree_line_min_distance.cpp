#include "kernel_operator.h"
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

using namespace AscendC;

namespace {
// Coordinates are finite reals, so a guard start point this far away always
// loses the reduction while every intermediate square stays finite in fp32.
constexpr float kGuardCoord = 1.0e19f;
constexpr uint32_t kLanes = 64;   // fp32 elements in one 256B vector repeat

// Single arena carved by hand. Segment parameters stay live across a whole
// batch; the two views below reuse one overlay because segment preparation
// always completes before the point tiles of that batch run.
constexpr uint32_t oAx = 0, oAy = 1024, oAz = 2048;
constexpr uint32_t oUx = 3072, oUy = 4096, oUz = 5120;
constexpr uint32_t oKx = 6144, oKy = 7168, oKz = 8192;
constexpr uint32_t oOverlay = 9216;
// preparation view
constexpr uint32_t oSegRaw = oOverlay;            // 6144 T
constexpr uint32_t oSegFlt = oOverlay + 6144;     // 6144 float
constexpr uint32_t oSegOff = oOverlay + 12288;    // 1024 uint32
constexpr uint32_t oTmp0 = oOverlay + 13312;      // 1024 float
constexpr uint32_t oTmp1 = oOverlay + 14336;      // 1024 float
// tile view
constexpr uint32_t oPtRaw = oOverlay;             // 1536 T
constexpr uint32_t oPtOff = oOverlay + 1536;      // 512 uint32
constexpr uint32_t oPx = oOverlay + 2048, oPy = oOverlay + 2560, oPz = oOverlay + 3072;
constexpr uint32_t oDx = oOverlay + 3584, oDy = oOverlay + 4608, oDz = oOverlay + 5632;
constexpr uint32_t oTt = oOverlay + 6656, oDsq = oOverlay + 7680;
constexpr uint32_t oBest = oOverlay + 8704, oFidx = oOverlay + 9216;
constexpr uint32_t oRes = oOverlay + 9728;        // 1024 float
constexpr uint32_t oRedW = oOverlay + 10752;      // 512 float
constexpr uint32_t oDist = oOverlay + 11264;      // 512 float
constexpr uint32_t oIdx = oOverlay + 11776;       // 512 int32
constexpr uint32_t oHalf = oOverlay + 12288;      // 512 half
constexpr uint32_t oMask = oOverlay + 12544;      // 256B compare mask
constexpr uint32_t oArena = oOverlay + 15360;     // arena size in floats

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
            if (cached != batch) {
                PrepareSegments(batch);
                cached = batch;
            }
            for (uint32_t done = 0; done < owned; done += tilePoints) {
                uint32_t count = owned - done;
                if (count > tilePoints) {
                    count = tilePoints;
                }
                if (mode == 1) {
                    PointLanes(batch, begin + done, count);
                } else {
                    SegmentLanes(batch, begin + done, count);
                }
            }
        }
    }

private:
    // Loads one batch of segments, splits the interleaved endpoints into lane
    // arrays and derives the direction, the guarded inverse length and the
    // fused projection coefficients once for the whole batch.
    __aicore__ inline void PrepareSegments(uint32_t batch)
    {
        LocalTensor<T> raw = arena[oSegRaw].template ReinterpretCast<T>();
        DataCopyPad(raw, segments[(uint64_t)batch * m * 6],
                    {1, (uint32_t)(m * 6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

        LocalTensor<float> ax = arena[oAx], ay = arena[oAy], az = arena[oAz];
        LocalTensor<float> ux = arena[oUx], uy = arena[oUy], uz = arena[oUz];
        LocalTensor<float> kx = arena[oKx], ky = arena[oKy], kz = arena[oKz];
        LocalTensor<float> t0 = arena[oTmp0], t1 = arena[oTmp1];
        LocalTensor<int32_t> offInt = arena[oSegOff].ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> off = arena[oSegOff].ReinterpretCast<uint32_t>();

        if (mAligned != m) {
            // Tail lanes must never win the reduction and must stay finite.
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
        CreateVecIndex(t1, 0.0f, mAligned);
        Muls(t1, t1, 24.0f, mAligned);
        Cast(offInt, t1, RoundMode::CAST_RINT, mAligned);

        PipeBarrier<PIPE_ALL>();

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

        Duplicate(t0, 1.0f, mAligned);
        Mul(t0, ux, ux, m);
        MulAddDst(t0, uy, uy, m);
        MulAddDst(t0, uz, uz, m);
        // Exact zero-length test on the quantised value; no division by L = 0.
        LocalTensor<uint8_t> keep = arena[oMask].ReinterpretCast<uint8_t>();
        CompareScalar(keep, t0, 0.0f, CMPMODE::NE, mAligned);
        Select(t0, keep, t0, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mAligned);
        Duplicate(t1, -1.0f, mAligned);
        Div(t1, t1, t0, mAligned);
        // Degenerate segments keep u = 0 exactly, so k = 0 and t clamps to 0.
        Mul(kx, ux, t1, m);
        Mul(ky, uy, t1, m);
        Mul(kz, uz, t1, m);

        PipeBarrier<PIPE_ALL>();
    }

    // Segment lanes: one point at a time, the whole segment set in vector
    // lanes, and ReduceMin supplying the tie-breaking argmin for free.
    __aicore__ inline void SegmentLanes(uint32_t batch, uint32_t start, uint32_t count)
    {
        LocalTensor<T> raw = arena[oPtRaw].template ReinterpretCast<T>();
        DataCopyPad(raw, points[((uint64_t)batch * n + start) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

        LocalTensor<float> ax = arena[oAx], ay = arena[oAy], az = arena[oAz];
        LocalTensor<float> ux = arena[oUx], uy = arena[oUy], uz = arena[oUz];
        LocalTensor<float> kx = arena[oKx], ky = arena[oKy], kz = arena[oKz];
        LocalTensor<float> dx = arena[oDx], dy = arena[oDy], dz = arena[oDz];
        LocalTensor<float> tt = arena[oTt], dsq = arena[oDsq];
        LocalTensor<float> res = arena[oRes], work = arena[oRedW];

        PipeBarrier<PIPE_ALL>();

        for (uint32_t p = 0; p < count; ++p) {
            const float px = (float)raw.GetValue(3 * p);
            const float py = (float)raw.GetValue(3 * p + 1);
            const float pz = (float)raw.GetValue(3 * p + 2);
            Adds(dx, ax, -px, m);
            Adds(dy, ay, -py, m);
            Adds(dz, az, -pz, m);
            Mul(tt, dx, kx, m);
            MulAddDst(tt, dy, ky, m);
            MulAddDst(tt, dz, kz, m);
            Maxs(tt, tt, 0.0f, m);
            Mins(tt, tt, 1.0f, m);
            MulAddDst(dx, tt, ux, m);
            MulAddDst(dy, tt, uy, m);
            MulAddDst(dz, tt, uz, m);
            Mul(dsq, dx, dx, m);
            MulAddDst(dsq, dy, dy, m);
            MulAddDst(dsq, dz, dz, m);
            ReduceMin(res[2 * p], dsq, work, m, true);
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
        DataCopyPad(raw, points[((uint64_t)batch * n + start) * 3],
                    {1, (uint32_t)(count * 3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

        LocalTensor<float> ax = arena[oAx], ay = arena[oAy], az = arena[oAz];
        LocalTensor<float> ux = arena[oUx], uy = arena[oUy], uz = arena[oUz];
        LocalTensor<float> kx = arena[oKx], ky = arena[oKy], kz = arena[oKz];
        LocalTensor<float> px = arena[oPx], py = arena[oPy], pz = arena[oPz];
        LocalTensor<float> dx = arena[oDx], dy = arena[oDy], dz = arena[oDz];
        LocalTensor<float> tt = arena[oTt], dsq = arena[oDsq];
        LocalTensor<float> best = arena[oBest], fidx = arena[oFidx];
        LocalTensor<int32_t> offInt = arena[oPtOff].ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> off = arena[oPtOff].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> keep = arena[oMask].ReinterpretCast<uint8_t>();

        CreateVecIndex(tt, 0.0f, wide);
        Muls(tt, tt, 12.0f, wide);
        Cast(offInt, tt, RoundMode::CAST_RINT, wide);
        // Padding lanes hold finite coordinates so no tail lane produces NaN.
        Duplicate(px, 0.0f, wide);
        Duplicate(py, 0.0f, wide);
        Duplicate(pz, 0.0f, wide);

        PipeBarrier<PIPE_ALL>();

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

        for (int32_t j = 0; j < (int32_t)m; ++j) {
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
            LocalTensor<T> wide = dist.template ReinterpretCast<T>();
            DataCopyPad(distances[(uint64_t)batch * n + start], wide,
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

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

#include "tree_line_min_distance_spatial.h"
#include "tree_line_min_distance_dense_points.h"

#include "tree_line_min_distance_segment_lanes.h"

#include "tree_line_min_distance_point_groups.h"

#include "tree_line_min_distance_scalar.h"

#include "tree_line_min_distance_small.h"

#include "tree_line_min_distance_single_axis.h"
#include "tree_line_min_distance_affine.h"
#include "tree_line_min_distance_shared_grid.h"

template <typename DT_TREE_POINTS, uint32_t PATH>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments,
                                                  GM_ADDR min_distance, GM_ADDR nearest_segment_idx,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    if constexpr (PATH == 220) {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, shortTiling, tiling);
        TPipe pipe;
        KernelPointGroups<DT_TREE_POINTS, 3, false, true, false, false, false, false,
            false, false, false, true> op;
        op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, shortTiling, &pipe);
        op.Process();
        return;
    }
    if constexpr (PATH >= 221 && PATH <= 226) {
        GET_TILING_DATA_MEMBER(TreeLineMinDistanceTilingData, n, numPoints, tiling);
        InitSocState();
        SmallScalarTwoPoints<DT_TREE_POINTS, PATH - 218>(tree_points, line_segments,
            min_distance, nearest_segment_idx, numPoints);
        return;
    }
    if constexpr (PATH == 252 || PATH == 253) {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, sharedTiling, tiling);
        {
            TPipe sharedPipe;
            TBuf<TPosition::VECCALC> sharedStorage;
            sharedPipe.InitBuffer(sharedStorage, SharedGridSearch::kArenaBytes);
            SharedGridSearch::Search<DT_TREE_POINTS,PATH == 253> search;
            if (search.Run(tree_points,line_segments,min_distance,nearest_segment_idx,
                           GetUserWorkspace(workspace),sharedTiling)) return;
        }
        PipeBarrier<PIPE_ALL>();
        TPipe fallbackPipe;
        TBuf<TPosition::VECCALC> fallbackStorage;
        fallbackPipe.InitBuffer(fallbackStorage,SharedGridSearch::kArenaBytes);
        SharedGridSearch::Historical<DT_TREE_POINTS>(tree_points,line_segments,
            min_distance,nearest_segment_idx,sharedTiling);
        return;
    }

    if constexpr (PATH == 244) {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, neighborTiling, tiling);
        InitSocState();
        KernelPointGroups<DT_TREE_POINTS, 3, true, false, true, true, false, true, false, false, true> neighbor;
        neighbor.Init(tree_points, line_segments, min_distance, nearest_segment_idx, neighborTiling, nullptr);
        neighbor.Process();
        return;
    }
    if constexpr (PATH == 251) {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, modelTiling, tiling);
        AffineOrFallback<DT_TREE_POINTS>(tree_points, line_segments, min_distance,
                                        nearest_segment_idx, modelTiling);
        return;
    }
    if constexpr (PATH == 241) {
        InitSocState();
        SinglePoint<DT_TREE_POINTS, 2, true, false, true, true>(
            tree_points, line_segments, min_distance, nearest_segment_idx);
        return;
    }
    if constexpr (PATH >= 233 && PATH <= 240) {
        InitSocState();
        TinySingleSegmentOverlap<DT_TREE_POINTS, PATH - 232, 1, true, false, true>(
            tree_points, line_segments, min_distance, nearest_segment_idx);
        return;
    }
    if constexpr (PATH >= 194 && PATH <= 201) {
        InitSocState();
        SinglePoint<DT_TREE_POINTS, PATH - 193, true, false, true>(tree_points, line_segments, min_distance, nearest_segment_idx);
        return;
    }
    if constexpr (PATH >= 202 && PATH <= 215) {
        InitSocState();
        FixedSmallBatch<DT_TREE_POINTS, (PATH - 202) % 7 + 2, (PATH - 202) / 7 + 1, true, false, true>(tree_points, line_segments, min_distance, nearest_segment_idx);
        return;
    }
    if constexpr (PATH == 1 || PATH == 2 || PATH == 3 || PATH == 4 ||
        (PATH >= 21 && PATH <= 26) || (PATH >= 60 && PATH <= 73) ||
        (PATH >= 90 && PATH <= 95) || (PATH >= 98 && PATH <= 99) ||
        (PATH >= 100 && PATH <= 113) || (PATH >= 120 && PATH <= 125) ||
        PATH == 156) {
        InitSocState();
    }
    if constexpr (PATH == 157 || PATH == 158 || PATH == 159) {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, spatialTiling, tiling);
        {
            TPipe candidatePipe;
            TBuf<TPosition::VECCALC> candidateStorage;
            candidatePipe.InitBuffer(candidateStorage, 150528);
            if (TreeDistanceSpatial::Run<DT_TREE_POINTS, true>(tree_points, line_segments,
                min_distance, nearest_segment_idx, spatialTiling)) return;
        }
        PipeBarrier<PIPE_ALL>();
        TPipe pipe;
        if (spatialTiling.chunkPoints < 1536) {
            auto blockedTiling = spatialTiling;
            blockedTiling.group = HoistedPointGroupLayout::kPoints;
            blockedTiling.tilePoints = HoistedPointGroupLayout::kPoints;
            KernelPointGroups<DT_TREE_POINTS, 3, true, false, true, true, true, false, true, true> op;
            op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, blockedTiling, &pipe);
            op.Process();
        } else {
            TBuf<TPosition::VECCALC> denseStorage;
            pipe.InitBuffer(denseStorage, 188416);
            DensePointTask<DT_TREE_POINTS>(tree_points, line_segments,
                min_distance, nearest_segment_idx, spatialTiling);
        }
        return;
    }
    if constexpr (PATH == 156) {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, spatialTiling, tiling);
        if (TreeDistanceSpatial::Run<DT_TREE_POINTS>(tree_points, line_segments,
            min_distance, nearest_segment_idx, spatialTiling)) return;
        PipeBarrier<PIPE_ALL>();
        TPipe pipe;
        KernelPointGroups<DT_TREE_POINTS, 3, true, false, true, true, false, false, true> op;
        op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, spatialTiling, &pipe);
        op.Process();
        return;
    }
    if constexpr (PATH >= 140 && PATH <= 153) {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, 512);
        FixedSmallBatch<DT_TREE_POINTS, (PATH - 140) % 7 + 2, (PATH - 140) / 7 + 1, false, true>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 130 && PATH <= 131) {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, 512);
        SinglePoint<DT_TREE_POINTS, PATH - 129, false, true>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 100 && PATH <= 113) {
        FixedSmallBatch<DT_TREE_POINTS, (PATH - 100) % 7 + 2, (PATH - 100) / 7 + 1, true>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 98 && PATH <= 99) {
        SinglePoint<DT_TREE_POINTS, PATH - 97, true>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 120 && PATH <= 125) {
        SinglePoint<DT_TREE_POINTS, PATH - 117, true, false, true>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 60 && PATH <= 73) {
        FixedSmallBatch<DT_TREE_POINTS, (PATH - 60) % 7 + 2, (PATH - 60) / 7 + 1>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 21 && PATH <= 26) {
        SinglePoint<DT_TREE_POINTS, PATH - 18>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH == 2) {
        SinglePoint<DT_TREE_POINTS, 1>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH == 3) {
        SinglePoint<DT_TREE_POINTS, 2>(tree_points, line_segments, min_distance, nearest_segment_idx);
    } else {
        GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, t, tiling);
        if constexpr (PATH == 128) {
            InitSocState();
            FusedPointSegments<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx, t.m);
        } else if constexpr (PATH == 53) {
            TPipe pipe;
            TBuf<TPosition::VECCALC> storage;
            pipe.InitBuffer(storage, 61728);
            SingleSegmentTask<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx, t);
        } else if constexpr (PATH >= 90 && PATH <= 95) {
            SmallScalarTask<DT_TREE_POINTS, PATH - 87>(tree_points, line_segments, min_distance, nearest_segment_idx, t);
        } else if constexpr (PATH == 1) {
            LegacyScalarTask<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx, t);
        } else if constexpr (PATH == 4) {
            ScalarTask<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx, t);
        } else {
            if constexpr (PATH == 10) ICachePreLoad(3);
            TPipe pipe;
            if constexpr (PATH != 5 && PATH != 10) {
            KernelPointGroups<DT_TREE_POINTS, PATH == 5 ? 1 : (PATH == 6 || PATH == 50 || PATH == 51 || PATH == 54 || PATH == 55 || PATH == 57 || PATH == 129 || PATH == 155 || PATH == 80 ? 3 : 0), (PATH == 50 || PATH == 51 || PATH == 54 || PATH == 55 || PATH == 57 || PATH == 129 || PATH == 155), PATH == 80, (PATH == 51 || PATH == 54 || PATH == 55 || PATH == 57 || PATH == 155), (PATH == 54 || PATH == 55 || PATH == 57 || PATH == 129 || PATH == 155), PATH == 55, PATH == 57, PATH == 155> op;
            op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, t, &pipe);
            if constexpr (PATH == 10) op.ProcessOne();
            else op.Process();
            } else {
            KernelTreeLineMinDistance<DT_TREE_POINTS, PATH == 5 ? 1 : (PATH == 6 || PATH == 50 ? 3 : 0), PATH == 50> op;
            op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, t, &pipe);
            if constexpr (PATH == 10) op.ProcessOne();
            else op.Process();
            }
        }
    }
}

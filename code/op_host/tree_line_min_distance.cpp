#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"
namespace {
constexpr uint32_t kLanes = 64;
constexpr uint32_t kTileCap = 512;
constexpr uint32_t kGroupCap = 64;
// Cost constants in AIV cycles, calibrated from PipeUtilization samples on
// ascend910b: a level-2 vector call costs a fixed issue overhead plus a few
// cycles per 256B repeat, and one scalar point/segment pair costs about as
// much as a short dependent float chain.
constexpr uint64_t kCall = 32;
constexpr uint64_t kRepeat = 3;
constexpr uint64_t kPair = 60;
// Measured cost of adding one more block to the launch: an empty kernel grows
// from 1.76us at one block to 3.60us at forty-eight.
constexpr uint64_t kBlock = 74;
// The blocked cost below tracks the call and repeat counts but not everything
// a group pays per point, and it drifts low enough to trade too much
// parallelism away: (1,33,31) runs in 4.12us across eleven blocks and 5.54us
// across four. So a blocked split also has to keep a floor of blocks busy,
// clamped to what the shape can actually supply. The other orientations are
// priced from measurements that held, and are left to the model alone.
constexpr uint32_t kBlockFloor = 10;
// Calls outside the inner loops: segment derivation, and the per-tile
// gather/reduce/emit tails of each orientation.
constexpr uint64_t kDeriveCalls = 18;
constexpr uint64_t kSegTailCalls = 3;
constexpr uint64_t kPointSetupCalls = 10;
constexpr uint64_t kScalarTailCalls = 4;
// One blocked chunk issues the fourteen lane operations, the window
// reduction and the carry across windows; one blocked point group also pays
// the coordinate spread and its own tail.
constexpr uint64_t kChunkCalls = 15;
constexpr uint64_t kCarryCalls = 7;
constexpr uint64_t kGroupCalls = 9;
bool Valid(const gert::Shape &p, const gert::Shape &s) {
    return p.GetDimNum()==3 && s.GetDimNum()==4 &&
        p.GetDim(0)>=1 && p.GetDim(0)<=8 && p.GetDim(0)==s.GetDim(0) &&
        p.GetDim(1)>=1 && p.GetDim(1)<=16384 && p.GetDim(2)==3 &&
        s.GetDim(1)>=1 && s.GetDim(1)<=1024 && s.GetDim(2)==2 && s.GetDim(3)==3;
}
inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
}
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto &p=context->GetInputShape(0)->GetStorageShape();
    const auto &s=context->GetInputShape(1)->GetStorageShape();
    const auto dtype=context->GetInputDesc(0)->GetDataType();
    if (!Valid(p,s) || dtype!=context->GetInputDesc(1)->GetDataType() ||
        (dtype!=ge::DT_FLOAT && dtype!=ge::DT_FLOAT16)) return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *t=context->GetTilingData<TreeLineMinDistanceTilingData>();
    const uint32_t b=static_cast<uint32_t>(p.GetDim(0));
    const uint32_t n=static_cast<uint32_t>(p.GetDim(1));
    const uint32_t m=static_cast<uint32_t>(s.GetDim(1));
    uint32_t cores=platform.GetCoreNumAiv();
    if(cores==0) cores=1;

    // Choose how many point chunks each batch is split into together with the
    // orientation, because the two trade against each other: more chunks keep
    // more cores busy but shrink the group a blocked call can amortise over,
    // and every extra block costs a fixed slice of the launch.
    uint32_t span=CeilDiv(cores,b);
    if(span>n) span=n;
    if(span==0) span=1;
    uint32_t chunks=span, chunkPoints=CeilDiv(n,span), tilePoints=1, group=1, mode=0;
    uint64_t cheapest=~0ULL;
    const uint32_t reach=b*span;
    const uint32_t least=kBlockFloor<reach?kBlockFloor:reach;
    for(uint32_t probe=1;probe<=span;++probe) {
        const uint32_t owned=CeilDiv(n,probe);
        const uint32_t split=CeilDiv(n,owned);
        const bool spread=b*split>=least;
        const uint32_t tile=owned<kTileCap?owned:kTileCap;
        const uint32_t pack=kGroupCap<owned?kGroupCap:owned;
        const uint64_t tiles=CeilDiv(owned,tile);
        const uint64_t segRepeat=CeilDiv(m,kLanes);
        const uint64_t pointRepeat=CeilDiv(tile,kLanes);
        const uint64_t segCall=kCall+kRepeat*segRepeat;
        const uint64_t derive=kDeriveCalls*segCall;
        const uint64_t costSegLanes=(kDeriveCalls+15ULL*owned)*segCall+tiles*kSegTailCalls*kCall;
        const uint64_t costPointLanes=derive+tiles*kPointSetupCalls*kCall+
            17ULL*m*tiles*(kCall+kRepeat*pointRepeat);
        const uint64_t costScalarLanes=(uint64_t)owned*m*kPair+tiles*kScalarTailCalls*kCall;
        // The repeat count a blocked call issues follows the points a core
        // owns, while the fixed issue cost follows the number of groups, so
        // the two are priced separately instead of rounding the work up to a
        // whole group.
        const uint64_t groups=CeilDiv(owned,pack);
        const uint64_t chunkCost=kChunkCalls*(kCall*groups+kRepeat*owned)+
            kCarryCalls*kCall*groups;
        const uint64_t costBlocked=derive+segRepeat*chunkCost+kGroupCalls*kCall*groups;
        uint32_t pick=0;
        uint64_t inner=costSegLanes;
        if(costPointLanes<inner) { inner=costPointLanes; pick=1; }
        if(costScalarLanes<inner) { inner=costScalarLanes; pick=2; }
        if(spread&&costBlocked<inner) { inner=costBlocked; pick=3; }
        const uint64_t total=kBlock*(uint64_t)b*split+inner;
        if(total<cheapest) {
            cheapest=total;
            chunks=split; chunkPoints=owned; tilePoints=tile; group=pack; mode=pick;
        }
    }
    if (n <= 8 && m >= 9 && m <= 64 && b * n <= cores) {
        mode = 3; chunks = 1; chunkPoints = n; tilePoints = n; group = n;
    }
    t->mode=mode;
    t->group=group;

    t->b=b; t->n=n; t->m=m;
    t->chunks=chunks; t->chunkPoints=chunkPoints; t->tilePoints=tilePoints;
    const uint32_t tasks=b*chunks;
    context->SetBlockDim(tasks<cores?tasks:cores);
    uint32_t DT_TREE_POINTS=static_cast<uint32_t>(dtype);
    uint32_t PATH = mode == 1 ? 5 : (mode == 3 ? 6 : 0);
    if (mode == 2) PATH = m > 2 ? 1 : (b * n == 1 ? m + 1 : 4);
    if (b * n == 1 && mode == 0) PATH = 10;
    if (b * n == 1 && m > 2 && m <= 8) PATH = m + 18;
    if (mode == 3 && chunkPoints >= 96 && m >= 128) {
        t->group = 96;
        PATH = 50;
    }
    if (mode == 3 && chunkPoints >= 128 && m >= 128) {
        t->group = 128; t->tilePoints = 128;
        PATH = 54;
    }
    if (n > 1 && n <= 8 && m <= 2) { PATH = 60 + (m - 1) * 7 + n - 2; context->SetBlockDim(b); }
    if (n <= 8 && m >= 9 && m <= 64 && b * n <= cores) PATH = 80;
    if (mode == 3 && m <= kLanes) PATH = 80;
    if (mode == 2 && m >= 3 && m <= 8 && b * n > 1) PATH = 87 + m;
    if (mode == 3 && chunkPoints >= 32 && chunkPoints < 64 && m >= 128) {
        t->group = chunkPoints < 64 ? chunkPoints : 64;
        t->tilePoints = t->group;
        PATH = 57;
    }
    if (m == 1 && mode == 1) {
        PATH = 53;
        t->chunks = CeilDiv(n, 1024); t->chunkPoints = 1024;
        t->tilePoints = n < 1024 ? n : 1024;
        const uint32_t singleTasks = b * t->chunks;
        context->SetBlockDim(singleTasks < cores ? singleTasks : cores);
    }
    if (b * n == 1 && mode == 0) PATH = 128;
    if (mode == 3 && chunkPoints >= 384 && m >= 128) PATH = 155;
    if (b * n == 1 && m >= 3 && m <= 8) PATH = 117 + m;
    if (n > 1 && n <= 8 && m == 1) PATH = 100 + n - 2;
    if (b * n == 1 && m <= 2) PATH = m + 1;
    if (mode == 3 && chunkPoints >= 768 && m >= 128) PATH = 159;
    ASCENDC_TPL_SEL_PARAM(context,DT_TREE_POINTS,PATH);
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(TreeLineMinDistance).Tiling(TilingFunc);
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const auto &p=*context->GetInputShape(0);
    const auto &s=*context->GetInputShape(1);
    if (!Valid(p,s)) return GRAPH_FAILED;
    for (int i=0;i<2;++i) {
        auto *out=context->GetOutputShape(i);
        out->SetDimNum(2); out->SetDim(0,p.GetDim(0)); out->SetDim(1,p.GetDim(1));
    }
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto dtype=context->GetInputDataType(0);
    if (dtype!=context->GetInputDataType(1) || (dtype!=DT_FLOAT && dtype!=DT_FLOAT16)) return GRAPH_FAILED;
    context->SetOutputDataType(0,dtype);
    context->SetOutputDataType(1,DT_INT32);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class TreeLineMinDistance:public OpDef {
public:
    explicit TreeLineMinDistance(const char *name):OpDef(name) {
        this->Input("tree_points").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Input("line_segments").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("min_distance").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("nearest_segment_idx").ParamType(REQUIRED).DataType({ge::DT_INT32,ge::DT_INT32}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(TreeLineMinDistance);
}

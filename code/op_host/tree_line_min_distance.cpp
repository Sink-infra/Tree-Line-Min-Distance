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
constexpr uint64_t kCall = 75;
constexpr uint64_t kRepeat = 4;
constexpr uint64_t kPair = 150;
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

    // Spread point chunks over the batch so every core owns one contiguous run
    // of points inside a single batch and prepares its segments only once.
    uint32_t chunks=CeilDiv(cores,b);
    if(chunks>n) chunks=n;
    if(chunks==0) chunks=1;
    uint32_t chunkPoints=CeilDiv(n,chunks);
    chunks=CeilDiv(n,chunkPoints);
    uint32_t tilePoints=chunkPoints<kTileCap?chunkPoints:kTileCap;

    // Four orientations priced on the same model, picked by the work one
    // core actually owns. No branch here looks at a particular shape.
    const uint64_t tiles=CeilDiv(chunkPoints,tilePoints);
    const uint64_t segRepeat=CeilDiv(m,kLanes);
    const uint64_t pointRepeat=CeilDiv(tilePoints,kLanes);
    const uint64_t segCall=kCall+kRepeat*segRepeat;
    const uint64_t pointCall=kCall+kRepeat*pointRepeat;
    const uint64_t costSegLanes=(kDeriveCalls+15ULL*chunkPoints)*segCall+tiles*kSegTailCalls*kCall;
    const uint64_t costPointLanes=kDeriveCalls*segCall+tiles*kPointSetupCalls*kCall+
        17ULL*m*tiles*pointCall;
    const uint64_t costScalarLanes=(uint64_t)chunkPoints*m*kPair+tiles*kScalarTailCalls*kCall;

    // Blocked lanes amortise the fixed issue cost of a call over a whole
    // group of points measured against the same 64-segment window.
    uint32_t group=kGroupCap<chunkPoints?kGroupCap:chunkPoints;
    const uint64_t chunkCost=kChunkCalls*(kCall+kRepeat*group)+kCarryCalls*kCall;
    const uint64_t groupCost=segRepeat*chunkCost+kGroupCalls*kCall;
    const uint64_t costBlocked=kDeriveCalls*segCall+CeilDiv(chunkPoints,group)*groupCost;

    uint32_t mode=0;
    uint64_t best=costSegLanes;
    if(costPointLanes<best) { best=costPointLanes; mode=1; }
    if(costScalarLanes<best) { best=costScalarLanes; mode=2; }
    if(costBlocked<best) { mode=3; }
    t->mode=mode;
    t->group=group;

    t->b=b; t->n=n; t->m=m;
    t->chunks=chunks; t->chunkPoints=chunkPoints; t->tilePoints=tilePoints;
    const uint32_t tasks=b*chunks;
    context->SetBlockDim(tasks<cores?tasks:cores);
    uint32_t DT_TREE_POINTS=static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context,DT_TREE_POINTS);
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

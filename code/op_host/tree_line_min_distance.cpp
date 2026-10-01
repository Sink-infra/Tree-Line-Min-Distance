// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_tree_points = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_line_segments = context->GetRequiredInputTensor(1);
        ge::DataType dtype_tree_points = tensor_tree_points->GetDataType(); // 获取数据类型
        int dtype_size_tree_points = ge::GetSizeByDataType(dtype_tree_points); // 获取数据类型的字长
        uint32_t length_tree_points = tensor_tree_points->GetShapeSize(); // 获取元素个数
        uint32_t size_tree_points = tensor_tree_points->GetSize(); // 获取内存大小
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_TREE_POINTS = static_cast<uint32_t>(dtype_tree_points);
        ASCENDC_TPL_SEL_PARAM(context, DT_TREE_POINTS);
        // 示例: 计算tiling方案并填充tiling结构体
        TreeLineMinDistanceTilingData *tiling = context->GetTilingData<TreeLineMinDistanceTilingData>();
        tiling->length = length_tree_points;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class TreeLineMinDistance : public OpDef {
    public:
        explicit TreeLineMinDistance(const char *name) : OpDef(name) {
            this->Input("tree_points")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("line_segments")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("min_distance")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("nearest_segment_idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(TreeLineMinDistance);
}  // namespace ops

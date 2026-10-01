// Kernel侧核函数实现
#include "kernel_operator.h"

#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

template <class DT_TREE_POINTS>
class KernelTreeLineMinDistance {
public:
    __aicore__ inline KernelTreeLineMinDistance() {}
    __aicore__ inline void Init(GM_ADDR tree_points, GM_ADDR line_segments, GM_ADDR min_distance, GM_ADDR nearest_segment_idx, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_TREE_POINTS>
 __global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments, GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, tiling_data, tiling);
    KernelTreeLineMinDistance<DT_TREE_POINTS> op;
    op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, tiling_data.length);
    op.Process();
}
